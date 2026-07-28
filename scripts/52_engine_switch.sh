#!/usr/bin/env bash
# Transactional profile switch for the unchanged :8010 auth chain.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly PROD_STATE=/home/dsv4/ds4-project/engine-switch
readonly PROD_SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
readonly PROD_GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly PROFILE_MANIFEST=$REPO/configs/glm52-profile.json
readonly PORT=8013
readonly AUTH_PORT=8010

if [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]]; then
    [[ -n ${ENGINE_SWITCH_TEST_ROOT:-} ]] || {
        echo "ENGINE_SWITCH_TEST_ROOT is required" >&2; exit 2;
    }
    STATE=$ENGINE_SWITCH_TEST_ROOT
    SRC=$STATE/source
    GGUF=$STATE/model.gguf
else
    unset ENGINE_SWITCH_TEST_ROOT ENGINE_PORT DS4_GLM_TOPK_KEEP \
        DS4_GLM_TOPK_SKIP_LOAD DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START || true
    STATE=$PROD_STATE
    SRC=$PROD_SRC
    GGUF=$PROD_GGUF
fi
readonly STATE SRC GGUF
readonly ACTIVE=$STATE/active.json
readonly LOCK=$STATE/switch.lock
readonly GLM_PROCESS=$STATE/glm52.process.json
readonly GLM_WATCHDOG_TARGET=$STATE/glm52.memwatch.target
readonly GLM_WATCHDOG_READY=$STATE/glm52.memwatch.ready
readonly GLM_WATCHDOG_LOG=$STATE/glm52.memwatch.log
rollback_needed=false
previous_profile=

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

clean_python() {
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        /usr/bin/python3 "$@"
}

clean_curl() {
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        /usr/bin/curl --disable "$@"
}

dsv4_launcher() {
    install -d -o dsv4 -g dsv4 -m 0700 /run/dsv4
    /usr/sbin/runuser -u dsv4 -- env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        DSV4_PORT="$PORT" \
        DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
        DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=14 \
        DSV4_MEM_FLOOR_GIB=14 DSV4_WATCHDOG_FLOOR_GIB=14 \
        DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=3 \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START=1 \
        DSV4_UBATCH=256 DSV4_BATCH=512 DSV4_UBATCH_LARGE=0 \
        CTX=1048576 DSV4_PARALLEL=1 DSV4_NO_MMAP=1 \
        DSV4_SPEC_TYPE=none \
        "$REPO/scripts/21_serve_llamacpp.sh" "$@"
}

sha256() { sha256sum -- "$1" | awk '{print $1}'; }

json_status() {
    clean_python - "$ACTIVE" <<'PY'
import json, os, sys
path = sys.argv[1]
profile = None
state = "inactive"
try:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") == 1 and value.get("profile") in {"dsv4", "glm52"}:
        profile = value["profile"]
        state = "recorded"
except (OSError, ValueError, TypeError):
    pass
print(json.dumps({"schema_version": 1, "active_profile": profile, "state": state},
                 separators=(",", ":"), sort_keys=True))
PY
}

read_active_profile() {
    clean_python - "$ACTIVE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        value = json.load(stream)
    profile = value.get("profile")
    print(profile if profile in {"dsv4", "glm52"} else "")
except (OSError, ValueError, TypeError):
    print("")
PY
}

glm_qualified() {
    [[ ${ENGINE_SWITCH_TESTING:-0} != 1 ]] || return 1
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        "$REPO/scripts/glm52_goal.py" release-check --json >/dev/null
}

verify_glm_hashes() {
    clean_python - "$PROFILE_MANIFEST" "$SRC/ds4-server" "$GGUF" <<'PY'
import hashlib, json, sys
manifest_path, binary_path, model_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
if digest(binary_path) != manifest["binary_sha256"]:
    raise SystemExit("GLM binary hash is not approved")
if digest(model_path) != manifest["model_sha256"]:
    raise SystemExit("GLM model hash is not approved")
if manifest.get("context_cap") != 1048576:
    raise SystemExit("GLM profile context cap is not 1048576")
PY
}

proc_identity() {
    local pid=$1 line
    [[ $pid =~ ^[0-9]+$ && $pid -gt 1 && -r /proc/$pid/stat ]] || return 1
    IFS= read -r line <"/proc/$pid/stat" || return 1
    line=${line##*) }
    local -a fields
    read -r -a fields <<<"$line"
    [[ ${fields[2]} =~ ^[0-9]+$ && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    printf '%s %s\n' "${fields[2]}" "${fields[19]}"
}

stop_glm_verified() {
    [[ -f $GLM_PROCESS ]] || return 0
    local values pid expected_pgid expected_ticks expected_sha memwatch_pid
    local memwatch_ticks current current_pgid current_ticks exe cmdline
    values=$(clean_python - "$GLM_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["exe_sha256"],
      value["memwatch_pid"], value["memwatch_start_ticks"])
PY
    ) || die "invalid GLM process record"
    read -r pid expected_pgid expected_ticks expected_sha memwatch_pid \
        memwatch_ticks <<<"$values"
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if [[ -n $current ]]; then
        read -r current_pgid current_ticks <<<"$current"
        exe=$(readlink -f "/proc/$pid/exe") ||
            die "cannot resolve recorded GLM executable"
        [[ $current_pgid == "$expected_pgid" &&
                $current_ticks == "$expected_ticks" ]] ||
            die "stale GLM PID identity; refusing to signal"
        [[ $(sha256 "$exe") == "$expected_sha" ]] ||
            die "GLM executable hash changed; refusing to signal"
        kill -TERM -- "-$expected_pgid"
        for _ in $(seq 1 600); do
            [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] && break
            sleep 0.1
        done
        if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$current" ]]; then
            kill -KILL -- "-$expected_pgid"
        fi
    fi
    if [[ $(proc_identity "$memwatch_pid" 2>/dev/null || true) == *" $memwatch_ticks" ]]; then
        cmdline=$(tr '\0' ' ' <"/proc/$memwatch_pid/cmdline")
        [[ $cmdline == *"$REPO/scripts/01_memwatch.sh"* &&
                $cmdline == *"$GLM_WATCHDOG_TARGET"* ]] ||
            die "GLM memwatch identity changed; refusing to disarm"
        printf 'DISARM %s %s %s\n' "$pid" "$expected_pgid" "$expected_ticks" \
            >"$GLM_WATCHDOG_TARGET.tmp"
        mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
        for _ in $(seq 1 50); do
            [[ -d /proc/$memwatch_pid ]] || break
            sleep 0.1
        done
        [[ ! -d /proc/$memwatch_pid ]] ||
            die "GLM memwatch did not accept authenticated disarm"
    fi
    rm -f -- "$GLM_PROCESS" "$GLM_WATCHDOG_TARGET" "$GLM_WATCHDOG_READY"
}

stop_profile() {
    case "$1" in
        dsv4) dsv4_launcher stop ;;
        glm52) stop_glm_verified ;;
        "") return 0 ;;
        *) die "unknown previous profile $1" ;;
    esac
}

start_dsv4() {
    dsv4_launcher start
}

start_glm52() {
    die "GLM switching remains disabled until its gated watchdog lifecycle passes fault injection"
}

api_key() {
    local file=${DSV4_API_KEY_FILE:-/etc/deepseek-v4-flash/api-key}
    [[ -r $file ]] || return 1
    IFS= read -r REPLY <"$file"
    [[ $REPLY =~ ^[A-Za-z0-9._-]{16,512}$ ]]
}

verify_dsv4_context() {
    local body path="/slots"
    body=$(clean_curl -fsS --max-time 5 \
        "http://127.0.0.1:$PORT$path") || return 1
    clean_python - "$body" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or len(value) != 1:
    raise SystemExit("DeepSeek slot topology is invalid")
slot = value[0]
if slot["n_ctx"] != 1048576:
    raise SystemExit("DeepSeek context is not 1048576")
PY
}

wait_model_ready() {
    local profile=$1 expected body deadline probe_count=0 available
    expected=deepseek-v4-flash
    [[ $profile == glm52 ]] && expected=glm-5.2
    deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        body=$(clean_curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" \
            2>/dev/null || true)
        if clean_python - "$expected" "$body" <<'PY' 2>/dev/null
import json, sys
expected, raw = sys.argv[1:]
value = json.loads(raw)
if not any(expected == item["id"].lower() for item in value["data"]):
    raise SystemExit("exact model identity mismatch")
PY
        then
            if [[ $profile != dsv4 ]] || verify_dsv4_context; then
                return 0
            fi
        fi
        probe_count=$((probe_count + 1))
        if (( probe_count % 15 == 0 )); then
            available=$(awk '$1 == "MemAvailable:" {printf "%.2f", $2 / 1048576}' \
                /proc/meminfo)
            printf 'Waiting for %s load: MemAvailable=%s GiB\n' \
                "$profile" "${available:-unknown}" >&2
        fi
        sleep 2
    done
    return 1
}

verify_serving() {
    local profile=$1 expected unauth code key body
    expected=deepseek-v4-flash
    [[ $profile == glm52 ]] && expected=glm-5.2
    body=$(clean_curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models") ||
        return 1
    clean_python - "$expected" "$body" <<'PY'
import json, sys
expected=sys.argv[1]
value=json.loads(sys.argv[2])
if not any(expected == item["id"].lower() for item in value["data"]):
    raise SystemExit("exact model identity mismatch")
PY
    if [[ $profile == dsv4 ]]; then
        verify_dsv4_context || return 1
    fi
    unauth=$(clean_curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:$AUTH_PORT/health" || true)
    [[ $unauth == 401 ]] || return 1
    api_key || return 1
    key=$REPLY
    code=$(
        printf 'header = "Authorization: Bearer %s"\n' "$key" |
            clean_curl --config - -sS -o "$STATE/probe.json.tmp" \
                -w '%{http_code}' --max-time 1800 \
                -H 'Content-Type: application/json' \
                -d '{"model":"default","messages":[{"role":"user","content":"Calculate 2+2. State the decimal answer clearly."}],"max_tokens":64,"temperature":0}' \
                "http://127.0.0.1:$AUTH_PORT/v1/chat/completions" || true
    )
    unset key REPLY
    [[ $code == 200 ]] || return 1
    clean_python - "$STATE/probe.json.tmp" <<'PY'
import json, sys
import re
with open(sys.argv[1], encoding="utf-8") as stream:
    value=json.load(stream)
message=value["choices"][0]["message"]
finish_reason=value["choices"][0]["finish_reason"]
if finish_reason not in {"stop", "length"}:
    raise SystemExit("semantic readiness finish reason is invalid")
usage=value["usage"]
if not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] < 1:
    raise SystemExit("semantic readiness completion count is invalid")
parts=[
    message.get(field)
    for field in ("reasoning_content", "content")
    if isinstance(message.get(field), str)
]
text="\n".join(parts)
if not text.strip() or re.search(r"(?<![0-9])4(?![0-9])", text) is None:
    raise SystemExit("semantic readiness probe failed")
PY
    mv -- "$STATE/probe.json.tmp" "$STATE/$profile.probe.json"
}

commit_active() {
    clean_python - "$ACTIVE.tmp" "$1" <<'PY'
import json, os, sys
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "profile":sys.argv[2]}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    mv -- "$ACTIVE.tmp" "$ACTIVE"
}

rollback() {
    local rc=$?
    "$rollback_needed" || return "$rc"
    rollback_needed=false
    stop_profile "${1:-}" || true
    if [[ -n $previous_profile ]]; then
        "start_$previous_profile" || true
        if verify_serving "$previous_profile"; then
            commit_active "$previous_profile"
        else
            echo "ROLLBACK VERIFICATION FAILED" >&2
        fi
    fi
    return "$rc"
}

command=${1:-status}
if [[ $command == status ]]; then
    [[ ${2:-} == --json ]] && { json_status; exit 0; }
    json_status
    exit 0
fi
[[ $command == restore || $command == dsv4 || $command == glm52 ]] ||
    die "usage: $0 status [--json]|restore|dsv4|glm52"
if [[ $command == restore ]]; then
    mkdir -p -- "$STATE"
    exec 9>"$LOCK"
    flock -x 9
    command=$(read_active_profile)
    [[ -n $command ]] || exit 0
    if [[ $command == glm52 ]] && ! glm_qualified; then
        die "recorded GLM-5.2 profile is no longer qualified"
    fi
    if verify_serving "$command"; then
        exit 0
    fi
    if [[ $command != dsv4 || -e /run/dsv4/llamacpp.state.json ]]; then
        stop_profile "$command"
    fi
    "start_$command"
    wait_model_ready "$command" ||
        die "$command boot restoration timed out or model identity is wrong"
    verify_serving "$command" ||
        die "$command boot restoration failed serving verification"
    exit 0
fi
if [[ $command == glm52 ]] && ! glm_qualified; then
    die "GLM-5.2 1M profile is not qualified"
fi
if [[ $command == glm52 ]]; then
    verify_glm_hashes
    die "GLM switching remains disabled until its gated watchdog lifecycle passes fault injection"
fi
mkdir -p -- "$STATE"
exec 9>"$LOCK"
flock -x 9
previous_profile=$(read_active_profile)
if [[ $previous_profile == "$command" ]] && verify_serving "$command"; then
    exit 0
fi
rollback_needed=true
trap 'rollback "$command"' EXIT
stop_profile "$previous_profile"
"start_$command"
wait_model_ready "$command" || die "$command readiness timed out or model identity is wrong"
verify_serving "$command"
commit_active "$command"
rollback_needed=false
trap - EXIT
