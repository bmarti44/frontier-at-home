#!/usr/bin/env bash
# Transactional profile switch for the unchanged :8010 auth chain.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly PROD_STATE=/home/dsv4/ds4-project/engine-switch
readonly PROD_BINARY=/home/bmarti44/.cache/glm52-dynexp2-patched/ds4-server
readonly PROD_GGUF=/home/bmarti44/models/glm52-full-denseq40.gguf
readonly PROFILE_MANIFEST=$REPO/configs/glm52-fullq4-production-profile.json
readonly QWEN_PROFILE_MANIFEST=$REPO/configs/qwen38-production-profile.json
readonly QWEN_1M_PROFILE_MANIFEST=$REPO/configs/qwen38-1m-production-profile.json
readonly PROD_QWEN_BINARY=/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17/src/build/bin/llama-server
readonly PROD_QWEN_MODEL=/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
readonly PROD_QWEN_MMPROJ=/home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf
readonly PORT=8013
readonly AUTH_PORT=8010

if [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]]; then
    [[ -n ${ENGINE_SWITCH_TEST_ROOT:-} ]] || {
        echo "ENGINE_SWITCH_TEST_ROOT is required" >&2; exit 2;
    }
    STATE=$ENGINE_SWITCH_TEST_ROOT
    BINARY=$STATE/source/ds4-server
    GGUF=$STATE/model.gguf
    QWEN_BINARY=$STATE/source/llama-server
    QWEN_MODEL=$STATE/qwen-model.gguf
    QWEN_MMPROJ=$STATE/qwen-mmproj.gguf
else
    unset ENGINE_SWITCH_TEST_ROOT ENGINE_PORT DS4_GLM_TOPK_KEEP \
        DS4_GLM_TOPK_SKIP_LOAD DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START || true
    STATE=$PROD_STATE
    BINARY=$PROD_BINARY
    GGUF=$PROD_GGUF
    QWEN_BINARY=$PROD_QWEN_BINARY
    QWEN_MODEL=$PROD_QWEN_MODEL
    QWEN_MMPROJ=$PROD_QWEN_MMPROJ
fi
readonly STATE BINARY GGUF QWEN_BINARY QWEN_MODEL QWEN_MMPROJ
readonly ACTIVE=$STATE/active.json
readonly LOCK=$STATE/switch.lock
readonly GLM_PROCESS=$STATE/glm52.process.json
readonly GLM_WATCHDOG_TARGET=$STATE/glm52.memwatch.target
readonly GLM_WATCHDOG_READY=$STATE/glm52.memwatch.ready
readonly GLM_WATCHDOG_LOG=$STATE/glm52.memwatch.log
readonly QWEN_PROCESS=$STATE/qwen38.process.json
readonly QWEN_UNIT=qwen38-engine.service
readonly QWEN_WATCHDOG_TARGET=$STATE/qwen38.memwatch.target
readonly QWEN_WATCHDOG_READY=$STATE/qwen38.memwatch.ready
readonly QWEN_WATCHDOG_LOG=$STATE/qwen38.memwatch.log
rollback_needed=false
previous_profile=
qwen_hashes_verified_profile=
qwen_verified_identities=

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
    install -d -o root -g dsv4 -m 1770 /run/dsv4
    /usr/sbin/runuser -u dsv4 -- env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        DSV4_PORT="$PORT" \
        DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
        DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=8 \
        DSV4_MEM_FLOOR_GIB=8 DSV4_WATCHDOG_FLOOR_GIB=8 \
        DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=12 \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START=1 \
        DSV4_UBATCH=2048 DSV4_BATCH=2048 DSV4_UBATCH_LARGE=1 \
        CTX=1048576 DSV4_PARALLEL=2 DSV4_NO_MMAP=1 \
        DSV4_SPEC_TYPE=none \
        DSV4_VERIFY_WEIGHTS=full \
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
    if value.get("schema_version") == 1 and value.get("profile") in {"dsv4", "glm52", "qwen38", "qwen38-1m"}:
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
    print(profile if profile in {"dsv4", "glm52", "qwen38", "qwen38-1m"} else "")
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
    clean_python - "$PROFILE_MANIFEST" "$BINARY" "$GGUF" <<'PY'
import hashlib, json, os, stat, sys
manifest_path, binary_path, model_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
if manifest.get("binary_path") != binary_path:
    raise SystemExit("GLM binary path is not approved")
if digest(binary_path) != manifest["binary_sha256"]:
    raise SystemExit("GLM binary hash is not approved")
if manifest.get("model_path") != model_path:
    raise SystemExit("GLM model path is not approved")
identity = manifest.get("model_identity", {})
if identity.get("first_bytes") != 1048576:
    raise SystemExit("GLM model prefix length is not approved")
with open(model_path, "rb") as stream:
    before = os.fstat(stream.fileno())
    first_bytes = stream.read(identity["first_bytes"])
    after = os.fstat(stream.fileno())
stable_before = (
    before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
    before.st_uid, before.st_gid, before.st_size, before.st_mtime_ns,
    before.st_ctime_ns,
)
stable_after = (
    after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
    after.st_uid, after.st_gid, after.st_size, after.st_mtime_ns,
    after.st_ctime_ns,
)
if stable_before != stable_after or not stat.S_ISREG(before.st_mode):
    raise SystemExit("GLM model identity changed during verification")
if (before.st_size, before.st_dev, before.st_ino) != (
    identity.get("size_bytes"), identity.get("device"), identity.get("inode")
):
    raise SystemExit("GLM model stat identity is not approved")
if len(first_bytes) != identity.get("first_bytes"):
    raise SystemExit("GLM model prefix is short")
if hashlib.sha256(first_bytes).hexdigest() != identity.get("first_bytes_sha256"):
    raise SystemExit("GLM model prefix hash is not approved")
if manifest.get("context_cap") != 32768:
    raise SystemExit("GLM profile context cap is not 32768")
PY
}

verify_qwen_profile_hashes() {
    local manifest_path=$1 expected_profile=$2 expected_context=$3
    qwen_verified_identities=$(clean_python - "$manifest_path" "$QWEN_BINARY" "$QWEN_MODEL" \
            "$QWEN_MMPROJ" "$expected_profile" "$expected_context" <<'PY'
import hashlib, json, os, stat, sys
manifest_path, binary_path, model_path, mmproj_path, expected_profile, expected_context = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)

def digest_and_identity(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    value = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"Qwen artifact is not a regular file: {path}")
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(chunk)
        after = os.fstat(stream.fileno())
    fields = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if fields(before) != fields(after):
        raise SystemExit(f"Qwen artifact changed during verification: {path}")
    return value.hexdigest(), fields(after)

identities = {}
for label, path, path_key, hash_key in (
    ("binary", binary_path, "binary_path", "binary_sha256"),
    ("model", model_path, "model_path", "model_sha256"),
    ("mmproj", mmproj_path, "mmproj_path", "mmproj_sha256"),
):
    if manifest.get(path_key) != path:
        raise SystemExit(f"Qwen {label} path is not approved")
    digest, identity = digest_and_identity(path)
    if digest != manifest.get(hash_key):
        raise SystemExit(f"Qwen {label} hash is not approved")
    identities[label] = identity
if manifest.get("profile") != expected_profile or manifest.get("schema_version") != 3:
    raise SystemExit("Qwen profile identity is not approved")
if manifest.get("port") != 8013 or manifest.get("context_cap") != int(expected_context):
    raise SystemExit("Qwen serving topology is not approved")
print(json.dumps(identities, separators=(",", ":"), sort_keys=True))
PY
    )
    qwen_hashes_verified_profile=$expected_profile
}

verify_qwen_hashes() {
    verify_qwen_profile_hashes "$QWEN_PROFILE_MANIFEST" qwen38 32768
}

verify_qwen_1m_hashes() {
    verify_qwen_profile_hashes "$QWEN_1M_PROFILE_MANIFEST" qwen38-1m 1048576
}

revalidate_qwen_identities() {
    [[ -n $qwen_verified_identities ]] || die "Qwen artifacts lack verified identities"
    clean_python - "$qwen_verified_identities" "$QWEN_BINARY" "$QWEN_MODEL" \
            "$QWEN_MMPROJ" <<'PY'
import json, os, stat, sys
expected = json.loads(sys.argv[1])
for label, path in zip(("binary", "model", "mmproj"), sys.argv[2:]):
    info = os.lstat(path)
    actual = [
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    ]
    if not stat.S_ISREG(info.st_mode) or actual != expected[label]:
        raise SystemExit(f"Qwen {label} identity changed after hash approval")
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

stop_qwen_verified() {
    local values pid expected_pgid expected_ticks expected_sha expected_unit
    local memwatch_pid memwatch_ticks current current_pgid current_ticks
    local exe unit_pid cmdline
    if [[ ! -f $QWEN_PROCESS ]]; then
        unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
            2>/dev/null || true)
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            die "Qwen unit is live without an identity record; refusing to continue"
        fi
        return 0
    fi
    values=$(clean_python - "$QWEN_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"],
      value["exe_sha256"], value["unit"], value["memwatch_pid"],
      value["memwatch_start_ticks"])
PY
    ) || die "invalid Qwen process record"
    read -r pid expected_pgid expected_ticks expected_sha expected_unit \
        memwatch_pid memwatch_ticks <<<"$values"
    [[ $expected_unit == "$QWEN_UNIT" ]] ||
        die "Qwen process record names an unexpected unit"
    current=$(proc_identity "$pid" 2>/dev/null || true)
    unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value 2>/dev/null || true)
    if [[ -n $current ]]; then
        read -r current_pgid current_ticks <<<"$current"
        exe=$(readlink -f "/proc/$pid/exe") ||
            die "cannot resolve recorded Qwen executable"
        [[ $unit_pid == "$pid" && $current_pgid == "$expected_pgid" &&
                $current_ticks == "$expected_ticks" ]] ||
            die "stale Qwen PID identity; refusing to stop unit"
        [[ $(sha256 "$exe") == "$expected_sha" ]] ||
            die "Qwen executable hash changed; refusing to stop unit"
        systemctl stop "$QWEN_UNIT"
        for _ in $(seq 1 600); do
            [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] && break
            sleep 0.1
        done
        [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] ||
            die "Qwen transient unit did not stop its recorded process"
    elif [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
        die "Qwen unit has an unrecorded live MainPID; refusing to stop it"
    fi
    if [[ $(proc_identity "$memwatch_pid" 2>/dev/null || true) == *" $memwatch_ticks" ]]; then
        cmdline=$(tr '\0' ' ' <"/proc/$memwatch_pid/cmdline")
        [[ $cmdline == *"$REPO/scripts/01_memwatch.sh"* &&
                $cmdline == *"$QWEN_WATCHDOG_TARGET"* ]] ||
            die "Qwen memwatch identity changed; refusing to disarm"
        printf 'DISARM %s %s %s\n' "$pid" "$expected_pgid" "$expected_ticks" \
            >"$QWEN_WATCHDOG_TARGET.tmp"
        mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET"
        for _ in $(seq 1 50); do
            [[ -d /proc/$memwatch_pid ]] || break
            sleep 0.1
        done
        [[ ! -d /proc/$memwatch_pid ]] ||
            die "Qwen memwatch did not accept authenticated disarm"
    fi
    rm -f -- "$QWEN_PROCESS" "$QWEN_WATCHDOG_TARGET" "$QWEN_WATCHDOG_READY"
}

stop_profile() {
    case "$1" in
        dsv4)
            if dsv4_launcher stop; then
                return 0
            fi
            # The launcher fails closed after removing a dead, identity-checked
            # state record. Treat only that exact absent-and-unreachable result
            # as already stopped; a live or ambiguous endpoint still fails.
            [[ ! -e /run/dsv4/llamacpp.state.json ]] || return 1
            if clean_curl -fsS --max-time 2 \
                    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                return 1
            fi
            return 0
            ;;
        glm52) stop_glm_verified ;;
        qwen38|qwen38-1m) stop_qwen_verified ;;
        "") return 0 ;;
        *) die "unknown previous profile $1" ;;
    esac
}

start_dsv4() {
    dsv4_launcher start
}

start_glm52() {
    local pid identity pgid ticks exe_sha current
    local memwatch_pid memwatch_ticks ready
    [[ ! -e $GLM_PROCESS ]] ||
        die "GLM process record already exists; refusing a second model"
    "$REPO/scripts/03_memory_guard.py" --required-gib 110 \
        --stable-samples 3 --interval-seconds 1 --timeout-seconds 180
    rm -f -- "$GLM_WATCHDOG_TARGET" "$GLM_WATCHDOG_READY" \
        "$GLM_PROCESS.tmp"
    "$REPO/scripts/01_memwatch.sh" \
        --target-file "$GLM_WATCHDOG_TARGET" \
        --ready-file "$GLM_WATCHDOG_READY" \
        --threshold-gib 18 --interval-sec 1 --log "$GLM_WATCHDOG_LOG" 9>&- &
    memwatch_pid=$!
    memwatch_ticks=
    ready=
    for _ in $(seq 1 50); do
        memwatch_ticks=$(proc_identity "$memwatch_pid" 2>/dev/null || true)
        memwatch_ticks=${memwatch_ticks#* }
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ -n $memwatch_ticks && $ready == READY ]] && break
        sleep 0.1
    done
    if [[ -z $memwatch_ticks || $ready != READY ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "GLM memory watchdog failed to initialize"
    fi
    setsid env -i HOME=/home/dsv4 LANG=C.UTF-8 \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        DS4_CUDA_EXPERT_CACHE_GB=94 \
        DS4_CUDA_EXPERT_CACHE_PIN=1 \
        DS4_CUDA_EXPERT_CACHE_SLRU=1 \
        DS4_CUDA_FETCH_THREADS=6 \
        DS4_CUDA_STABLE_MODEL_REMAP=1 \
        DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
        DS4_CUDA_EXPERT_DIRECT_SLOT=1 \
        "$BINARY" --cuda -m "$GGUF" -c 32768 \
        --host 127.0.0.1 --port "$PORT" --ssd-streaming \
        --ssd-streaming-cache-experts 40GB \
        >"$STATE/glm52.server.log" 2>&1 9>&- &
    pid=$!
    identity=
    for _ in $(seq 1 20); do
        identity=$(proc_identity "$pid" 2>/dev/null || true)
        [[ -n $identity ]] && break
        sleep 1
    done
    if [[ -z $identity ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "GLM startup died before identity capture"
    fi
    read -r pgid ticks <<<"$identity"
    exe_sha=$(sha256 "$BINARY")
    if ! clean_python - "$GLM_PROCESS.tmp" "$pid" "$pgid" "$ticks" \
            "$exe_sha" "$memwatch_pid" "$memwatch_ticks" <<'PY'
import json, os, sys
path, pid, pgid, ticks, digest, watchdog_pid, watchdog_ticks = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "pid":int(pid), "pgid":int(pgid),
               "start_ticks":int(ticks), "exe_sha256":digest,
               "memwatch_pid":int(watchdog_pid),
               "memwatch_start_ticks":int(watchdog_ticks)}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    then
        current=$(proc_identity "$pid" 2>/dev/null || true)
        if [[ $current == "$identity" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
            for _ in $(seq 1 100); do
                [[ $(proc_identity "$pid" 2>/dev/null || true) != "$identity" ]] && break
                sleep 0.1
            done
            if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$identity" ]]; then
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
        fi
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$GLM_PROCESS.tmp"
        if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$identity" ]]; then
            rollback_needed=false
            die "GLM cleanup failed after process record creation failure; rollback suppressed"
        fi
        die "GLM process identity record could not be created"
    fi
    mv -- "$GLM_PROCESS.tmp" "$GLM_PROCESS"
    printf '%s %s %s provisional\n' "$pid" "$pgid" "$ticks" \
        >"$GLM_WATCHDOG_TARGET.tmp"
    mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] ||
        die "GLM memory watchdog did not arm provisional process"
    printf '%s %s %s engine\n' "$pid" "$pgid" "$ticks" \
        >"$GLM_WATCHDOG_TARGET.tmp"
    mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
    for _ in $(seq 1 50); do
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks engine" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks engine" ]] ||
        die "GLM memory watchdog did not arm final process"
}

launch_qwen38() {
    systemd-run --unit=qwen38-engine --collect --quiet \
        --property Type=exec \
        --property User=bmarti44 \
        --property MemoryHigh=45G \
        --property MemoryMax=50G \
        --property MemorySwapMax=0 \
        --property OOMPolicy=kill \
        --property KillMode=control-group \
        --property Delegate=no \
        --property "StandardOutput=append:$STATE/qwen38.server.log" \
        --property "StandardError=append:$STATE/qwen38.server.log" \
        /usr/bin/flock --nonblock --no-fork \
        /run/lock/frontier-at-home/inference.lock \
        "$QWEN_BINARY" --model "$QWEN_MODEL" -ngl 99 -fa on \
        --no-mmap -c 32768 --mmproj "$QWEN_MMPROJ" --parallel 1 \
        --host 127.0.0.1 --port "$PORT" --alias qwen3.8-27b \
        --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.6 \
        --chat-template-kwargs '{"reasoning_effort":"low"}' \
        --cache-reuse 256
}

launch_qwen38-1m() {
    systemd-run --unit=qwen38-engine --collect --quiet \
        --property Type=exec \
        --property User=bmarti44 \
        --property MemoryHigh=88G \
        --property MemoryMax=95G \
        --property MemorySwapMax=0 \
        --property OOMPolicy=kill \
        --property KillMode=control-group \
        --property Delegate=no \
        --property "StandardOutput=append:$STATE/qwen38-1m.server.log" \
        --property "StandardError=append:$STATE/qwen38-1m.server.log" \
        /usr/bin/flock --nonblock --no-fork \
        /run/lock/frontier-at-home/inference.lock \
        "$QWEN_BINARY" --model "$QWEN_MODEL" -ngl 99 -fa on \
        --no-mmap -c 1048576 --mmproj "$QWEN_MMPROJ" --parallel 4 \
        --host 127.0.0.1 --port "$PORT" --alias qwen3.8-27b \
        --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.6 \
        --chat-template-kwargs '{"reasoning_effort":"low"}' \
        --cache-reuse 256
}

cleanup_qwen_killed_unit() {
    local load_state active_state
    load_state=$(systemctl show "$QWEN_UNIT" --property=LoadState --value \
        2>/dev/null || true)
    [[ -z $load_state || $load_state == not-found ]] && return 0
    active_state=$(systemctl show "$QWEN_UNIT" --property=ActiveState --value \
        2>/dev/null || true)
    if [[ $active_state == failed || $active_state == inactive ]]; then
        systemctl reset-failed "$QWEN_UNIT" 2>/dev/null || true
        for _ in $(seq 1 50); do
            load_state=$(systemctl show "$QWEN_UNIT" --property=LoadState --value \
                2>/dev/null || true)
            [[ -z $load_state || $load_state == not-found ]] && return 0
            sleep 0.1
        done
    fi
    die "Qwen transient unit already exists (LoadState=$load_state, ActiveState=$active_state)"
}

start_qwen_profile() {
    local profile=$1 manifest_path=$2 verify_function=$3
    local pid identity pgid ticks exe_sha approved_sha unit_pid current
    local memwatch_pid memwatch_identity memwatch_ticks ready
    # Watchdog floor: 18 GiB for the 32K profile. The 1m profile measured
    # MemAvailable bottoming at 11 GiB with all four slots filled to ~260K
    # (gate 4), so an 18 GiB floor would false-trip near full load; 8 GiB
    # mirrors the owner-accepted DSV4 1M floor and still protects the box.
    local watchdog_floor_gib=18
    [[ $profile != qwen38-1m ]] || watchdog_floor_gib=8
    [[ ! -e $QWEN_PROCESS ]] ||
        die "Qwen process record already exists; refusing a second model"
    [[ $qwen_hashes_verified_profile == "$profile" ]] || "$verify_function"
    "$REPO/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --interval-seconds 1 --timeout-seconds 180
    cleanup_qwen_killed_unit
    rm -f -- "$QWEN_PROCESS.tmp" "$QWEN_WATCHDOG_TARGET" \
        "$QWEN_WATCHDOG_READY"
    "$REPO/scripts/01_memwatch.sh" \
        --target-file "$QWEN_WATCHDOG_TARGET" \
        --ready-file "$QWEN_WATCHDOG_READY" \
        --threshold-gib "$watchdog_floor_gib" --interval-sec 1 --log "$QWEN_WATCHDOG_LOG" 9>&- &
    memwatch_pid=$!
    memwatch_ticks=
    ready=
    for _ in $(seq 1 50); do
        memwatch_identity=$(proc_identity "$memwatch_pid" 2>/dev/null || true)
        memwatch_ticks=${memwatch_identity#* }
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ -n $memwatch_ticks && $ready == READY ]] && break
        sleep 0.1
    done
    if [[ -z $memwatch_ticks || $ready != READY ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen memory watchdog failed to initialize"
    fi
    # The approved hashes were computed before stopping the old profile. Close
    # the pathname reopen window as far as systemd permits by checking the full
    # stat identities again immediately before systemd-run opens the paths.
    if ! revalidate_qwen_identities; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen artifact identity changed before execution"
    fi
    if ! "launch_$profile"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit failed to start"
    fi
    pid=
    identity=
    for _ in $(seq 1 100); do
        unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
            2>/dev/null || true)
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            pid=$unit_pid
            identity=$(proc_identity "$pid" 2>/dev/null || true)
            [[ -n $identity ]] && break
        fi
        sleep 0.1
    done
    if [[ -z $pid || -z $identity ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit died before identity capture"
    fi
    read -r pgid ticks <<<"$identity"
    if [[ $pgid != "$pid" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient-unit server is not its process-group leader"
    fi
    if [[ $(readlink -f "/proc/$pid/exe" 2>/dev/null || true) != \
            $(readlink -f "$QWEN_BINARY") ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit executable identity is wrong"
    fi
    exe_sha=$(sha256 "/proc/$pid/exe")
    approved_sha=$(clean_python - "$manifest_path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["binary_sha256"])
PY
    )
    if [[ $exe_sha != "$approved_sha" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit executed an unapproved binary"
    fi
    if ! clean_python - "$QWEN_PROCESS.tmp" "$pid" "$pgid" "$ticks" \
            "$exe_sha" "$QWEN_UNIT" "$memwatch_pid" "$memwatch_ticks" <<'PY'
import json, os, sys
path, pid, pgid, ticks, digest, unit, watchdog_pid, watchdog_ticks = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "pid":int(pid), "pgid":int(pgid),
               "start_ticks":int(ticks), "exe_sha256":digest,
               "unit":unit, "memwatch_pid":int(watchdog_pid),
               "memwatch_start_ticks":int(watchdog_ticks)}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS.tmp"
        die "Qwen process identity record could not be created"
    fi
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if [[ $current != "$identity" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS.tmp"
        die "Qwen process identity changed before record publication"
    fi
    mv -- "$QWEN_PROCESS.tmp" "$QWEN_PROCESS"
    printf '%s %s %s provisional\n' "$pid" "$pgid" "$ticks" \
        >"$QWEN_WATCHDOG_TARGET.tmp"
    mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] ||
        die "Qwen memory watchdog did not arm provisional process"
    printf '%s %s %s engine\n' "$pid" "$pgid" "$ticks" \
        >"$QWEN_WATCHDOG_TARGET.tmp"
    mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks engine" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks engine" ]] ||
        die "Qwen memory watchdog did not arm final process"
}

start_qwen38() {
    start_qwen_profile qwen38 "$QWEN_PROFILE_MANIFEST" verify_qwen_hashes
}

start_qwen38-1m() {
    start_qwen_profile qwen38-1m "$QWEN_1M_PROFILE_MANIFEST" \
        verify_qwen_1m_hashes
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
if not isinstance(value, list) or len(value) != 2:
    raise SystemExit("DeepSeek slot topology is invalid")
for slot in value:
    if slot["n_ctx"] != 524288:
        raise SystemExit("DeepSeek per-slot context is not 524288 (2 x 512k)")
PY
}

verify_qwen_1m_context() {
    local body path="/slots"
    body=$(clean_curl -fsS --max-time 5 \
        "http://127.0.0.1:$PORT$path") || return 1
    clean_python - "$body" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or len(value) != 4:
    raise SystemExit("Qwen 1M slot topology is invalid")
for slot in value:
    if slot["n_ctx"] != 262144:
        raise SystemExit("Qwen per-slot context is not 262144 (4 x 262K)")
PY
}

verify_qwen_process_ready() {
    local values pid expected_pgid expected_ticks expected_exe expected_unit
    local unit_pid identity pgid ticks live_exe sockets
    [[ -r $QWEN_PROCESS ]] || return 1
    values=$(clean_python - "$QWEN_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["unit"])
PY
    ) || return 1
    read -r pid expected_pgid expected_ticks expected_unit <<<"$values"
    [[ $expected_unit == "$QWEN_UNIT" ]] || return 1
    unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
        2>/dev/null || true)
    [[ $unit_pid == "$pid" ]] || return 1
    identity=$(proc_identity "$pid" 2>/dev/null || true)
    [[ -n $identity ]] || return 1
    read -r pgid ticks <<<"$identity"
    [[ $pgid == "$expected_pgid" && $ticks == "$expected_ticks" ]] || return 1
    live_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    expected_exe=$(readlink -f "$QWEN_BINARY" 2>/dev/null || true)
    [[ -n $live_exe && $live_exe == "$expected_exe" ]] || return 1
    sockets=$(ss -H -ltnp "sport = :$PORT" 2>/dev/null) || return 1
    [[ -n $sockets && $sockets == *"127.0.0.1:$PORT"* &&
            $sockets == *"pid=$pid,"* ]]
}

wait_model_ready() {
    local profile=$1 expected body deadline probe_count=0 available
    expected=deepseek-v4-flash
    [[ $profile == glm52 ]] && expected=glm-5.2
    [[ $profile == qwen38 || $profile == qwen38-1m ]] && expected=qwen3.8-27b
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
                if [[ $profile != qwen38 && $profile != qwen38-1m ]] || { \
                        clean_curl -fsS --max-time 3 \
                            "http://127.0.0.1:$PORT/health" >/dev/null &&
                        verify_qwen_process_ready; }; then
                    if [[ $profile != qwen38-1m ]] || verify_qwen_1m_context; then
                        return 0
                    fi
                fi
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
    [[ $profile == qwen38 || $profile == qwen38-1m ]] && expected=qwen3.8-27b
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
    if [[ $profile == qwen38 || $profile == qwen38-1m ]]; then
        clean_curl -fsS --max-time 5 \
            "http://127.0.0.1:$PORT/health" >/dev/null || return 1
        verify_qwen_process_ready || return 1
    fi
    if [[ $profile == qwen38-1m ]]; then
        verify_qwen_1m_context || return 1
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
        if "start_$previous_profile" && wait_model_ready "$previous_profile" &&
                verify_serving "$previous_profile"; then
            commit_active "$previous_profile"
        else
            echo "ROLLBACK VERIFICATION FAILED" >&2
        fi
    fi
    return "$rc"
}

restore_profile() {
    local profile=$1
    if [[ $profile == glm52 ]] && ! glm_qualified; then
        echo "recorded GLM-5.2 profile is no longer qualified" >&2
        return 1
    fi
    if [[ $profile == qwen38 ]]; then
        verify_qwen_hashes || return 1
        revalidate_qwen_identities || return 1
    fi
    if [[ $profile == qwen38-1m ]]; then
        verify_qwen_1m_hashes || return 1
        revalidate_qwen_identities || return 1
    fi
    verify_serving "$profile" && return 0
    if [[ $profile != dsv4 || -e /run/dsv4/llamacpp.state.json ]]; then
        stop_profile "$profile" || return 1
    fi
    "start_$profile" || return 1
    wait_model_ready "$profile" || return 1
    verify_serving "$profile"
}

# Exercise complete switch/restore control flow without loading a model. These
# overrides exist only in the explicit test mode and retain the real Qwen launch
# functions so their complete systemd-run argv is covered by subprocess tests.
if [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]]; then
    test_action() { printf '%s\n' "$*" >>"$STATE/actions.log"; }
    systemd-run() {
        test_action "SYSTEMD_RUN $*"
        [[ ! -e $STATE/fail-qwen-start ]] || return 1
        : >"$STATE/qwen-running"
    }
    systemctl() {
        local verb=${1:-} property=${2:-}
        if [[ $verb == show && $property == "$QWEN_UNIT" ]]; then
            if [[ $* == *--property=LoadState* ]]; then
                [[ -e $STATE/qwen-unit-killed ]] && printf 'loaded\n' || printf 'not-found\n'
            elif [[ $* == *--property=ActiveState* ]]; then
                [[ -e $STATE/qwen-unit-killed ]] && printf 'failed\n' || printf 'inactive\n'
            fi
            return 0
        fi
        test_action "SYSTEMCTL $*"
        if [[ $verb == reset-failed && $property == "$QWEN_UNIT" ]]; then
            rm -f -- "$STATE/qwen-unit-killed"
        fi
    }
    dsv4_launcher() {
        test_action "DSV4 $*"
        if [[ $1 == start ]]; then
            : >"$STATE/dsv4-running"
        else
            rm -f -- "$STATE/dsv4-running"
        fi
    }
    verify_qwen_profile_hashes() {
        local _manifest_path=$1 expected_profile=$2 _expected_context=$3
        if [[ ! -e $STATE/qwen-hashes-valid ]]; then
            echo "Qwen test artifact hashes are not approved" >&2
            return 1
        fi
        qwen_hashes_verified_profile=$expected_profile
        qwen_verified_identities=test
        test_action "HASHES $expected_profile"
    }
    revalidate_qwen_identities() {
        [[ $qwen_verified_identities == test ]]
    }
    stop_qwen_verified() {
        test_action "STOP qwen"
        rm -f -- "$STATE/qwen-running"
    }
    start_qwen_profile() {
        local profile=$1 _manifest_path=$2 verify_function=$3
        [[ $qwen_hashes_verified_profile == "$profile" ]] || "$verify_function"
        cleanup_qwen_killed_unit
        "launch_$profile" || die "Qwen transient unit failed to start"
        test_action "START $profile"
    }
    wait_model_ready() {
        test_action "WAIT $1"
        [[ -e $STATE/$([[ $1 == dsv4 ]] && printf dsv4 || printf qwen)-running ]]
    }
    verify_serving() {
        test_action "VERIFY $1"
        [[ ! -e $STATE/fail-$1-verify ]] &&
            [[ -e $STATE/$([[ $1 == dsv4 ]] && printf dsv4 || printf qwen)-running ]]
    }
fi

command=${1:-status}
if [[ $command == status ]]; then
    [[ ${2:-} == --json ]] && { json_status; exit 0; }
    json_status
    exit 0
fi
[[ $command == restore || $command == stop || $command == dsv4 || \
        $command == glm52 || $command == qwen38 || $command == qwen38-1m ]] ||
    die "usage: $0 status [--json]|stop|restore|dsv4|glm52|qwen38|qwen38-1m"
if [[ $command == stop ]]; then
    # Gate-window helper: stop the active engine but leave active.json
    # untouched so `restore` brings the same profile back afterwards.
    mkdir -p -- "$STATE"
    exec 9>"$LOCK"
    flock -x 9
    stop_target=$(read_active_profile)
    [[ -n $stop_target ]] || exit 0
    ( stop_profile "$stop_target" ) ||
        die "cannot safely stop active profile $stop_target"
    echo "STOPPED $stop_target (active.json unchanged; '$0 restore' restarts it)"
    exit 0
fi
if [[ $command == restore ]]; then
    mkdir -p -- "$STATE"
    exec 9>"$LOCK"
    flock -x 9
    command=$(read_active_profile)
    [[ -n $command ]] || exit 0
    if ( restore_profile "$command" ); then
        exit 0
    fi
    [[ $command != dsv4 ]] || die "dsv4 boot restoration failed"
    echo "RESTORE FAILED for recorded profile $command; falling back to dsv4" >&2
    ( stop_profile "$command" ) ||
        die "cannot safely stop failed $command profile before dsv4 fallback"
    if ( restore_profile dsv4 ); then
        commit_active dsv4
        echo "RESTORE FALLBACK committed dsv4 in active.json" >&2
        exit 0
    fi
    die "dsv4 boot restoration fallback failed after $command"
fi
if [[ $command == glm52 ]] && ! glm_qualified; then
    die "GLM-5.2 production profile is not qualified"
fi
if [[ $command == glm52 ]]; then
    verify_glm_hashes
fi
if [[ $command == qwen38 ]]; then
    verify_qwen_hashes
fi
if [[ $command == qwen38-1m ]]; then
    verify_qwen_1m_hashes
fi
mkdir -p -- "$STATE"
exec 9>"$LOCK"
flock -x 9
if [[ $command == qwen38 || $command == qwen38-1m ]]; then
    revalidate_qwen_identities
fi
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
