#!/usr/bin/env bash
# Generic profile-driven serving lifecycle (docs/PROFILE-SCHEMA.md).
#
# Resolves a declarative profile (scripts/lib/profile_resolver.py), verifies
# binary/weights digests, runs the membudget admission from the profile's
# memory model, and launches under the platform watchdog with the profile's
# mechanism. Dev-port serving on every platform; the production :8013 path
# stays with scripts/52_engine_switch.sh (which renders the same profiles).
#
# Usage: 93_profile_serve.sh --profile <model>/<file> [--host FILE]
#            [--port N] [start|stop|status]
#
# Memory-safety guarantees (docs/BACKEND-CONTRACT.md section 3) hold on all
# mechanisms: whole-system watchdog, single-residency lock, stable-memory
# release gate before load.
set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

PROFILE_ARG=
HOST_ARG=
PORT_OVERRIDE=
ACTION=start
while (( $# > 0 )); do
    case $1 in
        --profile) PROFILE_ARG=${2:?}; shift 2 ;;
        --host) HOST_ARG=${2:?}; shift 2 ;;
        --port) PORT_OVERRIDE=${2:?}; shift 2 ;;
        start|stop|status) ACTION=$1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $PROFILE_ARG ]] || { usage >&2; exit 2; }
if [[ -n $PORT_OVERRIDE ]]; then
    [[ $PORT_OVERRIDE =~ ^[0-9]+$ ]] && (( PORT_OVERRIDE >= 1024 && PORT_OVERRIDE <= 65535 )) \
        || die '--port must be 1024-65535'
fi

need_command python3

# Render the profile once; every later step reads this snapshot.
render_args=(render --profile "$PROFILE_ARG" --verb "$ACTION")
[[ -z $HOST_ARG ]] || render_args+=(--host "$HOST_ARG")
SNAPSHOT=$(python3 "$REPO_ROOT/scripts/92_resolve_profile.py" "${render_args[@]}") \
    || die "profile resolution failed for $PROFILE_ARG"
export SNAPSHOT

snap() {  # snap <python-expression over d>
    python3 - "$1" <<'PY'
import json, os, sys
d = json.loads(os.environ["SNAPSHOT"])
value = eval(sys.argv[1], {"d": d})
if isinstance(value, (dict, list)):
    print(json.dumps(value))
elif value is None:
    print("")
else:
    print(value)
PY
}

MECHANISM=$(snap 'd["mechanism"]')
PROFILE_ID=$(snap 'd["profile_id"]')
PORT=$(snap 'd["port"]')
[[ -z $PORT_OVERRIDE ]] || PORT=$PORT_OVERRIDE
SLUG=${PROFILE_ID//\//-}

# Host-local runtime layout (works unprivileged on any platform).
RUN_DIR=${FAH_RUN_DIR:-$HOME/.frontier-at-home/run}
LOG_DIR=${FAH_LOG_DIR:-$HOME/.frontier-at-home/logs}
STATE_FILE=$RUN_DIR/$SLUG.state.json
TARGET_FILE=$RUN_DIR/$SLUG.memwatch.target
WATCHDOG_READY=$RUN_DIR/$SLUG.memwatch.ready
LOCK_FILE=$RUN_DIR/inference.lock
SERVER_LOG=$LOG_DIR/$SLUG.server.log
MEMWATCH_LOG=$LOG_DIR/$SLUG.memwatch.log
mkdir -p -- "$RUN_DIR" "$LOG_DIR"
chmod 700 -- "$RUN_DIR" "$LOG_DIR"

pid_alive() { kill -0 "$1" 2>/dev/null; }

proc_start_ticks() {
    if [[ -r /proc/$1/stat ]]; then
        awk '{print $22}' "/proc/$1/stat" 2>/dev/null
    else
        # macOS: process start time string stands in for start ticks.
        ps -p "$1" -o lstart= 2>/dev/null | tr -s ' ' '_'
    fi
}

delegated() {
    # The delegate owns the full lifecycle; hand it the rendered env and verb.
    local runuser_name delegate
    runuser_name=$(snap 'd["runuser"]')
    delegate=$(snap 'd["delegate"]')
    mapfile -t env_pairs < <(snap '"\n".join(f"{k}={v}" for k, v in d["env"].items())')
    if [[ $runuser_name == "$(id -un)" || -z $runuser_name ]]; then
        env -i "${env_pairs[@]}" "$delegate" "$ACTION"
    else
        need_command runuser
        /usr/sbin/runuser -u "$runuser_name" -- env -i "${env_pairs[@]}" \
            "$delegate" "$ACTION"
    fi
}

verify_digests() {
    python3 - <<'PY'
import hashlib, json, os, sys
snapshot = json.loads(os.environ["SNAPSHOT"])
for check in snapshot["digest_checks"]:
    path = check["path"]
    if "sha256" in check:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != check["sha256"]:
            raise SystemExit(f"digest mismatch: {path}")
    elif "identity" in check:
        identity = check["identity"]
        import os
        if os.path.getsize(path) != identity["size_bytes"]:
            raise SystemExit(f"size mismatch: {path}")
        with open(path, "rb") as stream:
            head = stream.read(identity["first_bytes"])
        if hashlib.sha256(head).hexdigest() != identity["first_bytes_sha256"]:
            raise SystemExit(f"sampled-hash mismatch: {path}")
print("digests-ok")
PY
}

admission_gate() {
    local budget
    budget=$(python3 - <<'PY'
import json, os
snapshot = json.loads(os.environ["SNAPSHOT"])
budget = snapshot["membudget"]
print(" ".join([
    "--weights-gib", str(budget["weights_gib"]),
    "--ctx", str(budget["ctx"]),
    "--kv-bytes-per-token", str(budget["kv_bytes_per_token"]),
    "--overhead-gib", str(budget["overhead_gib"]),
    "--extra-gib", str(budget["extra_gib"]),
    "--floor-gib", str(budget["floor_gib"]),
]))
PY
    ) || die 'cannot assemble membudget arguments'
    if [[ -r /proc/meminfo ]]; then
        # shellcheck disable=SC2086
        python3 "$REPO_ROOT/scripts/02_membudget.py" $budget >/dev/null \
            || die 'membudget admission failed: profile does not fit current free memory'
    else
        local available
        available=$(python3 - <<'PY'
import subprocess
out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
import re
page = 16384
pages = 0
for name in ("Pages free", "Pages inactive", "Pages purgeable"):
    match = re.search(rf"{name}:\s+(\d+)", out)
    if match:
        pages += int(match.group(1))
print(round(pages * page / 2**30, 2))
PY
        ) || die 'cannot read available memory on this platform'
        # shellcheck disable=SC2086
        python3 "$REPO_ROOT/scripts/02_membudget.py" $budget \
            --mem-available-gib "$available" >/dev/null \
            || die 'membudget admission failed: profile does not fit current free memory'
    fi
}

release_gate() {
    local required
    required=$(snap 'd["membudget"]["weights_gib"] + 10')
    if [[ -r /proc/meminfo ]]; then
        python3 "$REPO_ROOT/scripts/03_memory_guard.py" \
            --required-gib "${required%.*}" --stable-samples 3 \
            --timeout-seconds 240 \
            || die 'stable-memory release gate failed'
    else
        python3 "$REPO_ROOT/scripts/07_memory_guard_macos.py" \
            --required-gib "${required%.*}" --stable-samples 3 \
            --timeout-seconds 240 \
            || die 'stable-memory release gate failed'
    fi
}

arm_memwatch() {
    local memwatch=$REPO_ROOT/scripts/01_memwatch.sh
    local threshold
    threshold=$(snap 'd.get("memwatch", {}).get("threshold_gib") or d["membudget"]["floor_gib"]')
    if [[ ! -r /proc/meminfo ]]; then
        memwatch=$REPO_ROOT/scripts/06_memwatch_macos.sh
        [[ -r $memwatch ]] || die 'macOS memory watchdog is missing'
    fi
    rm -f -- "$TARGET_FILE" "$WATCHDOG_READY"
    setsid bash "$memwatch" --target-file "$TARGET_FILE" \
        --ready-file "$WATCHDOG_READY" --threshold-gib "$threshold" \
        --interval-sec 1 --log "$MEMWATCH_LOG" 9>&- >/dev/null 2>&1 &
    MEMWATCH_PID=$!
    local attempt
    for ((attempt = 0; attempt < 100; attempt++)); do
        [[ -r $WATCHDOG_READY ]] && return 0
        pid_alive "$MEMWATCH_PID" || die 'memory watchdog exited during initialization'
        sleep 0.05
    done
    die 'memory watchdog initialization timed out'
}

launch_server() {
    local binary
    binary=$(snap 'd["binary"]')
    mapfile -t argv < <(snap '"\n".join(d["argv"])')
    mapfile -t env_pairs < <(snap '"\n".join(f"{k}={v}" for k, v in d["env"].items())')
    # Honor a port override by rewriting the rendered port in argv.
    if [[ -n $PORT_OVERRIDE ]]; then
        local original
        original=$(snap 'd["port"]')
        for index in "${!argv[@]}"; do
            [[ ${argv[$index]} == "$original" ]] && argv[index]=$PORT_OVERRIDE
        done
    fi
    printf '\n===== %s session start %s =====\n' "$PROFILE_ID" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$SERVER_LOG"
    if [[ ${#env_pairs[@]} -gt 0 ]]; then
        setsid env -i "${env_pairs[@]}" "$binary" "${argv[@]}" \
            >>"$SERVER_LOG" 2>&1 9>&- &
    else
        setsid "$binary" "${argv[@]}" >>"$SERVER_LOG" 2>&1 9>&- &
    fi
    SERVER_PID=$!
}

do_start() {
    for command_name in curl date flock setsid; do need_command "$command_name"; done
    [[ ! -e $STATE_FILE ]] || die "$PROFILE_ID appears to be running (state file exists); run stop or status first"

    verify_digests >/dev/null || die 'artifact digest verification failed'
    exec 9>"$LOCK_FILE"
    flock -n 9 || die 'another inference server holds the residency lock'
    admission_gate
    release_gate
    arm_memwatch
    launch_server

    local ticks identity
    ticks=$(proc_start_ticks "$SERVER_PID") || die 'cannot identify server process'
    printf '%s %s %s engine\n' "$SERVER_PID" "$SERVER_PID" "$ticks" \
        >"$TARGET_FILE.tmp"
    mv -- "$TARGET_FILE.tmp" "$TARGET_FILE"

    python3 - "$STATE_FILE" "$SERVER_PID" "$ticks" "$MEMWATCH_PID" "$PORT" \
        "$PROFILE_ID" <<'PY'
import json, sys
path, pid, ticks, memwatch_pid, port, profile_id = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({
        "schema_version": 1,
        "profile_id": profile_id,
        "pid": int(pid),
        "start_ticks": ticks,
        "memwatch_pid": int(memwatch_pid),
        "port": int(port),
    }, stream)
PY

    local attempt
    for ((attempt = 0; attempt < 900; attempt++)); do
        pid_alive "$SERVER_PID" || {
            rm -f -- "$STATE_FILE"
            die "server exited during startup; see $SERVER_LOG"
        }
        if curl --silent --fail --max-time 3 \
                "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            printf '{"ok":true,"profile":"%s","pid":%d,"port":%d}\n' \
                "$PROFILE_ID" "$SERVER_PID" "$PORT"
            return 0
        fi
        sleep 2
    done
    die "readiness timed out; see $SERVER_LOG"
}

do_stop() {
    [[ -r $STATE_FILE ]] || die "$PROFILE_ID is not running (state file absent)"
    local pid ticks memwatch_pid current
    pid=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['pid'])")
    ticks=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['start_ticks'])")
    memwatch_pid=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['memwatch_pid'])")
    current=$(proc_start_ticks "$pid" 2>/dev/null || true)
    if [[ -n $current && $current == "$ticks" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        local attempt
        for ((attempt = 0; attempt < 150; attempt++)); do
            pid_alive "$pid" || break
            sleep 0.2
        done
        if pid_alive "$pid"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    pid_alive "$memwatch_pid" && kill -TERM "$memwatch_pid" 2>/dev/null || true
    rm -f -- "$STATE_FILE" "$TARGET_FILE" "$WATCHDOG_READY"
    printf '{"ok":true,"profile":"%s","stopped":true}\n' "$PROFILE_ID"
}

do_status() {
    [[ -r $STATE_FILE ]] || die "$PROFILE_ID is not running (state file absent)"
    local pid ticks alive=false healthy=false
    pid=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['pid'])")
    ticks=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['start_ticks'])")
    [[ $(proc_start_ticks "$pid" 2>/dev/null || true) == "$ticks" ]] && alive=true
    if "$alive" && curl --silent --fail --max-time 3 \
            "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        healthy=true
    fi
    printf '{"profile":"%s","pid":%s,"alive":%s,"healthy":%s}\n' \
        "$PROFILE_ID" "$pid" "$alive" "$healthy"
    "$alive" && "$healthy"
}

if [[ $MECHANISM == delegated-launcher ]]; then
    delegated
    exit $?
fi

case $ACTION in
    start) do_start ;;
    stop) do_stop ;;
    status) do_status ;;
esac
