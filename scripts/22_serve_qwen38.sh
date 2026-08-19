#!/usr/bin/env bash
# Start, stop, or inspect the Qwen3.8 llama.cpp development server.
set -Eeuo pipefail
umask 077

readonly STACK=qwen38
readonly PORT=8015
readonly RUNTIME_DIR=/run/dsv4
readonly LOCK_FILE=/run/lock/frontier-at-home/inference.lock
readonly STATE_FILE=$RUNTIME_DIR/qwen38.state.json
readonly TARGET_FILE=$RUNTIME_DIR/qwen38.engine.target
readonly WATCHDOG_READY=$RUNTIME_DIR/qwen38.memwatch.ready
readonly SERVER_UNIT=qwen38-server.service
readonly MODEL_ROOT=/home/bmarti44/models/qwen3.8-27b
readonly MMPROJ=$MODEL_ROOT/mmproj-Qwen3.8-27B-f16.gguf
readonly BINARY=/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17/src/build/bin/llama-server

server_pid=
server_pgid=
server_start_ticks=
memwatch_pid=
memwatch_start_ticks=
state_boot_id=
startup_armed=false
unit_started=false

usage() {
    cat <<'EOF'
Usage: 22_serve_qwen38.sh [start|stop|status] [--help]

Start (the default), stop, or inspect Qwen3.8-27B on 127.0.0.1:8015.

Environment:
  QWEN_QUANT  GGUF filename below /home/bmarti44/models/qwen3.8-27b
              (default: Qwen3.8-27B-Q4_K_M.gguf)
  QWEN_MTP    Set to 1 to enable two-token built-in MTP drafting (default: 0)
  QWEN_REASONING_EFFORT  low | medium | xhigh (default: low, owner directive)
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

pid_alive() {
    [[ ${1:-} =~ ^[0-9]+$ ]] && (( $1 > 1 )) && kill -0 "$1" 2>/dev/null
}

proc_identity() {
    local line
    local -a fields
    [[ ${1:-} =~ ^[0-9]+$ && -r /proc/$1/stat ]] || return 1
    IFS= read -r line <"/proc/$1/stat" || return 1
    line=${line##*) }
    read -r -a fields <<<"$line"
    (( ${#fields[@]} > 19 )) || return 1
    [[ ${fields[2]} =~ ^[0-9]+$ && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    printf '%s %s\n' "${fields[2]}" "${fields[19]}"
}

proc_start_ticks() {
    local identity
    identity=$(proc_identity "$1") || return 1
    printf '%s\n' "${identity#* }"
}

verify_server_identity() {
    local identity current_pgid current_ticks current_boot unit_pid
    identity=$(proc_identity "$server_pid") || return 1
    read -r current_pgid current_ticks <<<"$identity"
    current_boot=$(< /proc/sys/kernel/random/boot_id) || return 1
    unit_pid=$(systemctl --user show "$SERVER_UNIT" --property=MainPID --value 2>/dev/null) \
        || return 1
    [[ $unit_pid == "$server_pid" &&
       $current_pgid == "$server_pgid" &&
       $current_ticks == "$server_start_ticks" &&
       $current_boot == "$state_boot_id" ]]
}

read_state() {
    local output
    output=$(python3 - "$STATE_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        state = json.load(stream)
    for key in ("server_pid", "server_pgid", "server_start_ticks", "memwatch_pid", "memwatch_start_ticks"):
        value = state[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
            raise ValueError(f"invalid {key}")
    if state["port"] != 8015:
        raise ValueError("invalid port")
    if state["unit"] != "qwen38-server.service":
        raise ValueError("invalid unit")
    boot_id = state["boot_id"]
    if not isinstance(boot_id, str) or not boot_id or any(c.isspace() for c in boot_id):
        raise ValueError("invalid boot_id")
    for key in ("server_pid", "server_pgid", "server_start_ticks", "memwatch_pid", "memwatch_start_ticks", "boot_id"):
        print(state[key])
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"invalid state file: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
    ) || die "cannot read state file: $STATE_FILE"
    mapfile -t values <<<"$output"
    server_pid=${values[0]}
    server_pgid=${values[1]}
    server_start_ticks=${values[2]}
    memwatch_pid=${values[3]}
    memwatch_start_ticks=${values[4]}
    state_boot_id=${values[5]}
}

write_state() {
    local temporary=$STATE_FILE.tmp.$$
    state_boot_id=$(< /proc/sys/kernel/random/boot_id) || die 'cannot read boot ID'
    python3 - "$temporary" "$STATE_FILE" "$server_pid" "$server_pgid" \
        "$server_start_ticks" "$memwatch_pid" "$memwatch_start_ticks" \
        "$state_boot_id" "$MODEL" "$MMPROJ" "$mtp" <<'PY'
import json
import os
import sys

(temporary, output, server_pid, server_pgid, server_ticks, watchdog_pid,
 watchdog_ticks, boot_id, model, mmproj, mtp) = sys.argv[1:]
value = {
    "stack": "qwen38",
    "unit": "qwen38-server.service",
    "server_pid": int(server_pid),
    "server_pgid": int(server_pgid),
    "server_start_ticks": int(server_ticks),
    "memwatch_pid": int(watchdog_pid),
    "memwatch_start_ticks": int(watchdog_ticks),
    "boot_id": boot_id,
    "host": "127.0.0.1",
    "port": 8015,
    "context": 32768,
    "model": model,
    "mmproj": mmproj,
    "mtp": mtp == "1",
}
with open(temporary, "x", encoding="utf-8") as stream:
    json.dump(value, stream, separators=(",", ":"), allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY
}

publish_target() {
    local role=$1 temporary=$TARGET_FILE.tmp.$$
    printf '%s %s %s %s\n' "$server_pid" "$server_pgid" \
        "$server_start_ticks" "$role" >"$temporary"
    mv -- "$temporary" "$TARGET_FILE"
}

publish_disarm() {
    local temporary=$TARGET_FILE.tmp.$$
    printf 'DISARM %s %s %s\n' "$server_pid" "$server_pgid" \
        "$server_start_ticks" >"$temporary"
    mv -- "$temporary" "$TARGET_FILE"
}

stop_verified_processes() {
    local attempt
    if verify_server_identity; then
        systemctl --user stop "$SERVER_UNIT" || return 1
        for ((attempt=0; attempt < 600; attempt++)); do
            pid_alive "$server_pid" || break
            sleep 0.1
        done
        pid_alive "$server_pid" && return 1
        publish_disarm
    fi
    for ((attempt=0; attempt < 50; attempt++)); do
        pid_alive "$memwatch_pid" || break
        sleep 0.1
    done
    if pid_alive "$memwatch_pid" &&
            [[ $(proc_start_ticks "$memwatch_pid" 2>/dev/null || true) == "$memwatch_start_ticks" ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
    fi
    return 0
}

cleanup_failed_start() {
    local rc=$?
    "$startup_armed" || return "$rc"
    startup_armed=false
    trap - ERR EXIT
    stop_verified_processes || true
    if "$unit_started"; then
        systemctl --user stop "$SERVER_UNIT" >/dev/null 2>&1 || true
    fi
    # If identity capture itself failed immediately after spawning memwatch,
    # this PID is still our just-created child and cannot yet be a reused PID.
    if pid_alive "$memwatch_pid" && [[ -z $memwatch_start_ticks ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
    fi
    rm -f -- "$STATE_FILE" "$WATCHDOG_READY"
    exit "$rc"
}

verify_build() {
    python3 - "$BUILD_MANIFEST" "$BINARY" <<'PY'
import hashlib
import json
import os
import sys


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["commit"] != "9d77fa17254e1dee4b9e92504c91611a60b1359f":
        raise ValueError("unexpected llama.cpp commit in build manifest")
    expected_flags = [
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=121",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if manifest.get("cmake_flags") != expected_flags or manifest.get("jobs") != 2:
        raise ValueError("unexpected CMake flags or build parallelism")
    binary = sys.argv[2]
    if os.path.islink(binary):
        raise ValueError("llama-server must not be a symlink")
    expected = manifest["binaries"]["llama-server"]["sha256"]
    if digest(binary) != expected:
        raise ValueError("llama-server SHA-256 mismatch")
    libraries = manifest.get("shared_libraries")
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError("build manifest has no shared-library hashes")
    binary_dir = os.path.dirname(binary)
    for name, entry in libraries.items():
        if os.path.basename(name) != name:
            raise ValueError(f"unsafe library name: {name!r}")
        if digest(os.path.join(binary_dir, name)) != entry["sha256"]:
            raise ValueError(f"shared-library SHA-256 mismatch: {name}")
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"build verification failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

verify_weights() {
    [[ -r $WEIGHTS_MANIFEST ]] || die "weights manifest is unreadable: $WEIGHTS_MANIFEST"
    printf 'Verifying model and mmproj SHA-256 against %s (this reads both GGUFs)...\n' \
        "$WEIGHTS_MANIFEST" >&2
    python3 - "$WEIGHTS_MANIFEST" "$MODEL" "$MMPROJ" <<'PY'
import hashlib
import json
import os
import stat
import sys

try:
    manifest_name, *artifact_names = sys.argv[1:]
    with open(manifest_name, encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest["files"]
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    for artifact_name in artifact_names:
        basename = os.path.basename(artifact_name)
        matches = [item for item in files if item.get("name") == basename]
        if len(matches) != 1:
            raise ValueError(f"manifest must contain exactly one entry for {basename}")
        entry = matches[0]
        expected = entry["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid SHA-256 in manifest for {basename}")
        info = os.stat(artifact_name)
        if not stat.S_ISREG(info.st_mode) or os.path.islink(artifact_name):
            raise ValueError(f"artifact is not a non-symlink regular file: {basename}")
        if "bytes" in entry and info.st_size != entry["bytes"]:
            raise ValueError(f"byte size does not match manifest for {basename}")
        value = hashlib.sha256()
        with open(artifact_name, "rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                value.update(chunk)
        if value.hexdigest() != expected:
            raise ValueError(f"SHA-256 does not match manifest for {basename}")
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"weight verification failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

verify_listener_owned() {
    local sockets
    sockets=$(ss -H -ltnp "sport = :$PORT") || return 1
    [[ -n $sockets && $sockets == *"127.0.0.1:$PORT"* &&
       $sockets == *"pid=$server_pid,"* ]]
}

verify_expected_model() {
    local document
    document=$(curl --silent --show-error --fail --max-time 3 \
        "http://127.0.0.1:$PORT/v1/models") || return 1
    python3 - "$quant" "$document" <<'PY'
import json
import sys

expected = sys.argv[1]
try:
    document = json.loads(sys.argv[2])
    data = document["data"]
    if not isinstance(data, list) or len(data) != 1 or data[0].get("id") != expected:
        raise ValueError("unexpected model identity")
except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"model identity verification failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

do_start() {
    local attempt identity current_exe expected_exe mtp_json unit_load_state flock_path
    local preflight_sockets
    local -a command
    for command_name in basename curl date flock python3 readlink setsid ss stat \
            systemctl systemd-run; do
        need_command "$command_name"
    done
    [[ -d $RUNTIME_DIR && ! -L $RUNTIME_DIR ]] \
        || die "runtime directory is absent or unsafe: $RUNTIME_DIR"
    [[ $(stat -Lc '%U:%G:%a' -- "$RUNTIME_DIR") == root:dsv4:1770 ]] \
        || die 'runtime directory ownership or mode is unsafe'

    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) \
        || die 'cannot resolve script directory'
    REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) \
        || die 'cannot resolve repository root'
    BUILD_MANIFEST=$REPO_ROOT/configs/build-manifests/llamacpp-qwen38-9d77fa17.json
    WEIGHTS_MANIFEST=$REPO_ROOT/weights/qwen3.8-27b/manifest.json
    MEMWATCH=$REPO_ROOT/scripts/01_memwatch.sh
    LOG_DIR=/home/bmarti44/logs
    SERVER_LOG=$LOG_DIR/qwen38-llamacpp-server.log
    MEMWATCH_LOG=$LOG_DIR/memwatch-qwen38.log

    quant=${QWEN_QUANT:-Qwen3.8-27B-Q4_K_M.gguf}
    [[ -n $quant && $quant == "$(basename -- "$quant")" &&
       $quant != *$'\n'* && $quant != *$'\r'* ]] \
        || die 'QWEN_QUANT must be a plain filename without path components'
    MODEL=$MODEL_ROOT/$quant
    mtp=${QWEN_MTP:-0}
    [[ $mtp == 0 || $mtp == 1 ]] || die 'QWEN_MTP must be 0 or 1'
    reasoning_effort=${QWEN_REASONING_EFFORT:-low}
    [[ $reasoning_effort == low || $reasoning_effort == medium ||
       $reasoning_effort == xhigh ]] \
        || die 'QWEN_REASONING_EFFORT must be low, medium, or xhigh'

    [[ -x $BINARY ]] || die "llama-server is missing or not executable: $BINARY"
    [[ -r $BUILD_MANIFEST ]] || die "build manifest is missing: $BUILD_MANIFEST"
    [[ -r $WEIGHTS_MANIFEST ]] || die "weights manifest is missing: $WEIGHTS_MANIFEST"
    [[ -f $MODEL && -r $MODEL ]] || die "model is missing or unreadable: $MODEL"
    [[ -f $MMPROJ && -r $MMPROJ ]] || die "mmproj is missing or unreadable: $MMPROJ"
    [[ -r $MEMWATCH ]] || die "memory watchdog is missing: $MEMWATCH"
    mkdir -p -- "$LOG_DIR"
    chmod 700 -- "$LOG_DIR"

    if [[ -e $STATE_FILE ]]; then
        read_state
        if verify_server_identity; then
            die "$STACK is already running with pid $server_pid"
        fi
        printf 'WARNING: removing stale Qwen3.8 state without signaling.\n' >&2
        rm -f -- "$STATE_FILE"
    fi

    unit_load_state=$(systemctl --user show "$SERVER_UNIT" \
        --property=LoadState --value 2>/dev/null) \
        || die 'cannot query the Qwen3.8 transient user unit'
    [[ -z $unit_load_state || $unit_load_state == not-found ]] \
        || die "transient unit already exists: $SERVER_UNIT (LoadState=$unit_load_state)"
    preflight_sockets=$(ss -H -ltn "sport = :$PORT") \
        || die "cannot inspect listeners on 127.0.0.1:$PORT"
    [[ -z $preflight_sockets ]] || die "127.0.0.1:$PORT is already listening"

    verify_build
    verify_weights

    exec 9>"$LOCK_FILE"
    flock -n 9 || die 'another inference server holds the residency lock'
    python3 "$REPO_ROOT/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --timeout-seconds 240 \
        || die 'pre-load 100 GiB stable-memory release gate failed'

    rm -f -- "$TARGET_FILE" "$WATCHDOG_READY"
    startup_armed=true
    trap cleanup_failed_start ERR EXIT
    setsid bash "$MEMWATCH" --target-file "$TARGET_FILE" \
        --ready-file "$WATCHDOG_READY" --threshold-gib 18 --interval-sec 0.25 \
        --log "$MEMWATCH_LOG" 9>&- >/dev/null 2>&1 &
    memwatch_pid=$!
    memwatch_start_ticks=$(proc_start_ticks "$memwatch_pid") \
        || die 'cannot identify memory watchdog process'
    for ((attempt=0; attempt < 100; attempt++)); do
        [[ -r $WATCHDOG_READY ]] && break
        pid_alive "$memwatch_pid" || die 'memory watchdog exited during initialization'
        sleep 0.05
    done
    [[ -r $WATCHDOG_READY ]] || die 'memory watchdog initialization timed out'

    command=(
        "$BINARY" --model "$MODEL" -ngl 99 -fa on --no-mmap -c 32768
        --mmproj "$MMPROJ" --parallel 1 --host 127.0.0.1 --port "$PORT"
        --alias "$quant"
        # Owner directive 2026-08-18: Qwen3.8 serves with low reasoning
        # effort by default (template levels: low|medium|xhigh).
        --chat-template-kwargs "{\"reasoning_effort\":\"$reasoning_effort\"}"
    )
    (( mtp == 0 )) || command+=(--spec-type draft-mtp --spec-draft-n-max 2)
    printf '\n===== Qwen3.8 session start %s mtp=%s =====\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mtp" >>"$SERVER_LOG"
    state_boot_id=$(< /proc/sys/kernel/random/boot_id) || die 'cannot read boot ID'
    flock_path=$(command -v flock)
    flock -u 9
    exec 9>&-
    systemd-run --user --unit qwen38-server --collect --quiet \
        --property Type=exec \
        --property MemoryHigh=45G \
        --property MemoryMax=50G \
        --property MemorySwapMax=0 \
        --property OOMPolicy=kill \
        --property Delegate=no \
        --property KillMode=control-group \
        --property "StandardOutput=append:$SERVER_LOG" \
        --property "StandardError=append:$SERVER_LOG" \
        "$flock_path" --nonblock --no-fork "$LOCK_FILE" "${command[@]}" \
        || die 'failed to launch Qwen3.8 transient user unit'
    unit_started=true
    server_pid=
    for ((attempt=0; attempt < 100; attempt++)); do
        server_pid=$(systemctl --user show "$SERVER_UNIT" \
            --property=MainPID --value 2>/dev/null || true)
        [[ $server_pid =~ ^[0-9]+$ ]] && (( server_pid > 1 )) && break
        sleep 0.05
    done
    [[ $server_pid =~ ^[0-9]+$ ]] && (( server_pid > 1 )) \
        || die "transient unit did not publish a server MainPID; see $SERVER_LOG"
    server_start_ticks=$(proc_start_ticks "$server_pid") \
        || die 'cannot identify llama-server process'
    identity=$(proc_identity "$server_pid") || die 'cannot read llama-server identity'
    read -r server_pgid _ <<<"$identity"
    [[ $server_pgid == "$server_pid" ]] \
        || die "transient-unit server is not its process-group leader: pid=$server_pid pgid=$server_pgid"
    publish_target engine

    for ((attempt=0; attempt < 100; attempt++)); do
        read -r marker ack_pid ack_pgid ack_ticks ack_role extra \
            <"$WATCHDOG_READY" 2>/dev/null || true
        if [[ ${marker:-} == ARMED && ${ack_pid:-} == "$server_pid" &&
              ${ack_pgid:-} == "$server_pgid" && ${ack_ticks:-} == "$server_start_ticks" &&
              ${ack_role:-} == engine && -z ${extra:-} ]]; then
            break
        fi
        pid_alive "$server_pid" || die "llama-server exited during startup; see $SERVER_LOG"
        pid_alive "$memwatch_pid" || die 'memory watchdog exited before arming'
        sleep 0.05
    done
    [[ ${marker:-} == ARMED && ${ack_pid:-} == "$server_pid" ]] \
        || die 'memory watchdog did not arm for llama-server'

    current_exe=$(readlink -f -- "/proc/$server_pid/exe") \
        || die 'cannot resolve live llama-server executable'
    expected_exe=$(readlink -f -- "$BINARY") \
        || die 'cannot resolve expected llama-server executable'
    [[ $current_exe == "$expected_exe" ]] \
        || die "live process executable mismatch: $current_exe"
    write_state

    for ((attempt=0; attempt < 300; attempt++)); do
        pid_alive "$server_pid" || die "llama-server exited during startup; see $SERVER_LOG"
        if curl --silent --show-error --fail --max-time 3 \
                "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 &&
           verify_listener_owned && verify_expected_model; then
            startup_armed=false
            trap - ERR EXIT
            mtp_json=false
            [[ $mtp == 0 ]] || mtp_json=true
            printf '{"ok":true,"stack":"qwen38","pid":%d,"port":8015,"mtp":%s}\n' \
                "$server_pid" "$mtp_json"
            return 0
        fi
        sleep 2
    done
    die "Qwen3.8 readiness timed out; see $SERVER_LOG"
}

do_stop() {
    [[ -r $STATE_FILE ]] || die "$STACK is not running (state file absent)"
    read_state
    verify_server_identity || {
        rm -f -- "$STATE_FILE"
        die 'stale Qwen3.8 state removed without signaling'
    }
    stop_verified_processes || die 'Qwen3.8 server did not stop cleanly'
    rm -f -- "$STATE_FILE" "$WATCHDOG_READY"
    printf '{"ok":true,"stack":"qwen38","stopped":true}\n'
}

do_status() {
    [[ -r $STATE_FILE ]] || die "$STACK is not running (state file absent)"
    read_state
    local alive=false healthy=false watchdog_alive=false
    verify_server_identity && alive=true
    if "$alive" && curl --silent --fail --max-time 3 \
            "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        healthy=true
    fi
    if pid_alive "$memwatch_pid" &&
            [[ $(proc_start_ticks "$memwatch_pid" 2>/dev/null || true) == "$memwatch_start_ticks" ]]; then
        watchdog_alive=true
    fi
    python3 - "$STATE_FILE" "$alive" "$healthy" "$watchdog_alive" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    state = json.load(stream)
state["server_alive"] = sys.argv[2] == "true"
state["healthy"] = sys.argv[3] == "true"
state["memwatch_alive"] = sys.argv[4] == "true"
print(json.dumps(state, separators=(",", ":")))
PY
    "$alive" && "$healthy" && "$watchdog_alive"
}

action=start
action_seen=false
while (( $# > 0 )); do
    case $1 in
        start|stop|status)
            "$action_seen" && { usage >&2; exit 2; }
            action=$1
            action_seen=true
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

case $action in
    start) do_start ;;
    stop)
        for command_name in python3 systemctl; do need_command "$command_name"; done
        do_stop
        ;;
    status)
        for command_name in curl python3 systemctl; do need_command "$command_name"; done
        do_status
        ;;
esac
