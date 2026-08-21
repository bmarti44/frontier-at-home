#!/usr/bin/env bash
# Start, stop, or inspect the Laguna S 2.1 llama.cpp development server.
set -Eeuo pipefail
umask 077

readonly STACK=laguna
readonly PORT=8016
readonly RUNTIME_DIR=/run/user/1000
readonly LOCK_FILE=/run/lock/frontier-at-home/inference.lock
readonly STATE_FILE=$RUNTIME_DIR/laguna.state.json
readonly TARGET_FILE=$RUNTIME_DIR/laguna.engine.target
readonly WATCHDOG_READY=$RUNTIME_DIR/laguna.memwatch.ready
readonly SERVER_UNIT=laguna-server.service
readonly MODEL_ROOT=/home/bmarti44/models/laguna-s-2.1
readonly DFLASH_MODEL=$MODEL_ROOT/poolside/laguna-s-2.1-DFlash-BF16.gguf
readonly BINARY=/home/bmarti44/.cache/llamacpp-laguna-06f8cebd/src/build/bin/llama-server

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
Usage: 25_serve_laguna.sh [start|stop|status] [--help]

Start (the default), stop, or inspect Laguna S 2.1 on 127.0.0.1:8016.

Environment:
  LAGUNA_QUANT     ud-q4 | ud-q5 | ud-q3 (default: ud-q4)
  LAGUNA_DFLASH    0 | 1 (default: 0)
  LAGUNA_SPEC_N_MAX  Positive draft-token limit (default: 7)
  LAGUNA_THINKING  off | max (default: max)
  LAGUNA_CTX       Positive context size (default: 65536)
  LAGUNA_PARALLEL  Positive parallel-slot count (default: 1)
  LAGUNA_MEM_MAX   Transient-unit memory cap (default: 100G)
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
    if state["port"] != 8016:
        raise ValueError("invalid port")
    if state["unit"] != "laguna-server.service":
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
        "$state_boot_id" "$MODEL" "$quant" "$dflash" "$thinking" \
        "$context" "$parallel" "$memory_max" <<'PY'
import json
import os
import sys

(temporary, output, server_pid, server_pgid, server_ticks, watchdog_pid,
 watchdog_ticks, boot_id, model, quant, dflash, thinking, context, parallel,
 memory_max) = sys.argv[1:]
value = {
    "stack": "laguna",
    "unit": "laguna-server.service",
    "server_pid": int(server_pid),
    "server_pgid": int(server_pgid),
    "server_start_ticks": int(server_ticks),
    "memwatch_pid": int(watchdog_pid),
    "memwatch_start_ticks": int(watchdog_ticks),
    "boot_id": boot_id,
    "host": "127.0.0.1",
    "port": 8016,
    "context": int(context),
    "model": model,
    "quant": quant,
    "parallel": int(parallel),
    "dflash": dflash == "1",
    "thinking": thinking,
    "memory_max": memory_max,
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
    if manifest["commit"] != "06f8cebd7fe728687be3d19f8bdedb70d75883af":
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
    local -a artifacts=("${model_files[@]}")
    (( dflash == 0 )) || artifacts+=("$DFLASH_MODEL")
    printf 'Verifying selected Laguna GGUF SHA-256 values against %s...\n' \
        "$WEIGHTS_MANIFEST" >&2
    python3 - "$WEIGHTS_MANIFEST" "$MODEL_ROOT" "${artifacts[@]}" <<'PY'
import hashlib
import json
import os
import stat
import sys

try:
    manifest_name, model_root, *artifact_names = sys.argv[1:]
    with open(manifest_name, encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest["files"]
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    for artifact_name in artifact_names:
        basename = os.path.basename(artifact_name)
        relative = os.path.relpath(artifact_name, model_root)
        matches = [
            item for item in files
            if item.get("name") in (basename, relative)
        ]
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
    local attempt identity current_exe expected_exe dflash_json thinking_json
    local unit_load_state flock_path preflight_sockets quant_dir
    local -a command model_files
    for command_name in basename curl date flock python3 readlink setsid ss stat \
            systemctl systemd-run; do
        need_command "$command_name"
    done
    [[ -d $RUNTIME_DIR && ! -L $RUNTIME_DIR ]] \
        || die "runtime directory is absent or unsafe: $RUNTIME_DIR"
    [[ $(stat -Lc '%U:%G:%a' -- "$RUNTIME_DIR") == "$(id -un):$(id -gn):700" ]] \
        || die 'runtime directory ownership or mode is unsafe'

    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) \
        || die 'cannot resolve script directory'
    REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) \
        || die 'cannot resolve repository root'
    BUILD_MANIFEST=$REPO_ROOT/configs/build-manifests/llamacpp-laguna-06f8cebd.json
    WEIGHTS_MANIFEST=$REPO_ROOT/weights/laguna-s-2.1/manifest.json
    MEMWATCH=$REPO_ROOT/scripts/01_memwatch.sh
    LOG_DIR=/home/bmarti44/logs
    SERVER_LOG=$LOG_DIR/laguna-llamacpp-server.log
    MEMWATCH_LOG=$LOG_DIR/memwatch-laguna.log

    quant=${LAGUNA_QUANT:-ud-q4}
    case $quant in
        ud-q4) quant_dir=unsloth/UD-Q4_K_XL ;;
        ud-q5) quant_dir=unsloth/UD-Q5_K_XL ;;
        ud-q3) quant_dir=unsloth/UD-Q3_K_XL ;;
        *) die 'LAGUNA_QUANT must be ud-q4, ud-q5, or ud-q3' ;;
    esac
    shopt -s nullglob
    model_files=("$MODEL_ROOT/$quant_dir"/*.gguf)
    shopt -u nullglob
    (( ${#model_files[@]} > 0 )) \
        || die "no GGUF files found below $MODEL_ROOT/$quant_dir"
    MODEL=${model_files[0]}
    dflash=${LAGUNA_DFLASH:-0}
    [[ $dflash == 0 || $dflash == 1 ]] || die 'LAGUNA_DFLASH must be 0 or 1'
    spec_n_max=${LAGUNA_SPEC_N_MAX:-7}
    [[ $spec_n_max =~ ^[1-9][0-9]*$ ]] \
        || die 'LAGUNA_SPEC_N_MAX must be a positive integer'
    thinking=${LAGUNA_THINKING:-max}
    [[ $thinking == off || $thinking == max ]] \
        || die 'LAGUNA_THINKING must be off or max'
    context=${LAGUNA_CTX:-65536}
    [[ $context =~ ^[1-9][0-9]*$ ]] || die 'LAGUNA_CTX must be a positive integer'
    parallel=${LAGUNA_PARALLEL:-1}
    [[ $parallel =~ ^[1-9][0-9]*$ ]] \
        || die 'LAGUNA_PARALLEL must be a positive integer'
    memory_max=${LAGUNA_MEM_MAX:-100G}
    [[ $memory_max =~ ^[1-9][0-9]*[KMGT]$ ]] \
        || die 'LAGUNA_MEM_MAX must be a positive systemd size ending in K, M, G, or T'

    [[ -x $BINARY ]] || die "llama-server is missing or not executable: $BINARY"
    [[ -r $BUILD_MANIFEST ]] || die "build manifest is missing: $BUILD_MANIFEST"
    [[ -f $MODEL && -r $MODEL ]] || die "model is missing or unreadable: $MODEL"
    if (( dflash == 1 )); then
        [[ -f $DFLASH_MODEL && -r $DFLASH_MODEL ]] \
            || die "DFlash model is missing or unreadable: $DFLASH_MODEL"
    fi
    [[ -r $MEMWATCH ]] || die "memory watchdog is missing: $MEMWATCH"
    mkdir -p -- "$LOG_DIR"
    chmod 700 -- "$LOG_DIR"

    if [[ -e $STATE_FILE ]]; then
        read_state
        if verify_server_identity; then
            die "$STACK is already running with pid $server_pid"
        fi
        printf 'WARNING: removing stale Laguna state without signaling.\n' >&2
        rm -f -- "$STATE_FILE"
    fi

    unit_load_state=$(systemctl --user show "$SERVER_UNIT" \
        --property=LoadState --value 2>/dev/null) \
        || die 'cannot query the Laguna transient user unit'
    [[ -z $unit_load_state || $unit_load_state == not-found ]] \
        || die "transient unit already exists: $SERVER_UNIT (LoadState=$unit_load_state)"
    preflight_sockets=$(ss -H -ltn "sport = :$PORT") \
        || die "cannot inspect listeners on 127.0.0.1:$PORT"
    [[ -z $preflight_sockets ]] || die "127.0.0.1:$PORT is already listening"

    verify_build
    if [[ -e $WEIGHTS_MANIFEST ]]; then
        [[ -r $WEIGHTS_MANIFEST ]] \
            || die "weights manifest exists but is unreadable: $WEIGHTS_MANIFEST"
        verify_weights
    else
        printf 'WARNING: weights manifest not present; skipping weights verification: %s\n' \
            "$WEIGHTS_MANIFEST" >&2
    fi

    exec 9>"$LOCK_FILE"
    flock -n 9 || die 'another inference server holds the residency lock'
    python3 "$REPO_ROOT/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --timeout-seconds 240 \
        || die 'pre-load 100 GiB stable-memory release gate failed'

    rm -f -- "$TARGET_FILE" "$WATCHDOG_READY"
    startup_armed=true
    trap cleanup_failed_start ERR EXIT
    setsid bash "$MEMWATCH" --target-file "$TARGET_FILE" \
        --ready-file "$WATCHDOG_READY" --threshold-gib 8 --interval-sec 0.25 \
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

    thinking_json=true
    [[ $thinking == max ]] || thinking_json=false
    command=(
        "$BINARY" --model "$MODEL" -ngl 99 -fa on --no-mmap -c "$context"
        --parallel "$parallel" --host 127.0.0.1 --port "$PORT"
        --alias "$quant" --jinja
        --chat-template-kwargs "{\"enable_thinking\":$thinking_json}"
    )
    if (( dflash == 1 )); then
        command+=(
            -md "$DFLASH_MODEL" --spec-type draft-dflash
            --spec-draft-n-max "$spec_n_max"
        )
    fi
    printf '\n===== Laguna S 2.1 session start %s quant=%s dflash=%s thinking=%s =====\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$quant" "$dflash" "$thinking" \
        >>"$SERVER_LOG"
    state_boot_id=$(< /proc/sys/kernel/random/boot_id) || die 'cannot read boot ID'
    flock_path=$(command -v flock)
    flock -u 9
    exec 9>&-
    # On GB10, cgroup accounting is blind to CUDA unified memory. This cap is
    # only a backstop; the real guard is the 8 GiB MemAvailable watchdog floor.
    systemd-run --user --unit laguna-server --collect --quiet \
        --property Type=exec \
        --property "MemoryMax=$memory_max" \
        --property MemorySwapMax=0 \
        --property OOMPolicy=kill \
        --property Delegate=no \
        --property KillMode=control-group \
        --property "StandardOutput=append:$SERVER_LOG" \
        --property "StandardError=append:$SERVER_LOG" \
        "$flock_path" --nonblock --no-fork "$LOCK_FILE" "${command[@]}" \
        || die 'failed to launch Laguna transient user unit'
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
            dflash_json=false
            [[ $dflash == 0 ]] || dflash_json=true
            printf '{"ok":true,"stack":"laguna","pid":%d,"port":8016,"dflash":%s}\n' \
                "$server_pid" "$dflash_json"
            return 0
        fi
        sleep 2
    done
    die "Laguna readiness timed out; see $SERVER_LOG"
}

do_stop() {
    [[ -r $STATE_FILE ]] || die "$STACK is not running (state file absent)"
    read_state
    verify_server_identity || {
        rm -f -- "$STATE_FILE"
        die 'stale Laguna state removed without signaling'
    }
    stop_verified_processes || die 'Laguna server did not stop cleanly'
    rm -f -- "$STATE_FILE" "$WATCHDOG_READY"
    printf '{"ok":true,"stack":"laguna","stopped":true}\n'
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
