#!/usr/bin/env bash
# Detached, watchdog-protected DeepSeek context qualification worker.
set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
readonly LIVE_REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly LAUNCHER=$REPO/scripts/21_serve_llamacpp.sh
readonly PROBE=$REPO/scripts/57_dsv4_context_probe.py
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'
readonly PREEXISTING_FATAL_PATTERN='NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'
readonly CAPS=(131072 262144 524288 1048576)
readonly TARGETS=(130000 260000 520000 1000000)
readonly SAFE_CTX=8192 # CTX=8192 is the proven production recovery profile.
readonly START_HOLD=/home/dsv4/.dsv4-start-hold
readonly WORKER_START_HOLD=/run/dsv4/context-worker.start-hold
readonly START_FAILURE_MARKER=/run/dsv4/llamacpp.start-failed
readonly USER_RESTORE=$REPO/scripts/61_restore_dsv4_user.sh
readonly USER_ENGINE_UNIT=dsv4-safe-engine.service
readonly ENGINE_LOG=/home/dsv4/logs/llamacpp-server.log
readonly PYTHON_BIN=$LIVE_REPO/.venv-harness/bin/python
readonly LIVE_MODEL_PATH=$LIVE_REPO/weights/unsloth-ud-q2_k_xl/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf
readonly PROTECTED_MODEL_PATH=/var/lib/dsv4-context/models/deepseek-v4-flash/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf
# Fixed acceptance authority: w11.context.v1 in glm52_goal.py.

[[ $# == 5 ]] || {
    echo "usage: $0 OUT TAG SEED_SHA256 CANDIDATE_HASH MODE" >&2
    exit 2
}
OUT=$1
TAG=$2
SEED_REQUEST=$3
CANDIDATE_HASH=$4
MODE=$5
RUN_MODE=root
if (( EUID != 0 )); then
    [[ $EUID == 1000 && $(id -un) == bmarti44 ]] ||
        { echo "context worker must run as root or bmarti44" >&2; exit 2; }
    RUN_MODE=user
    sudo -n -u dsv4 true ||
        { echo "passwordless dsv4 service-account delegation is unavailable" >&2; exit 2; }
fi
if [[ $RUN_MODE == root ]]; then
    valid_out_pattern=/var/lib/dsv4-context/attempts/dsv4-context-\*
    MODEL_PATH=$PROTECTED_MODEL_PATH
else
    valid_out_pattern=/home/bmarti44/.local/state/dsv4-context/dsv4-context-\*
    MODEL_PATH=$LIVE_MODEL_PATH
fi
readonly MODEL_PATH
[[ $OUT == $valid_out_pattern && -d $OUT && ! -L $OUT ]] ||
    { echo "invalid evidence directory" >&2; exit 2; }
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || { echo "invalid tag" >&2; exit 2; }
[[ $SEED_REQUEST == auto ]] || { echo "seed request must be auto" >&2; exit 2; }
SEED_SHA256=$(printf '%064d' 0)
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] ||
    { echo "invalid candidate hash" >&2; exit 2; }
[[ $MODE == one-million || $MODE == graduated ]] ||
    { echo "mode must be one-million or graduated" >&2; exit 2; }

verify_frozen_candidate() {
    "$PYTHON_BIN" \
        -I \
        "$REPO/scripts/62_score_dsv4_context.py" \
        --candidate-hash "$CANDIDATE_HASH" --verify-only
}
verify_frozen_candidate
if [[ $RUN_MODE == root ]]; then
    chmod -R a+rX "$REPO"
    chmod -R a-w "$REPO"
fi

ENGINE_ACTIVE=false
TELEMETRY_PID=
KERNEL_CURSOR=
ORIGINAL_RUNNING=false
RESTORE_ALLOWED=true
HOLD_OWNED=false

run_as_dsv4() {
    if [[ $RUN_MODE == root ]]; then
        /usr/sbin/runuser -u dsv4 -- "$@"
    else
        sudo -n -u dsv4 -- "$@"
    fi
}

start_failure_exists() {
    run_as_dsv4 test -e "$START_FAILURE_MARKER"
}

dsv4_launcher() {
    local context=$1 measured=$2
    shift 2
    # Normal/recovery contract: DSV4_MEM_FLOOR_GIB=18 and
    # DSV4_WATCHDOG_FLOOR_GIB=18. The exact 1M branch alone exports
    # DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=14.
    local floor=18 qualification_floor=0 batch=2048 ubatch=512 retry=0
    local -a hold_override=()
    if (( context == 1048576 && measured == 3 )); then
        floor=14
        qualification_floor=14
        batch=512
        ubatch=256
    fi
    (( context == SAFE_CTX && measured == 0 )) && retry=1
    if [[ $RUN_MODE == user ]]; then
        hold_override=(DSV4_START_HOLD_FILE="$WORKER_START_HOLD")
    fi
    run_as_dsv4 env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        DSV4_PORT=8013 \
        DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
        MODEL_PATH="$MODEL_PATH" DSV4_VERIFY_WEIGHTS=full \
        DSV4_EVICT_VERIFIED_WEIGHT_CACHE=1 \
        DSV4_MEM_FLOOR_GIB="$floor" DSV4_WATCHDOG_FLOOR_GIB="$floor" \
        DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB="$qualification_floor" \
        DSV4_MEASURED_HEADLESS_OVERHEAD_GIB="$measured" \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START="$retry" \
        DSV4_UBATCH="$ubatch" DSV4_BATCH="$batch" DSV4_UBATCH_LARGE=0 \
        CTX="$context" DSV4_PARALLEL=1 DSV4_NO_MMAP=1 \
        DSV4_SPEC_TYPE=none "${hold_override[@]}" \
        "$LAUNCHER" "$@"
}

run_context_probe() {
    local cap=$1 target=$2
    local -a command=(
        "$PYTHON_BIN" -I "$PROBE"
        --base-url http://127.0.0.1:8013
        --context-cap "$cap" --target-tokens "$target"
        --seed-sha256 "$SEED_SHA256" --out "$OUT/stage-$cap.json"
    )
    if [[ $RUN_MODE == root ]]; then
        env -i \
            HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 \
            PYTHONNOUSERSITE=1 PYTHONPATH= \
            PATH=/usr/bin:/bin \
            "${command[@]}"
    else
        env -i \
            HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 LANG=C.UTF-8 \
            PYTHONNOUSERSITE=1 PYTHONPATH= \
            PATH=/usr/bin:/bin \
            "${command[@]}"
    fi
}

stop_telemetry() {
    [[ ${TELEMETRY_PID:-} =~ ^[0-9]+$ ]] || return 0
    kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=
}

start_telemetry() {
    local cgroup_path
    cgroup_path=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup)
    [[ $cgroup_path == /* &&
        -r /sys/fs/cgroup$cgroup_path/memory.swap.current ]] ||
        { echo "cannot locate worker cgroup swap counter" >&2; return 1; }
    (
        trap 'exit 0' TERM INT
        while true; do
            swap_current=$(</sys/fs/cgroup$cgroup_path/memory.swap.current)
            awk -v now="$(date +%s.%N)" -v swap="$swap_current" '
                $1 == "MemAvailable:" { available = $2 / 1048576 }
                END {
                    printf "{\"timestamp_seconds\":%s,\"available_gib\":%.6f,\"swap_current_bytes\":%d}\n",
                           now, available, swap
                }' /proc/meminfo
            sleep 0.25
        done
    ) >>"$OUT/memory.jsonl" &
    TELEMETRY_PID=$!
}

require_no_kernel_fault() {
    journalctl -k --after-cursor "$KERNEL_CURSOR" --no-pager \
        >"$OUT/kernel-current.log" 2>&1
    if grep -Eiq "$FAULT_PATTERN" "$OUT/kernel-current.log"; then
        RESTORE_ALLOWED=false
        echo "new GPU/OOM kernel fault detected; refusing to advance" >&2
        return 1
    fi
}

activate_user_hold() {
    [[ $RUN_MODE == user ]] || return 0
    run_as_dsv4 bash -c \
        'set -Eeuo pipefail; set -o noclobber; umask 077; : >"$1"' \
        dsv4-context-hold "$START_HOLD"
    HOLD_OWNED=true
    for _ in {1..60}; do
        systemctl is-active --quiet dsv4-guard.service || return 0
        sleep 1
    done
    echo "dsv4 guard did not quiesce after maintenance hold" >&2
    return 1
}

prepare_safe_retry() {
    start_failure_exists || return 0
    /usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
        --interval-seconds 1 --timeout-seconds 180 \
        >"$OUT/recovery-admission.json" || return 1
    run_as_dsv4 rm -f -- "$START_FAILURE_MARKER"
}

restore_safe_profile() {
    "$RESTORE_ALLOWED" || {
        echo "safe-profile restore suppressed after a kernel/OOM fault" \
            >>"$OUT/restore.log"
        return 1
    }
    prepare_safe_retry >>"$OUT/restore.log" 2>&1 || return 1
    if "$ORIGINAL_RUNNING"; then
        if [[ $RUN_MODE == root ]]; then
            # A launcher started directly from a system transient unit inherits
            # that unit's cgroup. Restore through the persistent root unit.
            systemctl restart dsv4-engine-restore.service \
                >>"$OUT/restore.log" 2>&1 || return 1
        else
            # Keep the restored engine in a separate persistent user-systemd
            # cgroup. The qualification unit retains KillMode=control-group,
            # so a timeout still kills every 1M process without killing 8K
            # after this worker exits.
            systemd-run --user --unit="$USER_ENGINE_UNIT" \
                --property=Type=oneshot \
                --property=RemainAfterExit=yes \
                --property=KillMode=control-group \
                "$USER_RESTORE" >>"$OUT/restore.log" 2>&1 || return 1
            dsv4_launcher "$SAFE_CTX" 0 status \
                >>"$OUT/restore.log" 2>&1 || return 1
        fi
    fi
    if [[ $RUN_MODE == root ]]; then
        systemctl start dsv4-guard.timer >>"$OUT/restore.log" 2>&1
    elif "$HOLD_OWNED"; then
        run_as_dsv4 rm -f -- "$START_HOLD" >>"$OUT/restore.log" 2>&1 ||
            return 1
        HOLD_OWNED=false
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    stop_telemetry
    if "$ENGINE_ACTIVE"; then
        dsv4_launcher "$SAFE_CTX" 0 stop >>"$OUT/stop-cleanup.log" 2>&1
        ENGINE_ACTIVE=false
    fi
    if [[ -n $KERNEL_CURSOR ]]; then
        journalctl -k --after-cursor "$KERNEL_CURSOR" --no-pager \
            >"$OUT/kernel.log" 2>&1
        if grep -Eiq "$FAULT_PATTERN" "$OUT/kernel.log"; then
            RESTORE_ALLOWED=false
            rc=1
        fi
    fi
    start_failure_exists && rc=1
    restore_safe_profile || rc=1
    env -i HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 LANG=C.UTF-8 \
        PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONPATH= \
        INVOCATION_ID="${INVOCATION_ID:-}" \
        "$PYTHON_BIN" -I \
        "$REPO/scripts/62_score_dsv4_context.py" \
        --out "$OUT" --candidate-hash "$CANDIDATE_HASH" \
        --seed-sha256 "$SEED_SHA256" --mode "$MODE" \
        --lifecycle-exit-status "$rc" || rc=1
    env -i HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 LANG=C.UTF-8 \
        PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONPATH= \
        INVOCATION_ID="${INVOCATION_ID:-}" \
        "$PYTHON_BIN" -I \
        "$REPO/scripts/62_score_dsv4_context.py" \
        --out "$OUT" --candidate-hash "$CANDIDATE_HASH" \
        --seed-sha256 "$SEED_SHA256" --mode "$MODE" \
        --lifecycle-exit-status "$rc" --witness-final || rc=1
    chmod -R a+rX "$OUT" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT

SEED_SHA256=$(
    env -i HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 LANG=C.UTF-8 \
        PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONPATH= \
        INVOCATION_ID="${INVOCATION_ID:-}" \
        "$PYTHON_BIN" -I \
        "$REPO/scripts/62_score_dsv4_context.py" \
        --candidate-hash "$CANDIDATE_HASH" \
        --capture-lineage "$OUT/lineage.json"
)
[[ $SEED_SHA256 =~ ^[0-9a-f]{64}$ ]] ||
    { echo "captured seed is invalid" >&2; exit 1; }

KERNEL_CURSOR=$(
    journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p'
)
[[ -n $KERNEL_CURSOR ]] || { echo "cannot freeze kernel cursor" >&2; exit 1; }
if journalctl -k -b --no-pager | grep -Eiq "$PREEXISTING_FATAL_PATTERN"; then
    echo "pre-existing fatal kernel GPU/OOM fault on current boot" >&2
    exit 1
fi
# The root scheduler pauses the guard. The user scheduler instead creates the
# guard's existing persistent maintenance hold and waits out any in-flight
# oneshot. This grants no root capability to the worker.
if [[ $RUN_MODE == root ]]; then
    systemctl stop dsv4-guard.timer
    systemctl stop dsv4-guard.service
else
    activate_user_hold
fi

if dsv4_launcher "$SAFE_CTX" 0 status >/dev/null 2>&1; then
    ORIGINAL_RUNNING=true
    dsv4_launcher "$SAFE_CTX" 0 stop >"$OUT/original-stop.log" 2>&1
    if [[ $RUN_MODE == user ]]; then
        systemctl --user stop "$USER_ENGINE_UNIT" >>"$OUT/original-stop.log" 2>&1 ||
            true
    fi
elif [[ $RUN_MODE == user ]]; then
    echo "verified 8K DeepSeek engine is not running" >&2
    exit 1
fi
if [[ $RUN_MODE == root ]]; then
    systemctl stop display-manager.service >>"$OUT/display.log" 2>&1 || true
fi
systemctl is-active --quiet display-manager.service &&
    { echo "display manager remained active" >&2; exit 1; }
/usr/bin/python3 "$GUARD" --required-gib 115.0 --stable-samples 3 \
    --interval-seconds 1 --timeout-seconds 180 >"$OUT/admission.json"

start_telemetry
indices=(0 1 2 3)
[[ $MODE == graduated ]] || indices=(3)
for index in "${indices[@]}"; do
    verify_frozen_candidate
    cap=${CAPS[$index]}
    target=${TARGETS[$index]}
    dsv4_launcher "$cap" 3 start >"$OUT/start-$cap.log" 2>&1
    ENGINE_ACTIVE=true
    require_no_kernel_fault
    engine_log_bytes=$(run_as_dsv4 stat -c %s "$ENGINE_LOG")
    # Narrow the verified-path TOCTOU window to the probe exec boundary.
    verify_frozen_candidate
    set +e
    run_context_probe "$cap" "$target"
    probe_rc=$?
    set -e
    run_as_dsv4 tail -c "+$((engine_log_bytes + 1))" "$ENGINE_LOG" \
        >"$OUT/engine-$cap.log"
    env -i HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 LANG=C.UTF-8 \
        PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONPATH= \
        INVOCATION_ID="${INVOCATION_ID:-}" \
        "$PYTHON_BIN" -I \
        "$REPO/scripts/62_score_dsv4_context.py" \
        --out "$OUT" --candidate-hash "$CANDIDATE_HASH" \
        --seed-sha256 "$SEED_SHA256" --witness-stage "$cap"
    require_no_kernel_fault
    (( probe_rc == 0 ))
    dsv4_launcher "$cap" 3 stop >"$OUT/stop-$cap.log" 2>&1
    ENGINE_ACTIVE=false
    /usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
        --interval-seconds 1 --timeout-seconds 180 \
        >"$OUT/release-$cap.json"
done
