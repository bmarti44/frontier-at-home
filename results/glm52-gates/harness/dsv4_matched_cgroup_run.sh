#!/usr/bin/env bash
# Dedicated current-user containment for the owner-accepted DSV4 matched arm.
# It deliberately does not alter glm_safe_run.sh's >=18 GiB GLM floor.
set -Eeuo pipefail
umask 077

if [[ ${1:-} == --inner ]]; then
    shift
    DIR=${1:?missing crash directory}
    UNIT=${2:?missing cgroup unit}
    FLOOR_GIB=${3:?missing kill floor}
    shift 3
    [[ ${1:-} == -- ]] && shift
    (( $# > 0 )) || exit 2
    MAIN=$DIR/main.log
    SAMPLES=$DIR/samples.log
    KERNEL=$DIR/kernel.log
    CGROUP_PATH=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
    [[ $CGROUP_PATH == */"$UNIT.service" ]] || exit 2
    CGROUP_DIR=/sys/fs/cgroup$CGROUP_PATH
    [[ -r $CGROUP_DIR/memory.current && -r $CGROUP_DIR/memory.swap.current &&
       -r $CGROUP_DIR/memory.events.local ]] || exit 2
    declare -A BEFORE=()
    while read -r key value; do BEFORE[$key]=$value; done <"$CGROUP_DIR/memory.events.local"
    printf '%s cgroup_verified path=%s memory_high=%s memory_max=%s memory_swap_max=%s\n' \
        "$(date -u --iso-8601=ns)" "$CGROUP_DIR" \
        "$(<"$CGROUP_DIR/memory.high")" "$(<"$CGROUP_DIR/memory.max")" \
        "$(<"$CGROUP_DIR/memory.swap.max")" >>"$MAIN"
    cursor=$(journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p')
    [[ -n $cursor ]] || exit 2
    ulimit -v 419430400
    setsid timeout --signal=TERM --kill-after=30 "${DSV4_MATCHED_TIMEOUT_S}s" \
        "$@" >"$DIR/cmd.log" 2>&1 &
    wrapper_pid=$!
    pgid=$wrapper_pid
    killed=no
    while kill -0 "$wrapper_pid" 2>/dev/null; do
        mem_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        current=$(<"$CGROUP_DIR/memory.current")
        swap=$(<"$CGROUP_DIR/memory.swap.current")
        printf '%s mem_avail_kb=%s cgroup_current_bytes=%s cgroup_swap_current_bytes=%s\n' \
            "$(date -u --iso-8601=ns)" "$mem_kib" "$current" "$swap" >>"$SAMPLES"
        sync -d "$SAMPLES" 2>/dev/null || true
        if (( mem_kib < FLOOR_GIB * 1048576 )); then
            killed=floor
            kill -KILL -- "-$pgid" 2>/dev/null || true
            break
        fi
        sleep 0.25
    done
    set +e
    wait "$wrapper_pid"
    rc=$?
    set -e
    [[ $killed == no ]] || rc=137
    survivors=()
    for _ in $(seq 1 20); do
        survivors=()
        while read -r cgroup_pid; do
            [[ $cgroup_pid == "$$" ]] || survivors+=("$cgroup_pid")
        done <"$CGROUP_DIR/cgroup.procs"
        (( ${#survivors[@]} == 0 )) && break
        sleep 0.05
    done
    if (( ${#survivors[@]} != 0 )); then
        printf '%s FATAL contained descendants survived command exit pids=%s\n' \
            "$(date -u --iso-8601=ns)" "${survivors[*]}" >>"$MAIN"
        for survivor in "${survivors[@]}"; do
            kill -KILL "$survivor" 2>/dev/null || true
        done
        rc=15
    fi
    event_failure=0
    events=()
    while read -r key value; do
        before=${BEFORE[$key]:-0}
        delta=$((value - before))
        events+=("$key=$value" "${key}_delta=$delta")
        if [[ $key =~ ^(high|max|oom|oom_kill|oom_group_kill)$ ]] && (( delta != 0 )); then
            event_failure=1
        fi
    done <"$CGROUP_DIR/memory.events.local"
    swap=$(<"$CGROUP_DIR/memory.swap.current")
    current=$(<"$CGROUP_DIR/memory.current")
    peak=$(<"$CGROUP_DIR/memory.peak")
    (( swap == 0 && event_failure == 0 )) || rc=14
    journalctl -k --after-cursor "$cursor" --no-pager >"$KERNEL"
    if grep -Eiq 'NVRM.*Xid|NV_ERR_NO_MEMORY|oom-kill|Out of memory: Killed process' \
        "$KERNEL" "$DIR/cmd.log"; then
        rc=16
    fi
    printf '%s cgroup_final current_bytes=%s peak_bytes=%s swap_current_bytes=%s events=%s\n' \
        "$(date -u --iso-8601=ns)" "$current" "$peak" "$swap" \
        "$(IFS=,; echo "${events[*]}")" >>"$MAIN"
    tail -25 "$DIR/cmd.log" >>"$MAIN" 2>/dev/null || true
    printf '%s SAFE_RUN end rc=%s killed=%s (124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n' \
        "$(date -u --iso-8601=ns)" "$rc" "$killed" >>"$MAIN"
    sync -d "$MAIN" "$SAMPLES" "$KERNEL" 2>/dev/null || true
    main_sha=$(sha256sum "$MAIN" | awk '{print $1}')
    samples_sha=$(sha256sum "$SAMPLES" | awk '{print $1}')
    kernel_sha=$(sha256sum "$KERNEL" | awk '{print $1}')
    printf 'SAFE_RUN_DONE rc=%s killed=%s dir=%s main_sha256=%s samples_sha256=%s kernel_sha256=%s\n' \
        "$rc" "$killed" "$DIR" "$main_sha" "$samples_sha" "$kernel_sha"
    exit "$rc"
fi

[[ $# -ge 3 && ${1:-} == --tag && -n ${2:-} ]] || {
    echo "usage: $0 --tag NAME -- command..." >&2
    exit 2
}
TAG=$2
shift 2
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$ ]] || exit 2
[[ ${1:-} == -- ]] && shift
(( $# > 0 )) || exit 2
[[ $(id -u) != 0 && $(id -un) == bmarti44 ]] || exit 2

FLOOR=${DSV4_MATCHED_KILL_FLOOR_GIB:-}
MIN_START=${DSV4_MATCHED_MIN_START_GIB:-}
HIGH=${DSV4_MATCHED_MEMORY_HIGH_GIB:-}
MAX=${DSV4_MATCHED_MEMORY_MAX_GIB:-}
TIMEOUT=${DSV4_MATCHED_TIMEOUT_S:-}
[[ $FLOOR == 8 && $MIN_START == 110 && $HIGH == 105 && $MAX == 107 &&
   $TIMEOUT == 5400 ]] || {
    echo "invalid DSV4 matched containment envelope" >&2
    exit 2
}

PARENT=${GLM_SAFE_PARENT_LOCK_PID:-}
TICKS=${GLM_SAFE_PARENT_LOCK_START_TICKS:-}
FD=${GLM_SAFE_PARENT_LOCK_FD:-}
DEVINO=${GLM_SAFE_PARENT_LOCK_DEV_INO:-}
KEY=${GLM_SAFE_PARENT_LOCK_KERNEL_KEY:-}
[[ $PARENT == "$PPID" && $PARENT =~ ^[1-9][0-9]*$ && $TICKS =~ ^[1-9][0-9]*$ &&
   $FD =~ ^[0-9]+$ && $DEVINO =~ ^[0-9]+:[0-9]+$ &&
   $KEY =~ ^[0-9a-f]+:[0-9a-f]+:[0-9]+$ ]] || exit 2
[[ $(awk '{print $22}' "/proc/$PARENT/stat") == "$TICKS" &&
   $(stat -Lc '%d:%i' "/proc/$PARENT/fd/$FD") == "$DEVINO" &&
   $(stat -Lc '%d:%i' /run/lock/frontier-at-home/inference.lock) == "$DEVINO" ]] || exit 2
awk -v key="$KEY" '
  $1 == "lock:" && $3 == "FLOCK" && $5 == "WRITE" && $7 == key { count++ }
  END { exit(count == 1 ? 0 : 1) }
' "/proc/$PARENT/fdinfo/$FD" || exit 2

available_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( available_mib >= MIN_START * 1024 && available_mib >= (MAX + FLOOR) * 1024 )) || {
    echo "DSV4 matched envelope cannot preserve the owner-accepted floor" >&2
    exit 8
}
CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
mkdir -p -- "$CRASH_ROOT"
DIR=$CRASH_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$TAG
mkdir -- "$DIR"
UNIT=dsv4-matched-${TAG//./-}-$$
SELF=$(readlink -f -- "$0")
RUN_CLIENT_PID=
UNIT_ACTIVE=0
stop_unit() {
    trap - INT TERM HUP
    if (( UNIT_ACTIVE )); then
        systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
        for _ in $(seq 1 100); do
            [[ $(systemctl --user is-active "$UNIT.service" 2>/dev/null || true) != active ]] && break
            sleep 0.05
        done
    fi
    if [[ ${RUN_CLIENT_PID:-} =~ ^[1-9][0-9]*$ ]] && kill -0 "$RUN_CLIENT_PID" 2>/dev/null; then
        kill -TERM "$RUN_CLIENT_PID" 2>/dev/null || true
        wait "$RUN_CLIENT_PID" 2>/dev/null || true
    fi
    UNIT_ACTIVE=0
}
handle_signal() {
    local signal_name=$1 signal_rc=$2
    printf 'DSV4_MATCHED_INTERRUPTED signal=%s unit=%s\n' "$signal_name" "$UNIT" >&2
    stop_unit
    exit "$signal_rc"
}
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
env_args=(
    HOME=/home/bmarti44
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    "DSV4_MATCHED_TIMEOUT_S=$TIMEOUT"
)
for name in DSV4_MATCHED_BINARY DSV4_MATCHED_BINARY_SHA256 \
    DSV4_MATCHED_MODEL_FIRST DSV4_MATCHED_SHARDS_JSON MATCHED_BENCH_PATH \
    MATCHED_PYTHON_PATH MATCHED_TOKENIZER_NATIVE_PATH \
    MATCHED_TOKENIZER_NATIVE_SHA256 MATCHED_PORT
do
    [[ -v $name ]] && env_args+=("$name=${!name}")
done
set +e
UNIT_ACTIVE=1
systemd-run --user --wait --collect --pipe --quiet --expand-environment=no \
    --working-directory="$(pwd -P)" --unit="$UNIT" --service-type=exec \
    -p KillMode=control-group -p SendSIGKILL=yes -p TimeoutStopSec=45s \
    -p "RuntimeMaxSec=$((TIMEOUT + 60))s" -p MemoryAccounting=yes \
    -p "MemoryHigh=${HIGH}G" -p "MemoryMax=${MAX}G" -p MemorySwapMax=0 \
    -p OOMPolicy=kill -p TasksMax=4096 -- \
    /usr/bin/env -i "${env_args[@]}" /usr/bin/bash "$SELF" \
        --inner "$DIR" "$UNIT" "$FLOOR" -- "$@" &
RUN_CLIENT_PID=$!
wait "$RUN_CLIENT_PID"
rc=$?
set -e
stop_unit
exit "$rc"
