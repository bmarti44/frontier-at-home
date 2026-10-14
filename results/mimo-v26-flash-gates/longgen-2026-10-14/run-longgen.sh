#!/usr/bin/env bash
# MiMo long-generation diagnostic driver (PREREG.md here; DIAGNOSTIC). One
# production-down window, three fresh starts on :8019, in a fixed order:
#   a  the qualified mimo-v2.6-flash/cuda-spark-128g-262k launch;
#   b  the same with --output-chunking false;
#   c  the same without the DFlash drafter.
# Each start runs the restart probe, then longgen.py's three fixtures. The
# stop/lock/identity/gate/containment/restore machinery is the holdout's
# (../holdout-2026-10-13/run-holdout.sh, reviewed). Production qwen38fn is
# stopped through the switch first and restored at the end, also on failure.
set -Eeuo pipefail
WT=/home/bmarti44/mimo26
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/mimo-v26-flash-gates/longgen-2026-10-14
C3=$WT/results/mimo-v26-flash-gates/qualify-2026-10-13-c1
R1=$WT/results/mimo-v26-flash-gates/smoke-2026-10-13
BASE_PROFILE=mimo-v2.6-flash/cuda-spark-128g-262k
declare -A ARM_PROFILE=([a]=$BASE_PROFILE [b]=$BASE_PROFILE-longgen-b [c]=$BASE_PROFILE-longgen-c)
ARM_ORDER=(a b c)
PORT=8019
BASE=http://127.0.0.1:$PORT
HOST=$R1/host-worktree.json
RT=/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001
PACK=/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash
DRAFT=/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash
OFFICIAL=/home/bmarti44/models/mimo-v2.6-flash-mopd
LOGS=$HOME/.frontier-at-home/logs
OBS=$EV/observe
RELEASE_ROOTS=(/home/bmarti44/models/qwen3.8-flash-next-nvfp4 /home/bmarti44/.cache/qwen38-flash-next
               "$PACK" "$DRAFT" "$OFFICIAL" /home/bmarti44/models/mimo-v2.6-flash-mopd-exl3-2.52-src
               /home/bmarti44/models/mimo-v2.6-flash-dflash-exl3-src "$RT")
CAP=3600   # per arm
# The restore's output goes outside the evidence: production's memwatch keeps it open.
RESTORE_LOG=$LOGS/mimo26-longgen-restore-$$.log
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
GIB=1048576
KILL_KERNEL='out of memory|oom-kill|oom_reaper|NVRM|Xid'
KILL_MEMWATCH='BREACH|EMERGENCY_STOP|FAIL_CLOSED'
[[ ! -e $OBS ]] || { echo "REFUSED $OBS exists; this window has run" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]] || { echo "REFUSED tracked edits; commit first" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain -- "$EV") ]] || { echo "REFUSED uncommitted files under $EV" >&2; exit 2; }
for arm in "${ARM_ORDER[@]}"; do
    python3 "$WT/scripts/92_resolve_profile.py" check --profile "${ARM_PROFILE[$arm]}" >/dev/null \
        || { echo "REFUSED arm $arm profile ${ARM_PROFILE[$arm]} does not resolve" >&2; exit 2; }
done
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
meminfo() { awk -v k="$1" '$1 == k":" {print $2}' /proc/meminfo; }
unit_of() { echo "mimo26-longgen-$1-$$"; }
slug_of() { local p=${ARM_PROFILE[$1]}; echo "${p//\//-}"; }
serve() {  # serve <arm> start|stop
    bash "$WT/scripts/93_profile_serve.sh" --profile "${ARM_PROFILE[$1]}" --host "$HOST" --port "$PORT" "$2"
}
mkdir -p "$OBS"
log() {  # the file first; the terminal copy runs in a subshell, so a closed reader fails only it
    local line
    line="$(stamp) $*"
    printf '%s\n' "$line" >>"$OBS/driver.log"
    (printf '%s\n' "$line") 2>/dev/null || true
}

( set +e
  while :; do
      printf '%s ' "$(stamp)"
      awk '/^(MemFree|MemAvailable|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Dirty):/{printf "%s=%s ", $1, $2}' /proc/meminfo
      for arm in a b c; do
          cg=$(systemctl --user show -P ControlGroup "mimo26-longgen-$arm-$$.scope" 2>/dev/null)
          [[ -n $cg && -r /sys/fs/cgroup$cg/memory.stat ]] \
              && awk -v a="$arm" '$1 == "anon" || $1 == "file" {printf "cg_%s_%s=%s ", a, $1, $2}' "/sys/fs/cgroup$cg/memory.stat"
      done
      printf '\n'
      sleep 1
  done >>"$OBS/meminfo.log" ) 8>&- &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 8>&- &
JOURNAL=$!

restore_production() {
    local attempt
    for attempt in 1 2; do
        for ((i = 0; i < 180; i++)); do
            (( $(meminfo MemAvailable) >= 110 * GIB )) && break
            sleep 1
        done
        log "restore attempt $attempt: MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
        # Its own session, default signal handling (holdout review B9, C2-1).
        if ( trap - INT TERM HUP; exec setsid -w sudo -n "$PROD/scripts/52_engine_switch.sh" qwen38fn ) \
            >>"$RESTORE_LOG" 2>&1 </dev/null; then
            log "restore rc=0"
            return 0
        fi
        log "restore attempt $attempt failed"
        sleep 30
    done
    log "RESTORE_FAILED after 2 attempts"
    return 1
}
close_kernel_journal() {
    [[ -n ${JOURNAL:-} ]] || return 0
    if kill -0 "$JOURNAL" 2>/dev/null; then
        log "kernel journal alive after the last engine stop"
    else
        log "FAILED_ATTEMPT kernel journal not running after the last engine stop"
    fi
    kill "$JOURNAL" 2>/dev/null || true
    JOURNAL=
    ( trap - INT TERM HUP; exec journalctl -k -f -o short-iso --since now ) >"$OBS/kernel-restore.log" 2>&1 8>&- &
    RESTORE_JOURNAL=$!
}
SERVER_OFFSET=0
MEMWATCH_OFFSET=0
collect_arm_logs() {  # collect_arm_logs <arm>
    local slug
    slug=$(slug_of "$1")
    mkdir -p "$OBS/arm-$1"
    tail -n +"$((SERVER_OFFSET + 1))" "$LOGS/$slug.server.log" >"$OBS/arm-$1/server.log" 2>/dev/null
    tail -n +"$((MEMWATCH_OFFSET + 1))" "$LOGS/$slug.memwatch.log" >"$OBS/arm-$1/memwatch.log" 2>/dev/null
    return 0
}
stop_arm() {  # stop_arm <arm>: stop the server, wait for the empty scope, keep its logs
    local unit
    unit=$(unit_of "$1")
    serve "$1" stop 8>&- 2>&1 | tee -a "$OBS/driver.log"
    for ((i = 0; i < 60; i++)); do
        systemctl --user is-active -q "$unit.scope" || { log "arm $1: scope empty after stop"; break; }
        sleep 1
    done
    if systemctl --user is-active -q "$unit.scope"; then
        log "FAILED_ATTEMPT arm $1: surviving descendants in $unit.scope"
        systemctl --user stop "$unit.scope" >>"$OBS/driver.log" 2>&1
    fi
    collect_arm_logs "$1"
}
CURRENT_ARM=
cleanup() {
    local rc=$?
    # Nothing interrupts the stop and the restore: no second signal, no errexit.
    trap '' INT TERM HUP
    trap - EXIT
    set +e
    if [[ -n $CURRENT_ARM ]]; then
        serve "$CURRENT_ARM" stop >>"$OBS/driver.log" 2>&1
        systemctl --user stop "$(unit_of "$CURRENT_ARM").scope" >>"$OBS/driver.log" 2>&1
        collect_arm_logs "$CURRENT_ARM"
    fi
    [[ -n ${RELEASER:-} ]] && kill "$RELEASER" 2>/dev/null
    exec 8>&-   # the switch's engine takes the inference lock itself
    [[ -n ${JOURNAL:-} ]] && sleep 5
    close_kernel_journal
    restore_production || rc=1
    cp "$RESTORE_LOG" "$OBS/restore.log" 2>/dev/null
    sleep 5
    kill "$SAMPLER" ${RESTORE_JOURNAL:-} 2>/dev/null
    log "driver exit rc=$rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP INT

release_and_gate() {
    PYTHONPATH=$WT/scripts/lib python3 - "${RELEASE_ROOTS[@]}" <<'PY' | tee -a "$OBS/driver.log"
import sys
from checkpoint_cache import release_tree
for root in sys.argv[1:]:
    print(f"released {release_tree(root)} files under {root}")
PY
    local i run=0 free=0 avail=0
    for ((i = 0; i < 180; i++)); do
        free=$(meminfo MemFree)
        avail=$(meminfo MemAvailable)
        if (( free >= 108 * GIB && avail >= 114 * GIB )); then
            (( ++run >= 3 )) && { log "MemFree gate pass ${free} kB, MemAvailable ${avail} kB (3 samples)"; return 0; }
        else
            run=0
        fi
        sleep 1
    done
    log "REFUSED MemFree ${free} kB / MemAvailable ${avail} kB below 108 / 114 GiB after release"
    return 1
}

log "head $(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)"
log "prod-stop: sudo -n 52_engine_switch.sh stop"
sudo -n "$PROD/scripts/52_engine_switch.sh" stop 2>&1 | tee -a "$OBS/driver.log"
# Only this shell holds the lock: every long-lived child starts with fd 8 closed.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { log "REFUSED $INFERENCE_LOCK is held after the production stop"; exit 1; }
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
python3 "$C3/verify_identity.py" "$WT" 8>&- 2>&1 | tee -a "$OBS/driver.log" \
    || { log "REFUSED artifact identity; nothing was started"; exit 1; }

for arm in "${ARM_ORDER[@]}"; do
    slug=$(slug_of "$arm")
    unit=$(unit_of "$arm")
    release_and_gate || exit 1
    ( set +e
      while :; do
          sleep 1
          PYTHONPATH=$WT/scripts/lib python3 -c 'import sys; from checkpoint_cache import release_tree; [release_tree(r) for r in sys.argv[1:]]' "$PACK" "$DRAFT"
      done ) 8>&- &
    RELEASER=$!
    SERVER_OFFSET=$(( $(wc -l <"$LOGS/$slug.server.log" 2>/dev/null || echo 0) ))
    MEMWATCH_OFFSET=$(( $(wc -l <"$LOGS/$slug.memwatch.log" 2>/dev/null || echo 0) ))
    CURRENT_ARM=$arm
    t0=$SECONDS
    log "arm $arm: launching ${ARM_PROFILE[$arm]} in scope $unit"
    if ! systemd-run --user --scope -q --unit="$unit" \
        -p MemoryHigh=8G -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec="$CAP" \
        -- bash "$WT/scripts/93_profile_serve.sh" --profile "${ARM_PROFILE[$arm]}" --host "$HOST" --port "$PORT" start \
        8>&- 2>&1 | tee -a "$OBS/driver.log"; then
        log "FAILED_ATTEMPT arm $arm: launch did not reach ready"
        exit 1
    fi
    kill "$RELEASER" 2>/dev/null || true
    RELEASER=
    log "arm $arm: ready after $((SECONDS - t0)) s; idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
    timeout 900 python3 "$R1/probe.py" --base-url "$BASE" --out "$EV/probe-$arm.json" 8>&- 2>&1 \
        | tee -a "$OBS/driver.log" || log "arm $arm: probe exit nonzero rc=${PIPESTATUS[0]}"
    timeout 2700 python3 "$EV/longgen.py" run --arm "$arm" --base-url "$BASE" --out "$EV/arm-$arm.json" 8>&- 2>&1 \
        | tee -a "$OBS/driver.log" || log "arm $arm: longgen exit nonzero rc=${PIPESTATUS[0]}"
    stop_arm "$arm"
    CURRENT_ARM=
    log "arm $arm: closed; MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
    # A kill line or a memwatch breach ends the window: no further arm starts.
    if grep -E -i -q "$KILL_KERNEL" "$OBS/kernel.log" || grep -E -q "$KILL_MEMWATCH" "$OBS/arm-$arm/memwatch.log"; then
        log "FAILED_ATTEMPT arm $arm: kill line or memwatch breach; no further arms"
        break
    fi
done
log "window closed; postrun MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
sleep 5
close_kernel_journal
