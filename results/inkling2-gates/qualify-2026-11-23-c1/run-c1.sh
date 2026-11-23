#!/usr/bin/env bash
# Inkling 2 Mini M3 candidate-1 driver (acceptance; PREREG.md here). One fresh start of
# inkling-2-mini/cuda-spark-128g-1m on :8018, contained like the screen arms (scope
# MemoryHigh 92G / MemoryMax 96G, swapless, OOMPolicy=kill), with the host page-cache
# release worker in the profile; MemFree >= 100 GiB and MemAvailable >= 110 GiB before the
# load; the system inference lock held by this shell for the whole start.
# After ready: probe, agent gate, then the kit's speed, toolcall, vision, context (the direct
# 4 x 250,128-token fill) and soak cells into this directory. Production qwen38fn is stopped
# through the switch first and restored at the end, also on failure.
# Adapted from results/mimo-v26-flash-gates/qualify-2026-10-13-c1/run-c1.sh.
#
# Launch only after any other GPU window has exited, with stdout to a file, never a pipe
# (a closed pipe fails the `| tee` steps; review B6):
#   setsid -f bash run-c1.sh > ~/.frontier-at-home/logs/inkling2-c1.log 2>&1
# ABORT: `systemctl --user stop inkling2-c1-<driver pid>.scope`. The cells then fail fast and
# the driver stops, closes the journal and restores production itself. Never SIGKILL the
# driver: bash defers a trapped TERM until the running foreground step returns (up to the
# kit's 12,600 s), and a killed driver leaves the engine up and production down (review B1).
set -Eeuo pipefail
WT=/home/bmarti44/inkling2
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/inkling2-gates/qualify-2026-11-23-c1
R1=$WT/results/inkling2-gates/smoke-2026-11-23
PROFILE=inkling-2-mini/cuda-spark-128g-1m
SERVED=inkling-2-mini
PORT=8018
# The agent gate's reuse floors: vLLM caches whole blocks of the KV groups' lcm (64 tokens,
# "kv lcm block sizes 64", also with DSpark on in screen arm a6); DSpark is a speculative drafter,
# so the floors allow one more trailing block, as qwen38fn c10 did for MTP (--spec-block-drop).
AGENT_BLOCK_ARGS=(--prefix-block-tokens 64 --spec-block-drop)
BASE=http://127.0.0.1:$PORT
HOST=$R1/host-worktree.json
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
KIT=(python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT"
     --no-launch --out "$EV")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
OBS=$EV/observe
RELEASE_ROOTS=(/home/bmarti44/models/qwen3.8-flash-next-nvfp4 /home/bmarti44/.cache/qwen38-flash-next
               /home/bmarti44/models/inkling-2-mini /home/bmarti44/.cache/inkling-2-mini)
UNIT=inkling2-c1-$$
# The restore's output goes outside the evidence: production's memwatch keeps it open.
RESTORE_LOG=$HOME/.frontier-at-home/logs/inkling2-c1-restore-$$.log
# regression-suite.py reads its agent-gate fixture from the production checkout.
AGENT_FIXTURE=$PROD/fixtures/ctx-32k.txt
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
GIB=1048576
[[ ! -e $OBS ]] || { echo "REFUSED $OBS exists; a rerun needs a new candidate directory" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]] || { echo "REFUSED tracked edits; commit first" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain -- "$EV") ]] || { echo "REFUSED uncommitted files under $EV" >&2; exit 2; }
# The scorer's own check: the working tree is the freeze's under every frozen path.
python3 "$EV/gate_verdict.py" --preflight || { echo "REFUSED the tree is not the freeze's under FROZEN" >&2; exit 2; }
cmp -s "$AGENT_FIXTURE" "$WT/fixtures/ctx-32k.txt" \
    || { echo "REFUSED $AGENT_FIXTURE differs from the frozen fixtures/ctx-32k.txt" >&2; exit 2; }
# Another window's engine must be gone before production is stopped (review B3).
[[ -z $(ss -Htln "sport = :$PORT") ]] || { echo "REFUSED :$PORT is listening; another window is running" >&2; exit 2; }
[[ -z $(systemctl --user list-units --state=active --no-legend 'inkling2-*' 2>/dev/null) ]] \
    || { echo "REFUSED an inkling2-* user unit is active; another window is running" >&2; exit 2; }
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
meminfo() { awk -v k="$1" '$1 == k":" {print $2}' /proc/meminfo; }
mkdir -p "$OBS"
log() {  # the file first; the terminal copy runs in a subshell, so a closed reader (EPIPE or SIGPIPE) fails only it
    local line
    line="$(stamp) $*"
    printf '%s\n' "$line" >>"$OBS/driver.log"
    (printf '%s\n' "$line") 2>/dev/null || true
}

( set +e
  cg=
  while :; do
      [[ -n $cg ]] || cg=$(systemctl --user show -P ControlGroup "$UNIT.scope" 2>/dev/null)
      printf '%s ' "$(stamp)"
      awk '/^(MemFree|MemAvailable|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Dirty):/{printf "%s=%s ", $1, $2}' /proc/meminfo
      [[ -n $cg && -r /sys/fs/cgroup$cg/memory.stat ]] \
          && awk '$1 == "anon" || $1 == "file" {printf "cg_%s=%s ", $1, $2}' "/sys/fs/cgroup$cg/memory.stat"
      printf '\n'
      sleep 1
  done >>"$OBS/meminfo.log" ) 8>&- &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 8>&- &
JOURNAL=$!
SERVER_OFFSET=$(( $(wc -l <"$SERVER_LOG" 2>/dev/null || echo 0) ))
MEMWATCH_OFFSET=$(( $(wc -l <"$MEMWATCH_LOG" 2>/dev/null || echo 0) ))

restore_production() {
    local attempt
    for attempt in 1 2; do
        for ((i = 0; i < 180; i++)); do
            (( $(meminfo MemAvailable) >= 110 * GIB )) && break
            sleep 1
        done
        log "restore attempt $attempt: MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
        # Its own session: production's memwatch must not join this driver's process group.
        # Default signal handling: cleanup ignores INT/TERM/HUP, and ignored signals are
        # inherited, which would disarm the memwatch's TERM trap (review round 2, C2-1).
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
# Clause 1's kernel log ends 5 s after the engine stop; the restore's kernel log
# is recorded in kernel-restore.log, not scored.
close_kernel_journal() {
    [[ -n ${JOURNAL:-} ]] || return 0
    if kill -0 "$JOURNAL" 2>/dev/null; then
        log "kernel journal alive after the engine stop"
    else
        log "FAILED_ATTEMPT kernel journal not running after the engine stop"
    fi
    kill "$JOURNAL" 2>/dev/null || true
    JOURNAL=
    ( trap - INT TERM HUP; exec journalctl -k -f -o short-iso --since now ) >"$OBS/kernel-restore.log" 2>&1 8>&- &
    RESTORE_JOURNAL=$!
}
cleanup() {
    local rc=$?
    # Nothing interrupts the stop and the restore: no second signal, no errexit.
    trap '' INT TERM HUP
    trap - EXIT
    set +e
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1
    systemctl --user stop "$UNIT.scope" >>"$OBS/driver.log" 2>&1
    tail -n +"$((SERVER_OFFSET + 1))" "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null
    tail -n +"$((MEMWATCH_OFFSET + 1))" "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null
    exec 8>&-   # the switch's engine takes the inference lock itself
    # Clause 1's kernel log closes 5 s after the engine stop on this path too (C2-2).
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
    # MemFree >= 100 GiB: the worker's arena check needs MemFree >= 63.33 + 2 GiB after weights and
    # graphs (the screen's a0 met it at 90.55 GiB); MemAvailable >= 110 GiB is the screen's guard.
    local i run=0 free=0 avail=0
    # 600 s, as the screen allowed: memory returns slowly after the production stop (review B2).
    for ((i = 0; i < 600; i++)); do
        free=$(meminfo MemFree)
        avail=$(meminfo MemAvailable)
        if (( free >= 100 * GIB && avail >= 110 * GIB )); then
            (( ++run >= 3 )) && { log "MemFree gate pass ${free} kB, MemAvailable ${avail} kB (3 samples)"; return 0; }
        else
            run=0
        fi
        sleep 1
    done
    log "REFUSED MemFree ${free} kB / MemAvailable ${avail} kB below 100 / 110 GiB after release; nothing was started"
    return 1
}

cgroup_snapshot() {  # cgroup_snapshot <label>
    local cg
    cg=$(systemctl --user show -P ControlGroup "$UNIT.scope" 2>/dev/null) || return 0
    [[ -n $cg ]] || return 0
    {
        printf '%s %s memory.peak=%s memory.current=%s\n' "$(stamp)" "$1" \
            "$(cat /sys/fs/cgroup$cg/memory.peak 2>/dev/null)" "$(cat /sys/fs/cgroup$cg/memory.current 2>/dev/null)"
        grep -E '^(anon|file|shmem|file_mapped) ' /sys/fs/cgroup$cg/memory.stat 2>/dev/null | sed "s/^/  /"
    } >>"$OBS/cgroup.log"
}

log "head $(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)"
log "agent-gate fixture sha256_16 $(sha256sum "$AGENT_FIXTURE" | cut -c1-16)"
log "prod-stop: sudo -n 52_engine_switch.sh stop"
sudo -n "$PROD/scripts/52_engine_switch.sh" stop 2>&1 | tee -a "$OBS/driver.log"
# Only this shell holds the lock: every long-lived child starts with fd 8 closed.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { log "REFUSED $INFERENCE_LOCK is held after the production stop"; exit 1; }
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
# The interpreter, the patched vLLM tree, the patch files and every weight file, against their pins.
python3 "$EV/verify_identity.py" "$WT" 8>&- 2>&1 | tee -a "$OBS/driver.log" \
    || { log "REFUSED artifact identity; nothing was started"; exit 1; }

release_and_gate
# The profile's HostCacheReleaseWorker evicts the checkpoint's page cache during the load and
# refuses the arena unless MemFree covers it (clause 12).
t0=$SECONDS
log "launching in scope $UNIT"
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=21600 \
    -- "${SERVE[@]}" start 8>&- 2>&1 | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT launch did not reach ready"
    exit 1
fi
log "ready after $((SECONDS - t0)) s; idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
cgroup_snapshot "ready"

# Rendered at effort low, as a6's reference probe was (review A-M3-1).
timeout 1800 python3 "$EV/probe_low.py" --base-url "$BASE" --out "$EV/probe-c1.json" 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "probe exit nonzero"
log "cell agent-gate"
timeout 1800 python3 "$WT/scripts/dev/regression-suite.py" agent-gate --base "$BASE" \
    --engine vllm --model "$SERVED" "${AGENT_BLOCK_ARGS[@]}" --json "$EV/agent-gate.json" 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "agent-gate exit nonzero"
cgroup_snapshot "after agent-gate"
log "cell kit speed,toolcall,vision,context,soak"
timeout 12600 "${KIT[@]}" --cells speed,toolcall,vision,context --with-soak 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "kit exit nonzero"
cgroup_snapshot "before stop"
"${SERVE[@]}" stop 8>&- 2>&1 | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "scope empty after stop"; break; }
    sleep 1
done
if systemctl --user is-active -q "$UNIT.scope"; then
    log "FAILED_ATTEMPT surviving descendants in $UNIT.scope"
fi
log "window closed; postrun MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
sleep 5
close_kernel_journal
