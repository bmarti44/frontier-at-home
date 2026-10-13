#!/usr/bin/env bash
# M3 candidate-1 driver (acceptance; PREREG.md here). One fresh start of
# mimo-v2.6-flash/cuda-spark-128g-262k on :8019, contained like the preflights
# (scope MemoryHigh 8G / MemoryMax 16G, as the profile's unit; a load-time
# page-cache release every 1 s; MemFree >= 108 GiB before the load), with the
# system inference lock held by this shell for the whole start.
# After ready: probe, agent gate, then the kit's speed, toolcall, vision,
# context and soak cells into this directory. Production qwen38fn is stopped
# through the switch first and restored at the end, also on failure.
set -Eeuo pipefail
WT=/home/bmarti44/mimo26
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/mimo-v26-flash-gates/qualify-2026-10-13-c1
R1=$WT/results/mimo-v26-flash-gates/smoke-2026-10-13
PROFILE=mimo-v2.6-flash/cuda-spark-128g-262k
SERVED=mimo-v2.6-flash
PORT=8019
BASE=http://127.0.0.1:$PORT
HOST=$R1/host-worktree.json
RT=/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001
PACK=/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash
DRAFT=/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash
OFFICIAL=/home/bmarti44/models/mimo-v2.6-flash-mopd
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
KIT=(python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT"
     --no-launch --out "$EV")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
OBS=$EV/observe
RELEASE_ROOTS=(/home/bmarti44/models/qwen3.8-flash-next-nvfp4 /home/bmarti44/.cache/qwen38-flash-next
               "$PACK" "$DRAFT" "$OFFICIAL" /home/bmarti44/models/mimo-v2.6-flash-mopd-exl3-2.52-src
               /home/bmarti44/models/mimo-v2.6-flash-dflash-exl3-src "$RT")
UNIT=mimo26-c1-$$
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
GIB=1048576
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new candidate directory" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]] || { echo "tracked edits; commit first" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain -- "$EV") ]] || { echo "uncommitted files under $EV" >&2; exit 2; }
# The scorer's own check: the working tree is the freeze's under every frozen path.
python3 "$EV/gate_verdict.py" --preflight || { echo "the tree is not the freeze's under FROZEN" >&2; exit 2; }
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
meminfo() { awk -v k="$1" '$1 == k":" {print $2}' /proc/meminfo; }
mkdir -p "$OBS"
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }

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
        if sudo -n "$PROD/scripts/52_engine_switch.sh" qwen38fn >>"$OBS/restore.log" 2>&1; then
            log "restore rc=0"
            return 0
        fi
        log "restore attempt $attempt failed"
        sleep 30
    done
    log "RESTORE_FAILED after 2 attempts"
    return 1
}
cleanup() {
    local rc=$?
    trap - EXIT
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    systemctl --user stop "$UNIT.scope" >>"$OBS/driver.log" 2>&1 || true
    [[ -n ${RELEASER:-} ]] && kill "$RELEASER" 2>/dev/null || true
    tail -n +"$((SERVER_OFFSET + 1))" "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null || true
    tail -n +"$((MEMWATCH_OFFSET + 1))" "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    exec 8>&-   # the switch's engine takes the inference lock itself
    restore_production || rc=1
    sleep 5
    if kill -0 "$JOURNAL" 2>/dev/null; then
        log "kernel journal alive at exit"
    else
        log "FAILED_ATTEMPT kernel journal not running at exit"
    fi
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
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
    local i run=0 free=0
    for ((i = 0; i < 120; i++)); do
        free=$(meminfo MemFree)
        if (( free >= 108 * GIB )); then
            (( ++run >= 3 )) && { log "MemFree gate pass ${free} kB (3 samples)"; return 0; }
        else
            run=0
        fi
        sleep 1
    done
    log "REFUSED MemFree ${free} kB < 108 GiB after release; nothing was started"
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
log "prod-stop: sudo -n 52_engine_switch.sh stop"
sudo -n "$PROD/scripts/52_engine_switch.sh" stop 2>&1 | tee -a "$OBS/driver.log"
# Only this shell holds the lock: every child starts with fd 8 closed.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { log "REFUSED $INFERENCE_LOCK is held after the production stop"; exit 1; }
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
# Runtime sources, the installed exllamav3 wheel and every served weight file, against their pins.
python3 "$EV/verify_identity.py" "$WT" 8>&- 2>&1 | tee -a "$OBS/driver.log" \
    || { log "REFUSED artifact identity; nothing was started"; exit 1; }

release_and_gate
# Keep the pack's page cache from crowding MemFree while the weights load
# (GB10 CUDA allocations come from MemFree and are not cgroup-charged).
( set +e
  while :; do
      sleep 1
      PYTHONPATH=$WT/scripts/lib python3 -c 'import sys; from checkpoint_cache import release_tree; [release_tree(r) for r in sys.argv[1:]]' "$PACK" "$DRAFT"
  done ) 8>&- &
RELEASER=$!
t0=$SECONDS
log "launching in scope $UNIT"
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=8G -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=14400 \
    -- "${SERVE[@]}" start 8>&- | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT launch did not reach ready"
    exit 1
fi
kill "$RELEASER" 2>/dev/null || true
RELEASER=
log "ready after $((SECONDS - t0)) s; idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
cgroup_snapshot "ready"

timeout 1800 python3 "$R1/probe.py" --base-url "$BASE" --out "$EV/probe-c1.json" 8>&- \
    | tee -a "$OBS/driver.log" || log "probe exit nonzero"
log "cell agent-gate"
timeout 1800 python3 "$WT/scripts/dev/regression-suite.py" agent-gate --base "$BASE" \
    --engine vllm --model "$SERVED" --prefix-block-tokens 256 --json "$EV/agent-gate.json" 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "agent-gate exit nonzero"
cgroup_snapshot "after agent-gate"
log "cell kit speed,toolcall,vision,context,soak"
timeout 12600 "${KIT[@]}" --cells speed,toolcall,vision,context --with-soak 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "kit exit nonzero"
cgroup_snapshot "before stop"
"${SERVE[@]}" stop 8>&- | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "scope empty after stop"; break; }
    sleep 1
done
if systemctl --user is-active -q "$UNIT.scope"; then
    log "FAILED_ATTEMPT surviving descendants in $UNIT.scope"
fi
log "window closed; postrun MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
