#!/usr/bin/env bash
# MiMo holdout driver (README.md here; owner-authorized, spend-once). One fresh
# start of the qualified mimo-v2.6-flash/cuda-spark-128g-262k on :8019 with M3
# candidate 3's containment and run-c1.sh's reviewed stop/lock/identity/gate/
# restore machinery (../qualify-2026-10-13-c1/run-c1.sh). After ready: the
# restart probe; only if its three greedy hashes equal candidate 3's, the kit's
# accuracy cell on the holdout split. Production qwen38fn is stopped through
# the switch first and restored at the end, also on failure.
set -Eeuo pipefail
WT=/home/bmarti44/mimo26
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/mimo-v26-flash-gates/holdout-2026-10-13
C3=$WT/results/mimo-v26-flash-gates/qualify-2026-10-13-c1
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
UNIT=mimo26-holdout-$$
CAP=43200
# The restore's output goes outside the evidence: production's memwatch keeps it open.
RESTORE_LOG=$HOME/.frontier-at-home/logs/mimo26-holdout-restore-$$.log
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
GIB=1048576
[[ ! -e $OBS ]] || { echo "REFUSED $OBS exists; the holdout is spend-once" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]] || { echo "REFUSED tracked edits; commit first" >&2; exit 2; }
[[ -z $(git -C "$WT" status --porcelain -- "$EV") ]] || { echo "REFUSED uncommitted files under $EV" >&2; exit 2; }
# README: the ledger namespace is "" (31_bench_accuracy.py reads DSV4_LEDGER_NAMESPACE).
unset DSV4_LEDGER_NAMESPACE
# HumanEval's sandbox fails closed at its suite start, after the holdout is drawn:
# check it with the bench's own resolver before anything stops.
python3 - "$WT/scripts/31_bench_accuracy.py" <<'PY' || { echo "REFUSED the HumanEval runtime does not resolve" >&2; exit 2; }
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bench", sys.argv[1])
bench = importlib.util.module_from_spec(spec)
sys.modules["bench"] = bench
spec.loader.exec_module(bench)
print("humaneval runtime", bench.resolve_humaneval_runtime_digest())
PY
# The draw runs on candidate 3's launch: M4 changed only the profile's status,
# admission note and memory model. The kit's dry render at this head must equal
# candidate 3's recorded launch (config_hash covers argv and env).
RENDER=$(mktemp -d)
python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT" --no-launch \
    --out "$RENDER" --dry-run --cells accuracy --accuracy-split holdout --reasoning-effort '' \
    >"$RENDER/dry-run.log" 2>&1 || { echo "REFUSED the kit's dry render failed (see $RENDER/dry-run.log)" >&2; exit 2; }
RENDER_LINE=$(python3 - "$RENDER/manifest.json" "$C3/manifest.json" 2>&1 <<'PY'
import json, sys
a, b = (json.load(open(p))["profile"] for p in sys.argv[1:3])
keys = ("config_hash", "argv", "env", "containment", "served_model", "model_path", "tokenizer_sha256",
        "slots", "tokens_per_slot", "context_cap", "minimum_start_gib", "kill_floor_gib", "encoder")
diff = [k for k in keys if a.get(k) != b.get(k)]
if diff:
    sys.exit("differs from candidate 3 in " + ", ".join(diff))
print(f"render identical to candidate 3: config_hash {a['config_hash']} (status {a['profile_status']})")
PY
) || { echo "REFUSED the launch render: $RENDER_LINE" >&2; exit 2; }
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
    [[ -n ${RELEASER:-} ]] && kill "$RELEASER" 2>/dev/null
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
    # MemAvailable >= 114 GiB is the profile's minimum_start_gib: preflight r2 used 105.52 GiB
    # over its 8 GiB floor (review round 1, B3).
    local i run=0 free=0 avail=0
    for ((i = 0; i < 120; i++)); do
        free=$(meminfo MemFree)
        avail=$(meminfo MemAvailable)
        if (( free >= 108 * GIB && avail >= 114 * GIB )); then
            (( ++run >= 3 )) && { log "MemFree gate pass ${free} kB, MemAvailable ${avail} kB (3 samples)"; return 0; }
        else
            run=0
        fi
        sleep 1
    done
    log "REFUSED MemFree ${free} kB / MemAvailable ${avail} kB below 108 / 114 GiB after release; nothing was started"
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
log "$RENDER_LINE"
cp "$RENDER/manifest.json" "$OBS/render-manifest.json"
log "prod-stop: sudo -n 52_engine_switch.sh stop"
sudo -n "$PROD/scripts/52_engine_switch.sh" stop 2>&1 | tee -a "$OBS/driver.log"
# Only this shell holds the lock: every long-lived child starts with fd 8 closed.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { log "REFUSED $INFERENCE_LOCK is held after the production stop"; exit 1; }
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
# Runtime sources, the installed exllamav3 wheel and every served weight file, against their pins.
python3 "$C3/verify_identity.py" "$WT" 8>&- 2>&1 | tee -a "$OBS/driver.log" \
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
    -p MemoryHigh=8G -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec="$CAP" \
    -- "${SERVE[@]}" start 8>&- 2>&1 | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT launch did not reach ready"
    exit 1
fi
kill "$RELEASER" 2>/dev/null || true
RELEASER=
log "ready after $((SECONDS - t0)) s; idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
cgroup_snapshot "ready"

timeout 1800 python3 "$R1/probe.py" --base-url "$BASE" --out "$EV/probe-holdout.json" 8>&- 2>&1 \
    | tee -a "$OBS/driver.log" || log "probe exit nonzero"
# The holdout is drawn only on the configuration candidate 3 qualified: the
# three greedy hashes must equal candidate 3's (restart identity).
if python3 - "$EV/probe-holdout.json" "$C3/probe-c1.json" <<'PY'
import json, sys
a, b = ([g["sha256"] for g in json.load(open(p))["greedy"]] for p in sys.argv[1:3])
sys.exit(0 if len(a) == 3 and a == b else 1)
PY
then
    log "probe identical to candidate 3; drawing the holdout"
    timeout 41400 "${KIT[@]}" --cells accuracy --accuracy-split holdout --reasoning-effort '' 8>&- 2>&1 \
        | tee -a "$OBS/driver.log" || log "kit exit nonzero"
else
    log "FAILED_ATTEMPT probe differs from candidate 3; holdout not drawn"
fi
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
