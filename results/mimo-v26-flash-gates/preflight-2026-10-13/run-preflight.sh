#!/usr/bin/env bash
# M3 preflight driver (DIAGNOSTIC; PREREG.md here). One fresh start of the
# profile on :8019, contained like smoke r3 (scope MemoryHigh 8G / MemoryMax 16G,
# load-time page-cache release every 1 s, MemFree >= 108 GiB before the load).
# It runs the M3 cells the smoke never exercised against TabbyAPI: the agent
# gate, kit vision (20 cases) and context (direct 250,128-token fill), and the
# accuracy encoder path on the GSM8K dev split (cut at 1200 s; no ledger).
# Production qwen38fn is stopped through the switch first and restored at the
# end, also on failure.
set -Eeuo pipefail
WT=/home/bmarti44/mimo26
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/mimo-v26-flash-gates/preflight-2026-10-13
R1=$WT/results/mimo-v26-flash-gates/smoke-2026-10-13
R3=$WT/results/mimo-v26-flash-gates/smoke-2026-10-13-r3
PROFILE=mimo-v2.6-flash/cuda-spark-128g-262k
LABEL=mimo-v2.6-flash-cuda-spark-128g-262k
SERVED=mimo-v2.6-flash
PORT=8019
BASE=http://127.0.0.1:$PORT
HOST=$R1/host-worktree.json
RT=/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001
PACK=/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash
DRAFT=/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash
OFFICIAL=/home/bmarti44/models/mimo-v2.6-flash-mopd
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
OBS=$EV/observe
RELEASE_ROOTS=(/home/bmarti44/models/qwen3.8-flash-next-nvfp4 /home/bmarti44/.cache/qwen38-flash-next
               "$PACK" "$DRAFT" "$OFFICIAL" /home/bmarti44/models/mimo-v2.6-flash-mopd-exl3-2.52-src
               /home/bmarti44/models/mimo-v2.6-flash-dflash-exl3-src "$RT")
UNIT=mimo26-preflight-$$
GIB=1048576
# The pinned tokenizer digest, from the weights manifest (the served pack's artifact).
TOKENIZER_SHA256=$(python3 -c 'import json, sys
m = json.load(open(sys.argv[1]))
print(next(f["sha256"] for f in m["artifacts"]["exl3-2.52-mikeroz"]["files"] if f["name"] == "tokenizer.json"))' \
    "$WT/weights/mimo-v2.6-flash/manifest.json")
[[ ! -e $OBS ]] || { echo "evidence directory $OBS exists; use a fresh one for a rerun" >&2; exit 1; }
[[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]] || { echo "tracked edits; commit first" >&2; exit 1; }
[[ -z $(git -C "$WT" status --porcelain -- "$EV") ]] || { echo "uncommitted files under $EV; commit first" >&2; exit 1; }
if systemctl --user is-active -q mimo-download.scope; then
    echo "the MiMo download scope is still active; refusing" >&2
    exit 1
fi
mkdir -p "$OBS"
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
meminfo() { awk -v k="$1" '$1 == k":" {print $2}' /proc/meminfo; }

# HumanEval's sandbox fails closed only at its suite start; M3 needs it, so check it before any load.
python3 - "$WT/scripts/31_bench_accuracy.py" <<'PY' | tee -a "$OBS/driver.log"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bench", sys.argv[1])
bench = importlib.util.module_from_spec(spec)
sys.modules["bench"] = bench
spec.loader.exec_module(bench)
print("humaneval runtime", bench.resolve_humaneval_runtime_digest())
PY

# 1 Hz host sampler, plus the engine scope's anon and file pages once it exists.
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
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 &
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
    restore_production || rc=1
    sleep 5
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
    log "FAILED_ATTEMPT MemFree ${free} kB < 108 GiB after release"
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
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"

release_and_gate
# Keep the pack's page cache from crowding MemFree while the weights load
# (GB10 CUDA allocations come from MemFree and are not cgroup-charged).
( set +e
  while :; do
      sleep 1
      PYTHONPATH=$WT/scripts/lib python3 -c 'import sys; from checkpoint_cache import release_tree; [release_tree(r) for r in sys.argv[1:]]' "$PACK" "$DRAFT"
  done ) &
RELEASER=$!
t0=$SECONDS
log "launching in scope $UNIT"
systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=8G -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=10800 \
    -- "${SERVE[@]}" start | tee -a "$OBS/driver.log"
kill "$RELEASER" 2>/dev/null || true
RELEASER=
log "ready after $((SECONDS - t0)) s; idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
cgroup_snapshot "ready"

timeout 1800 python3 "$R1/probe.py" --base-url "$BASE" --out "$EV/probe.json" \
    | tee -a "$OBS/driver.log" || log "probe exit nonzero"
log "cell agent-gate"
timeout 1800 python3 "$WT/scripts/dev/regression-suite.py" agent-gate --base "$BASE" \
    --engine vllm --model "$SERVED" --prefix-block-tokens 256 --json "$EV/agent-gate.json" 2>&1 \
    | tee -a "$OBS/driver.log" || log "agent-gate exit nonzero"
cgroup_snapshot "after agent-gate"
log "cell kit vision,context"
timeout 7200 python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT" \
    --no-launch --cells vision,context --vision-limit 20 --reasoning-effort '' --out "$EV/kit" 2>&1 \
    | tee -a "$OBS/driver.log" || log "kit exit nonzero"
cgroup_snapshot "after kit"
# The kit's context cell asks with a 512-token budget and MiMo thinks by default. If the
# short round fails, the full fill never runs, so run the probe itself with a 4,096-token
# budget: the 262K fill's memory and retrieval are what M3 depends on.
if ! python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["checks"].get("full_phase_ran") else 1)' \
        "$EV/kit/context/summary.json" 2>/dev/null; then
    log "cell context-budget4096: the kit's full fill did not run"
    timeout 5400 python3 "$WT/scripts/50_probe_context.py" --base-url "$BASE" --model "$SERVED" \
        --out "$EV/context-budget4096" --stack-label "$LABEL" --slots 1 --tokens-per-slot 250128 \
        --tokenizer-path "$PACK/tokenizer.json" \
        --tokenizer-sha256 "$TOKENIZER_SHA256" \
        --phase both --profile-id "$PROFILE" --mem-floor-gib 8 --max-tokens 4096 2>&1 \
        | tee -a "$OBS/driver.log" || log "context-budget4096 exit nonzero"
    cgroup_snapshot "after context-budget4096"
fi
log "cell accuracy encoder path (gsm8k dev, cut at 1200 s)"
timeout 1200 python3 "$WT/scripts/31_bench_accuracy.py" --base-url "$BASE" \
    --out "$EV/accuracy/acc-gsm8k-dev.json" --stack-label "$LABEL" --suite gsm8k --split dev \
    --transcripts-dir "$EV/accuracy/transcripts/gsm8k" --encoder mimo26 --max-tokens 16384 \
    --request-timeout 2700 --profile-id "$PROFILE" 2>&1 | tee -a "$OBS/driver.log" \
    || log "accuracy exit nonzero (expected at the 1200 s cut)"
cgroup_snapshot "before stop"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "scope empty after stop"; break; }
    sleep 1
done
if systemctl --user is-active -q "$UNIT.scope"; then
    log "FAILED_ATTEMPT surviving descendants in $UNIT.scope"
fi
log "window closed; postrun MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
