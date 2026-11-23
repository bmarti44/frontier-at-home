#!/usr/bin/env bash
# Inkling 2 Mini first-light smoke (DIAGNOSTIC). Usage: run-smoke.sh <run-id>
# Adapted from results/qwen38fn-gates/tune-screen-2026-10-11/run-arm.sh.
# One fresh start of inkling-2-mini/cuda-spark-128g-smoke on :8018 inside a user
# scope (MemorySwapMax=0, OOMPolicy=kill, RuntimeMaxSec), with a 1 Hz
# /proc/meminfo sampler and the kernel log. Runs the runtime GPU self-test,
# greedy probes, a cold-length prefill probe, the repo speed bench, tool-call,
# and 10 MMMU cases, then stops the engine. Production must already be stopped
# (52_engine_switch.sh stop): the driver holds the system inference lock.
set -Eeuo pipefail
RUN=${1:?usage: run-smoke.sh <run-id>}
WT=/home/bmarti44/inkling2
EV=$WT/results/inkling2-gates/smoke-2026-11-23
PORT=8018
HOST=$EV/host-worktree.json
PROFILE=${PROFILE:-inkling-2-mini/cuda-spark-128g-smoke}
RUNTIME_PY=/home/bmarti44/.cache/inkling-2-mini/native-runtime-001/runtime/bin/python3.12
TOKENIZER=/home/bmarti44/models/inkling-2-mini/tokenizer.json
TOKENIZER_SHA=$(python3 -c 'import json,sys; print(next(f["sha256"] for f in json.load(open(sys.argv[1]))["files"] if f["name"] == "tokenizer.json"))' "$WT/weights/inkling-2-mini/manifest.json")
STACK=inkling-2-mini-cuda-spark-128g-smoke
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=inkling2-smoke-$RUN-$$
OBS=$EV/$RUN
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new run id" >&2; exit 2; }
cd "$WT"

DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
[[ $DIRTY -eq 0 ]] || { echo "$WT has $DIRTY tracked changes; commit before a run" >&2; exit 2; }
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { echo "$INFERENCE_LOCK is held: an engine is serving; stop it first" >&2; exit 2; }
RELEASED=$(python3 - "$WT/scripts/lib" <<'PY2'
import sys
sys.path.insert(0, sys.argv[1])
from checkpoint_cache import release_tree
for root in ("/home/bmarti44/.cache/inkling-2-mini", "/home/bmarti44/models/inkling-2-mini",
             "/home/bmarti44/.cache/qwen38-flash-next", "/home/bmarti44/models/qwen3.8-flash-next-nvfp4"):
    print(f"page cache released under {root}: {release_tree(root)} files")
PY2
) || { echo "page-cache release failed" >&2; exit 2; }
GUARD=$(python3 "$WT/scripts/03_memory_guard.py" --required-gib 110 --stable-samples 3 --timeout-seconds 600 2>&1) \
    || { printf '%s\n' "$GUARD" >&2; echo "memory guard did not pass; no run directory was created" >&2; exit 2; }
MEMFREE_GIB=$(awk '/^MemFree:/{printf "%.2f", $2/1048576}' /proc/meminfo)
awk -v f="$MEMFREE_GIB" 'BEGIN{exit !(f >= 95)}' \
    || { echo "MemFree ${MEMFREE_GIB} GiB < 95 GiB: page cache would starve CUDA allocations; ask for drop_caches" >&2; exit 2; }
mkdir -p "$OBS"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
size_of() { stat -c %s -- "$1" 2>/dev/null || echo 0; }
SERVER_OFF=$(size_of "$SERVER_LOG")
MEMWATCH_OFF=$(size_of "$MEMWATCH_LOG")

( cg=
  while :; do
    if [[ -z $cg ]]; then
        cg=$(systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null || true)
    fi
    printf '%s ' "$(stamp)"
    awk '/^(MemAvailable|MemFree|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Dirty):/{printf "%s=%s ", $1, $2}' /proc/meminfo
    [[ -n $cg && -r /sys/fs/cgroup$cg/memory.current ]] && printf 'cgroup_memory_current=%s ' "$(<"/sys/fs/cgroup$cg/memory.current")"
    printf '\n'
    sleep 1
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 &
JOURNAL=$!
cleanup() {
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    sleep 5
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
    tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null || true
    tail -c +$((MEMWATCH_OFF + 1)) -- "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    local i
    for ((i = 0; i < 30; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || break
        sleep 1
    done
    if systemctl --user is-active -q "$UNIT.scope"; then
        log "FAILED_ATTEMPT $RUN: scope still active at exit; stopping it"
        systemctl --user stop "$UNIT.scope" || true
    fi
    nvidia-smi >"$OBS/nvidia-smi-end.txt" 2>&1 || true
    log "driver exit"
}
trap cleanup EXIT

log "run $RUN profile $PROFILE stack $STACK"
log "head $(git -C "$WT" rev-parse HEAD) dirty=$DIRTY"
log "inference lock $INFERENCE_LOCK held by the driver"
printf '%s\n' "$RELEASED" | sed "s/^/$(stamp) /" | tee -a "$OBS/driver.log"
printf '%s\n' "$GUARD" | tee -a "$OBS/driver.log"
log "preload $(awk '/MemAvailable|MemFree|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"

log "gpu self-test"
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- \
    "$RUNTIME_PY" "$EV/gpu_selftest.py" 8>&- 2>&1 | tail -3 | tee "$OBS/gpu-selftest.json" | tee -a "$OBS/driver.log" \
    || log "gpu self-test exit nonzero"
sleep 2

log "$RUN: launching in scope $UNIT"
t0=$SECONDS
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=7200 \
    -- "${SERVE[@]}" start 8>&- | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT $RUN: launch did not reach ready"
    exit 1
fi
log "$RUN: ready after $((SECONDS - t0)) s"
awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
NEW_LOG() { tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG"; }
NEW_LOG | grep -E 'GPU KV cache size|Maximum concurrency|block size|INKLING2_MAX_CTAS_POW2|Available KV cache memory|Model loading took|CUDA graph|torch.compile|non_default' \
    | cut -c1-400 | sed 's/^/serverlog /' | tee -a "$OBS/driver.log" || true

python3 "$EV/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/probe.json" 2>&1 | tee -a "$OBS/driver.log" \
    || log "probe exit nonzero"
python3 "$EV/cold_ttft.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/cold.json" 2>&1 | tee -a "$OBS/cold.log" \
    | tail -1 | tee -a "$OBS/driver.log" || log "cold probe exit nonzero"
python3 "$WT/scripts/30_bench_speed.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/speed.json" \
    --stack-label "$STACK" --reps 2 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 \
    --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body '{"min_tokens":320}' \
    --model-id inkling-2-mini \
    --output-tokenizer-path "$TOKENIZER" --output-tokenizer-sha256 "$TOKENIZER_SHA" 2>&1 \
    | tee -a "$OBS/speed.log" | tail -5 | tee -a "$OBS/driver.log" || log "speed exit nonzero"
python3 "$WT/scripts/39_bench_toolcall.py" --base-url "http://127.0.0.1:$PORT/v1" --model inkling-2-mini \
    --out "$OBS/toolcall.json" --temperature 0 2>&1 | tee -a "$OBS/toolcall.log" | tail -3 | tee -a "$OBS/driver.log" \
    || log "toolcall exit nonzero"
python3 "$WT/scripts/38_bench_vision.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/vision" \
    --thinking-mode chat --limit 10 --model-id inkling-2-mini 2>&1 | tee -a "$OBS/vision.log" | tail -3 \
    | tee -a "$OBS/driver.log" || log "vision exit nonzero"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "$RUN: scope empty after stop"; break; }
    sleep 1
done
systemctl --user is-active -q "$UNIT.scope" && log "FAILED_ATTEMPT $RUN: surviving descendants in $UNIT.scope"
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
