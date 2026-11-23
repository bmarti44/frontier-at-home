#!/usr/bin/env bash
# Inkling 2 Mini lever screen driver (DIAGNOSTIC, PREREG.md). Usage: run-arm.sh <arm from arms.json>
# Adapted from ../smoke-2026-11-23/run-smoke.sh. One fresh start of the arm's
# profile on :8018 inside a user scope (MemorySwapMax=0, OOMPolicy=kill,
# RuntimeMaxSec), with a 1 Hz /proc/meminfo sampler and the kernel log.
# Measures readiness, greedy probes, cold-length prefill, block-aware speed
# (with vLLM's spec-decode counters around it), code-shaped decode and
# tool-call (plus 10 MMMU cases on a0 and the combined arm), then stops the
# engine. Production must already be stopped (52_engine_switch.sh stop): the
# driver holds the system inference lock that a serving engine holds.
set -Eeuo pipefail
ARM=${1:?usage: run-arm.sh <arm>}
WT=/home/bmarti44/inkling2
EV=$WT/results/inkling2-gates/tune-screen-2026-11-23
SMOKE=$WT/results/inkling2-gates/smoke-2026-11-23
PORT=8018
HOST=$SMOKE/host-worktree.json
# Every arm's speed cell uses one bench and one K (DSpark streams up to 8 tokens per event).
K=8
PROFILE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["profile"])' "$EV/arms.json" "$ARM")
read -r SLUG_DIR PROFILE_FILE <<<"${PROFILE/\// }"
read -r STACK TOKENIZER TOKENIZER_SHA < <(python3 - "$WT" "$SLUG_DIR" "$PROFILE_FILE.json" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/lib")
import profile_resolver as r
p = r.load_profile(sys.argv[2], sys.argv[3])
m = r.load_model(sys.argv[2])
art = m["artifacts"][p["artifact_roles"]["model"]]
shard = next(s for s in art["shards"] if s["path"].endswith("/tokenizer.json"))
path = art["path"].replace("{model_root}", "/home/bmarti44/models") + "/tokenizer.json"
print(p["bench"]["stack_label"], path, shard["sha256"])
PY
)
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=inkling2-tune-$ARM-$$
OBS=$EV/$ARM
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new arm id" >&2; exit 2; }
cd "$WT"

# Preflight. Nothing here creates the arm directory, so an abort leaves no arm
# behind for the scorer and the same arm id can be retried.
DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
[[ $DIRTY -eq 0 ]] || { echo "$WT has $DIRTY tracked changes; commit before an arm" >&2; exit 2; }
python3 "$EV/screen_verdict.py" --may-run "$EV" "$ARM" || { echo "$ARM is not due" >&2; exit 2; }
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
    || { printf '%s\n' "$GUARD" >&2; echo "memory guard did not pass; no arm directory was created" >&2; exit 2; }
MEMFREE_GIB=$(awk '/^MemFree:/{printf "%.2f", $2/1048576}' /proc/meminfo)
awk -v f="$MEMFREE_GIB" 'BEGIN{exit !(f >= 100)}' \
    || { echo "MemFree ${MEMFREE_GIB} GiB < 100 GiB: page cache would starve the 63 GiB arena; drop caches first" >&2; exit 2; }
mkdir -p "$OBS"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
size_of() { stat -c %s -- "$1" 2>/dev/null || echo 0; }
metrics() { curl -sS --max-time 10 "http://127.0.0.1:$PORT/metrics" >"$1" 2>&1 || log "metrics scrape failed: $1"; }
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
    sleep 5   # teardown kernel lines reach the journal capture
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
    tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null || true
    tail -c +$((MEMWATCH_OFF + 1)) -- "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    local i
    for ((i = 0; i < 30; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || break
        sleep 1
    done
    if systemctl --user is-active -q "$UNIT.scope"; then
        log "FAILED_ATTEMPT $ARM: scope still active at exit; stopping it"
        systemctl --user stop "$UNIT.scope" || true
    fi
    nvidia-smi >"$OBS/nvidia-smi-end.txt" 2>&1 || true
    log "driver exit"
}
trap cleanup EXIT

log "arm $ARM profile $PROFILE stack $STACK tokens_per_event $K"
log "head $(git -C "$WT" rev-parse HEAD) dirty=$DIRTY"
log "inference lock $INFERENCE_LOCK held by the driver"
printf '%s\n' "$RELEASED" | sed "s/^/$(stamp) /" | tee -a "$OBS/driver.log"
printf '%s\n' "$GUARD" | tee -a "$OBS/driver.log"
log "preload $(awk '/MemAvailable|MemFree|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
sleep 2   # at least one meminfo sample precedes the launching line

log "$ARM: launching in scope $UNIT"
t0=$SECONDS
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=7200 \
    -- "${SERVE[@]}" start 8>&- | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT $ARM: launch did not reach ready"
    exit 1
fi
log "$ARM: ready after $((SECONDS - t0)) s"
awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
NEW_LOG() { tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG"; }
conc=$(NEW_LOG | grep -oP 'Maximum concurrency for 262,144 tokens per request: \K[0-9.]+(?=x)' | tail -1 || true)
log "$ARM: KV concurrency ${conc:-unknown}x at 262,144 per request"
NEW_LOG | grep -E 'GPU KV cache size|kv cache group sizes|non-default args|INKLING2_|Encoder cache can hold|Raising encoder_cache_size|host page cache released|Model loading took|speculative|Humming|humming' \
    | cut -c1-600 | sed 's/^/serverlog /' | tee -a "$OBS/driver.log" || true
if [[ -z $conc ]] || ! awk -v c="$conc" 'BEGIN{exit !(c >= 4.0)}'; then
    log "FAILED_ATTEMPT $ARM: KV concurrency ${conc:-unknown}x at 262,144 (want >= 4.0)"
    exit 1
fi

python3 "$SMOKE/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/probe.json" 2>&1 | tee -a "$OBS/driver.log" \
    || log "probe exit nonzero"
python3 "$SMOKE/cold_ttft.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/cold.json" 2>&1 | tee -a "$OBS/cold.log" \
    | tail -1 | tee -a "$OBS/driver.log" || log "cold probe exit nonzero"
metrics "$OBS/metrics-before.txt"
python3 "$WT/scripts/30_bench_speed_blocks.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/speed.json" \
    --stack-label "$STACK" --reps 3 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 \
    --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body '{"min_tokens":320}' \
    --model-id inkling-2-mini --max-tokens-per-event "$K" \
    --output-tokenizer-path "$TOKENIZER" --output-tokenizer-sha256 "$TOKENIZER_SHA" >"$OBS/speed.log" 2>&1 \
    || log "speed exit nonzero"
tail -5 "$OBS/speed.log" | tee -a "$OBS/driver.log"
metrics "$OBS/metrics-after.txt"
python3 "$EV/code_decode.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/code.json" --max-tokens-per-event "$K" \
    2>&1 | tee -a "$OBS/code.log" | tail -1 | tee -a "$OBS/driver.log" || log "code decode exit nonzero"
python3 "$WT/scripts/39_bench_toolcall.py" --base-url "http://127.0.0.1:$PORT/v1" --model inkling-2-mini \
    --out "$OBS/toolcall.json" --temperature 0 2>&1 | tee -a "$OBS/toolcall.log" | tail -3 | tee -a "$OBS/driver.log" \
    || log "toolcall exit nonzero"
if [[ $ARM == a0 || $ARM == a6* ]]; then
    # Patch 0004's live check (images up to 4096 px) and the declared 4096-token answer budget.
    python3 "$WT/scripts/38_bench_vision.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/vision" \
        --thinking-mode chat --limit 10 --max-tokens 4096 --model-id inkling-2-mini 2>&1 \
        | tee -a "$OBS/vision.log" | tail -3 | tee -a "$OBS/driver.log" || log "vision exit nonzero"
fi
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "$ARM: scope empty after stop"; break; }
    sleep 1
done
systemctl --user is-active -q "$UNIT.scope" && log "FAILED_ATTEMPT $ARM: surviving descendants in $UNIT.scope"
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
