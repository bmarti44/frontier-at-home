#!/usr/bin/env bash
# Lever screen driver (DIAGNOSTIC, PREREG.md). Usage: run-arm.sh <arm from arms.json>
# One fresh start of the arm's profile on :8017 inside a user scope
# (MemorySwapMax=0, OOMPolicy=kill, RuntimeMaxSec), with a 1 Hz /proc/meminfo
# sampler and the kernel log. Measures greedy probes, block-aware speed (with
# vLLM spec-decode counters around it) and tool-call, then stops the engine.
# Production must already be stopped (52_engine_switch.sh stop): the arm holds
# the system inference lock that a serving engine holds, so it refuses to start
# beside one and a concurrent switch cannot start an engine beside the arm.
set -Eeuo pipefail
ARM=${1:?usage: run-arm.sh <arm>}
WT=/home/bmarti44/qwen38fn-tune
EV=$WT/results/qwen38fn-gates/tune-screen-2026-10-11
SMOKE=$WT/results/qwen38fn-gates/smoke-2026-10-11
PORT=8017
BLOCK_TOKENS=3200
HOST=$EV/host-worktree.json
PROFILE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["profile"])' "$EV/arms.json" "$ARM")
read -r SLUG_DIR PROFILE_FILE <<<"${PROFILE/\// }"
read -r STACK K TOKENIZER TOKENIZER_SHA < <(python3 - "$WT" "$SLUG_DIR" "$PROFILE_FILE.json" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/lib")
import profile_resolver as r
p = r.load_profile(sys.argv[2], sys.argv[3])
m = r.load_model(sys.argv[2])
shard = next(s for s in m["artifacts"][p["artifact_roles"]["model"]]["shards"]
             if s["path"].endswith("/tokenizer.json"))
path = m["artifacts"][p["artifact_roles"]["model"]]["path"].replace(
    "{model_root}", "/home/bmarti44/models") + "/tokenizer.json"
print(p["bench"]["stack_label"], p["qualification_options"]["speed_max_tokens_per_event"],
      path, shard["sha256"])
PY
)
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=qwen38fn-tune-$ARM-$$
OBS=$EV/$ARM
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new arm id" >&2; exit 2; }
cd "$WT"

# Preflight. Nothing here creates the arm directory, so an abort leaves no arm
# behind for the scorer and the same arm id can be retried.
DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
[[ $DIRTY -eq 0 ]] || { echo "$WT has $DIRTY tracked changes; commit before an arm" >&2; exit 2; }
python3 "$EV/screen_verdict.py" --may-run "$EV" "$ARM" || { echo "$ARM is not the next due arm" >&2; exit 2; }
# Only this shell holds the lock: the engine and memwatch start with fd 8
# closed, so nothing that outlives an arm can block production's restore.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { echo "$INFERENCE_LOCK is held: an engine is serving; stop it first" >&2; exit 2; }
# Release every Flash-Next runtime tree's unmapped page cache: the engine's arena
# guard releases only the model root, and a previous arm on the other runtime
# leaves ~2.5 GiB of its libraries cached (ATTEMPTS.md, a1 attempt 1).
RELEASED=$(python3 - "$WT/scripts/lib" "$HOME/.cache/qwen38-flash-next" <<'PY2'
import glob, sys
sys.path.insert(0, sys.argv[1])
from checkpoint_cache import release_tree
for root in sorted(glob.glob(sys.argv[2] + "/native-runtime-*")):
    print(f"page cache released under {root}: {release_tree(root)} files")
PY2
) || { echo "runtime page-cache release failed" >&2; exit 2; }
GUARD=$(python3 "$WT/scripts/03_memory_guard.py" --required-gib 114.5 --stable-samples 3 --timeout-seconds 600 2>&1) \
    || { printf '%s\n' "$GUARD" >&2; echo "memory guard did not pass; no arm directory was created" >&2; exit 2; }
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
log "preload $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
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
block=$(NEW_LOG | grep -oP 'Setting attention block size to \K[0-9]+' | tail -1 || true)
kv=$(NEW_LOG | grep -oP 'GPU KV cache size: \K[0-9,]+(?= tokens, Maximum concurrency for 262,144 tokens per request)' \
    | tail -1 | tr -d , || true)
log "$ARM: attention block size ${block:-unknown} tokens"
log "$ARM: KV capacity ${kv:-unknown} tokens at 262,144 per request"
NEW_LOG | grep -E 'QWEN38FN_PLE_WILLNEED|draft vocabulary|draft head|Fused multi-step draft decode|Using V2 Model Runner' \
    | cut -c1-300 | sed 's/^/switchlog /' | tee -a "$OBS/driver.log" || true
if [[ $block != "$BLOCK_TOKENS" || -z $kv || $kv -lt 1048576 ]]; then
    log "FAILED_ATTEMPT $ARM: block ${block:-unknown} (want $BLOCK_TOKENS), KV ${kv:-unknown} (want >= 1048576)"
    exit 1
fi

python3 "$SMOKE/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/probe.json" | tee -a "$OBS/driver.log" \
    || log "probe exit nonzero"
metrics "$OBS/metrics-before.txt"
python3 "$WT/scripts/30_bench_speed_blocks.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/speed.json" \
    --stack-label "$STACK" --reps 3 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 \
    --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body '{"min_tokens":320}' \
    --model-id qwen3.8-flash-next --max-tokens-per-event "$K" \
    --output-tokenizer-path "$TOKENIZER" --output-tokenizer-sha256 "$TOKENIZER_SHA" 2>&1 \
    | tee -a "$OBS/speed.log" | tail -5 | tee -a "$OBS/driver.log" || log "speed exit nonzero"
metrics "$OBS/metrics-after.txt"
python3 "$WT/scripts/39_bench_toolcall.py" --base-url "http://127.0.0.1:$PORT/v1" --model qwen3.8-flash-next \
    --out "$OBS/toolcall.json" --temperature 0 2>&1 | tee -a "$OBS/toolcall.log" | tail -3 | tee -a "$OBS/driver.log" \
    || log "toolcall exit nonzero"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "$ARM: scope empty after stop"; break; }
    sleep 1
done
systemctl --user is-active -q "$UNIT.scope" && log "FAILED_ATTEMPT $ARM: surviving descendants in $UNIT.scope"
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
