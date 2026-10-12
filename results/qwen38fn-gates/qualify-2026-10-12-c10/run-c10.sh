#!/usr/bin/env bash
# Candidate 10 driver (acceptance, PREREG.md). Usage: run-c10.sh start1 | blocks
#   start1  one fresh start of the tune-c10 profile on :8017 inside a user scope
#           (MemorySwapMax=0, OOMPolicy=kill, RuntimeMaxSec), external 1 Hz
#           sampler and kernel log, after priming qwen38-1m's GGUF and
#           runtime-001's largest libraries into page cache; then probes, the
#           kit and the agent gate.
#   blocks  run 2: the five matched ABBA/BAAB blocks of plan.json through
#           scripts/97_matched_speed_blocks.py, only after a clean run 1
#           (gate_verdict.py --run1-check). Rerunning it resumes the campaign.
# Production must already be stopped (52_engine_switch.sh stop).
set -Eeuo pipefail
MODE=${1:?usage: run-c10.sh start1 | blocks}
WT=/home/bmarti44/qwen38fn-tune
EV=$WT/results/qwen38fn-gates/qualify-2026-10-12-c10
SMOKE=$WT/results/qwen38fn-gates/smoke-2026-10-11
PROFILE=qwen3.8-flash-next/cuda-spark-128g-1m-tune-c10
PORT=8017
BLOCK_TOKENS=3200
CAP=18000
# Differs from configs/hosts/spark-aba1.json only in paths.repo, so the
# server's PYTHONPATH reaches this worktree's scripts/lib.
HOST=$EV/host-worktree.json
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
RUNTIME_ROOT=$HOME/.cache/qwen38-flash-next
R1=$RUNTIME_ROOT/native-runtime-001/runtime/lib/python3.12/site-packages
R2=$RUNTIME_ROOT/native-runtime-002/runtime/lib/python3.12/site-packages
# What a switch from qwen38-1m leaves (candidate 9), and what a production
# qwen38fn start on runtime-001 leaves: its nine largest libraries (2.98 GiB).
PRIME=(/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
       /home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf
       "$R1/nvidia/cu13/lib/libcublasLt.so.13"
       "$R1/triton/_C/libtriton.so"
       "$R1/torch/lib/libtorch_cuda.so"
       "$R1/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
       "$R1/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9"
       "$R1/nvidia/cu13/lib/libcufft.so.12"
       "$R1/nvidia/cusparselt/lib/libcusparseLt.so.0"
       "$R1/torch/lib/libtorch_cpu.so"
       "$R1/nvidia/nccl/lib/libnccl.so.2")
RESIDENCY=$WT/results/qwen38fn-gates/qualify-2026-10-11-c7/pagecache_residency.py
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
preflight() {
    local dirty
    dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
    [[ $dirty -eq 0 ]] || { echo "$WT has $dirty tracked changes" >&2; exit 2; }
}
# One "runtime-tree <name> <first 16 hex> <files>" line per runtime: its site-packages/vllm digest by
# the runtime-002 manifest's vllm_tree method (the contract test's tree_sha256). The gate compares
# them with the frozen manifest. Reading a tree fills page cache, so each is released afterwards.
runtime_trees() {
    python3 -I - "$WT" "$R1" "$R2" <<'PY'
import importlib.util
import sys
from pathlib import Path
wt = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("contract", wt / "scripts/tests/test_qwen38fn_tune_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)   # puts scripts/lib on sys.path
from checkpoint_cache import release_tree
for site in map(Path, sys.argv[2:]):
    digest, files = contract.tree_sha256(site)
    release_tree(str(site / "vllm"))
    print(f"runtime-tree {site.parents[3].name} {digest[:16]} {files}")
PY
}
log_runtime_trees() {
    local trees line
    trees=$(runtime_trees)
    while read -r line; do log "$line"; done <<<"$trees"
}

if [[ $MODE == blocks ]]; then
    OUT=$EV/blocks
    [[ -f $EV/plan.json ]] || { echo "no $EV/plan.json: draw it with 97 plan --drand-after <freeze>" >&2; exit 2; }
    preflight
    python3 "$EV/gate_verdict.py" --run1-check "$EV" \
        || { echo "run 1 is not clean (gate_verdict.py --run1-check): run 2 is not run" >&2; exit 2; }
    mkdir -p "$OUT"
    log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OUT/driver.log"; }
    # Each invocation logs its head; the gate requires one clean commit across all of them.
    log "head $(git -C "$WT" rev-parse HEAD) dirty=0"
    log_runtime_trees
    log "blocks: $(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["blocks"], p["drand"]["round"])' "$EV/plan.json")"
    # 97 holds the system inference lock itself, and --resume continues after a
    # guard timeout (the guard runs before a start's directory exists).
    python3 "$WT/scripts/97_matched_speed_blocks.py" run --plan "$EV/plan.json" --out "$OUT" \
        --host "$HOST" --port "$PORT" --release-tree "$RUNTIME_ROOT" --runtime-max-sec 3600 --resume \
        2>&1 | tee -a "$OUT/driver.log"
    python3 "$WT/scripts/97_matched_speed_blocks.py" score --out "$OUT" | tee -a "$OUT/driver.log" || true
    log_runtime_trees
    log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
    exit 0
fi
[[ $MODE == start1 ]] || { echo "unknown mode: $MODE" >&2; exit 2; }

SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
KIT=(python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT"
     --no-launch --out "$EV")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=qwen38fn-c10-$MODE-$$
OBS=$EV/observe/$MODE
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new candidate directory" >&2; exit 2; }
preflight
# Only this shell holds the lock: the engine and memwatch start with fd 8 closed.
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { echo "$INFERENCE_LOCK is held: an engine is serving; stop it first" >&2; exit 2; }
# The guard runs before $OBS exists: a timeout leaves nothing behind, and the start can be retried.
GUARD=$(python3 "$WT/scripts/03_memory_guard.py" --required-gib 114.5 --stable-samples 3 --timeout-seconds 600 2>&1) \
    || { printf '%s\n' "$GUARD" >&2; echo "memory guard did not pass; nothing was started" >&2; exit 2; }
mkdir -p "$OBS"

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
  done >>"$OBS/meminfo.log" ) 8>&- &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 8>&- &
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
        log "FAILED_ATTEMPT $MODE: scope still active at exit; stopping it"
        systemctl --user stop "$UNIT.scope" || true
    fi
    nvidia-smi >"$OBS/nvidia-smi-end.txt" 2>&1 || true
    log "driver exit"
}
trap cleanup EXIT

check_no_survivors() {
    local i
    for ((i = 0; i < 60; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || { log "$MODE: scope empty after stop"; return 0; }
        sleep 1
    done
    log "FAILED_ATTEMPT $MODE: surviving descendants in $UNIT.scope; stopping scope"
    systemctl --user stop "$UNIT.scope" || true
    return 1
}

printf '%s\n' "$GUARD" | tee -a "$OBS/driver.log"
log "preload $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
log "head $(git -C "$WT" rev-parse HEAD) dirty=0"
log_runtime_trees
cat -- "${PRIME[@]}" >/dev/null
log "primed $(awk '/^(MemFree|MemAvailable|Cached):/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
while read -r gib unit path; do
    log "precondition $gib $unit $path"
done < <(python3 -I "$RESIDENCY" "${PRIME[@]}")

log "$MODE: launching in scope $UNIT (RuntimeMaxSec=$CAP)"
t0=$SECONDS
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec="$CAP" \
    -- "${SERVE[@]}" start 8>&- | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT $MODE: launch did not reach ready"
    exit 1
fi
log "$MODE: ready after $((SECONDS - t0)) s"
awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
block=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" | grep -oP 'Setting attention block size to \K[0-9]+' | tail -1 || true)
log "$MODE: attention block size ${block:-unknown} tokens"
kv=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" \
    | grep -oP 'GPU KV cache size: \K[0-9,]+(?= tokens, Maximum concurrency for 262,144 tokens per request)' \
    | tail -1 | tr -d , || true)
log "$MODE: KV capacity ${kv:-unknown} tokens at 262,144 per request"
if [[ $block != "$BLOCK_TOKENS" || -z $kv || $kv -lt 1048576 ]]; then
    log "FAILED_ATTEMPT $MODE: block ${block:-unknown} (want $BLOCK_TOKENS), KV ${kv:-unknown} (want >= 1048576)"
    exit 1
fi

python3 "$SMOKE/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-$MODE.json" | tee -a "$OBS/driver.log" \
    || log "probe exit nonzero"
# The kit's fixed cell order runs vision before context, so the encoder memory
# the allocator still holds counts against the 1M fill's low point.
"${KIT[@]}" --cells speed,toolcall,vision,teacher,context --with-soak 2>&1 | tee -a "$OBS/driver.log" \
    || log "kit exit nonzero"
python3 "$WT/scripts/dev/regression-suite.py" agent-gate --base "http://127.0.0.1:$PORT" \
    --engine vllm --model default --prefix-block-tokens "$BLOCK_TOKENS" --spec-block-drop --json "$EV/agent-gate.json" \
    2>&1 | tee -a "$OBS/driver.log" || log "agent-gate exit nonzero"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
check_no_survivors
log_runtime_trees
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
