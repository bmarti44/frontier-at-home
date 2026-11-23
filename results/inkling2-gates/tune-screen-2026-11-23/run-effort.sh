#!/usr/bin/env bash
# Inkling 2 Mini reasoning-effort pick (DIAGNOSTIC, PREREG.md). Usage: run-effort.sh <carrier arm>
# run-arm.sh's preflight, containment and monitoring, unchanged. One fresh start
# of the carrier the screen named; then the dev split (no ledger rows) of GSM8K
# and MMLU-Pro at reasoning effort low and medium, the four suite runs
# concurrently against the one server (4 slots), each with its own transcripts.
# The encoder renders the effort into every prompt (/v1/completions), so the
# profile's chat default does not enter these rows.
set -Eeuo pipefail
ARM=${1:?usage: run-arm.sh <arm>}
WT=/home/bmarti44/inkling2
EV=$WT/results/inkling2-gates/tune-screen-2026-11-23
SMOKE=$WT/results/inkling2-gates/smoke-2026-11-23
PORT=8018
HOST=$SMOKE/host-worktree.json
SITE=/home/bmarti44/.cache/inkling-2-mini/native-runtime-001/runtime/lib/python3.12/site-packages
TRITON_CACHE=/home/bmarti44/.cache/inkling-2-mini/home/.triton/cache
# Every arm's speed cell uses one bench and one K (DSpark streams up to 8 tokens per event).
K=8
PROFILE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["profile"])' "$EV/arms.json" "$ARM")
# The readiness floor: 4.0x unless arms.json lowers it (DSpark's draft KV shares the arena; PREREG).
MIN_CONC=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]].get("min_concurrency", 4.0))' "$EV/arms.json" "$ARM")
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
UNIT=inkling2-effort-$ARM-$$
OBS=$EV/effort-$ARM
INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new arm id" >&2; exit 2; }
cd "$WT"

# Preflight. Nothing here creates the arm directory, so an abort leaves no arm
# behind for the scorer and the same arm id can be retried.
DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
[[ $DIRTY -eq 0 ]] || { echo "$WT has $DIRTY tracked changes; commit before an arm" >&2; exit 2; }
python3 "$EV/screen_verdict.py" --effort-due "$EV" "$ARM" || { echo "the screen did not name $ARM as its carrier" >&2; exit 2; }
exec 8>"$INFERENCE_LOCK"
flock -n 8 || { echo "$INFERENCE_LOCK is held: an engine is serving; stop it first" >&2; exit 2; }
# The vLLM tree this arm loads, by the build manifest's vllm_tree method (sorted
# relpath NUL sha256 LF, no __pycache__/*.pyc); hashed before the release below drops its pages.
TREE=$(python3 - "$SITE/vllm" <<'PY3'
import hashlib, os, sys
root = sys.argv[1]
rows = []
for d, _, files in os.walk(root):
    for n in files:
        rel = os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
        if "__pycache__" in rel.split("/") or rel.endswith(".pyc"):
            continue
        with open(os.path.join(d, n), "rb") as f:
            rows.append(f"{rel}\0{hashlib.file_digest(f, 'sha256').hexdigest()}\n")
rows.sort()
print(hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows))
PY3
) || { echo "vLLM tree hash failed" >&2; exit 2; }
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
# Every start compiles its Triton kernels from an empty cache, so no arm inherits
# another's compiled variants (the cold-length probe measures those compiles).
TRITON_ENTRIES=0
TRITON_ASIDE=none
if [[ -d $TRITON_CACHE ]]; then
    TRITON_ENTRIES=$(ls -A "$TRITON_CACHE" | wc -l)
    TRITON_ASIDE=$TRITON_CACHE.pre-${OBS##*/}-$(date -u +%Y%m%dT%H%M%SZ)
    mv -- "$TRITON_CACHE" "$TRITON_ASIDE"
fi
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
pids=()
cleanup() {
    # The suites never hold the inference lock (fd 8 closed) and never outlive the driver.
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
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
log "vllm_tree ${TREE% *} files=${TREE#* }"
log "triton cache: $TRITON_ENTRIES entries moved aside to $TRITON_ASIDE; this start compiles from empty"
printf '%s\n' "$RELEASED" | sed "s/^/$(stamp) /" | tee -a "$OBS/driver.log"
printf '%s\n' "$GUARD" | tee -a "$OBS/driver.log"
log "preload $(awk '/MemAvailable|MemFree|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
sleep 2   # at least one meminfo sample precedes the launching line

log "$ARM: launching in scope $UNIT"
t0=$SECONDS
if ! systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=50400 \
    -- "${SERVE[@]}" start 8>&- | tee -a "$OBS/driver.log"; then
    log "FAILED_ATTEMPT $ARM: launch did not reach ready"
    exit 1
fi
log "$ARM: ready after $((SECONDS - t0)) s"
awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
NEW_LOG() { tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG"; }
# A kernel NVRM/Xid/OOM line ends the run before the next step (the EXIT trap stops the engine).
kernel_check() {
    if grep -Eiq 'NVRM|Xid|oom-kill|Out of memory' "$OBS/kernel.log"; then
        log "FAILED_ATTEMPT $ARM: kernel kill line before $1"
        exit 1
    fi
}
conc=$(NEW_LOG | grep -oP 'Maximum concurrency for 262,144 tokens per request: \K[0-9.]+(?=x)' | tail -1 || true)
log "$ARM: KV concurrency ${conc:-unknown}x at 262,144 per request"
NEW_LOG | grep -E 'GPU KV cache size|kv cache group sizes|non-default args|INKLING2_|Encoder cache can hold|Raising encoder_cache_size|host page cache released|Model loading took|speculative|Humming|humming' \
    | cut -c1-600 | sed 's/^/serverlog /' | tee -a "$OBS/driver.log" || true
if [[ -z $conc ]] || ! awk -v c="$conc" -v m="$MIN_CONC" 'BEGIN{exit !(c >= m)}'; then
    log "FAILED_ATTEMPT $ARM: KV concurrency ${conc:-unknown}x at 262,144 (want >= $MIN_CONC)"
    exit 1
fi

kernel_check accuracy
for effort in low medium; do
    for suite in gsm8k mmlu-pro; do
        mkdir -p "$OBS/$effort"
        python3 "$WT/scripts/31_bench_accuracy.py" --base-url "http://127.0.0.1:$PORT" \
            --out "$OBS/$effort/acc-$suite.json" --stack-label "$STACK" --suite "$suite" --split dev \
            --transcripts-dir "$OBS/$effort/transcripts/$suite" --encoder inkling2 --reasoning-effort "$effort" \
            --max-tokens 16384 --request-timeout 2700 --profile-id "$PROFILE" \
            >"$OBS/$effort/acc-$suite.log" 2>&1 8>&- &
        pids+=($!)
        log "started $suite dev at effort $effort (pid $!)"
    done
done
# Watch the kernel log while the suites run (hours): a kill line stops them all.
while :; do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    ((alive)) || break
    if grep -Eiq 'NVRM|Xid|oom-kill|Out of memory' "$OBS/kernel.log"; then
        kill "${pids[@]}" 2>/dev/null
        log "FAILED_ATTEMPT $ARM: kernel kill line during the accuracy suites"
        exit 1
    fi
    sleep 30
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
log "accuracy runs finished (any nonzero exit: $rc)"
# An engine that died mid-run (memwatch, an EngineCore exception, RuntimeMaxSec) turns every
# remaining row into a fast transport error; the run then decides nothing.
if ! curl -sf --max-time 10 -o /dev/null "http://127.0.0.1:$PORT/health"; then
    log "FAILED_ATTEMPT $ARM: engine not healthy after the accuracy suites"
    exit 1
fi
log "effort-$ARM: engine healthy after the accuracy suites"
for effort in low medium; do
    for suite in gsm8k mmlu-pro; do
        python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], sys.argv[3], d.get("correct"), "/", d.get("n"))' \
            "$OBS/$effort/acc-$suite.json" "$effort" "$suite" 2>&1 | tee -a "$OBS/driver.log"
    done
done
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
for ((i = 0; i < 60; i++)); do
    systemctl --user is-active -q "$UNIT.scope" || { log "effort-$ARM: scope empty after stop"; break; }
    sleep 1
done
systemctl --user is-active -q "$UNIT.scope" && log "FAILED_ATTEMPT effort-$ARM: surviving descendants in $UNIT.scope"
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
