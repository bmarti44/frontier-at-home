#!/usr/bin/env bash
# DIAGNOSTIC (not acceptance, never a headline). One load of the 1m profile
# with images enabled, at the commit that enables them, before candidate 6
# is frozen. Records the startup low and idle MemAvailable, vLLM's KV
# capacity, encoder-profiling and attention-backend lines, the three greedy
# probe hashes against candidate 5 (did enabling images change the text
# path?), and the first 10 MMMU-val-100 cases in chat mode as the kit runs
# them. Same containment as run-phase2.sh, plus an external stop at 10.5 GiB
# ahead of the profile's 10 GiB memwatch.
# Run 2 adds a per-process RSS sampler, all 100 MMMU cases, then the kit's
# exact 4-slot direct-1M fill (50_probe_context.py), to measure whether
# images and the 1M fill fit above the floor. A stop at 10.5 GiB means the
# margin is too thin for candidate 6.
# Usage: run-diag.sh [label=run] [mmmu cases=10] [fill 0|1=0]
set -Eeuo pipefail
LABEL=${1:-run}
CASES=${2:-10}
FILL=${3:-0}
WT=/home/bmarti44/qwen38fn
DIAG=$WT/results/qwen38fn-gates/diag-vision-2026-10-11
OBS=$DIAG/$LABEL
C5=$WT/results/qwen38fn-gates/qualify-2026-10-11-c4
SMOKE=$WT/results/qwen38fn-gates/smoke-2026-10-11
PROFILE=qwen3.8-flash-next/cuda-spark-128g-1m
PORT=8017
STOP_KIB=$((21 * 2**20 / 2))
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$C5/host-worktree.json" --port "$PORT")
SLUG=qwen3.8-flash-next-cuda-spark-128g-1m
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=qwen38fn-diag-vision-$$
[[ ! -e $OBS ]] || { echo "$OBS exists" >&2; exit 2; }
mkdir -p "$OBS"
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
size_of() { stat -c %s -- "$1" 2>/dev/null || echo 0; }
avail_kib() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }
SERVER_OFF=$(size_of "$SERVER_LOG")
MEMWATCH_OFF=$(size_of "$MEMWATCH_LOG")

( while :; do
    printf '%s ' "$(stamp)"
    awk '/^(MemAvailable|MemFree|SwapTotal|SwapFree|Cached|AnonPages|Mapped):/{printf "%s=%s ", $1, $2}' /proc/meminfo
    printf '\n'
    sleep 1
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
( while sleep 5; do
    cg=$(systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null || true)
    [[ -n $cg && -r /sys/fs/cgroup$cg/cgroup.procs ]] || continue
    printf '%s' "$(stamp)"
    while read -r pid; do
        awk -v p="$pid" '/^Name:/{n=$2} /^RssAnon:/{a=$2} /^RssFile:/{f=$2} /^RssShmem:/{s=$2}
            END{if (a + f + s > 102400) printf " %s:%s:anon=%s:file=%s:shmem=%s", p, n, a, f, s}' "/proc/$pid/status" 2>/dev/null || true
    done <"/sys/fs/cgroup$cg/cgroup.procs"
    # CUDA allocations are invisible to RssAnon/RssFile; log the driver's per-process view too.
    printf ' gpu=%s\n' "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  done >>"$OBS/procs.log" ) &
PROCS=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 &
JOURNAL=$!
# Armed only once the scope exists, so a dip during the guard wait cannot
# fire it early and disarm it for the run.
( while sleep 1; do
    systemctl --user is-active -q "$UNIT.scope" 2>/dev/null || continue
    if (( $(avail_kib) < STOP_KIB )); then
        printf '%s STOP MemAvailable below 10.5 GiB: SIGKILL scope\n' "$(stamp)" >>"$OBS/driver.log"
        systemctl --user kill --signal=SIGKILL "$UNIT.scope" 2>/dev/null || true
        exit 0
    fi
  done ) &
STOPPER=$!
cleanup() {
    kill "$STOPPER" 2>/dev/null || true
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    kill "$SAMPLER" "$PROCS" "$JOURNAL" 2>/dev/null || true
    tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null || true
    tail -c +$((MEMWATCH_OFF + 1)) -- "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    local i
    for ((i = 0; i < 30; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || break
        sleep 1
    done
    if systemctl --user is-active -q "$UNIT.scope"; then
        log "scope still active at exit; stopping it"
        systemctl --user stop "$UNIT.scope" || true
    fi
    log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
    log "driver exit"
}
trap cleanup EXIT

log "DIAGNOSTIC vision $LABEL profile $PROFILE mmmu_cases $CASES fill $FILL"
log "head $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
log "preload $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
python3 "$WT/scripts/03_memory_guard.py" --required-gib 114 --stable-samples 3 --timeout-seconds 600 \
    | tee -a "$OBS/driver.log"
log "launching in scope $UNIT"
t0=$SECONDS
systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=3600 \
    -- "${SERVE[@]}" start 2>&1 | tee -a "$OBS/driver.log"
log "ready after $((SECONDS - t0)) s; sampling idle memory for 60 s"
sleep 60
log "idle_mem_available_kib=$(avail_kib)"
tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" \
    | grep -E "Model loading took|GPU KV cache size|Encoder cache will be initialized|text-only|mm_prefix|MMEncoderAttention|[Aa]ttention backend|Using .*[Bb]ackend" \
    | cut -c1-400 | tee -a "$OBS/driver.log" || true
block=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" | grep -oP 'Setting attention block size to \K[0-9]+' | tail -1 || true)
kv=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" \
    | grep -oP 'GPU KV cache size: \K[0-9,]+(?= tokens, Maximum concurrency for 262,144 tokens per request)' \
    | tail -1 | tr -d , || true)
log "attention block size ${block:-unknown}; KV capacity ${kv:-unknown} tokens at 262,144"
if [[ $block != 3200 || -z $kv || $kv -lt 1048576 ]]; then
    log "DIAG FAIL: block ${block:-unknown} (want 3200), KV ${kv:-unknown} (want >= 1048576)"
    exit 1
fi
python3 "$SMOKE/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/probe.json" | tee -a "$OBS/driver.log"
python3 - "$OBS/probe.json" "$C5/probe-start1.json" <<'PY' | tee -a "$OBS/driver.log"
import json, sys
a, b = ([g["sha256"] for g in json.load(open(p))["greedy"]] for p in sys.argv[1:3])
print("probe vs candidate 5 start 1:", "identical" if a == b else "DIFFERENT", [x == y for x, y in zip(a, b)])
PY
python3 "$WT/scripts/38_bench_vision.py" --base-url "http://127.0.0.1:$PORT" --out "$OBS/vision" \
    --thinking-mode chat --limit "$CASES" --model-id qwen3.8-flash-next 2>&1 | tail -3 | tee -a "$OBS/driver.log"
log "after vision mem_available_kib=$(avail_kib)"
if [[ $FILL == 1 ]]; then
    # The kit's context cell argv (qualify-2026-10-11-c4/context/log.txt line 1), output here;
    # the tokenizer digest comes from the pinned model.json.
    toksha=$(python3 -c 'import json, sys; m = json.load(open(sys.argv[1]))
print(next(s["sha256"] for a in m["artifacts"].values() for s in a.get("shards", []) if s["path"].endswith("/tokenizer.json")))' \
        "$WT/configs/profiles/qwen3.8-flash-next/model.json")
    python3 "$WT/scripts/50_probe_context.py" --base-url "http://127.0.0.1:$PORT" --model qwen3.8-flash-next \
        --out "$OBS/context" --stack-label qwen3.8-flash-next-cuda-spark-128g-1m --slots 4 --tokens-per-slot 250128 \
        --tokenizer-path "$HOME/models/qwen3.8-flash-next-nvfp4/tokenizer.json" \
        --tokenizer-sha256 "$toksha" \
        --phase both --profile-id "$PROFILE" --mem-floor-gib 10 2>&1 | tail -5 | tee -a "$OBS/driver.log"
    log "after fill mem_available_kib=$(avail_kib)"
fi
