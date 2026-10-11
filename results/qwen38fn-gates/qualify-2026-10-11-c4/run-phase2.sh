#!/usr/bin/env bash
# Phase-2 driver (acceptance, PREREG.md). Usage: run-phase2.sh start1|start2
# One fresh start of the 1m profile on :8017 inside a user scope
# (MemorySwapMax=0, OOMPolicy=kill, RuntimeMaxSec). External 1 Hz sampler
# (/proc/meminfo + the scope's memory.current) and kernel log per start.
set -Eeuo pipefail
START=${1:?usage: run-phase2.sh start1|start2}
case $START in
    start1) CAP=18000 ;;
    start2) CAP=57600 ;;
    *) echo "unknown start: $START" >&2; exit 2 ;;
esac
WT=/home/bmarti44/qwen38fn
EV=$WT/results/qwen38fn-gates/qualify-2026-10-11-c4
SMOKE=$WT/results/qwen38fn-gates/smoke-2026-10-11
PROFILE=qwen3.8-flash-next/cuda-spark-128g-1m
PORT=8017
BLOCK_TOKENS=3200
# Differs from configs/hosts/spark-aba1.json only in paths.repo, so the
# server's PYTHONPATH reaches this worktree's scripts/lib.
HOST=$EV/host-worktree.json
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
KIT=(python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT"
     --no-launch --out "$EV")
SLUG=qwen3.8-flash-next-cuda-spark-128g-1m
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
UNIT=qwen38fn-p2-$START-$$
OBS=$EV/observe/$START
[[ ! -e $OBS ]] || { echo "$OBS exists; a rerun needs a new candidate directory" >&2; exit 2; }
mkdir -p "$OBS"
cd "$WT"

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
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
    tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" >"$OBS/server.log" 2>/dev/null || true
    tail -c +$((MEMWATCH_OFF + 1)) -- "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    local i
    for ((i = 0; i < 30; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || break
        sleep 1
    done
    if systemctl --user is-active -q "$UNIT.scope"; then
        log "FAILED_ATTEMPT $START: scope still active at exit; stopping it"
        systemctl --user stop "$UNIT.scope" || true
    fi
    nvidia-smi >"$OBS/nvidia-smi-end.txt" 2>&1 || true
    log "driver exit"
}
trap cleanup EXIT

check_no_survivors() {
    local i
    for ((i = 0; i < 60; i++)); do
        systemctl --user is-active -q "$UNIT.scope" || { log "$START: scope empty after stop"; return 0; }
        sleep 1
    done
    log "FAILED_ATTEMPT $START: surviving descendants in $UNIT.scope; stopping scope"
    systemctl --user stop "$UNIT.scope" || true
    return 1
}

log "preload $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
log "head $(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)"
python3 "$WT/scripts/03_memory_guard.py" --required-gib 112 --stable-samples 3 --timeout-seconds 600 \
    | tee -a "$OBS/driver.log"

log "$START: launching in scope $UNIT (RuntimeMaxSec=$CAP)"
t0=$SECONDS
systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec="$CAP" \
    -- "${SERVE[@]}" start | tee -a "$OBS/driver.log"
log "$START: ready after $((SECONDS - t0)) s"
awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
block=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" | grep -oP 'Setting attention block size to \K[0-9]+' | tail -1 || true)
log "$START: attention block size ${block:-unknown} tokens"
kv=$(tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" \
    | grep -oP 'GPU KV cache size: \K[0-9,]+(?= tokens, Maximum concurrency for 262,144 tokens per request)' \
    | tail -1 | tr -d , || true)
log "$START: KV capacity ${kv:-unknown} tokens at 262,144 per request"
if [[ $block != "$BLOCK_TOKENS" || -z $kv || $kv -lt 1048576 ]]; then
    log "FAILED_ATTEMPT $START: block ${block:-unknown} (want $BLOCK_TOKENS), KV ${kv:-unknown} (want >= 1048576)"
    exit 1
fi

python3 "$SMOKE/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-$START.json" | tee -a "$OBS/driver.log"
if [[ $START == start1 ]]; then
    "${KIT[@]}" --cells speed,toolcall,teacher,context --with-soak 2>&1 | tee -a "$OBS/driver.log" \
        || log "kit exit nonzero"
else
    python3 "$WT/scripts/dev/regression-suite.py" agent-gate --base "http://127.0.0.1:$PORT" \
        --engine vllm --model default --prefix-block-tokens "$BLOCK_TOKENS" --spec-block-drop --json "$EV/agent-gate.json" \
        2>&1 | tee -a "$OBS/driver.log" || log "agent-gate exit nonzero"
    "${KIT[@]}" --cells accuracy --accuracy-split dev 2>&1 | tee -a "$OBS/driver.log" \
        || log "kit exit nonzero"
fi
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
check_no_survivors
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
