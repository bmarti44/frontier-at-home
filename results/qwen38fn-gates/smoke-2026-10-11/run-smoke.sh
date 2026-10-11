#!/usr/bin/env bash
# Phase-1 smoke driver (DIAGNOSTIC). Two fresh starts of the smoke profile on
# :8017 inside a user scope (MemorySwapMax=0, OOMPolicy=kill); start 1 runs the
# probes plus the kit's speed and tool-call cells, start 2 repeats the probes
# for greedy identity. External 1 Hz /proc/meminfo sampler and kernel log.
set -Eeuo pipefail
WT=/home/bmarti44/qwen38fn
EV=$WT/results/qwen38fn-gates/smoke-2026-10-11
PROFILE=qwen3.8-flash-next/cuda-spark-128g-smoke
PORT=8017
# The committed host file differs from configs/hosts/spark-aba1.json only in
# paths.repo, so PYTHONPATH reaches this worktree's scripts/lib middleware.
HOST=$EV/host-worktree.json
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SERVER_LOG=$HOME/.frontier-at-home/logs/qwen3.8-flash-next-cuda-spark-128g-smoke.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/qwen3.8-flash-next-cuda-spark-128g-smoke.memwatch.log
OBS=$EV/observe
mkdir -p "$OBS"
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }

( while :; do
    printf '%s ' "$(stamp)"
    awk '/^(MemAvailable|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Dirty):/{printf "%s=%s ", $1, $2}' /proc/meminfo
    printf '\n'
    sleep 1
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 &
JOURNAL=$!
cleanup() {
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
    [[ -e $OBS/server-start2.log ]] || cp -- "$SERVER_LOG" "$OBS/server-last.log" 2>/dev/null || true
    cp -- "$MEMWATCH_LOG" "$OBS/memwatch.log" 2>/dev/null || true
    nvidia-smi >"$OBS/nvidia-smi-end.txt" 2>&1 || true
    log "driver exit"
}
trap cleanup EXIT

check_no_survivors() {  # check_no_survivors <n>
    local unit="qwen38fn-smoke-$1-$$.scope" i
    for ((i = 0; i < 60; i++)); do
        systemctl --user is-active -q "$unit" || { log "start $1: scope empty after stop"; return 0; }
        sleep 1
    done
    log "FAILED_ATTEMPT start $1: surviving descendants in $unit; stopping scope"
    systemctl --user stop "$unit" || true
    return 1
}

start_server() {  # start_server <n>
    log "start $1: launching in scope qwen38fn-smoke-$1"
    local t0=$SECONDS
    systemd-run --user --scope -q --unit="qwen38fn-smoke-$1-$$" \
        -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=7200 \
        -- "${SERVE[@]}" start | tee -a "$OBS/driver.log"
    log "start $1: ready after $((SECONDS - t0)) s"
    awk '/MemAvailable/{printf "%s idle_mem_available_kib=%s\n", "'"$(stamp)"'", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
}

log "preload $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
log "head $(git -C "$WT" rev-parse HEAD)"

start_server 1
python3 "$EV/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-start1.json" | tee -a "$OBS/driver.log"
python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT" --no-launch \
    --cells speed,toolcall --out "$EV/kit" 2>&1 | tee -a "$OBS/driver.log" || log "kit exit nonzero"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
check_no_survivors 1
cp -- "$SERVER_LOG" "$OBS/server-start1.log"

python3 "$WT/scripts/03_memory_guard.py" --required-gib 100 --stable-samples 3 --timeout-seconds 300 | tee -a "$OBS/driver.log"
start_server 2
python3 "$EV/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-start2.json" | tee -a "$OBS/driver.log"
"${SERVE[@]}" stop | tee -a "$OBS/driver.log"
check_no_survivors 2
cp -- "$SERVER_LOG" "$OBS/server-start2.log"
log "postrun $(awk '/MemAvailable|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
