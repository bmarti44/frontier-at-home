#!/usr/bin/env bash
# DIAGNOSTIC (not acceptance, never a headline). Reloads a Flash-Next profile
# to name the drafter-step allocation that killed Phase 2 start 1
# (../qualify-2026-10-11/RESULT.md). Same containment as run-phase2.sh, plus
# watch.py (stack dumps via faulthandler on SIGUSR1; SIGKILL below 15 GiB).
# The hook reaches the engine only through a scratch PYTHONPATH: a host file
# whose paths.repo is a scratch dir of symlinks to this worktree's
# scripts/lib plus sitecustomize.py. Production files are unchanged.
# Usage: run-diag.sh <label> <scratch dir> [profile]
set -Eeuo pipefail
LABEL=${1:?usage: run-diag.sh <label> <scratch dir> [profile]}
SCRATCH=${2:?scratch dir}
PROFILE=${3:-qwen3.8-flash-next/cuda-spark-128g-1m}
WT=/home/bmarti44/qwen38fn
DIAG=$WT/results/qwen38fn-gates/diag-drafter-2026-10-11
OBS=$DIAG/$LABEL
PORT=8017
UNIT=qwen38fn-diag-$LABEL-$$
[[ ! -e $OBS ]] || { echo "$OBS exists" >&2; exit 2; }
mkdir -p "$OBS"
FAKE=$SCRATCH/repo-$LABEL
mkdir -p "$FAKE/scripts/lib"
for f in "$WT"/scripts/lib/*.py; do ln -s "$f" "$FAKE/scripts/lib/"; done
ln -s "$DIAG/sitecustomize.py" "$FAKE/scripts/lib/sitecustomize.py"
python3 - "$WT/results/qwen38fn-gates/qualify-2026-10-11/host-worktree.json" "$FAKE" "$OBS/host-diag.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["paths"]["repo"] = sys.argv[2]
json.dump(d, open(sys.argv[3], "w"), indent=2)
PY
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$OBS/host-diag.json" --port "$PORT")
SLUG=$(echo "$PROFILE" | tr / -)
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
size_of() { stat -c %s -- "$1" 2>/dev/null || echo 0; }
SERVER_OFF=$(size_of "$SERVER_LOG")
MEMWATCH_OFF=$(size_of "$MEMWATCH_LOG")

( while :; do
    printf '%s ' "$(stamp)"
    awk '/^(MemAvailable|MemFree|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Unevictable|Mlocked):/{printf "%s=%s ", $1, $2}' /proc/meminfo
    printf '\n'
    sleep 0.5
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel.log" 2>&1 &
JOURNAL=$!
WATCH=
cleanup() {
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    [[ -z $WATCH ]] || kill "$WATCH" 2>/dev/null || true
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
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

log "DIAGNOSTIC $LABEL profile $PROFILE"
log "head $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
log "sha256 $(sha256sum "configs/profiles/$PROFILE.json" "$DIAG/sitecustomize.py" "$DIAG/watch.py" "$0" | awk '{printf "%s:%s ", $2, $1}')"
python3 "$WT/scripts/03_memory_guard.py" --required-gib 112 --stable-samples 3 --timeout-seconds 600 \
    | tee -a "$OBS/driver.log"
python3 "$DIAG/watch.py" "$SERVER_LOG" "$SERVER_OFF" "$OBS/watch.log" "$UNIT" &
WATCH=$!
log "launching in scope $UNIT"
systemd-run --user --scope -q --unit="$UNIT" \
    -p MemoryHigh=92G -p MemoryMax=96G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=1800 \
    -- "${SERVE[@]}" start 2>&1 | tee -a "$OBS/driver.log" || log "start exit nonzero"
wait "$WATCH" || true
WATCH=
tail -c +$((SERVER_OFF + 1)) -- "$SERVER_LOG" | grep -E "Model loading took|GPU KV cache size|Initial free memory" \
    | tee -a "$OBS/driver.log" || true
awk '/MemAvailable/{printf "idle_mem_available_kib=%s\n", $2}' /proc/meminfo | tee -a "$OBS/driver.log"
