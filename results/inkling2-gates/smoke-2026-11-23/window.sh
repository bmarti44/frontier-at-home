#!/usr/bin/env bash
# GPU window wrapper (DIAGNOSTIC): stop production, run one smoke, restore production.
# Usage: setsid -f bash window.sh <run-id> >~/.frontier-at-home/logs/inkling2-window-<run-id>.log 2>&1
# Production's restored root memwatch inherits this script's stdout, so stdout must be a
# file outside the repository, never a pipe.
set -uo pipefail
RUN=${1:?usage: window.sh <run-id>}
EV=/home/bmarti44/inkling2/results/inkling2-gates/smoke-2026-11-23
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
echo "$(date -u +%FT%TZ) window $RUN open (kernel log since $SINCE local)"
# shellcheck source=/dev/null
source /home/bmarti44/spark-deepseek-v4-flash/scripts/lib/gate_window.sh
# A signal to this wrapper must not start the restore while the smoke engine is still
# loaded (bash runs the EXIT trap at once and leaves the foreground child running).
# Abort procedure: kill -TERM <run-smoke.sh pid>; its EXIT trap stops the engine and
# this wrapper then restores production. A handler (not '') so run-smoke.sh does not
# inherit an ignored signal.
trap 'echo "$(date -u +%FT%TZ) window: signal deferred until run-smoke.sh exits"' TERM INT HUP
# Arm the restore before the stop: a stop that dies part-way must still restore.
_gate_window_armed=true
trap gate_restore_production EXIT
echo "[gate] stopping production"
sudo /home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh stop \
    || echo "$(date -u +%FT%TZ) window: production stop failed (run-smoke.sh refuses while the inference lock is held)"
bash "$EV/run-smoke.sh" "$RUN"
rc=$?
echo "$(date -u +%FT%TZ) smoke $RUN exit $rc"
gate_window_close
echo "$(date -u +%FT%TZ) production restore attempted; switch status:"
/home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh status --json
echo "kernel lines matching NVRM|Xid|oom|Out of memory since window open:"
journalctl -k --since "$SINCE" --no-pager -o short-iso 2>/dev/null | grep -Ei 'NVRM|Xid|oom-kill|Out of memory' || echo "(none)"
echo "$(date -u +%FT%TZ) window $RUN closed: $(awk '/MemAvailable|MemFree|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
