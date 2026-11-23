#!/usr/bin/env bash
# GPU window wrapper (DIAGNOSTIC): stop production, run screen arms in order, restore production.
# Usage: setsid -f bash window.sh <tag> <arm|effort:arm> [...] >~/.frontier-at-home/logs/inkling2-screen-<tag>.log 2>&1
# "effort:<arm>" runs run-effort.sh <arm> (the reasoning-effort pick on the carrier).
# Adapted from ../smoke-2026-11-23/window.sh. Production's restored root memwatch inherits
# this script's stdout, so stdout must be a file outside the repository, never a pipe.
# Any kernel NVRM/Xid/OOM line after an arm stops the window before the next arm.
set -uo pipefail
TAG=${1:?usage: window.sh <tag> <arm> [<arm> ...]}
shift
(($# > 0)) || { echo "no arms given" >&2; exit 2; }
EV=/home/bmarti44/inkling2/results/inkling2-gates/tune-screen-2026-11-23
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
KILL='NVRM|Xid|oom-kill|Out of memory'
echo "$(date -u +%FT%TZ) window $TAG open for arms $* (kernel log since $SINCE local)"
# shellcheck source=/dev/null
source /home/bmarti44/spark-deepseek-v4-flash/scripts/lib/gate_window.sh
# A signal must not start the restore while an arm's engine is still loaded. Abort
# procedure: touch $EV/ABORT (no further arm starts), then kill -TERM the running
# run-arm.sh; its EXIT trap stops the engine and this wrapper then restores production.
trap 'echo "$(date -u +%FT%TZ) window: signal deferred until the running arm exits"' TERM INT HUP
_gate_window_armed=true
trap gate_restore_production EXIT
echo "[gate] stopping production"
sudo /home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh stop \
    || echo "$(date -u +%FT%TZ) window: production stop failed (run-arm.sh refuses while the inference lock is held)"
for arm in "$@"; do
    if [[ -e $EV/ABORT ]]; then
        echo "$(date -u +%FT%TZ) window: ABORT file present; no further arms"
        break
    fi
    echo "$(date -u +%FT%TZ) arm $arm start"
    if [[ $arm == effort:* ]]; then
        bash "$EV/run-effort.sh" "${arm#effort:}"
    else
        bash "$EV/run-arm.sh" "$arm"
    fi
    echo "$(date -u +%FT%TZ) arm $arm exit $?"
    if journalctl -k --since "$SINCE" --no-pager -o short-iso 2>/dev/null | grep -Eiq "$KILL"; then
        echo "$(date -u +%FT%TZ) window: kernel kill line after $arm; no further arms"
        break
    fi
done
gate_window_close
echo "$(date -u +%FT%TZ) production restore attempted; switch status:"
/home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh status --json
echo "kernel lines matching NVRM|Xid|oom|Out of memory since window open:"
journalctl -k --since "$SINCE" --no-pager -o short-iso 2>/dev/null | grep -Ei "$KILL" || echo "(none)"
echo "$(date -u +%FT%TZ) window $TAG closed: $(awk '/MemAvailable|MemFree|SwapFree/{printf "%s=%s ", $1, $2}' /proc/meminfo)"
