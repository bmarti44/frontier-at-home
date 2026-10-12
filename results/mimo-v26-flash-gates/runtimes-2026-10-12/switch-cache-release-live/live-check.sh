#!/usr/bin/env bash
# Live check of PR #22 (switch cache release + MemFree gate): one qwen38fn
# stop + restart through the new path with kernel and meminfo capture.
# Usage: live-check.sh <expected-merge-sha>
set -uo pipefail
S=/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/a369cc3c-eabf-4c1d-b667-1b57dbab2d3a/scratchpad/live-check
REPO=/home/bmarti44/spark-deepseek-v4-flash
SWITCH=$REPO/scripts/52_engine_switch.sh
want=$1
mkdir -p "$S"
stamp() { date -u +%FT%TZ; }
mem() { awk '/^(MemFree|MemAvailable|Cached|Buffers|SReclaimable|SwapFree):/ {printf "%s=%.2f ", $1, $2/1048576}' /proc/meminfo; }

git -C "$REPO" pull --ff-only -q origin main || exit 11
head=$(git -C "$REPO" rev-parse HEAD)
echo "$(stamp) checkout now $head"
[[ $head == "$want"* ]] || { echo "unexpected head"; exit 12; }
(cd "$REPO" && sha256sum --check --quiet verification/MANIFEST.sha256) || exit 13
grep -q '^release_host_caches() {' "$SWITCH" || exit 14

journalctl -k -f -o short-iso --since now >"$S/kernel.log" 2>&1 &
J=$!
(
    while :; do
        printf '%s %s\n' "$(date -u +%FT%TZ)" \
            "$(awk '/^(MemAvailable|MemFree|Cached|Buffers|SReclaimable|SwapFree):/ {printf "%s=%s ", $1, $2}' /proc/meminfo)"
        sleep 1
    done
) >"$S/meminfo.log" 2>&1 &
M=$!
trap 'kill $J $M 2>/dev/null' EXIT

echo "$(stamp) pre: $($SWITCH status --json) $(mem)"
sudo -n "$SWITCH" stop >"$S/stop.log" 2>&1
rc=$?
echo "$(stamp) stop rc=$rc: $(tail -1 "$S/stop.log") $(mem)"
[[ $rc == 0 ]] || exit 10
sleep 5
echo "$(stamp) before switch: $(mem)"
sudo -n "$SWITCH" qwen38fn >"$S/switch.log" 2>&1
rc=$?
echo "$(stamp) switch rc=$rc"
grep -E 'host caches released|free-memory gate|WARNING' "$S/switch.log"
tail -2 "$S/switch.log"
sleep 5
echo "$(stamp) post: $($SWITCH status --json) $(mem)"
grep -iE 'nvrm|xid|oom|out of memory' "$S/kernel.log" && echo "KERNEL: KILL LINES" || echo "kernel: no NVRM/Xid/OOM lines"
exit $rc
