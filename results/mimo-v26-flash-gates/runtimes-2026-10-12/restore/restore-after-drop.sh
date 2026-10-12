#!/usr/bin/env bash
# Wait until the owner drops the kernel caches (Buffers and reclaimable slab
# small, MemFree high), then restore qwen38fn through the switch with kernel
# and meminfo capture. Gives up after 8 h without touching anything.
set -uo pipefail
S=/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/a369cc3c-eabf-4c1d-b667-1b57dbab2d3a/scratchpad/restore
SWITCH=/home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh
mkdir -p "$S"
stamp() { date -u +%FT%TZ; }
mem() { awk '/^(MemFree|MemAvailable|Cached|Buffers|SReclaimable|SwapFree):/ {printf "%s=%.2f ", $1, $2/1048576}' /proc/meminfo; }
ready() {
    awk '/^MemFree:/ {f=$2} /^Buffers:/ {b=$2} /^SReclaimable:/ {s=$2}
         END {exit !(f >= 108*1048576 && b < 2*1048576 && s < 3*1048576)}' /proc/meminfo
}

echo "$(stamp) waiting for cache drop: $(mem)"
deadline=$(( $(date +%s) + 8 * 3600 ))
until ready; do
    (( $(date +%s) < deadline )) || { echo "$(stamp) gave up waiting: $(mem)"; exit 20; }
    sleep 5
done
echo "$(stamp) caches dropped: $(mem)"
sleep 10
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
sudo -n "$SWITCH" qwen38fn >"$S/switch.log" 2>&1
rc=$?
echo "$(stamp) switch rc=$rc"
tail -3 "$S/switch.log"
sleep 5
echo "$(stamp) post: $($SWITCH status --json) $(mem)"
grep -iE 'nvrm|xid|oom|out of memory' "$S/kernel.log" && echo "KERNEL: KILL LINES" || echo "kernel: no NVRM/Xid/OOM lines"
exit $rc
