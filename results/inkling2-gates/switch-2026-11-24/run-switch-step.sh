#!/usr/bin/env bash
# One live switch step for the inkling2mini round trip (qwen38fn -> inkling2mini -> qwen38fn).
# Usage: run-switch-step.sh <step name> <alias>
# Uses only the owner-installed NOPASSWD grant for the main checkout's scripts/52_engine_switch.sh.
# Switch output goes to a file. A 1 Hz meminfo sampler and the kernel log cover the step.
set -uo pipefail
repo=/home/bmarti44/spark-deepseek-v4-flash
out=$(cd "$(dirname "$0")" && pwd)
switch=$repo/scripts/52_engine_switch.sh
state=/home/dsv4/ds4-project/engine-switch
name=$1 alias=$2
mkdir -p "$out/observe"
log=$out/switch-log.txt
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$*" >>"$log"; }
(
    while :; do
        printf '%s %s\n' "$(stamp)" "$(awk '/^(MemFree|MemAvailable|SwapFree):/ {printf "%s=%s ", $1, $2}' /proc/meminfo)"
        sleep 1
    done
) >>"$out/observe/meminfo-$name.log" &
sampler=$!
journalctl -k -f -n 0 -o short-iso >>"$out/observe/kernel-$name.log" 2>&1 &
kernel=$!
trap 'kill "$sampler" "$kernel" 2>/dev/null' EXIT
sleep 1

note "head $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" branch --show-current)"
note "status before $name: $("$switch" status --json)"
note "MemAvailable before $name: $(awk '/^MemAvailable:/ {printf "%.2f GiB", $2 / 1048576}' /proc/meminfo)"
started=$SECONDS
note "BEGIN $name: sudo -n scripts/52_engine_switch.sh $alias"
sudo -n "$switch" "$alias" >>"$log" 2>&1
rc=$?
note "END $name: exit $rc in $((SECONDS - started)) s"
note "status after $name: $("$switch" status --json)"
note "MemAvailable after $name: $(awk '/^MemAvailable:/ {printf "%.2f GiB", $2 / 1048576}' /proc/meminfo)"
note "models on 8013: $(curl -fsS --max-time 5 http://127.0.0.1:8013/v1/models 2>&1)"
note "unauthenticated 8010/health: HTTP $(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8010/health)"
[[ -f "$state/$alias.probe.json" ]] && cp -- "$state/$alias.probe.json" "$out/$name.probe.json" && note "copied the switch's semantic probe response"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >"$out/observe/nvidia-smi-$name.txt" 2>&1
sleep 2
note "kernel NVRM/Xid/OOM lines during $name: $(grep -cE 'NVRM|Xid|Out of memory|oom-kill' "$out/observe/kernel-$name.log")"
exit "$rc"
