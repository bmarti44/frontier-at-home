#!/usr/bin/env bash
# Production switch run for the qwen38fn alias (owner choice 2026-10-11:
# optional engine, qwen38-1m stays the default). Uses only the owner-installed
# NOPASSWD grant for scripts/52_engine_switch.sh. Steps:
#   1. qwen38fn       (load, digest, health, model card, auth, semantics)
#   2. qwen38-1m      (restore the default in active.json)
#   3. stop           (leave the host as found: no engine, active qwen38-1m)
# Steps 2 and 3 always run, so the end state does not depend on step 1.
set -uo pipefail

repo=/home/bmarti44/spark-deepseek-v4-flash
out=$repo/results/qwen38fn-gates/switch-2026-10-11
switch=$repo/scripts/52_engine_switch.sh
state=/home/dsv4/ds4-project/engine-switch
mkdir -p "$out/observe"
log=$out/switch-log.txt
: >"$log"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$log"; }

# 1 Hz host memory and swap, plus kernel NVRM/Xid/OOM lines, for the whole run.
(
    while :; do
        printf '%s %s\n' "$(stamp)" "$(awk '/^(MemFree|MemAvailable|SwapFree):/ {printf "%s=%s ", $1, $2}' /proc/meminfo)"
        sleep 1
    done
) >>"$out/observe/meminfo.log" &
sampler=$!
journalctl -k -f -n 0 -o short-iso >>"$out/observe/kernel.log" 2>&1 &
kernel=$!
trap 'kill "$sampler" "$kernel" 2>/dev/null' EXIT

note "head $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" branch --show-current)"
note "status before: $("$switch" status --json)"

step() {
    local name=$1 started rc
    shift
    started=$SECONDS
    note "BEGIN $name: sudo -n scripts/52_engine_switch.sh $*"
    sudo -n "$switch" "$@" >>"$log" 2>&1
    rc=$?
    note "END $name: exit $rc in $((SECONDS - started)) s"
    note "status after $name: $("$switch" status --json)"
    note "MemAvailable after $name: $(awk '/^MemAvailable:/ {printf "%.2f GiB", $2 / 1048576}' /proc/meminfo)"
    return "$rc"
}

if step qwen38fn qwen38fn; then
    note "models on 8013: $(curl -fsS --max-time 5 http://127.0.0.1:8013/v1/models)"
    note "unauthenticated 8010/health: HTTP $(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8010/health)"
    cp -- "$state/qwen38fn.probe.json" "$out/qwen38fn.probe.json" &&
        note "copied the switch's semantic probe response"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >"$out/observe/nvidia-smi-qwen38fn.txt" 2>&1
fi
step qwen38-1m qwen38-1m
step stop stop
note "units: qwen38fn-engine=$(systemctl is-active qwen38fn-engine.service) qwen38-engine=$(systemctl is-active qwen38-engine.service)"
note "kernel NVRM/Xid/OOM lines: $(grep -cE 'NVRM|Xid|Out of memory|oom-kill' "$out/observe/kernel.log")"
note "done"
