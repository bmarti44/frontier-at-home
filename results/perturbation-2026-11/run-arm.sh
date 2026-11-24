#!/usr/bin/env bash
# One model's run of both arms against the serving production engine (PREREG "Protocol").
# Usage: run-arm.sh <name> <encoder> <expected served model id>
#   generate (pass 1), generate --retry-invalid (pass 2: one retry of non-timeout transport errors),
#   score-humaneval twice (the second pass re-executes sandbox infrastructure failures once).
# A 1 Hz meminfo sampler and the kernel log run for the whole arm. Refuses a tree that differs from
# the freeze (FROZEN.sha256, BUILT.sha256) or a dirty tracked tree.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
name=$1 encoder=$2 model=$3
out=$here/runs/$name
mkdir -p "$out/observe"
log=$out/arm.log
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$*" >>"$log"; }
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

note "head $(git -C "$repo" rev-parse HEAD) arm $name encoder $encoder model $model"
git -C "$repo" diff --quiet HEAD -- || { note "REFUSED: tracked tree differs from HEAD"; exit 2; }
(cd "$here" && sha256sum --quiet -c FROZEN.sha256 && sha256sum --quiet -c BUILT.sha256) >>"$log" 2>&1 \
    || { note "REFUSED: frozen or built files differ"; exit 2; }
note "switch status: $(/home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh status --json 2>&1)"
common=(--repo "$repo" --out "$out")
gen=(python3 -B "$here/run_perturb.py" generate "${common[@]}" --items "$here/items.jsonl"
     --base-url http://127.0.0.1:8013 --api-key-file /etc/deepseek-v4-flash/api-key
     --encoder "$encoder" --expect-model "$model" --concurrency 3)
note "generate pass 1"
"${gen[@]}" >>"$out/generate.log" 2>&1; rc=$?
note "generate pass 1 exit $rc"
[[ $rc -eq 0 ]] || exit "$rc"
note "generate pass 2 (--retry-invalid)"
"${gen[@]}" --retry-invalid >>"$out/generate.log" 2>&1; rc=$?
note "generate pass 2 exit $rc"
[[ $rc -eq 0 ]] || exit "$rc"
for pass in 1 2; do
    note "score-humaneval pass $pass"
    python3 -B "$here/run_perturb.py" score-humaneval "${common[@]}" --min-available-gib 10 >>"$out/score.log" 2>&1; rc=$?
    note "score-humaneval pass $pass exit $rc"
    [[ $rc -eq 0 ]] || exit "$rc"
done
note "kernel NVRM/Xid/OOM lines: $(grep -cE 'NVRM|Xid|Out of memory|oom-kill' "$out/observe/kernel.log")"
note "done"
