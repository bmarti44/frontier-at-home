#!/usr/bin/env bash
# Contained DeepSeek V4.1 Flash CUDA smoke runner (fidelity/correctness only).
# usage: run_smoke.sh text|fidelity OUTDIR
#        run_smoke.sh diag OUTDIR CACHE   (diagnostic only: -n 16, expert cache CACHE e.g. 4gb)
# Preregistration: ../smoke-2026-09-25/PREREGISTRATION.md
set -euo pipefail
readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SRC=/home/bmarti44/.cache/ds4-v41-0aaea5a2
readonly MODEL=/home/bmarti44/models/deepseek-v4.1-flash/DeepSeek-V4.1-Flash-Q2.gguf
readonly FIX=$SRC/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router
readonly WRAPPER=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly LOCK=/run/lock/frontier-at-home/inference.lock
readonly PROMPT='Explain in three sentences why the sky is blue.'
readonly LONG_PROMPT='Write a detailed, multi-paragraph explanation of how a CPU pipeline works, covering fetch, decode, execute, memory access, write-back, hazards, and branch prediction.'

arm=${1:?arm}; out=${2:?outdir}
mkdir -p "$out"
# Match real engine executables (exe basename) or python vLLM servers; never
# match shell command lines that merely mention an engine name.
engines=$(for d in /proc/[0-9]*; do
  exe=$(readlink "$d/exe" 2>/dev/null) || continue
  case ${exe##*/} in
    ds4|ds4-server|ds4-agent|ds4-bench|ds4-eval|score_official|llama-server) echo "${d#/proc/} $exe" ;;
    python3*) tr '\0' ' ' < "$d/cmdline" 2>/dev/null | grep -q 'vllm' && echo "${d#/proc/} vllm" ;;
  esac
done)
[[ -z $engines ]] || { echo "another engine is running: $engines" >&2; exit 3; }
python3 "$REPO/scripts/03_memory_guard.py" --required-gib 110 --timeout-seconds 600

case $arm in
  text)
    timeout_s=900
    cmd=("$SRC/ds4" --cuda -m "$MODEL" --ssd-streaming --ssd-streaming-cache-experts 42gb
         -c 8192 --nothink --temp 0 -n 256 -p "$LONG_PROMPT") ;;
  fidelity)
    timeout_s=5400
    cmd=("$SRC/gguf-tools/quality-testing/score_official" "$MODEL" "$FIX/manifest.tsv"
         "$out/cuda.tsv" 34816 --ssd-streaming --ssd-streaming-cache-experts 42gb) ;;
  diag)
    timeout_s=900
    cache=${3:?cache size, e.g. 4gb}
    [[ $cache =~ ^[0-9]{1,3}gb$ ]] || { echo "bad cache $cache" >&2; exit 2; }
    cmd=("$SRC/ds4" --cuda -m "$MODEL" --ssd-streaming --ssd-streaming-cache-experts "$cache"
         -c 8192 --nothink --temp 0 -n 16 -p "$PROMPT") ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

unit=glm52-dsv41-smoke-$arm-$(date -u +%Y%m%dT%H%M%S)
printf '%s\n' "$unit" > "$out/unit.txt"
printf '%q ' "${cmd[@]}" > "$out/command.txt"; echo >> "$out/command.txt"
date -u --iso-8601=ns > "$out/started_at.txt"
set +e
/usr/bin/flock -n -E 75 "$LOCK" \
  systemd-run --user --unit "$unit" --wait --collect --pipe --quiet \
    -p MemorySwapMax=0 -p OOMPolicy=kill -p KillMode=control-group \
    -p MemoryHigh=72G -p MemoryMax=76G \
    -E GLM_SAFE_RUN_AS_CURRENT_USER=1 -E GLM_SAFE_REQUIRE_CGROUP=1 \
    -E GLM_SAFE_CGROUP_UNIT="$unit" -E GLM_SAFE_KILL_FLOOR_GIB=40 \
    -E GLM_SAFE_MIN_START_GIB=110 -E GLM_SAFE_TIMEOUT_S="$timeout_s" \
    /bin/bash "$WRAPPER" --tag "$unit" -- "${cmd[@]}" \
    > "$out/stdout.txt" 2> "$out/stderr.txt"
rc=$?
set -e
date -u --iso-8601=ns > "$out/finished_at.txt"
echo "$rc" > "$out/exit_code.txt"
crash=$(ls -d /home/bmarti44/.local/state/glm52-crashlog/*-"$unit" 2>/dev/null | tail -1 || true)
[[ -n $crash ]] && cp -a "$crash" "$out/containment"
journalctl -k --since "$(date -d "$(sed 's/,/./' "$out/started_at.txt")" '+%Y-%m-%d %H:%M:%S')" --no-pager 2>/dev/null | grep -E 'NVRM|Xid|oom' > "$out/kernel-events.txt" || true
echo "arm=$arm rc=$rc out=$out"
exit "$rc"
