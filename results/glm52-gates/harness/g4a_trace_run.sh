#!/bin/bash
# G4a lever-0 trace driver: serve GLM with DS4_CUDA_EXPERT_TRACE=1 and drive
# a varied, deterministic prompt mix (shared agent-style prefix + distinct
# tasks) to collect per-layer expert-selection traces for cache design.
# Run under glm_safe_run.sh as dsv4.
set -u
OUT=/home/dsv4/ds4-project/glm52-g4a-trace
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
PORT=8024
mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "trace run start binary=$(sha256sum "$SRC/ds4-server" | cut -c1-12)"

DS4_CUDA_EXPERT_TRACE=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  > "$OUT/server-trace.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "server died"; exit 1; }
  sleep 2
done
note "server ready pid=$SPID"

PREFIX="You are a careful engineering assistant working inside a build system. Follow instructions exactly, prefer concise correct answers, and never invent file contents. Context notes: the project uses C11, CUDA 13, aarch64 Linux; tests run under a deterministic harness; disk I/O uses O_DIRECT with 4 KiB alignment; the reviewer requires exact byte counts in tables."
TASKS=(
 "Write a C function that reverses a null-terminated string in place, then explain its complexity."
 "Explain the difference between mmap with MAP_PRIVATE and MAP_SHARED for a read-only file."
 "Given f(n)=3n^2+2n+7, compute f(1) through f(6) and the second differences."
 "Draft a git commit message for a change that fixes an off-by-one in a ring buffer."
 "List the steps to bisect a performance regression across 40 commits."
 "Convert 211075856448 bytes to GiB and GB, showing the arithmetic."
 "Write a bash loop that checksums every *.log file in a directory tree."
 "Summarize why atomic float addition is non-deterministic on GPUs."
 "Write a Python function to parse lines like 'XTRACE L12 N8: 1 2 3' into (layer, ids)."
 "Explain LRU vs LFU eviction for a cache of 4000 slots with skewed access."
)
for r in 1 2; do
  for t in "${!TASKS[@]}"; do
    note "request round=$r task=$t"
    python3 - "$PREFIX" "${TASKS[$t]}" > /tmp/req.json <<'EOF'
import json, sys
print(json.dumps({"model":"glm-5.2","prompt":sys.argv[1]+"\n\nTask: "+sys.argv[2],
                  "max_tokens":64,"temperature":0,"seed":42}))
EOF
    curl -s -o "$OUT/resp-r${r}-t${t}.json" --max-time 1200 \
      -H 'Content-Type: application/json' -d @/tmp/req.json \
      "http://127.0.0.1:$PORT/v1/completions"
  done
done
note "requests done; trace lines: $(grep -c '^XTRACE' "$OUT/server-trace.log" || true)"
kill -TERM $SPID 2>/dev/null; sleep 5
chmod -R a+rX "$OUT"
echo "TRACE_RUN_DONE lines=$(grep -c '^XTRACE' "$OUT/server-trace.log" || true)"
