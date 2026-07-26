#!/bin/bash
# Warm-TTFT probe: does forcing indexed batch prefill for small suffixes
# (DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1) beat the 19 s warm TTFT?
set -u
OUT=/home/dsv4/ds4-project/glm52-ttft2
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
FIX=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/fixture-glm-long8.json
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-ttft
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
python3 - "$FIX" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); j["max_tokens"] = 1
json.dump(j, open(sys.argv[2]+"/fix1.json","w"))
EOF
for MODE in tokmajor batchall; do
  ENVX=()
  [[ $MODE == batchall ]] && ENVX=(DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1)
  env "${ENVX[@]}" DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
    DS4_CUDA_EXPERT_CACHE_GB=68 DS4_CUDA_FETCH_THREADS=6 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port 8029 \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 8192 --kv-cache-boundary-align-tokens 64 \
    > "$OUT/server-$MODE.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8029/v1/models)" == 200 ]] && break
    kill -0 $SPID 2>/dev/null || { echo "DEAD $MODE"; exit 1; }; sleep 2
  done
  t0=$(date +%s%3N)
  curl -s -o "$OUT/cold-$MODE.json" --max-time 3600 -H 'Content-Type: application/json' -d @"$OUT/fix1.json" http://127.0.0.1:8029/v1/completions
  t1=$(date +%s%3N)
  curl -s -o "$OUT/warm-$MODE.json" --max-time 3600 -H 'Content-Type: application/json' -d @"$OUT/fix1.json" http://127.0.0.1:8029/v1/completions
  t2=$(date +%s%3N)
  echo "$MODE cold_ms=$((t1-t0)) warm_ms=$((t2-t1)) text_sha=$(python3 -c "import json,hashlib,sys;print(hashlib.sha256(json.load(open('$OUT/warm-$MODE.json'))['choices'][0]['text'].encode()).hexdigest()[:12])" 2>/dev/null)"
  kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
done
chmod -R a+rX "$OUT"
echo TTFT2_DONE
