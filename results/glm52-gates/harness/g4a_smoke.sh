#!/bin/bash
set -u
OUT=/home/dsv4/ds4-project/glm52-g4a-smoke
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
FIX=${G4A_FIXTURE:-/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/fixture-glm-short.json}
rm -rf "$OUT"; mkdir -p "$OUT"
CACHE_GB=${G4A_CACHE_GB:-40}
REPS=${G4A_REPS:-3}
DS4_GLM_TP_DEBUG=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GB \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port 8025 \
  --ssd-streaming --ssd-streaming-cache-experts 40GB > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 200); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8025/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { echo DEAD; tail -5 "$OUT/server.log"; exit 1; }
  sleep 2
done
for r in $(seq 1 $REPS); do
  T0=$(date +%s)
  cat /proc/$SPID/io | awk '/^read_bytes/{print $2}' > "$OUT/rb_before_$r"
  curl -s -o "$OUT/resp$r.json" --max-time 1200 -H 'Content-Type: application/json' \
    -d @"$FIX" http://127.0.0.1:8025/v1/completions
  cat /proc/$SPID/io | awk '/^read_bytes/{print $2}' > "$OUT/rb_after_$r"
  python3 -c "import json,sys;sys.stdout.write(json.load(open('$OUT/resp$r.json'))['choices'][0]['text'])" > "$OUT/text$r"
  echo "req$r wall=$(( $(date +%s) - T0 ))s" >> "$OUT/walls"
done
kill -TERM $SPID 2>/dev/null; sleep 5
for r in $(seq 1 $REPS); do
  echo "req$r read_delta=$(( $(cat $OUT/rb_after_$r) - $(cat $OUT/rb_before_$r) )) sha=$(sha256sum $OUT/text$r | cut -c1-12)"
done
grep "expert cache enabled" "$OUT/server.log"
grep "expert-cache stats" "$OUT/server.log" | tail -2
chmod -R a+rX "$OUT"
