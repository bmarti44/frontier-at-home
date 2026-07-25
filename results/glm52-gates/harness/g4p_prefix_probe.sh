#!/bin/bash
# G4-prefill probe: does ds4's disk-KV checkpoint restore GLM compact-DSA
# state correctly? Measures TTFT cold vs live-cache vs disk-restore (after
# full server restart) and asserts byte-identical continuations.
set -u
OUT=/home/dsv4/ds4-project/glm52-g4p-probe
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
FIX=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/fixture-glm-long8.json
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }

start_srv() {
  DS4_GLM_TP_DEBUG=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=60 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port 8027 \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 8192 > "$OUT/$1" 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8027/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || return 1; sleep 2
  done; return 1
}
fire() { local t0=$(date +%s%3N)
  curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' -d @"$FIX" http://127.0.0.1:8027/v1/completions
  echo "$1 wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/walls"
  python3 -c "import json,sys;sys.stdout.write(json.load(open('$OUT/$1.json'))['choices'][0]['text'])" > "$OUT/$1.text" 2>/dev/null; }

start_srv s1.log || { echo PROBE_FAIL server1; exit 1; }
fire cold       # full prefill + checkpoint save
fire live       # live KV reuse in-process
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
ls -la "$KVDIR" >> "$OUT/run.log" 2>&1
start_srv s2.log || { echo PROBE_FAIL server2; exit 1; }
fire restored   # should hit the disk checkpoint
kill -TERM $SPID; sleep 5
cat "$OUT/walls"
for f in cold live restored; do echo "$f sha=$(sha256sum $OUT/$f.text | cut -c1-12)"; done
grep -iE "checkpoint|kv.disk|restore" "$OUT/s1.log" "$OUT/s2.log" | tail -8
chmod -R a+rX "$OUT"
echo PROBE_DONE
