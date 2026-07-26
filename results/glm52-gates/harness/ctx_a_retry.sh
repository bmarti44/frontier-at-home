#!/bin/bash
# Phase-A retry: same >8192-row dependency fixture as ctx_regate.sh but
# max_tokens=64 (the 12-token budget was consumed by reasoning preamble
# before the passphrase could be emitted). Serves the exact adopted
# profile; reuses the regate disk-KV so a warm resume covers the prefix.
set -u
OUT=/home/dsv4/ds4-project/glm52-ctx-regate
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-regate
PORT=8016
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "phase-A retry (max_tokens 64)"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8
python3 - "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1] + "/fa.json")); j["max_tokens"] = 64
json.dump(j, open(sys.argv[1] + "/fa64.json", "w"))
EOF
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
  --kv-cache-boundary-align-tokens 4 \
  --kv-cache-boundary-trim-tokens 0 \
  > "$OUT/server-a64.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "a64 server died"; echo A64_FAIL; exit 1; }
  sleep 2
done
for r in a64_1 a64_2; do
  t0=$(date +%s%3N)
  code=$(curl -s -o "$OUT/$r.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$OUT/fa64.json" http://127.0.0.1:$PORT/v1/completions)
  echo "$r http=$code wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"
done
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT
python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
sec = open(out + "/secret.txt").read().strip()
a1 = json.load(open(out + "/a64_1.json"))["choices"][0]["text"]
a2 = json.load(open(out + "/a64_2.json"))["choices"][0]["text"]
print(f"A64 run1 sha={hashlib.sha256(a1.encode()).hexdigest()[:12]} contains_secret={sec in a1}")
print(f"A64 text: {a1[:200]!r}")
print(f"A64 run2 identical={a1==a2}")
print(f"A64 VERDICT: {'PASS' if (sec in a1 and a1==a2) else 'FAIL'}")
EOF
grep "GLM sync start=" "$OUT/server-a64.log" | head -3 >> "$OUT/summary"
chmod -R a+rX "$OUT"
echo A64_DONE
