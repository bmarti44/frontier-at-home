#!/bin/bash
# keep-N fidelity quick probe: serve with DS4_GLM_TOPK_KEEP=$KEEPN, fire the
# 8 saved mtp_ab prompts (96-tok greedy), compare against the committed
# keep-8 baseline outputs (glm52-mtp-ab/off-p*.json). Expected per theory:
# visible degradation (kill lever 2). Coherent+similar outputs would instead
# justify the full NLL gate.
set -u
KEEPN=${KEEPN:-6}
OUT=/home/dsv4/ds4-project/glm52-keepn-$KEEPN
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
BASE=/home/dsv4/ds4-project/glm52-mtp-ab
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for keep-$KEEPN window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

DS4_GLM_TOPK_KEEP=$KEEPN DS4_CUDA_EXPERT_CACHE_SLRU=1 \
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "died"; echo KEEPN_FAIL; exit 1; }
  sleep 2
done
for i in 0 1 2 3 4 5 6 7; do
  t0=$(date +%s%3N)
  curl -s -o "$OUT/p$i.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$BASE/p$i.json" http://127.0.0.1:$PORT/v1/completions > /dev/null
  echo "p$i wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"
done
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

python3 - "$OUT" "$BASE" "$KEEPN" <<'EOF' | tee "$OUT/summary"
import json, sys, hashlib
out, base, keepn = sys.argv[1], sys.argv[2], sys.argv[3]
def tx(p):
    return json.load(open(p))["choices"][0]["text"]
same = 0
for i in range(8):
    a = tx(f"{base}/off-p{i}.json")
    b = tx(f"{out}/p{i}.json")
    ident = a == b
    same += ident
    print(f"p{i}: identical={ident}")
    if not ident:
        print(f"  keep8: {a[:110]!r}")
        print(f"  keep{keepn}: {b[:110]!r}")
print(f"identical {same}/8 (divergence expected; judge COHERENCE of keep-{keepn} texts above)")
EOF
cat "$OUT/timings"
chmod -R a+rX "$OUT"
note "keep-$KEEPN window done (caller restores DSV4)"
echo KEEPN_DONE
