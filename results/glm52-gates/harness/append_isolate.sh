#!/bin/bash
# Discriminating experiment for the append-resume bug: ap3 fired as the
# FIRST request on a fresh server + wiped disk-KV = true canonical cold.
# Compare vs y_ap3 (which resumed from ap1's checkpoint lineage) and
# x_ap3 (resumed from base checkpoint). Also re-fire ap1 second here so
# we get ap1-after-ap3 lineage for the reverse contamination check.
set -u
OUT=/home/dsv4/ds4-project/glm52-append-probe
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-append-iso
PORT=8016
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "append-isolate window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
  --kv-cache-boundary-align-tokens 4 \
  --kv-cache-boundary-trim-tokens 0 \
  > "$OUT/server-iso.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "iso server died"; echo ISO_FAIL; exit 1; }
  sleep 2
done
fire() { local t0=$(date +%s%3N)
  local code=$(curl -s -o "$OUT/$1.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$2" http://127.0.0.1:$PORT/v1/completions)
  echo "$1 http=$code wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }
fire z_ap3_cold "$OUT/ap3.json"
fire z_ap1_after "$OUT/ap1.json"
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT
python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
def tx(n): return json.load(open(f"{out}/{n}.json"))["choices"][0]["text"]
def sh(t): return hashlib.sha256(t.encode()).hexdigest()[:12]
z3, z1 = tx("z_ap3_cold"), tx("z_ap1_after")
y3, y1, x3 = tx("y_ap3"), tx("y_ap1"), tx("x_ap3")
print("=== ISOLATE ===")
print(f"z_ap3_cold(TRUE COLD) = {sh(z3)} {z3[:50]!r}")
print(f"y_ap3(resumed lineage)= {sh(y3)}  equal_to_true_cold={z3==y3}")
print(f"x_ap3(base resume)    = {sh(x3)}  equal_to_true_cold={z3==x3}")
print(f"z_ap1_after(resumed from ap3 lineage) = {sh(z1)}  vs y_ap1(cold) {sh(y1)} equal={z1==y1}")
print("INTERPRETATION: if z_ap3_cold != y_ap3 -> resume contamination proven;")
print("if z_ap3_cold == y_ap3 -> y equality was generic-response coincidence")
EOF
grep -E "GLM sync start=" "$OUT/server-iso.log" >> "$OUT/summary"
chmod -R a+rX "$OUT"
echo ISO_DONE
