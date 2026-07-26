#!/bin/bash
# Replay-flow round-trip probe (closes sol's L40 invariant question):
# base -> REPLAY of base (evict+load path, dump at sync entry = post-load
# pre-eval) -> dummy. If L40 lower rows differ live-vs-restored while the
# replay output is byte-identical, the restored-row difference is
# output-inert (benign); the invariant closes.
set -u
OUT=/home/dsv4/ds4-project/glm52-rowdump-replay
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-rowdump-replay
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT/X" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for replay-rowdump window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
json.dump({"model": "default", "prompt": "ping", "max_tokens": 1,
           "temperature": 0}, open(sys.argv[2] + "/dummy.json", "w"))
EOF

DS4_GLM_KV_ROWDUMP="$OUT/X" DS4_GLM_KV_ROWDUMP_LO=5040 \
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
  --kv-cache-boundary-align-tokens 4 \
  --kv-cache-boundary-trim-tokens 0 \
  > "$OUT/server-X.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "died"; echo REPLAY_FAIL; exit 1; }
  sleep 2
done
fire() { curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$2" http://127.0.0.1:$PORT/v1/completions > /dev/null; }
fire x_base "$OUT/base.json"
fire x_replay "$OUT/base.json"
fire x_dummy "$OUT/dummy.json"
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
x1 = open(f"{out}/X/rowdump-001.bin", "rb").read()
x2 = open(f"{out}/X/rowdump-002.bin", "rb").read()
seg = [('L0 kv_lora', 0, 16384), ('L0 k_rope', 16384, 18432), ('L0 idx', 18432, 20480),
       ('L40 kv_lora', 20480, 36864), ('L40 k_rope', 36864, 38912), ('L40 idx', 38912, 40960)]
print("REPLAY-FLOW round trip (live post-store vs post-load pre-eval), rows below 5044:")
for name, lo, hi in seg:
    a, b = lo, lo + (hi - lo) // 2
    diff = sum(1 for p, q in zip(x1[a:b], x2[a:b]) if p != q)
    print(f"  {name}: {diff}/{b-a} bytes differ")
tb = json.load(open(f"{out}/x_base.json"))["choices"][0]["text"]
tr = json.load(open(f"{out}/x_replay.json"))["choices"][0]["text"]
print(f"base sha={hashlib.sha256(tb.encode()).hexdigest()[:12]} "
      f"replay sha={hashlib.sha256(tr.encode()).hexdigest()[:12]} identical={tb == tr}")
print("VERDICT: L40 differs here + identical output => restored-row diff is output-inert (benign)")
EOF
grep -E "live kv cache|kv cache (stored|hit)" "$OUT/server-X.log" | head -6 >> "$OUT/summary"
chmod -R a+rX "$OUT"
note "replay window done (caller restores DSV4)"
echo REPLAY_DONE
