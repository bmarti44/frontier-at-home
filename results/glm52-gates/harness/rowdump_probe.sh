#!/bin/bash
# GPU row-content probe for the resume bug (write-cursor / restore-content
# discriminator). Dumps 8 compact-cache rows (layers 0+40) at sync entry of
# every request (DS4_GLM_KV_ROWDUMP), rows [LO..LO+8).
# Server X: base(5047,gen16) -> ap1 (load+resume+suffix write) -> dummy
#   dump0 = virgin, dump1 = post-load restored rows, dump2 = post-ap1 rows
# Server Y (fresh kv): ap1 cold -> dummy
#   dump0 = virgin, dump1 = post-cold-ap1 rows
# VERDICTS: X.dump2 vs Y.dump1 (post-write content at logical rows);
#           X.dump1 vs Y.dump1 lower half (restored vs cold same-lineage).
set -u
OUT=/home/dsv4/ds4-project/glm52-rowdump
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-rowdump
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT/X" "$OUT/Y" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for rowdump window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
base = j["prompt"]
json.dump({"model": "default", "prompt": base, "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
open(sys.argv[2] + "/base_prompt.txt", "w").write(base)
json.dump({"model": "default", "prompt": "ping", "max_tokens": 1,
           "temperature": 0}, open(sys.argv[2] + "/dummy.json", "w"))
EOF

serve() { # $1 = X|Y
  DS4_GLM_KV_ROWDUMP="$OUT/$1" DS4_GLM_KV_ROWDUMP_LO=5040 \
  DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --kv-cache-boundary-align-tokens 4 \
    --kv-cache-boundary-trim-tokens 0 \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || { note "$1 died"; return 1; }
    sleep 2
  done; return 1
}
stop_srv() { kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; }
fire() { curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$2" http://127.0.0.1:$PORT/v1/completions > /dev/null; }
trap 'kill -TERM $SPID 2>/dev/null' EXIT

serve X || { echo ROWDUMP_FAIL; exit 1; }
fire x_base "$OUT/base.json"
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
base = open(out + "/base_prompt.txt").read()
gen = json.load(open(out + "/x_base.json"))["choices"][0]["text"]
json.dump({"model": "default", "prompt": base + gen + "ological analysis shows",
           "max_tokens": 24, "temperature": 0}, open(out + "/ap1.json", "w"))
EOF
fire x_ap1 "$OUT/ap1.json"
fire x_dummy "$OUT/dummy.json"
stop_srv

rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve Y || { echo ROWDUMP_FAIL; exit 1; }
fire y_ap1 "$OUT/ap1.json"
fire y_dummy "$OUT/dummy.json"
stop_srv
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import hashlib, os, sys
out = sys.argv[1]
def sh(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12], os.path.getsize(p)
    except Exception: return "MISSING", 0
for tag in ("X", "Y"):
    for f in sorted(os.listdir(os.path.join(out, tag))):
        h, n = sh(os.path.join(out, tag, f))
        print(f"{tag}/{f}: sha={h} bytes={n}")
def rd(tag, n): return open(f"{out}/{tag}/rowdump-{n:03d}.bin", "rb").read()
try:
    x2, y1 = rd("X", 2), rd("Y", 1)
    print(f"POSTWRITE X.dump2 == Y.dump1: {x2 == y1}")
    if x2 != y1 and len(x2) == len(y1):
        k = next(i for i in range(len(x2)) if x2[i] != y1[i])
        print(f"  first diff at byte {k} of {len(x2)}")
    x1 = rd("X", 1)
    half = min(len(x1), len(y1)) // 2
    print(f"RESTORED-LOWER X.dump1[:half] == Y.dump1[:half]: {x1[:half] == y1[:half]}")
except Exception as e:
    print("compare error:", e)
EOF
chmod -R a+rX "$OUT"
note "rowdump window done (caller restores DSV4)"
echo ROWDUMP_DONE
