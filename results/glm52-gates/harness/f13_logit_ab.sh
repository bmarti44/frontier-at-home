#!/bin/bash
# FIX-A step 2: compare LOGITS (not text) for the same appended prompt,
# resumed vs cold. Distinguishes the two surviving F13 hypotheses:
#   - tiny logit deltas that greedy amplifies  => evaluation-order numerics
#   - large/structured logit deltas           => state/content corruption
# Uses the engine's existing DS4_GLM_LOGIT_DUMP (first sync completion only).
# Smaller arena (24GB) so the run fits under the safety floor.
set -u
OUT=/home/dsv4/ds4-project/glm52-f13-logit
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-f13
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "F13 logit A/B start"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 5

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
base = j["prompt"]
json.dump({"model": "default", "prompt": base, "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
open(sys.argv[2] + "/base_prompt.txt", "w").write(base)
EOF

serve() { # $1 tag, $2 logit-dump path
  DS4_GLM_LOGIT_DUMP="$2" DS4_GLM_RESUME_GUARD_OFF=1 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=24 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 20GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 0 \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || { note "$1 died"; return 1; }
    sleep 2
  done; return 1
}
stop_srv() { kill -TERM ${SPID:-0} 2>/dev/null
  for i in $(seq 1 60); do kill -0 ${SPID:-0} 2>/dev/null || break; sleep 2; done; }
fire() { curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$2" http://127.0.0.1:$PORT/v1/completions > /dev/null; }
trap 'stop_srv' EXIT

# --- run A: resumed. base (checkpoints + generates), then the append. The
#     logit dump fires on the FIRST sync (the base), so restart to capture
#     the append's logits: phase A1 primes the disk-KV, phase A2 resumes.
serve prime /dev/null || { echo F13_FAIL; exit 1; }
fire prime_base "$OUT/base.json"
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
base = open(out + "/base_prompt.txt").read()
gen = json.load(open(out + "/prime_base.json"))["choices"][0]["text"]
json.dump({"model": "default", "prompt": base + gen + "ological analysis shows",
           "max_tokens": 1, "temperature": 0}, open(out + "/ap.json", "w"))
EOF
fire resumed_ap "$OUT/ap.json"          # genuine resume (same process, live+disk state)
stop_srv
cp "$OUT/logit_resumed.bin" /dev/null 2>/dev/null || true

# --- run B: cold. fresh process AND wiped kv-dir, same appended prompt first.
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve cold "$OUT/logit_cold.bin" || { echo F13_FAIL; exit 1; }
fire cold_ap "$OUT/ap.json"
stop_srv

# --- run C: resumed WITH logit dump armed (dump fires on its first sync,
#     which is the append because the prime phase ran in a prior process)
serve resumed "$OUT/logit_resumed.bin" || { echo F13_FAIL; exit 1; }
fire resumed_ap2 "$OUT/ap.json"
stop_srv
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, os, struct, sys
out = sys.argv[1]
def load(p):
    if not os.path.exists(p): return None
    b = open(p, 'rb').read()
    return struct.unpack(f'<{len(b)//4}f', b[:len(b)//4*4])
a, c = load(f"{out}/logit_resumed.bin"), load(f"{out}/logit_cold.bin")
if not a or not c:
    print("missing dumps:", os.listdir(out)); raise SystemExit
n = min(len(a), len(c))
d = [abs(a[i]-c[i]) for i in range(n)]
mx = max(d); mean = sum(d)/n
ta = max(range(n), key=lambda i: a[i]); tc = max(range(n), key=lambda i: c[i])
sa, sc = sorted(a, reverse=True)[:2], sorted(c, reverse=True)[:2]
print(f"vocab compared: {n}")
print(f"max |delta| = {mx:.6g}   mean |delta| = {mean:.6g}")
print(f"argmax resumed={ta} (top1={sa[0]:.5f}, margin={sa[0]-sa[1]:.5f})")
print(f"argmax cold   ={tc} (top1={sc[0]:.5f}, margin={sc[0]-sc[1]:.5f})")
print(f"same argmax: {ta == tc}")
print()
print("INTERPRETATION: max|delta| ~<1e-2 with equal argmax => evaluation-order")
print("numerics. Large deltas / different argmax with a wide margin => state or")
print("content corruption.")
EOF
chmod -R a+rX "$OUT"
note "F13 logit A/B done"
echo F13_DONE
