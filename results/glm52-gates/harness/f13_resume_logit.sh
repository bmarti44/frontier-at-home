#!/bin/bash
# FIX-A step 3 (decisive): compare LOGITS on a GENUINELY RESUMED append vs a
# cold evaluation of the same prompt.
#   X phase: prime process runs the base (checkpoints + generates), then the
#            process is restarted on the same kv-dir so the FIRST sync of the
#            new process is the appended request -> the engine's logit dump
#            captures the RESUMED evaluation.
#   Y phase: fresh process + wiped kv-dir -> the same appended prompt is
#            evaluated cold; its first sync is the same request.
# Verdict rule: max|delta|==0 -> no corruption; <1e-2 -> evaluation-order
# numerics; large -> state/content corruption in the resume path.
set -u
OUT=/home/dsv4/ds4-project/glm52-f13-resume-logit
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-f13r
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "F13 resume-logit start"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 5

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
open(sys.argv[2] + "/base_prompt.txt", "w").write(j["prompt"])
PYEOF

serve() { # $1 tag (logit dump named after it)
  DS4_GLM_LOGIT_DUMP="$OUT/logit-$1.bin" DS4_GLM_RESUME_GUARD_OFF=1 \
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

# ---- X: prime the lineage, then restart so the append is the first sync
serve Xprime || { echo F13R_FAIL; exit 1; }
fire x_base "$OUT/base.json"
python3 - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
base = open(out + "/base_prompt.txt").read()
gen = json.load(open(out + "/x_base.json"))["choices"][0]["text"]
json.dump({"model": "default", "prompt": base + gen + "ological analysis shows",
           "max_tokens": 8, "temperature": 0}, open(out + "/ap.json", "w"))
PYEOF
stop_srv
serve X || { echo F13R_FAIL; exit 1; }
fire x_ap "$OUT/ap.json"
stop_srv

# ---- Y: cold control, wiped kv-dir, same appended prompt first
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve Y || { echo F13R_FAIL; exit 1; }
fire y_ap "$OUT/ap.json"
stop_srv
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, struct, sys, hashlib
out = sys.argv[1]
def load(p):
    if not os.path.exists(p): return None
    b = open(p, 'rb').read()
    return struct.unpack('<%df' % (len(b)//4), b[:len(b)//4*4])
def tx(n):
    try: return json.load(open("%s/%s.json" % (out, n)))["choices"][0]["text"]
    except Exception as e: return "<ERR %s>" % e
xr, yr = tx("x_ap"), tx("y_ap")
print("text resumed sha=%s" % hashlib.sha256(xr.encode()).hexdigest()[:12])
print("text cold    sha=%s" % hashlib.sha256(yr.encode()).hexdigest()[:12])
print("text identical: %s" % (xr == yr))
print("resumed: %r" % xr[:90])
print("cold   : %r" % yr[:90])
r, c = load("%s/logit-X.bin" % out), load("%s/logit-Y.bin" % out)
if r and c:
    n = min(len(r), len(c))
    d = [abs(r[i]-c[i]) for i in range(n)]
    mx, mean = max(d), sum(d)/n
    ir = max(range(n), key=lambda i: r[i]); ic = max(range(n), key=lambda i: c[i])
    sr, sc = sorted(r, reverse=True)[:2], sorted(c, reverse=True)[:2]
    print("\nLOGITS compared: %d" % n)
    print("max|delta|=%.6g  mean|delta|=%.6g" % (mx, mean))
    print("argmax resumed=%d top1=%.5f margin=%.5f" % (ir, sr[0], sr[0]-sr[1]))
    print("argmax cold   =%d top1=%.5f margin=%.5f" % (ic, sc[0], sc[0]-sc[1]))
    print("same argmax: %s" % (ir == ic))
    if mx == 0:
        print("VERDICT: resumed evaluation BIT-IDENTICAL to cold -> no corruption here")
    elif mx < 1e-2:
        print("VERDICT: tiny deltas -> evaluation-order numerics (greedy amplifies near-ties)")
    else:
        print("VERDICT: LARGE logit deltas -> state/content corruption in the resume path")
else:
    print("logit dumps missing:", [f for f in os.listdir(out) if f.endswith('.bin')])
PYEOF
echo "--- sync traces ---" >> "$OUT/summary"
grep -h "GLM sync start=" "$OUT/server-X.log" 2>/dev/null | head -2 >> "$OUT/summary"
grep -h "GLM sync start=" "$OUT/server-Y.log" 2>/dev/null | head -2 >> "$OUT/summary"
chmod -R a+rX "$OUT"
note "F13 resume-logit done"
echo F13R_DONE
