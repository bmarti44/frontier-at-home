#!/bin/bash
# FIX-A step 7 (decisive): reproduce the IDENTIFIED failing regime and take
# the logit comparison inside it.
#
# Regime (from KVDECIDE): a live-cache miss caused by BPE re-merge at the
# generation junction (common < live), followed by a DISK-KV load of a
# shorter base checkpoint, then a LONG suffix extension. Divergence has only
# ever been observed here.
#
# Recipe: (1) prime — base request, VERIFY a "kv cache stored" line;
#         (2) keep KVDIR; fire the append in the SAME process so the live
#             session mismatches at the junction and the server disk-loads;
#         (3) verify the sync trace shows start=<checkpoint> with a LONG
#             suffix (the failing shape) before trusting any comparison;
#         (4) cold arm: fresh process + wiped KVDIR, append only.
# Dumps are tagged with (start,prompt,suffix) so the pair is chosen by
# evidence. If the regime is not reached, the script says so and emits NO
# verdict.
set -u
OUT=/home/dsv4/ds4-project/glm52-f13-regime
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-f13reg
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "F13 regime probe start"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 5

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
open(sys.argv[2] + "/base_prompt.txt", "w").write(j["prompt"])
PYEOF

serve() { # $1 tag
  DS4_GLM_LOGIT_DUMP="$OUT/lg-$1" DS4_GLM_LOGIT_DUMP_ALL=1 \
  DS4_GLM_RESUME_GUARD_OFF=1 DS4_KV_DECIDE_LOG=1 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
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

# ---- X: prime the disk store, then append in the SAME process
serve X || { echo F13REG_FAIL; exit 1; }
fire x_base "$OUT/base.json"
# a SECOND base-shaped request forces the cold-boundary store to land before
# the append (the store is written on completion of a qualifying request)
fire x_base2 "$OUT/base.json"
STORED=$(grep -c "kv cache stored" "$OUT/server-X.log" || true)
note "checkpoints stored after prime: $STORED"
python3 - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
base = open(out + "/base_prompt.txt").read()
gen = json.load(open(out + "/x_base.json"))["choices"][0]["text"]
json.dump({"model": "default", "prompt": base + gen + "ological analysis shows",
           "max_tokens": 24, "temperature": 0}, open(out + "/ap.json", "w"))
PYEOF
fire x_ap "$OUT/ap.json"
stop_srv

# ---- Y: cold reference
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve Y || { echo F13REG_FAIL; exit 1; }
fire y_ap "$OUT/ap.json"
stop_srv
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, re, struct, sys, hashlib
out = sys.argv[1]
def parse(fn):
    m = re.match(r'lg-([XY])\.(\d+)\.s(-?\d+)_p(-?\d+)_x(-?\d+)$', fn)
    if not m: return None
    return {"arm": m.group(1), "seq": int(m.group(2)), "start": int(m.group(3)),
            "prompt": int(m.group(4)), "suffix": int(m.group(5)), "file": fn}
dumps = sorted((d for d in (parse(f) for f in os.listdir(out)) if d),
               key=lambda d: (d["arm"], d["seq"]))
for d in dumps:
    print("  %s start=%d prompt=%d suffix=%d" % (d["file"], d["start"], d["prompt"], d["suffix"]))
ap_len = max((d["prompt"] for d in dumps), default=0)
# the FAILING regime: X dump at the append length, start>0, LONG suffix (>4)
resumed = [d for d in dumps if d["arm"] == "X" and d["prompt"] == ap_len
           and d["start"] > 0 and d["suffix"] > 4]
cold = [d for d in dumps if d["arm"] == "Y" and d["prompt"] == ap_len and d["start"] == 0]
print("\nappend length=%d" % ap_len)
print("REGIME-MATCHING resumed dumps (start>0, suffix>4): %s" % [d["file"] for d in resumed])
print("cold dumps: %s" % [d["file"] for d in cold])
def load(fn):
    b = open(os.path.join(out, fn), 'rb').read()
    return struct.unpack('<%df' % (len(b)//4), b[:len(b)//4*4])
if resumed and cold:
    r, c = load(resumed[0]["file"]), load(cold[0]["file"])
    n = min(len(r), len(c))
    dd = [abs(r[i]-c[i]) for i in range(n)]
    mx, mean = max(dd), sum(dd)/n
    ir = max(range(n), key=lambda i: r[i]); ic = max(range(n), key=lambda i: c[i])
    sr, sc = sorted(r, reverse=True)[:2], sorted(c, reverse=True)[:2]
    print("\nIN-REGIME COMPARISON %s vs %s" % (resumed[0]["file"], cold[0]["file"]))
    print("logits=%d max|delta|=%.6g mean|delta|=%.6g" % (n, mx, mean))
    print("argmax resumed=%d top1=%.5f margin=%.5f" % (ir, sr[0], sr[0]-sr[1]))
    print("argmax cold   =%d top1=%.5f margin=%.5f" % (ic, sc[0], sc[0]-sc[1]))
    print("same argmax: %s" % (ir == ic))
    if mx == 0:
        print("VERDICT: BIT-IDENTICAL even in the failing regime")
    elif mx < 1e-2:
        print("VERDICT: NUMERICS — tiny deltas amplified by greedy at near-ties")
    else:
        print("VERDICT: CORRUPTION — large logit deltas in the resume path")
else:
    print("\nREGIME NOT REACHED — no verdict. (need X: start>0 & suffix>4 at the")
    print("append length, plus a cold Y dump). Report as harness limitation.")
def tx(n):
    try: return json.load(open("%s/%s.json" % (out, n)))["choices"][0]["text"]
    except Exception as e: return "<ERR %s>" % e
xr, yr = tx("x_ap"), tx("y_ap")
print("\ntext resumed sha=%s cold sha=%s identical=%s" % (
    hashlib.sha256(xr.encode()).hexdigest()[:12],
    hashlib.sha256(yr.encode()).hexdigest()[:12], xr == yr))
print("resumed: %r" % xr[:80])
print("cold   : %r" % yr[:80])
PYEOF
{ echo "--- X sync ---"; grep -h "GLM sync start=" "$OUT/server-X.log" | head -6
  echo "--- X kv ---"; grep -hE "KVDECIDE|kv cache stored|kv cache hit" "$OUT/server-X.log" | head -8
} >> "$OUT/summary"
chmod -R a+rX "$OUT"
note "F13 regime probe done"
echo F13REG_DONE
