#!/bin/bash
# FIX-A step 5: run the EXACT sequence of ttft_append_probe.sh (the only
# harness observed to reach a genuine resume: sync shows start=5044
# suffix=22) with TAGGED per-sync logit dumps. Each dump is named
#   <path>.<seq>.s<start>_p<prompt>_x<suffix>
# so the resumed evaluation is selected by its sync parameters (start>0),
# never by index — the mistake that produced a retracted verdict earlier.
#
# X: base then append in one process (live-session resume path)
# Y: fresh process + wiped kv, append only (cold reference)
# Compare X's dump with start>0 and prompt==append_len against Y's dump
# with start==0 and the same prompt length.
set -u
OUT=/home/dsv4/ds4-project/glm52-f13-tagged
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-f13tag
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "F13 tagged-logit start"
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
  DS4_GLM_RESUME_GUARD_OFF=1 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  DS4_KV_DECIDE_LOG=1 \
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

serve X || { echo F13T_FAIL; exit 1; }
fire x_base "$OUT/base.json"
# build the append EXACTLY as ttft_append_probe.sh does (mid-word glue onto
# the generated tail) — that harness's shape is what reaches a real resume
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

rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve Y || { echo F13T_FAIL; exit 1; }
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
dumps = [d for d in (parse(f) for f in os.listdir(out)) if d]
dumps.sort(key=lambda d: (d["arm"], d["seq"]))
print("tagged dumps:")
for d in dumps:
    print("  %s arm=%s seq=%d start=%d prompt=%d suffix=%d" %
          (d["file"], d["arm"], d["seq"], d["start"], d["prompt"], d["suffix"]))
ap_len = max((d["prompt"] for d in dumps), default=0)
resumed = [d for d in dumps if d["arm"] == "X" and d["start"] > 0 and d["prompt"] == ap_len]
cold    = [d for d in dumps if d["arm"] == "Y" and d["start"] == 0 and d["prompt"] == ap_len]
print("\nappend prompt length inferred: %d" % ap_len)
print("resumed candidates (X, start>0): %s" % [d["file"] for d in resumed])
print("cold candidates (Y, start==0):   %s" % [d["file"] for d in cold])
def load(fn):
    b = open(os.path.join(out, fn), 'rb').read()
    return struct.unpack('<%df' % (len(b)//4), b[:len(b)//4*4])
if resumed and cold:
    r, c = load(resumed[0]["file"]), load(cold[0]["file"])
    n = min(len(r), len(c))
    d_ = [abs(r[i]-c[i]) for i in range(n)]
    mx, mean = max(d_), sum(d_)/n
    ir = max(range(n), key=lambda i: r[i]); ic = max(range(n), key=lambda i: c[i])
    sr, sc = sorted(r, reverse=True)[:2], sorted(c, reverse=True)[:2]
    print("\nCOMPARING %s (resumed) vs %s (cold)" % (resumed[0]["file"], cold[0]["file"]))
    print("logits=%d  max|delta|=%.6g  mean|delta|=%.6g" % (n, mx, mean))
    print("argmax resumed=%d top1=%.5f margin=%.5f" % (ir, sr[0], sr[0]-sr[1]))
    print("argmax cold   =%d top1=%.5f margin=%.5f" % (ic, sc[0], sc[0]-sc[1]))
    print("same argmax: %s" % (ir == ic))
    if mx == 0:
        print("VERDICT: resumed evaluation BIT-IDENTICAL to cold")
    elif mx < 1e-2:
        print("VERDICT: tiny deltas -> evaluation-order numerics")
    else:
        print("VERDICT: LARGE deltas -> state/content difference in the resume path")
else:
    print("\nNO VALID PAIR: the append did not resume in arm X (no start>0 dump at")
    print("the append length). Report as harness limitation, NOT as a finding.")
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
echo "--- X sync traces ---" >> "$OUT/summary"
grep -h "GLM sync start=" "$OUT/server-X.log" 2>/dev/null | head -4 >> "$OUT/summary"
echo "--- KV decisions (X) ---" >> "$OUT/summary"
grep -h "KVDECIDE" "$OUT/server-X.log" 2>/dev/null >> "$OUT/summary"
chmod -R a+rX "$OUT"
note "F13 tagged-logit done"
echo F13T_DONE
