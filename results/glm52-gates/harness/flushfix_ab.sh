#!/bin/bash
# Long-prompt A/B for the expert-cache flush fix -- the regime where the bug
# actually bites.
#
# WHAT THE EVIDENCE SHOWED. The flush lines interleave with batch-PREFILL
# loads (slots=23992), one per routed layer, and never with decode loads
# (slots=8). So the storm is a prefill pathology: every prefill layer wiped
# the whole 72 GB expert cache, and decode then started from a cold cache
# (resident 245/7398 observed). A short-prompt harness cannot see this --
# my first decode A/B recorded 0 flushes in BOTH arms and the "bug" arm was
# therefore not a bug arm at all. This one uses the 5047-token fixture.
#
# Arms, same binary, ABBA order, two passes:
#   bug  DS4_CUDA_MODEL_GEN_ALWAYS_BUMP=1  (pre-fix behaviour)
#   fix  default                            (bump only on real map change)
#
# Metrics: cold TTFT (5047-token prefill, 1 token out), warm TTFT (replay),
# decode rate isolated as 32/(t33 - t1) on the warm server, plus flush count,
# final hit%/resident, and output shas (the fix must not change any output).
set -u
OUT=/home/dsv4/ds4-project/glm52-flushfix-ab
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "flush-fix A/B start binary_sha12=$(sha256sum $SRC/ds4-server | cut -c1-12)"
unset DS4_CUDA_MODEL_GEN_ALWAYS_BUMP DS4_GLM_TOPK_KEEP DS4_GLM_TOPK_SKIP_LOAD \
      DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL 2>/dev/null || true

wait_gone() {
  for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || return 0; sleep 2; done
  pkill -KILL -x ds4-server; sleep 5
}
pkill -TERM -f "llama-server.*8011" 2>/dev/null
pkill -TERM -x ds4-server 2>/dev/null; wait_gone

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
for n in (1, 33):
    json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": n,
               "temperature": 0}, open("%s/long%d.json" % (sys.argv[2], n), "w"))
PYEOF

run_arm() { # $1 tag, $2 pass, $3.. env
  local tag=$1 pass=$2; shift 2
  local envs=("$@") key="$1x"
  key="${tag}-p${pass}"
  note "arm $key env=[${envs[*]:-default}]"
  wait_gone
  env "${envs[@]}" \
    DS4_GLM_TP_DEBUG=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    > "$OUT/server-$key.log" 2>&1 &
  SPID=$!
  local up=0
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && { up=1; break; }
    kill -0 $SPID 2>/dev/null || { note "$key died"; return 1; }
    sleep 2
  done
  [[ $up == 1 ]] || { note "$key never up"; kill -TERM $SPID; return 1; }
  fire() {
    local t0=$(date +%s%3N)
    local code=$(curl -s -o "$OUT/$key-$1.json" -w '%{http_code}' --max-time 3600 \
      -H 'Content-Type: application/json' -d @"$2" \
      http://127.0.0.1:$PORT/v1/completions)
    echo "$key $1 ms=$(( $(date +%s%3N) - t0 )) http=$code" >> "$OUT/timings"
  }
  fire cold  "$OUT/long1.json"     # 5047-token prefill from a cold cache
  fire warm  "$OUT/long1.json"     # replay: live KV + whatever cache survived
  fire t33   "$OUT/long33.json"    # decode 32 tokens on top of the warm prefix
  {
    echo "$key flushes=$(grep -c 'expert cache flushed' "$OUT/server-$key.log")"
    echo "$key stats=$(grep 'expert-cache stats' "$OUT/server-$key.log" | tail -1)"
  } >> "$OUT/timings"
  kill -TERM $SPID; wait_gone
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT
run_arm bug 1 DS4_CUDA_MODEL_GEN_ALWAYS_BUMP=1
run_arm fix 1 DS4_CUDA_EXPERT_CACHE_SLRU=1
run_arm fix 2 DS4_CUDA_EXPERT_CACHE_SLRU=1
run_arm bug 2 DS4_CUDA_MODEL_GEN_ALWAYS_BUMP=1
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, re, sys, hashlib, statistics
out = sys.argv[1]
ms, http, flush, stats = {}, {}, {}, {}
for line in open(os.path.join(out, "timings")):
    p = line.split()
    if len(p) >= 3 and p[2].startswith("ms="):
        ms[(p[0], p[1])] = int(p[2].split("=")[1])
        http[(p[0], p[1])] = p[3].split("=")[1] if len(p) > 3 else "?"
    elif len(p) >= 2 and p[1].startswith("flushes="):
        flush[p[0]] = int(p[1].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("stats="):
        stats[p[0]] = line.split("stats=", 1)[1].strip()
def resp(key, lbl):
    try:
        d = json.loads(open(os.path.join(out, "%s-%s.json" % (key, lbl)), 'rb')
                       .read().decode('utf-8', 'replace'))
        return d["choices"][0]["text"], d["usage"]["completion_tokens"]
    except Exception:
        return "", 0
print("%-7s %10s %10s %10s %8s %6s  %s" % (
    "arm", "cold s", "warm s", "dec t/s", "flushes", "http", "cache"))
agg = {}
for tag in ("bug", "fix"):
    for ps in ("1", "2"):
        key = "%s-p%s" % (tag, ps)
        cold, warm, t33 = (ms.get((key, x)) for x in ("cold", "warm", "t33"))
        _, n = resp(key, "t33")
        dec = (n - 1) / ((t33 - warm) / 1000.0) if (t33 and warm and t33 > warm and n > 1) else None
        st = stats.get(key, "")
        m = re.search(r"hit%=([0-9.]+).*resident=(\d+)/(\d+)", st)
        codes = ",".join(sorted({v for (k, l), v in http.items() if k == key}))
        print("%-7s %10s %10s %10s %8s %6s  %s" % (
            key, "%.1f" % (cold/1000) if cold else "--",
            "%.3f" % (warm/1000) if warm else "--",
            "%.3f" % dec if dec else "--",
            flush.get(key, "?"), codes,
            "hit%%=%s res=%s/%s" % m.groups() if m else "-"))
        a = agg.setdefault(tag, {"cold": [], "warm": [], "dec": []})
        if cold: a["cold"].append(cold / 1000)
        if warm: a["warm"].append(warm / 1000)
        if dec: a["dec"].append(dec)
print()
if "bug" in agg and "fix" in agg:
    for metric, better in (("cold", "lower"), ("warm", "lower"), ("dec", "higher")):
        b, f = agg["bug"][metric], agg["fix"][metric]
        if not b or not f: continue
        mb, mf = statistics.mean(b), statistics.mean(f)
        delta = 100 * (mf / mb - 1)
        print("%-5s bug mean=%.3f %s   fix mean=%.3f %s   -> %+.1f%% (%s is better)" % (
            metric, mb, ["%.3f" % x for x in b], mf, ["%.3f" % x for x in f], delta, better))
print()
print("output identity (the fix must not change a single byte):")
for tag in ("bug", "fix"):
    for ps in ("1", "2"):
        key = "%s-p%s" % (tag, ps)
        t, n = resp(key, "t33")
        print("  %-8s n=%-3d sha=%s %r" % (
            key, n, hashlib.sha256(t.encode()).hexdigest()[:12], t[:44]))
PYEOF
chmod -R a+rX "$OUT"
note "flush-fix A/B done"
echo FLUSHFIX_AB_DONE
