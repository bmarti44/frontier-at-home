#!/bin/bash
# FIX-C: the SLRU claim needs a CAUSAL A/B, not a single pair.
#
# The audit (F07) found the SLRU summary conflated two different measurements
# and rested on one A/B pair with no committed digests. SLRU is already in the
# serving profile, so this either confirms it or removes it.
#
# Arms (same binary, ABBA over two passes):
#   slru   DS4_CUDA_EXPERT_CACHE_SLRU=1   (current serving default)
#   lru    SLRU disabled                   (plain LRU)
#
# Decode is isolated the same way as decode_ab.sh: 64/(t65 - t1) against one
# warm server so prefill cancels. Recorded per arm: final hit%/resident from
# the engine's own counter, a DIGEST of the expert access stream so the two
# arms can be shown to have seen the SAME sequence of requests (the thing that
# makes a hit-rate comparison meaningful), HTTP status, and the output sha.
set -u
OUT=/home/dsv4/ds4-project/glm52-slru-ab
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "SLRU A/B start binary_sha12=$(sha256sum $SRC/ds4-server | cut -c1-12)"
unset DS4_CUDA_EXPERT_CACHE_SLRU DS4_GLM_TOPK_KEEP DS4_GLM_TOPK_SKIP_LOAD \
      DS4_CUDA_MODEL_GEN_ALWAYS_BUMP 2>/dev/null || true

wait_gone() {
  for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || return 0; sleep 2; done
  pkill -KILL -x ds4-server; sleep 5
}
pkill -TERM -x ds4-server 2>/dev/null; wait_gone

python3 - "$OUT" <<'PYEOF'
import json, sys
p = ("Explain, in careful technical detail, how a write-back cache decides "
     "which line to evict and why that policy matters for throughput.")
for n in (1, 65):
    json.dump({"model": "default", "prompt": p, "max_tokens": n,
               "temperature": 0}, open("%s/q%d.json" % (sys.argv[1], n), "w"))
PYEOF

run_arm() { # $1 tag, $2 pass, $3 slru(1/0)
  local key="$1-p$2"
  local envs=()
  [[ "$3" == "1" ]] && envs+=("DS4_CUDA_EXPERT_CACHE_SLRU=1")
  note "arm $key slru=$3"
  wait_gone
  env "${envs[@]}" DS4_CUDA_EXPERT_TRACE=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
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
  fire warm "$OUT/q65.json"
  local mark=$(wc -l < "$OUT/server-$key.log")
  fire t1  "$OUT/q1.json"
  fire t65 "$OUT/q65.json"
  # DIGEST of the expert access stream after warmup: identical digests across
  # arms prove both caches saw the SAME request sequence, which is what makes
  # the hit-rate difference attributable to the POLICY.
  local dig=$(tail -n +$((mark+1)) "$OUT/server-$key.log" | grep '^XTRACE' | sha256sum | cut -c1-16)
  local nacc=$(tail -n +$((mark+1)) "$OUT/server-$key.log" | grep -c '^XTRACE')
  {
    echo "$key access_digest=$dig accesses=$nacc"
    echo "$key stats=$(grep 'expert-cache stats' "$OUT/server-$key.log" | tail -1)"
  } >> "$OUT/timings"
  kill -TERM $SPID; wait_gone
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT
run_arm slru 1 1
run_arm lru  1 0
run_arm lru  2 0
run_arm slru 2 1
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, re, sys, hashlib, statistics
out = sys.argv[1]
ms, http, dig, acc, stats = {}, {}, {}, {}, {}
for line in open(os.path.join(out, "timings")):
    p = line.split()
    if len(p) >= 3 and p[2].startswith("ms="):
        ms[(p[0], p[1])] = int(p[2].split("=")[1])
        http[(p[0], p[1])] = p[3].split("=")[1] if len(p) > 3 else "?"
    elif len(p) >= 2 and p[1].startswith("access_digest="):
        dig[p[0]] = p[1].split("=")[1]
        if len(p) > 2 and p[2].startswith("accesses="): acc[p[0]] = int(p[2].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("stats="):
        stats[p[0]] = line.split("stats=", 1)[1].strip()
def tps(key):
    t1, t65 = ms.get((key, "t1")), ms.get((key, "t65"))
    if not t1 or not t65 or t65 <= t1: return None
    try:
        d = json.loads(open(os.path.join(out, "%s-t65.json" % key), 'rb').read().decode('utf-8','replace'))
        n = d["usage"]["completion_tokens"]
    except Exception: return None
    return (n - 1) / ((t65 - t1) / 1000.0) if n > 1 else None
def sha(key):
    try:
        d = json.loads(open(os.path.join(out, "%s-t65.json" % key), 'rb').read().decode('utf-8','replace'))
        return hashlib.sha256(d["choices"][0]["text"].encode()).hexdigest()[:12]
    except Exception: return "-"
print("%-9s %10s %9s %10s %18s %s" % ("arm", "dec t/s", "hit%", "resident", "access_digest", "out_sha"))
agg, hits = {}, {}
for tag in ("slru", "lru"):
    for ps in ("1", "2"):
        k = "%s-p%s" % (tag, ps)
        if k not in dig and (k, "t65") not in ms: continue
        v = tps(k)
        m = re.search(r"hit%=([0-9.]+).*resident=(\d+)/(\d+)", stats.get(k, ""))
        hp = float(m.group(1)) if m else float("nan")
        print("%-9s %10s %9s %10s %18s %s" % (
            k, "%.3f" % v if v else "--", "%.1f" % hp if m else "--",
            "%s/%s" % (m.group(2), m.group(3)) if m else "--",
            dig.get(k, "-"), sha(k)))
        if v: agg.setdefault(tag, []).append(v)
        if m: hits.setdefault(tag, []).append(hp)
print()
ds = set(dig.values())
print("access-stream digests: %s" % ("IDENTICAL across all arms -- the hit-rate"
      " difference is attributable to the eviction policy" if len(ds) == 1 else
      "DIFFER (%d distinct) -- arms did not see the same access sequence, so a"
      " hit-rate comparison is NOT causal" % len(ds)))
print("accesses per arm: %s" % acc)
print()
for metric, d_ in (("decode t/s", agg), ("hit%", hits)):
    if "slru" in d_ and "lru" in d_:
        a, b = statistics.mean(d_["lru"]), statistics.mean(d_["slru"])
        print("%-11s lru=%.3f %s  slru=%.3f %s  -> %+.1f%%" % (
            metric, a, ["%.3f" % x for x in d_["lru"]], b,
            ["%.3f" % x for x in d_["slru"]], 100 * (b / a - 1) if a else 0))
PYEOF
chmod -R a+rX "$OUT"
note "SLRU A/B done"
echo SLRU_AB_DONE
