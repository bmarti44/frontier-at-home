#!/bin/bash
# FIX-B part 2: what is expert-skipping WORTH in tokens/second?
#
# CRITICAL distinction this harness exists to make. DS4_GLM_TOPK_KEEP alone
# only zeroes router GATE WEIGHTS. The streaming loader fetches from
# router_selected, so all 8 experts are still read from SSD -- fidelity
# changes, bytes do not, speed does not. DS4_GLM_TOPK_SKIP_LOAD=1 (added
# 2026-07-26) additionally points every dropped slot at a KEPT expert id;
# the loader dedups (cuda_stream_selected_cache_begin_load -> compact_ids)
# so it fetches exactly `keep` unique experts. Only that arm can be faster.
#
# Arms (same binary, same server config, cold-equivalent start each time):
#   keep8   control, top-8, no truncation
#   keep7s  keep-7 + SKIP_LOAD   (real byte reduction, 7/8 = -12.5%)
#   keep6s  keep-6 + SKIP_LOAD   (real byte reduction, 6/8 = -25%)
#   keep6w  keep-6 weights-only  (NO byte reduction: isolates compute effect,
#                                 and proves the loader is what matters)
#
# DS4_GLM_TP_DEBUG=1 is on in EVERY arm (same stderr overhead everywhere) so
# the per-load "unique=N" counts give a deterministic verification that the
# byte reduction actually happened, rather than an inference from timings.
set -u
OUT=/home/dsv4/ds4-project/glm52-keepn-speed
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "keep-N speed start binary_sha12=$(sha256sum $SRC/ds4-server | cut -c1-12)"

wait_gone() { # do not repeat the stale-lock failure: wait for a real exit
  for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || return 0; sleep 2; done
  pkill -KILL -x ds4-server; sleep 5; return 0
}
pkill -TERM -f "llama-server.*8011" 2>/dev/null
pkill -TERM -x ds4-server 2>/dev/null; wait_gone

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": 32,
           "temperature": 0}, open(sys.argv[2] + "/long32.json", "w"))
json.dump({"model": "default", "prompt":
           "The three most important considerations when designing a distributed cache are",
           "max_tokens": 32, "temperature": 0},
          open(sys.argv[2] + "/short32.json", "w"))
PYEOF

run_arm() { # $1 tag, $2 keepN (0=off), $3 skip_load(1/0)
  local envs=()
  [[ "$2" != "0" ]] && envs+=("DS4_GLM_TOPK_KEEP=$2")
  [[ "$3" == "1" ]] && envs+=("DS4_GLM_TOPK_SKIP_LOAD=1")
  note "arm $1 keep=$2 skip_load=$3"
  wait_gone
  env "${envs[@]}" \
    DS4_GLM_TP_DEBUG=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
    kill -0 $SPID 2>/dev/null || { note "$1 died"; return 1; }
    sleep 2
  done
  kill -0 $SPID 2>/dev/null || { note "$1 never came up"; return 1; }
  # warm the expert cache on the long fixture, then time two decode runs
  curl -s -o /dev/null --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$OUT/long32.json" http://127.0.0.1:$PORT/v1/completions
  local mark=$(wc -l < "$OUT/server-$1.log")   # measure uniques post-warmup only
  echo "$1 mark=$mark" >> "$OUT/timings"
  for r in a b; do
    t0=$(date +%s%3N)
    curl -s -o "$OUT/$1-long-$r.json" --max-time 3600 -H 'Content-Type: application/json' \
      -d @"$OUT/long32.json" http://127.0.0.1:$PORT/v1/completions
    echo "$1 long-$r ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"
  done
  t0=$(date +%s%3N)
  curl -s -o "$OUT/$1-short.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$OUT/short32.json" http://127.0.0.1:$PORT/v1/completions
  echo "$1 short ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"
  kill -TERM $SPID; wait_gone
  return 0
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT
run_arm keep8  0 0
run_arm keep7s 7 1
run_arm keep6s 6 1
run_arm keep6w 6 0
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, re, sys, hashlib
out = sys.argv[1]
arms = ["keep8", "keep7s", "keep6s", "keep6w"]
t, mark = {}, {}
for line in open(os.path.join(out, "timings")):
    p = line.split()
    if len(p) >= 3 and p[2].startswith("ms="):
        t[(p[0], p[1])] = int(p[2].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("mark="):
        mark[p[0]] = int(p[1].split("=")[1])
def resp(f):
    try:
        raw = open(os.path.join(out, f), 'rb').read()
        bad = 0
        try: raw.decode('utf-8')
        except UnicodeDecodeError: bad = 1
        d = json.loads(raw.decode('utf-8', 'replace'))
        return d["usage"]["completion_tokens"], d["choices"][0]["text"], bad
    except Exception:
        return 0, "", 1
def repetition(x):
    w = x.split()
    if len(w) < 8: return 0.0
    return 1.0 - len(set(zip(w, w[1:], w[2:]))) / max(1, len(w) - 2)
UNIQ = re.compile(r"selected-expert batch layer=(\d+) slots=(\d+) unique=(\d+)")
def uniques(arm):
    """DETERMINISTIC VERIFICATION: how many distinct experts the loader
    actually fetched per routed layer, counted only after warmup."""
    path = os.path.join(out, "server-%s.log" % arm)
    hist, n, mb = {}, 0, 0.0
    try:
        for i, line in enumerate(open(path, errors="replace")):
            if i < mark.get(arm, 0): continue
            m = UNIQ.search(line)
            if not m: continue
            if int(m.group(2)) != 8: continue      # decode only, not prefill chunks
            u = int(m.group(3)); hist[u] = hist.get(u, 0) + 1; n += 1
            mm = re.search(r"load=([0-9.]+) MiB", line)
            if mm: mb += float(mm.group(1))
    except FileNotFoundError:
        pass
    return hist, n, mb
print("%-7s %11s %11s %10s %8s %8s  %s" % (
    "arm", "long t/s a", "long t/s b", "short t/s", "bad_utf8", "rep3gr", "uniq/load hist"))
base = None
rows = {}
for arm in arms:
    row, bad = [], 0
    for key, f in (("long-a", f"{arm}-long-a.json"), ("long-b", f"{arm}-long-b.json"),
                   ("short", f"{arm}-short.json")):
        ms = t.get((arm, key), 0)
        n, txt, b = resp(f); bad += b
        row.append(n / (ms / 1000) if ms and n else 0.0)
    rows[arm] = row
    hist, nn, mb = uniques(arm)
    hs = " ".join("%d:%d" % (k, v) for k, v in sorted(hist.items())) or "none"
    _, txt, _ = resp(f"{arm}-long-a.json")
    print("%-7s %11.3f %11.3f %10.3f %8d %8.3f  %s" % (
        arm, row[0], row[1], row[2], bad, repetition(txt), hs))
    if arm == "keep8": base = row
print()
print("%-7s %s" % ("arm", "decode speed vs keep8 control"))
for arm in arms[1:]:
    r = rows[arm]
    print("%-7s long-a %+6.1f%%  long-b %+6.1f%%  short %+6.1f%%" % (
        arm,
        100*(r[0]/base[0]-1) if base[0] else 0,
        100*(r[1]/base[1]-1) if base[1] else 0,
        100*(r[2]/base[2]-1) if base[2] else 0))
print()
print("expert bytes actually fetched per decode step (post-warmup, MiB total):")
for arm in arms:
    hist, nn, mb = uniques(arm)
    print("  %-7s loads=%d total=%.1f MiB avg=%.3f MiB/load" % (
        arm, nn, mb, mb / nn if nn else 0.0))
print()
for arm in arms:
    n, txt, b = resp(f"{arm}-long-a.json")
    print("%-7s sha=%s %r" % (arm, hashlib.sha256(txt.encode()).hexdigest()[:12], txt[:56]))
print()
print("VERIFY: keep7s must show uniq hist dominated by 7, keep6s by 6, and")
print("keep8/keep6w by 8. If keep6w shows 8 AND is not faster, that proves the")
print("gain comes from the loader (bytes), not from skipping the math.")
PYEOF
chmod -R a+rX "$OUT"
note "keep-N speed done"
echo KEEPN_SPEED_DONE
