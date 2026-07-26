#!/bin/bash
# Decode-path profile window: serve GLM-5.2 (ds4 streaming, full serving
# pins) with DS4_CUDA_LOAD_PROFILE + DS4_CUDA_MOE_PROFILE, run the short
# fixture twice (run2 = hits-dominant) then a 32-token decode, and
# aggregate per-token ms into loader (hit-copy / parallel-fetch /
# arena-fill) vs kernel (xq/sort/gateup/midq/down/sum) buckets.
# Answers: where do the ~714 ms/token of warm decode actually go?
# Caller stops/restores DSV4 around this window.
set -u
OUT=/home/dsv4/ds4-project/glm52-loadprof
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
PORT=8013
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for profile window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

DS4_CUDA_LOAD_PROFILE=${PROBE_LOAD_PROFILE:-1} DS4_CUDA_MOE_PROFILE=${PROBE_MOE_PROFILE:-1} \
DS4_CUDA_ATTN_OUTPUT_PROFILE=${PROBE_ATTN_PROFILE:-1} \
DS4_CUDA_EXPERT_CACHE_PIN=${PROBE_PIN:-1} \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=${PROBE_CACHE_GB:-68} \
DS4_CUDA_FETCH_THREADS=6 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  ${PROBE_EXTRA_ARGS:-} \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "server died"; tail -8 "$OUT/server.log" >> "$OUT/run.log"; echo PROF_FAIL; exit 1; }
  sleep 2
done
note "ready"

fire() { local t0=$(date +%s%3N)
  curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$2" http://127.0.0.1:$PORT/v1/completions
  echo "$1 wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-short.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
j["max_tokens"] = 32
json.dump(j, open(sys.argv[2] + "/f_dec32.json", "w"))
EOF
cp "$REPO/results/glm52-gates/harness/fixture-glm-short.json" "$OUT/f_short.json"

fire warm1 "$OUT/f_short.json"           # arena warmup for this working set
MARK1=$(grep -c LOADPROF "$OUT/server.log" || true)
fire warm2 "$OUT/f_short.json"           # hits-dominant repeat
MARK2=$(grep -c LOADPROF "$OUT/server.log" || true)
fire dec32 "$OUT/f_dec32.json"           # the measured decode window
echo "marks $MARK1 $MARK2 $(grep -c LOADPROF "$OUT/server.log" || true)" >> "$OUT/timings"
grep -iE "mtp|accept" "$OUT/server.log" | tail -12 >> "$OUT/timings" || true

kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

# aggregate: loader lines after MARK2 belong to dec32
python3 - "$OUT/server.log" "$MARK2" <<'EOF' | tee "$OUT/summary"
import re, sys
log, skip = open(sys.argv[1], errors="replace").read().splitlines(), int(sys.argv[2])
lp = [l for l in log if l.startswith("LOADPROF")][skip:]
mp = [l for l in log if "CUDA MoE profile tokens=1 " in l]
def f(l, k): return float(re.search(k + r"=([0-9.]+)", l).group(1))
def i(l, k): return int(re.search(k + r"=([0-9]+)", l).group(1))
n_layers = 79
if lp:
    tok = max(1, len(lp) // n_layers)
    tot = {k: sum(f(l, k) for l in lp) / tok for k in ("hit_ms", "fetch_ms", "fill_ms", "total_ms")}
    hits = sum(i(l, "hits") for l in lp) / tok
    miss = sum(i(l, "miss") for l in lp) / tok
    print(f"LOADER per-token over {tok} tokens: hits={hits:.1f} miss={miss:.1f} "
          f"hit_copy={tot['hit_ms']:.1f}ms fetch={tot['fetch_ms']:.1f}ms "
          f"fill={tot['fill_ms']:.1f}ms other={tot['total_ms']-tot['hit_ms']-tot['fetch_ms']-tot['fill_ms']:.1f}ms "
          f"TOTAL={tot['total_ms']:.1f}ms")
if mp:
    mp = mp[-32 * n_layers:]
    tok = max(1, len(mp) // n_layers)
    keys = ("xq", "sort", "gateup", "midq", "down", "sum", "total")
    agg = {k: sum(f(l, k) for l in mp) / tok for k in keys}
    print("KERNEL per-token: " + " ".join(f"{k}={agg[k]:.1f}ms" for k in keys))
ap = [l for l in log if "attention output profile tokens=1 " in l]
if ap:
    ap = ap[-32 * n_layers:]
    tok = max(1, len(ap) // n_layers)
    agg = {k: sum(f(l, k) for l in ap) / tok for k in ("A", "B", "total")}
    print(f"ATTN-OUT per-token: A={agg['A']:.1f}ms B={agg['B']:.1f}ms total={agg['total']:.1f}ms")
pin = [l for l in log if "arena pin" in l]
if pin:
    print(pin[-1].strip())
EOF
cat "$OUT/timings"
note "window done (caller restores DSV4)"
echo PROF_DONE
