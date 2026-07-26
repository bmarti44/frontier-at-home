#!/bin/bash
# MATCHED head-to-head: GLM-5.2 vs DSV4, identical fixtures, identical
# request sequence, identical measurement code, same box, back-to-back.
#
# Closes the gap every prior comparison had: DSV4's historical numbers came
# from different prompts/settings than GLM's (sol audit finding, repeatedly).
# Sequence per engine: cold TTFT -> warm TTFT -> warm TTFT -> decode(64) x2
# on the SAME 5047-token fixture, plus a short-prompt TTFT.
# ABBA-capable: MATCHED_ORDER=glm-first (default) or dsv4-first.
set -u
OUT=/home/dsv4/ds4-project/glm52-matched-ab
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-matched
PORT=8011   # DSV4's launcher hardcodes 8011, so both engines are
            # measured on the same port; prod is down for this window
ORDER=${MATCHED_ORDER:-glm-first}
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "matched A/B start (order=$ORDER)"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
base = j["prompt"]
for name, mt in (("ttft", 1), ("dec64", 64)):
    json.dump({"model": "default", "prompt": base, "max_tokens": mt,
               "temperature": 0}, open(f"{sys.argv[2]}/f_{name}.json", "w"))
json.dump({"model": "default", "prompt": "Explain in one sentence why the sky is blue.",
           "max_tokens": 1, "temperature": 0},
          open(sys.argv[2] + "/f_short.json", "w"))
EOF

fire() { # tag fixture
  local t0=$(date +%s%3N)
  local code=$(curl -s -o "$OUT/$1.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$OUT/$2" \
    http://127.0.0.1:$PORT/v1/completions)
  echo "$1 http=$code wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"
}

run_sequence() { # engine-tag
  fire "$1-short"      f_short.json     # short-prompt TTFT (matches DSV4's historical probe shape)
  fire "$1-ttft-cold"  f_ttft.json      # 5047-token cold prefill, 1 token out
  fire "$1-ttft-warm1" f_ttft.json      # first warm repeat
  fire "$1-ttft-warm2" f_ttft.json      # steady-state warm repeat
  fire "$1-dec64-a"    f_dec64.json     # decode throughput
  fire "$1-dec64-b"    f_dec64.json     # decode repeat
}

serve_glm() {
  DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 0 \
    > "$OUT/server-glm.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || { note "glm died"; return 1; }
    sleep 2
  done; return 1
}

serve_dsv4() {
  # the qualified production stack (its launcher pins 8011)
  "$REPO/scripts/21_serve_llamacpp.sh" start > "$OUT/server-dsv4.log" 2>&1
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    sleep 2
  done; return 1
}
stop_any() {
  kill -TERM ${SPID:-0} 2>/dev/null || true
  pkill -TERM -f "llama-server.*--port $PORT" 2>/dev/null || true
  pkill -TERM -f "ds4-server.*--port $PORT" 2>/dev/null || true
  for i in $(seq 1 90); do
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models" || break
    sleep 2
  done
}
trap 'stop_any' EXIT

if [[ "$ORDER" == "glm-first" ]]; then
  note "serving GLM"; serve_glm && run_sequence glm; stop_any
  note "serving DSV4"; serve_dsv4 && run_sequence dsv4; stop_any
else
  note "serving DSV4"; serve_dsv4 && run_sequence dsv4; stop_any
  note "serving GLM"; serve_glm && run_sequence glm; stop_any
fi
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, os, sys, hashlib
out = sys.argv[1]
t = {}
for line in open(f"{out}/timings"):
    p = line.split()
    t[p[0]] = {"http": p[1].split("=")[1], "ms": int(p[2].split("=")[1])}
def toks(tag):
    try:
        d = json.load(open(f"{out}/{tag}.json"))
        u = d.get("usage", {})
        return u.get("completion_tokens", 0)
    except Exception:
        return 0
print(f"{'metric':28s} {'GLM-5.2':>14s} {'DSV4':>14s}   ratio")
rows = [("short-prompt TTFT (ms)", "short"),
        ("5047-tok cold TTFT (ms)", "ttft-cold"),
        ("warm TTFT first (ms)", "ttft-warm1"),
        ("warm TTFT steady (ms)", "ttft-warm2")]
for label, key in rows:
    g, d = t.get(f"glm-{key}", {}).get("ms"), t.get(f"dsv4-{key}", {}).get("ms")
    if g and d:
        print(f"{label:28s} {g:>14d} {d:>14d}   {g/d:.2f}x")
for tag in ("dec64-a", "dec64-b"):
    g, d = t.get(f"glm-{tag}", {}), t.get(f"dsv4-{tag}", {})
    if g and d:
        gt, dt = toks(f"glm-{tag}"), toks(f"dsv4-{tag}")
        gs = gt / (g["ms"] / 1000) if gt else 0
        ds = dt / (d["ms"] / 1000) if dt else 0
        print(f"{'decode t/s ('+tag+')':28s} {gs:>14.2f} {ds:>14.2f}   "
              f"{(ds/gs if gs else 0):.1f}x DSV4 faster")
print()
for k in sorted(t):
    print(f"  raw {k}: http={t[k]['http']} wall_ms={t[k]['ms']}")
EOF
chmod -R a+rX "$OUT"
note "matched A/B done (caller restores DSV4 prod)"
echo MATCHED_DONE
