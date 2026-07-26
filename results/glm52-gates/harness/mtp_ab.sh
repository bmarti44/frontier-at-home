#!/bin/bash
# MTP generation A/B: same 8 prompts, greedy 96 tokens, --glm-mtp OFF vs ON.
# Measures per-config steady-state decode wall (pass 2 of 2, arena warm),
# output agreement, and acceptance counters. MTP is greedy-verified so any
# divergence should be a near-tie logit flip (2-token-batch FP order);
# this window quantifies how often that actually happens.
set -u
OUT=/home/dsv4/ds4-project/glm52-mtp-ab
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
PORT=8014
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for A/B window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$OUT" <<'EOF'
import json, sys
prompts = [
 "The three most important considerations when designing a distributed cache are",
 "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
 "In 1905, Albert Einstein published four papers that",
 "SELECT customers.name, SUM(orders.total)\nFROM customers\n",
 "The difference between TCP and UDP is that",
 "Once upon a time, in a village at the edge of a great forest,",
 "To compile a C program with debugging symbols and optimizations disabled, run",
 "The primary causes of the fall of the Roman Empire include",
]
for i, p in enumerate(prompts):
    json.dump({"model": "default", "prompt": p, "max_tokens": 96,
               "temperature": 0}, open(f"{sys.argv[1]}/p{i}.json", "w"))
EOF

serve() { # $1 = tag, $2 = extra args
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=68 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB $2 \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || { note "$1 server died"; return 1; }
    sleep 2
  done
  return 1
}
stop_engine() {
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
}
trap 'kill -TERM $SPID 2>/dev/null' EXIT

run_pass() { # $1 = tag, $2 = pass no (timing only recorded for pass 2)
  local t0 t1
  for i in 0 1 2 3 4 5 6 7; do
    t0=$(date +%s%3N)
    curl -s -o "$OUT/$1-p$i.json" --max-time 3600 -H 'Content-Type: application/json' \
      -d @"$OUT/p$i.json" http://127.0.0.1:$PORT/v1/completions
    t1=$(date +%s%3N)
    [[ "$2" == 2 ]] && echo "$1 p$i wall_ms=$((t1-t0))" >> "$OUT/timings"
  done
}

for cfg in off on; do
  if [[ $cfg == off ]]; then EXTRA=""; else EXTRA="--glm-mtp-timing"; fi
  note "serving cfg=$cfg"
  serve $cfg "$EXTRA" || { echo AB_FAIL; exit 1; }
  run_pass $cfg 1          # arena + KV warm for this working set
  run_pass $cfg 2          # timed pass
  stop_engine
done
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
tot = {"off": 0, "on": 0}
ntok = {"off": 0, "on": 0}
agree = disagree = 0
for i in range(8):
    row = {}
    for cfg in ("off", "on"):
        d = json.load(open(f"{out}/{cfg}-p{i}.json"))
        row[cfg] = d["choices"][0]["text"]
        u = d.get("usage", {})
        ntok[cfg] += u.get("completion_tokens", 0)
    same = row["off"] == row["on"]
    agree += same; disagree += not same
    if not same:
        a, b = row["off"], row["on"]
        pfx = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print(f"p{i} DIVERGES at char {pfx}: off={a[pfx:pfx+28]!r} on={b[pfx:pfx+28]!r}")
    else:
        print(f"p{i} identical ({hashlib.sha256(row['off'].encode()).hexdigest()[:12]})")
for l in open(f"{out}/timings"):
    cfg, _, w = l.split()
    tot[cfg] += int(w.split("=")[1])
print(f"identical {agree}/8")
for cfg in ("off", "on"):
    if ntok[cfg]:
        print(f"{cfg}: {tot[cfg]/1000:.1f}s for {ntok[cfg]} tokens = {ntok[cfg]/(tot[cfg]/1000):.2f} tok/s (pass-2)")
EOF
grep -c "ACCEPT" "$OUT/server-on.log" >> "$OUT/summary" 2>/dev/null || true
grep -c "reject" "$OUT/server-on.log" >> "$OUT/summary" 2>/dev/null || true
cat "$OUT/timings"
chmod -R a+rX "$OUT"
note "window done (caller restores DSV4)"
echo AB_DONE
