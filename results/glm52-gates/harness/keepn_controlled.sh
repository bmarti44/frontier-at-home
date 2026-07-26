#!/bin/bash
# FIX-B: expert-skip re-test with a VALID control (audit finding F04).
# Previous run compared keep-6 against a baseline from a DIFFERENT config
# (68GB cache / no SLRU / 8K ctx / no batchall) — invalid. Here every arm
# runs the SAME binary and SAME server config, differing only in the
# DS4_GLM_TOPK_KEEP value, and includes a non-renormalized arm to separate
# "skipping" from "renormalization amplification".
#
# Arms: keep-8 (control), keep-7, keep-6, keep-6-noremorm
# Scoring: identical 8 prompts, greedy, 96 tokens; report per-arm output
# hashes + a coherence signal (invalid-UTF8 bytes, repetition rate) and keep
# raw outputs for the NLL suite.
set -u
OUT=/home/dsv4/ds4-project/glm52-keepn-controlled
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "controlled keep-N start"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 5

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

run_arm() { # $1 arm-tag, $2 keep-N (0 = control/off), $3 norenorm(1/0)
  local envs=()
  [[ "$2" != "0" ]] && envs+=("DS4_GLM_TOPK_KEEP=$2")
  [[ "$3" == "1" ]] && envs+=("DS4_GLM_TOPK_NORENORM=1")
  note "arm $1 (keep=$2 norenorm=$3)"
  env "${envs[@]}" \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=24 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 20GB \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
    kill -0 $SPID 2>/dev/null || { note "$1 died"; return 1; }
    sleep 2
  done
  for i in 0 1 2 3 4 5 6 7; do
    curl -s -o "$OUT/$1-p$i.json" --max-time 3600 -H 'Content-Type: application/json' \
      -d @"$OUT/p$i.json" http://127.0.0.1:$PORT/v1/completions > /dev/null
  done
  kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  return 0
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT

run_arm keep8    0 0     # CONTROL: same binary, same config, no truncation
run_arm keep7    7 0
run_arm keep6    6 0
run_arm keep6nr  6 1     # skipping WITHOUT renormalization
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, os, sys, hashlib, re
out = sys.argv[1]
arms = ["keep8", "keep7", "keep6", "keep6nr"]
def read(arm, i):
    p = f"{out}/{arm}-p{i}.json"
    try:
        raw = open(p, 'rb').read()
        bad_utf8 = 0
        try: raw.decode('utf-8')
        except UnicodeDecodeError: bad_utf8 = 1
        d = json.loads(raw.decode('utf-8', 'replace'))
        return d["choices"][0]["text"], bad_utf8
    except Exception as e:
        return f"<ERR {e}>", 1
def repetition(t):
    w = t.split()
    if len(w) < 8: return 0.0
    return 1.0 - len(set(zip(w, w[1:], w[2:]))) / max(1, len(w) - 2)
print(f"{'arm':10s} {'match_ctl':>10s} {'bad_utf8':>9s} {'rep3gram':>9s}  sample")
ctl = {i: read("keep8", i)[0] for i in range(8)}
for arm in arms:
    same = bad = 0; reps = []
    for i in range(8):
        t, b = read(arm, i)
        same += (t == ctl[i]); bad += b; reps.append(repetition(t))
    s0 = read(arm, 0)[0][:60].replace("\n", "\\n")
    print(f"{arm:10s} {same:>7d}/8 {bad:>9d} {sum(reps)/8:>9.3f}  {s0!r}")
print()
print("keep8 is the same-binary control the previous experiment lacked.")
print("If keep6 collapses but keep6nr does not, the renormalization was the cause.")
EOF
chmod -R a+rX "$OUT"
note "controlled keep-N done"
echo KEEPN_CTL_DONE
