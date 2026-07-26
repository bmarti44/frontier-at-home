#!/bin/bash
# trim=0 append/BPE-boundary probe + appended-turn TTFT (sol findings).
# Server X: 5047-token base -> checkpoint -> three appended continuations
# whose appends start at merge-prone boundaries (each resumes from the
# base checkpoint; wall time = agent appended-turn TTFT).
# Server Y (fresh, wiped kv): ap1 fired FIRST = true cold canonical-
# tokenization control for the nastiest boundary; ap2/ap3 provide
# cross-history consistency (resume from a different checkpoint lineage).
# PASS = ap1 resumed == ap1 cold, byte-for-byte, and ap2/ap3 X==Y.
set -u
OUT=/home/dsv4/ds4-project/glm52-append-probe
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-append
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for append-probe window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
base = j["prompt"]
json.dump({"model": "default", "prompt": base, "max_tokens": 16,
           "temperature": 0}, open(sys.argv[2] + "/base.json", "w"))
open(sys.argv[2] + "/base_prompt.txt", "w").write(base)
EOF

serve() { # $1 = log suffix
  if [[ -n "${APPEND_GUARD_OFF:-}" ]]; then export DS4_GLM_RESUME_GUARD_OFF=1; fi
  DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=${APPEND_PROBE_CACHE_GB:-72} \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=${APPEND_PROBE_BATCHALL:-1} DS4_GLM_SYNC_TRACE=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --kv-cache-boundary-align-tokens 4 \
    --kv-cache-boundary-trim-tokens 0 \
    > "$OUT/server-$1.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || { note "$1 server died"; return 1; }
    sleep 2
  done; return 1
}
stop_srv() { kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; }
fire() { local t0=$(date +%s%3N)
  local code=$(curl -s -o "$OUT/$1.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$2" http://127.0.0.1:$PORT/v1/completions)
  echo "$1 http=$code wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }
trap 'kill -TERM $SPID 2>/dev/null' EXIT

# ---- Server X: resume path ----
serve X || { echo APPEND_FAIL; exit 1; }
fire x_base "$OUT/base.json"
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
base = open(out + "/base_prompt.txt").read()
gen = json.load(open(out + "/x_base.json"))["choices"][0]["text"]
open(out + "/gen.txt", "w").write(gen)
# Appends chosen to land on merge-prone tokenization boundaries:
# ap1: append starts mid-word gluing onto generated tail (worst case)
# ap2: unicode + NBSP + dash cluster
# ap3: whitespace/newline boundary into code-ish text
appends = ["ological analysis shows",
           " naïve café résumé — attaché",
           "\n\n```python\ndef f(x):"]
import os
mt = int(os.environ.get("APPEND_MAX_TOKENS", "24"))
for i, ap in enumerate(appends, 1):
    json.dump({"model": "default", "prompt": base + gen + ap,
               "max_tokens": mt, "temperature": 0},
              open(f"{out}/ap{i}.json", "w"))
EOF
fire x_ap1 "$OUT/ap1.json"
fire x_ap2 "$OUT/ap2.json"
fire x_ap3 "$OUT/ap3.json"
fire x_ap1r "$OUT/ap1.json"
stop_srv

# ---- Server Y: cold control (fresh kv + fresh process; ap1 first) ----
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
serve Y || { echo APPEND_FAIL; exit 1; }
fire y_ap1 "$OUT/ap1.json"
fire y_ap2 "$OUT/ap2.json"
fire y_ap3 "$OUT/ap3.json"
stop_srv
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
def tx(n): return json.load(open(f"{out}/{n}.json"))["choices"][0]["text"]
def sh(t): return hashlib.sha256(t.encode()).hexdigest()[:12]
ok = True
for i in (1, 2, 3):
    x, y = tx(f"x_ap{i}"), tx(f"y_ap{i}")
    same = x == y
    ok &= same
    lbl = "COLD-CONTROL" if i == 1 else "cross-history"
    print(f"ap{i} ({lbl}): X(resumed)={sh(x)} Y={sh(y)} identical={same}")
    if not same:
        k = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"  diverge@{k}: X={x[k:k+30]!r} Y={y[k:k+30]!r}")
print(f"ap1 repeat stable: {tx('x_ap1') == tx('x_ap1r')}")
print(f"TRIM0_BPE_VERDICT: {'PASS' if ok else 'FAIL'}")
EOF
grep -E "GLM sync start=" "$OUT/server-X.log" | head -6 >> "$OUT/summary"
cat "$OUT/timings"
chmod -R a+rX "$OUT"
note "append window done (caller restores DSV4)"
echo APPEND_DONE
