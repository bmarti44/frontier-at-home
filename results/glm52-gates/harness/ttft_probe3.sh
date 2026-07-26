#!/bin/bash
# Warm-TTFT root-cause probe: same 5047-token fixture fired 3x at one
# server (cold, warm, warm). DS4_GLM_SYNC_TRACE shows where each request
# resumes (start/prompt/suffix/checkpoint/dense_len); LOADPROF line counts
# show how many expert-loads each request performs. llama.cpp does the
# warm case in 1.56 s via in-RAM prefix reuse — this measures what ds4
# recomputes instead.
set -u
ALIGN=${TTFT3_ALIGN:-64}
TRIM=${TTFT3_TRIM:-32}
OUT=/home/dsv4/ds4-project/glm52-ttft3-a${ALIGN}t${TRIM}b${TTFT3_BATCHALL:-0}
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
FIX=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/fixture-glm-long8.json
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-ttft3-a${ALIGN}
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for TTFT window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$FIX" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); j["max_tokens"] = 1
json.dump(j, open(sys.argv[2] + "/fix1.json", "w"))
EOF

BA_ENV=()
[[ -n "${TTFT3_BATCHALL:-}" ]] && BA_ENV+=(DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1)
env DS4_GLM_SYNC_TRACE=1 DS4_CUDA_LOAD_PROFILE=1 \
  "${BA_ENV[@]}" \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=68 \
  DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 8192 \
  --kv-cache-boundary-align-tokens $ALIGN \
  --kv-cache-boundary-trim-tokens $TRIM \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "server died"; echo TTFT3_FAIL; exit 1; }
  sleep 2
done
note "ready"

fire() { local t0=$(date +%s%3N)
  curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$OUT/fix1.json" http://127.0.0.1:$PORT/v1/completions
  local t1=$(date +%s%3N)
  echo "$1 wall_ms=$((t1-t0)) loadprof_lines=$(grep -c LOADPROF "$OUT/server.log")" >> "$OUT/timings"; }

fire cold
fire warm1
fire warm2
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

grep -E "GLM sync|checkpoint" "$OUT/server.log" | grep -v "branch=" | tail -12 >> "$OUT/timings"
python3 - "$OUT" <<'PYC' >> "$OUT/timings"
import json, sys, hashlib
out = sys.argv[1]
for n in ("cold", "warm1", "warm2"):
    try:
        raw = open("%s/%s.json" % (out, n), "rb").read()
        bad = 0
        try: raw.decode("utf-8")
        except UnicodeDecodeError: bad = 1
        d = json.loads(raw.decode("utf-8", "replace"))
        t = d["choices"][0]["text"]
        print("%s sha=%s bad_utf8=%d text=%r" % (
            n, hashlib.sha256(t.encode()).hexdigest()[:12], bad, t[:50]))
    except Exception as e:
        print("%s ERR %s" % (n, e))
PYC
cat "$OUT/timings"
chmod -R a+rX "$OUT"
note "window done (caller restores DSV4)"
echo TTFT3_DONE
