#!/bin/bash
# REAP50 resident-variant benchmark window: stop DSV4, serve REAP via
# llama.cpp (mmap: file-backed pages are reclaimable -> desktop stays
# responsive), measure TTFT/prefill/decode + coherence, restore DSV4.
set -u
OUT=/home/dsv4/ds4-project/glm52-reap-bench
BIN=/home/dsv4/llamacpp-project/src/llama.cpp/build/bin/llama-server
M=/home/dsv4/ds4-project/gguf-reap/GLM-5.2-REAP50-Q2_K-00001-of-00004.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for bench window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8
"$BIN" -m "$M" -c 8192 -ngl 999 -b 2048 -ub 512 --host 127.0.0.1 --port 8030 \
  --no-warmup > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 600); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8030/health)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "server died"; tail -8 "$OUT/server.log" >> "$OUT/run.log"; echo REAP_FAIL; exit 1; }
  sleep 3
done
note "ready after startup"
fire() { local t0=$(date +%s%3N)
  curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' -d @"$2" http://127.0.0.1:8030/v1/completions
  echo "$1 wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }
python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
j["max_tokens"] = 1;   json.dump(j, open(sys.argv[2]+"/f_ttft.json","w"))
j["max_tokens"] = 128; json.dump(j, open(sys.argv[2]+"/f_dec.json","w"))
EOF
cp "$REPO/results/glm52-gates/harness/fixture-glm-short.json" "$OUT/f_short.json"
fire short1 "$OUT/f_short.json"
fire short2 "$OUT/f_short.json"
fire ttft_cold "$OUT/f_ttft.json"
fire ttft_warm "$OUT/f_ttft.json"
fire dec128 "$OUT/f_dec.json"
grep -E "prompt eval time|eval time" "$OUT/server.log" | tail -6 >> "$OUT/timings"
for r in short1 short2 dec128; do
  python3 -c "import json,sys;t=json.load(open('$OUT/$r.json'))['choices'][0]['text'];sys.stdout.write(f'$r sha={__import__(\"hashlib\").sha256(t.encode()).hexdigest()[:12]} len={len(t)}\n')" >> "$OUT/timings" 2>/dev/null
done
head -c 300 <(python3 -c "import json;print(json.load(open('$OUT/dec128.json'))['choices'][0]['text'])") > "$OUT/dec128.text" 2>/dev/null
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
cat "$OUT/timings"
note "restoring DSV4"
chmod -R a+rX "$OUT"
echo REAP_BENCH_DONE
