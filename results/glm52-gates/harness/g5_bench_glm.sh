#!/bin/bash
# GLM-5.2 benchmark: TTFT (cold/warm-restore), decode+prefill tok/s, and
# FULL 100-fixture official-continuation NLL fidelity run. Run under
# glm_safe_run.sh as dsv4. DSV4 baseline comes from recorded prod logs.
set -u
OUT=/home/dsv4/ds4-project/glm52-bench
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
FIXL8=$REPO/results/glm52-gates/harness/fixture-glm-long8.json
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-bench
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "bench start binary=$(sha256sum "$SRC/ds4-server" | cut -c1-12)"

DS4_GLM_TP_DEBUG=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port 8028 \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 8192 --kv-cache-boundary-align-tokens 64 \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 200); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8028/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { echo BENCH_FAIL; exit 1; }; sleep 2
done

fire() { local t0=$(date +%s%3N)
  curl -s -o "$OUT/$1.json" --max-time 3600 -H 'Content-Type: application/json' -d @"$2" http://127.0.0.1:8028/v1/completions
  echo "$1 wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }

python3 - "$FIXL8" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); j["max_tokens"] = 1
json.dump(j, open(sys.argv[2]+"/fix_ttft.json","w"))
j["max_tokens"] = 128
json.dump(j, open(sys.argv[2]+"/fix_decode.json","w"))
EOF
fire ttft_cold "$OUT/fix_ttft.json"       # 5047-token prefill, 1 token out
fire ttft_warm "$OUT/fix_ttft.json"       # live KV + warm cache
fire decode128 "$OUT/fix_decode.json"     # warm decode throughput
fire decode128b "$OUT/fix_decode.json"    # repeat (KV boundary differences)
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done

note "full-100 fidelity pass begin"
(cd "$SRC" && DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
  ./gguf-tools/quality-testing/score_official "$GGUF" \
  gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv \
  "$OUT/quality100.tsv" 8192 --ssd-streaming --ssd-streaming-cache-experts 40GB) \
  > "$OUT/quality100.log" 2>&1
echo "quality_exit=$?" >> "$OUT/run.log"
cat "$OUT/timings"
python3 - "$OUT/quality100.tsv" <<'EOF'
import sys, statistics
rows = [l.split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
nll = [float(r[4]) for r in rows if len(r) > 4]
print(f"quality100: n={len(nll)} avg_nll mean={statistics.mean(nll):.4f} median={statistics.median(nll):.4f} p90={sorted(nll)[int(len(nll)*0.9)]:.4f} max={max(nll):.4f}")
first = [int(r[5]) for r in rows if len(r) > 5]
print(f"quality100: first_token_match rate={sum(first)/len(first)*100:.1f}%")
EOF
chmod -R a+rX "$OUT"
echo BENCH_DONE
