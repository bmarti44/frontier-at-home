#!/bin/bash
# Cold-TTFT lever: prefill chunk size.
#
# WHERE THIS CAME FROM. The flush-fix A/B showed prefill is not cache-limited
# in a way flushing explains: one prefill chunk touches ~253 DISTINCT experts
# per routed layer (~2.35 GiB/layer, ~176 GB across 75 layers) no matter how
# many tokens are in the chunk. ds4_prefill_cap_for_prompt() caps the chunk at
# 4096 tokens for prompts longer than that, so a 5047-token prompt is split
# into two chunks and pays that ~176 GB expert sweep TWICE. Raising the cap so
# the prompt fits in ONE chunk should remove an entire sweep.
#
# Arms (same binary, ABBA, two passes), all on the 5047-token fixture:
#   c4096   default (server logs prefill_chunk=4096) -> two chunks
#   c8192   --prefill-chunk 8192 -> should be a single chunk
#
# v2 CORRECTION. v1 used DS4_METAL_PREFILL_CHUNK=8192 and BOTH arms ran with
# prefill_chunk=4096: ds4_prefill_cap_for_prompt() only consults that env var
# when the configured chunk is 0, and ds4-server always sets one. The two arms
# were the same configuration, exactly like the earlier flush "A/B". The
# harness now (a) uses the real CLI flag and (b) ASSERTS from the server's own
# "context buffers ... prefill_chunk=N" line that the arms differ, refusing to
# report if they do not.
#
# Measured: cold TTFT (the metric this targets), plus the per-layer selected
# load lines so the number of prefill sweeps is COUNTED, not assumed, and the
# output sha so a faster arm that changed the answer is caught.
set -u
OUT=/home/dsv4/ds4-project/glm52-prefill-chunk
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "prefill-chunk A/B start binary_sha12=$(sha256sum $SRC/ds4-server | cut -c1-12)"
unset DS4_METAL_PREFILL_CHUNK DS4_GLM_TOPK_KEEP DS4_GLM_TOPK_SKIP_LOAD 2>/dev/null || true

wait_gone() {
  for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || return 0; sleep 2; done
  pkill -KILL -x ds4-server; sleep 5
}
pkill -TERM -x ds4-server 2>/dev/null; wait_gone

python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
json.dump({"model": "default", "prompt": j["prompt"], "max_tokens": 1,
           "temperature": 0}, open(sys.argv[2] + "/long1.json", "w"))
PYEOF

run_arm() { # $1 tag, $2 pass, $3 chunk (0 = server default)
  local key="$1-p$2"
  local chunk_arg=()
  [[ "$3" != "0" ]] && chunk_arg=(--prefill-chunk "$3")
  note "arm $key chunk=$3"
  wait_gone
  env DS4_GLM_TP_DEBUG=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    "${chunk_arg[@]}" \
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
  # PRECONDITION: the server prints the chunk it actually used. Record it, and
  # the summary refuses to report a comparison unless the arms differ.
  local used=$(grep -o 'prefill_chunk=[0-9]*' "$OUT/server-$key.log" | head -1 | cut -d= -f2)
  echo "$key used_chunk=${used:-unknown}" >> "$OUT/timings"
  note "arm $key server reports prefill_chunk=${used:-unknown}"
  local t0=$(date +%s%3N)
  local code=$(curl -s -o "$OUT/$key-cold.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$OUT/long1.json" \
    http://127.0.0.1:$PORT/v1/completions)
  echo "$key cold ms=$(( $(date +%s%3N) - t0 )) http=$code" >> "$OUT/timings"
  # count prefill sweeps: a "sweep" is one pass over all routed layers with a
  # multi-token batch. layer=3 is the first routed layer, so its batch-load
  # count IS the number of chunks.
  echo "$key sweeps=$(grep -c 'selected-expert batch layer=3 slots=[0-9]\{3,\}' "$OUT/server-$key.log")" >> "$OUT/timings"
  echo "$key prefill_mib=$(grep 'selected-expert batch' "$OUT/server-$key.log" | grep -v 'slots=8 ' | sed 's/.*load=\([0-9.]*\) MiB.*/\1/' | paste -sd+ | bc 2>/dev/null || echo 0)" >> "$OUT/timings"
  kill -TERM $SPID; wait_gone
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT
run_arm c4096 1 0
run_arm c8192 1 8192
run_arm c8192 2 8192
run_arm c4096 2 0
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, sys, hashlib, statistics
out = sys.argv[1]
ms, sweeps, mib, used = {}, {}, {}, {}
for line in open(os.path.join(out, "timings")):
    p = line.split()
    if len(p) >= 3 and p[2].startswith("ms="): ms[p[0]] = int(p[2].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("sweeps="): sweeps[p[0]] = int(p[1].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("used_chunk="):
        used[p[0]] = p[1].split("=")[1]
    elif len(p) >= 2 and p[1].startswith("prefill_mib="):
        try: mib[p[0]] = float(p[1].split("=")[1])
        except ValueError: mib[p[0]] = 0.0
def txt(key):
    try:
        d = json.loads(open(os.path.join(out, key + "-cold.json"), 'rb').read().decode('utf-8','replace'))
        return d["choices"][0]["text"]
    except Exception: return ""
u = set(used.values())
print("server-reported prefill_chunk per arm: %s" % used)
if len(u) < 2:
    print()
    print("*** NO COMPARISON: every arm ran with the same prefill_chunk %s." % (u or "?"))
    print("*** The independent variable did not change; nothing is reported.")
    raise SystemExit(0)
print("%-10s %10s %8s %14s  %s" % ("arm", "cold s", "sweeps", "prefill GiB", "sha"))
agg = {}
for tag in ("c4096", "c8192"):
    for ps in ("1", "2"):
        k = "%s-p%s" % (tag, ps)
        if k not in ms: continue
        t = txt(k)
        print("%-10s %10.1f %8s %14.1f  %s" % (
            k, ms[k]/1000.0, sweeps.get(k, "?"), mib.get(k, 0)/1024.0,
            hashlib.sha256(t.encode()).hexdigest()[:12]))
        agg.setdefault(tag, []).append(ms[k]/1000.0)
print()
if "c4096" in agg and "c8192" in agg:
    a, b = statistics.mean(agg["c4096"]), statistics.mean(agg["c8192"])
    print("cold TTFT: default %.1f s -> single-chunk %.1f s = %+.1f%%" % (a, b, 100*(b/a-1)))
    print("(negative is faster; check the sweep count actually dropped 2 -> 1,")
    print(" and that the output sha is unchanged, before believing the number)")
PYEOF
chmod -R a+rX "$OUT"
note "prefill-chunk A/B done"
echo PREFILL_CHUNK_DONE
