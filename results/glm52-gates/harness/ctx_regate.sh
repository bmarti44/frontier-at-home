#!/bin/bash
# Re-gate per sol night audit: (A) functional 32K window — deterministic
# dependency on material BEYOND row 8192, cold + repeat, raw evidence;
# (B) warm-TTFT on the EXACT adopted glm52 profile (ctx 32768, arena 72,
# kv-disk 16GB, align 4 trim 0) — cold/warm1/warm2 with HTTP codes;
# (C) appended-turn TTFT + BPE-merge-boundary safety for trim 0: resumed
# continuation must equal cold full-prompt continuation byte-for-byte.
set -u
OUT=/home/dsv4/ds4-project/glm52-ctx-regate
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-regate
PORT=8016
rm -rf "$OUT" "$KVDIR"; mkdir -p "$OUT" "$KVDIR"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "stopping DSV4 for regate window"
pkill -TERM -f "llama-server.*8011" 2>/dev/null; sleep 8

python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
# Fixture A: secret placed ~10K tokens in (beyond row 8192), question at the
# end. Filler is varied deterministic prose (~4 chars/token -> ~12K tokens).
filler_unit = ("The %d%s survey of the northern watershed recorded %d distinct "
               "specimens along transect %d, of which %d were previously "
               "undocumented in the regional archive. ")
def filler(n_units, seed):
    return "".join(filler_unit % (i+seed, "th", (i*7+seed) % 90, i % 40,
                                  (i*3+seed) % 12) for i in range(n_units))
secret = "KESTREL-4491"
prompt_a = (filler(300, 1)
            + f"\nIMPORTANT RECORD: the expedition passphrase is {secret}. Memorize it.\n"
            + filler(60, 7)
            + "\nQuestion: what is the expedition passphrase stated in the "
              "IMPORTANT RECORD above? Answer with only the passphrase.\nAnswer:")
json.dump({"model": "default", "prompt": prompt_a, "max_tokens": 12,
           "temperature": 0}, open(out + "/fa.json", "w"))
open(out + "/secret.txt", "w").write(secret)
# Fixture C base: a prompt whose append lands on a merge-prone boundary.
base_c = ("Complete this sentence naturally. The scientific study of word "
          "formation is called morph")
json.dump({"model": "default", "prompt": base_c, "max_tokens": 24,
           "temperature": 0}, open(out + "/fc_base.json", "w"))
json.dump({"model": "default", "prompt": base_c + "ology, and the study of "
           "sentence structure is called synt", "max_tokens": 24,
           "temperature": 0}, open(out + "/fc_full.json", "w"))
EOF
python3 - "$REPO/results/glm52-gates/harness/fixture-glm-long8.json" "$OUT" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); j["max_tokens"] = 1
json.dump(j, open(sys.argv[2] + "/fb.json", "w"))
EOF

# EXACT adopted glm52 profile (mirror scripts/52_engine_switch.sh)
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
  --kv-cache-boundary-align-tokens 4 \
  --kv-cache-boundary-trim-tokens 0 \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "server died"; tail -5 "$OUT/server.log" >> "$OUT/run.log"; echo REGATE_FAIL; exit 1; }
  sleep 2
done
note "ready (adopted profile, ctx 32768)"

fire() { local t0=$(date +%s%3N)
  local code=$(curl -s -o "$OUT/$1.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$2" http://127.0.0.1:$PORT/v1/completions)
  echo "$1 http=$code wall_ms=$(( $(date +%s%3N) - t0 ))" >> "$OUT/timings"; }

fire a_cold  "$OUT/fa.json"       # >8192-row dependency, cold
fire a_rep   "$OUT/fa.json"       # repeat (determinism + warm at depth)
fire b_cold  "$OUT/fb.json"       # 5047-tok TTFT on adopted profile
fire b_warm1 "$OUT/fb.json"
fire b_warm2 "$OUT/fb.json"
fire c_base  "$OUT/fc_base.json"  # primes checkpoint ending mid-word "morph"
fire c_full  "$OUT/fc_full.json"  # append crosses BPE merge boundary; resumed
fire c_full2 "$OUT/fc_full.json"  # repeat of full (steady)

kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

# Control: fresh server + wiped disk-KV -> c_full with canonical whole-prompt
# tokenization (no resumed checkpoint). This is the byte-identity reference
# for the trim=0 BPE-boundary test.
note "control phase: fresh server for c_full cold reference"
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 DS4_GLM_SYNC_TRACE=1 \
  "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
  --kv-cache-boundary-align-tokens 4 \
  --kv-cache-boundary-trim-tokens 0 \
  > "$OUT/server-control.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
  kill -0 $SPID 2>/dev/null || { note "control server died"; echo REGATE_FAIL; exit 1; }
  sleep 2
done
fire c_control "$OUT/fc_full.json"
kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
trap - EXIT

python3 - "$OUT" <<'EOF' | tee "$OUT/summary"
import json, sys, hashlib
out = sys.argv[1]
sec = open(out + "/secret.txt").read().strip()
def tx(n):
    try: return json.load(open(f"{out}/{n}.json"))["choices"][0]["text"]
    except Exception as e: return f"<ERR {e}>"
def sh(t): return hashlib.sha256(t.encode()).hexdigest()[:12]
a1, a2 = tx("a_cold"), tx("a_rep")
print(f"A cold: sha={sh(a1)} contains_secret={sec in a1} text={a1[:60]!r}")
print(f"A rep : sha={sh(a2)} contains_secret={sec in a2} identical={a1==a2}")
print(f"A VERDICT: {'PASS' if (sec in a1 and a1==a2) else 'FAIL'} (deterministic dependency beyond row 8192)")
b1 = tx("b_cold")
print(f"B cold sha={sh(b1)}  warm1 sha={sh(tx('b_warm1'))}  warm2 sha={sh(tx('b_warm2'))}  all_eq={b1==tx('b_warm1')==tx('b_warm2')}")
cf, cf2 = tx("c_full"), tx("c_full2")
print(f"C full(resumed-after-base) sha={sh(cf)} text={cf[:48]!r}")
print(f"C full2(repeat)            sha={sh(cf2)} identical={cf==cf2}")
print("C NOTE: cold reference for c_full comes from the paired fresh-server control run")
EOF
grep -E "GLM sync" "$OUT/server.log" | tail -10 >> "$OUT/summary"
cat "$OUT/timings"
chmod -R a+rX "$OUT"
note "regate window done (caller restores DSV4; run ctx_regate_control.sh for C cold reference)"
echo REGATE_DONE
