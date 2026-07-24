#!/bin/bash
# G3 harness — GLM-5.2 POC via ds4 CUDA SSD streaming on one DGX Spark.
# Run as dsv4: sudo -u dsv4 bash results/glm52-gates/harness/g3_glm_poc.sh
# Engine tree: upstream pin 0a7ad77 + PR#513 LUT fix (commit recorded below).
# Gate: correctness/determinism only; tok/s + read amplification RECORDED.
set -u
OUT=/home/dsv4/ds4-project/glm52-gates-g3
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
MEMWATCH=$REPO/scripts/01_memwatch.sh
FIXS=$REPO/results/glm52-gates/harness/fixture-glm-short.json
FIXL=$REPO/results/glm52-gates/harness/fixture-glm-long.json
PORT=8023
THRESH_GIB=10
CACHE_ARG=40GB
rm -rf "$OUT"; mkdir -p "$OUT"
A="$OUT/assertions.log"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
assert() { local p=FAIL; [[ "$4" == 0 ]] && p=PASS
  echo "ASSERT name=$1 expected=[$2] actual=[$3] result=$p" >> "$A"; }

note "G3 start host=$(hostname)"
note "harness_sha12=$(sha256sum "$0" | cut -c1-12) fixS_sha12=$(sha256sum "$FIXS" | cut -c1-12) fixL_sha12=$(sha256sum "$FIXL" | cut -c1-12)"
note "binary_sha12=$(sha256sum "$SRC/ds4-server" | cut -c1-12)"
(cd "$SRC" && git log --oneline -2) >> "$OUT/run.log" 2>&1
note "gguf_sha_size=$(stat -c%s "$GGUF")"
grep MemAvailable /proc/meminfo >> "$OUT/run.log"

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
  assert port_free "no listener on $PORT" "listener present" 1; echo "G3_ABORT port busy"; exit 1
fi
assert port_free "no listener on $PORT" "no listener" 0

TF="$OUT/memwatch.target"; RF="$OUT/memwatch.ready"
rm -f "$TF" "$RF"
"$MEMWATCH" --target-file "$TF" --ready-file "$RF" --threshold-gib "$THRESH_GIB" \
  --interval-sec 2 --log "$OUT/memwatch.log" &
MWPID=$!

note "starting ds4-server (GLM streaming, cache $CACHE_ARG)"
"$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port "$PORT" \
  --ssd-streaming --ssd-streaming-cache-experts "$CACHE_ARG" \
  > "$OUT/server.log" 2>&1 &
SPID=$!
SPGID=$(ps -o pgid= -p "$SPID" | tr -d ' ')
STICKS=$(awk '{print $22}' "/proc/$SPID/stat")
echo "$SPID $SPGID $STICKS engine" > "$TF"
note "server pid=$SPID pgid=$SPGID ticks=$STICKS; memwatch ARM REQUESTED"

for i in $(seq 1 30); do [[ "$(awk 'NR==1{print $1}' "$RF" 2>/dev/null)" == ARMED ]] && break; sleep 1; done
assert memwatch_armed "ready-file ARMED $SPID" "$(cat "$RF" 2>/dev/null || echo missing)" \
  $([[ "$(awk 'NR==1{print $2}' "$RF" 2>/dev/null)" == "$SPID" ]] && echo 0 || echo 1)

READY=1; T0=$(date +%s)
while (( $(date +%s) - T0 < 1800 )); do
  if ! kill -0 "$SPID" 2>/dev/null; then note "server died during load"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/v1/models" || true)
  [[ "$code" == 200 ]] && { READY=0; break; }
  sleep 3
done
ELAPSED=$(( $(date +%s) - T0 ))
assert server_ready "GET /v1/models -> 200 within 1800s" "code=${code:-none} after ${ELAPSED}s" $READY
if [[ $READY != 0 ]]; then
  tail -60 "$OUT/server.log" >> "$OUT/run.log"
  echo "DISARM $SPID $SPGID $STICKS" > "$TF"; sleep 3
  kill -TERM -- -"$SPGID" 2>/dev/null; sleep 5; kill -KILL -- -"$SPGID" 2>/dev/null
  kill "$MWPID" 2>/dev/null; echo "G3_DONE result=FAIL"; exit 1
fi
curl -s "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"

: > "$OUT/fdinfo.txt"; DIRECT_OK=1
for fd in /proc/$SPID/fd/*; do
  tgt=$(readlink "$fd" 2>/dev/null) || continue
  [[ "$tgt" == "$GGUF" ]] || continue
  n=$(basename "$fd")
  { echo "== fd $n -> $tgt"; cat "/proc/$SPID/fdinfo/$n"; } >> "$OUT/fdinfo.txt"
  flags=$(awk '/^flags:/{print $2}' "/proc/$SPID/fdinfo/$n")
  if (( 8#$flags & 8#0200000 )); then DIRECT_OK=0; echo "== fd $n HAS O_DIRECT (flags octal $flags)" >> "$OUT/fdinfo.txt"; fi
done
assert o_direct_fd "gguf fd with O_DIRECT bit (octal 0200000)" "$(grep -c 'HAS O_DIRECT' "$OUT/fdinfo.txt") of $(grep -cE '^== fd [0-9]+ ->' "$OUT/fdinfo.txt") gguf fds" $DIRECT_OK

run_req() { # fixture out_prefix
  cat "/proc/$SPID/io" > "$OUT/io_${2}_before.txt"
  local t0=$(date +%s)
  local code=$(curl -s -o "$OUT/${2}.json" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d @"$1" "http://127.0.0.1:$PORT/v1/completions")
  local t1=$(date +%s)
  cat "/proc/$SPID/io" > "$OUT/io_${2}_after.txt"
  echo "$code $((t1-t0))"
}
note "request short1"; R=$(run_req "$FIXS" short1); CS1=${R% *}; TS1=${R#* }
note "request short2"; R=$(run_req "$FIXS" short2); CS2=${R% *}; TS2=${R#* }
note "request long1";  R=$(run_req "$FIXL" long1);  CL1=${R% *}; TL1=${R#* }
note "request long2";  R=$(run_req "$FIXL" long2);  CL2=${R% *}; TL2=${R#* }
note "timings s: short1=$TS1 short2=$TS2 long1=$TL1 long2=$TL2"

python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
for n in ("short1","short2","long1","long2"):
    try:
        j = json.load(open(f"{out}/{n}.json"))
        t = j["choices"][0]["text"]
        u = j.get("usage", {})
    except Exception as e:
        t, u = f"<EXTRACT_ERROR {e}>", {}
    open(f"{out}/{n}.text","w").write(t)
    open(f"{out}/{n}.usage","w").write(json.dumps(u))
EOF

for x in S1:$CS1 S2:$CS2 L1:$CL1 L2:$CL2; do
  assert http_200_${x%%:*} "200" "${x##*:}" $([[ "${x##*:}" == 200 ]] && echo 0 || echo 1)
done
cmp -s "$OUT/short1.text" "$OUT/short2.text"; BS=$?
assert short_byte_identical "identical" "cmp=$BS sha12=$(sha256sum "$OUT/short1.text" | cut -c1-12)/$(sha256sum "$OUT/short2.text" | cut -c1-12)" $BS
cmp -s "$OUT/long1.text" "$OUT/long2.text"; BL=$?
assert long_byte_identical "identical" "cmp=$BL sha12=$(sha256sum "$OUT/long1.text" | cut -c1-12)/$(sha256sum "$OUT/long2.text" | cut -c1-12)" $BL
FIRST=$(head -c 1 "$OUT/long1.text")
assert long_first_char_gt "\">\" (upstream glm_long_context_smoke criterion)" "[$FIRST]" $([[ "$FIRST" == ">" ]] && echo 0 || echo 1)

STUBS=$(grep -c 'CUDA stub called' "$OUT/server.log")
assert zero_cuda_stubs "0 in unfiltered server.log" "$STUBS" $([[ "$STUBS" == 0 ]] && echo 0 || echo 1)
grep -inE 'ssd|stream|o_direct|direct' "$OUT/server.log" | head -60 > "$OUT/streaming_lines.log"
assert streaming_engaged "streaming startup lines present" "$(wc -l < "$OUT/streaming_lines.log") lines" $([[ $(wc -l < "$OUT/streaming_lines.log") -gt 0 ]] && echo 0 || echo 1)
KILLS=$(grep -cE 'BREACH|TERMINAT|KILL' "$OUT/memwatch.log" || true)
MINMEM=$(awk -F'mem_available_gib=' '/mem_available_gib/{print $2}' "$OUT/memwatch.log" | sort -n | head -1)
assert memwatch_no_breach "0 breach lines" "breach=$KILLS min_gib=${MINMEM:-na}" $([[ "$KILLS" == 0 ]] && echo 0 || echo 1)

# read amplification recorded (not gated)
for n in short1 short2 long1 long2; do
  b=$(awk '/^read_bytes/{print $2}' "$OUT/io_${n}_before.txt")
  a=$(awk '/^read_bytes/{print $2}' "$OUT/io_${n}_after.txt")
  echo "read_bytes_delta $n: $((a-b))" >> "$A"
done

echo "DISARM $SPID $SPGID $STICKS" > "$TF"; sleep 4
kill -TERM "$SPID" 2>/dev/null
for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
if kill -0 "$SPID" 2>/dev/null; then kill -KILL -- -"$SPGID" 2>/dev/null; assert clean_shutdown "TERM exits <=120s" "needed SIGKILL" 1
else assert clean_shutdown "TERM exits <=120s" "exited" 0; fi
kill "$MWPID" 2>/dev/null; sleep 1

# Quality subset: 3 short official vectors, twice, byte-identical TSVs.
SUBSET="$OUT/manifest-short3.tsv"
head -1 "$SRC/tests/test-vectors/glm-openrouter/manifest.tsv" > "$SUBSET"
grep -E '^short_' "$SRC/tests/test-vectors/glm-openrouter/manifest.tsv" >> "$SUBSET"
note "score_official pass 1"
(cd "$SRC" && ./gguf-tools/quality-testing/score_official "$GGUF" "$SUBSET" "$OUT/quality1.tsv" 8192 \
  --ssd-streaming --ssd-streaming-cache-experts "$CACHE_ARG") > "$OUT/quality1.log" 2>&1
Q1=$?
note "score_official pass 2"
(cd "$SRC" && ./gguf-tools/quality-testing/score_official "$GGUF" "$SUBSET" "$OUT/quality2.tsv" 8192 \
  --ssd-streaming --ssd-streaming-cache-experts "$CACHE_ARG") > "$OUT/quality2.log" 2>&1
Q2=$?
assert quality_exit "0/0" "$Q1/$Q2" $([[ "$Q1$Q2" == 00 ]] && echo 0 || echo 1)
cmp -s "$OUT/quality1.tsv" "$OUT/quality2.tsv"; QI=$?
assert quality_deterministic "quality1.tsv == quality2.tsv" "cmp=$QI" $QI
QNAN=$(grep -ciE 'nan|inf' "$OUT/quality1.tsv" || true)
assert quality_finite "0 nan/inf rows" "$QNAN" $([[ "$QNAN" == 0 ]] && echo 0 || echo 1)

grep MemAvailable /proc/meminfo >> "$OUT/run.log"
note "G3 end"
FAILS=$(grep -c 'result=FAIL' "$A" || true)
chmod -R a+rX "$OUT"
echo "G3_DONE result=$([[ "$FAILS" == 0 ]] && echo PASS || echo "FAIL($FAILS)")"
