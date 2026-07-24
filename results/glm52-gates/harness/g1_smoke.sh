#!/bin/bash
# G1 harness — upstream ds4 CUDA SSD streaming smoke on DSV4 weights.
# Run as dsv4: sudo -u dsv4 bash results/glm52-gates/harness/g1_smoke.sh
# Raw evidence lands in /home/dsv4/ds4-project/glm52-gates-g1 (copied verbatim
# into results/glm52-gates/logs/g1/ — server.log committed UNFILTERED).
set -u
OUT=/home/dsv4/ds4-project/glm52-gates-g1
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
MEMWATCH=$REPO/scripts/01_memwatch.sh
FIX=$REPO/results/glm52-gates/harness/fixture-prime8.json
PORT=8022
THRESH_GIB=10
rm -rf "$OUT"; mkdir -p "$OUT"
A="$OUT/assertions.log"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
assert() { # name expected actual pass(0/1)
  local p=FAIL; [[ "$4" == 0 ]] && p=PASS
  echo "ASSERT name=$1 expected=[$2] actual=[$3] result=$p" >> "$A"
}

note "G1 start host=$(hostname)"
note "harness_sha12=$(sha256sum "$0" | cut -c1-12) fixture_sha12=$(sha256sum "$FIX" | cut -c1-12)"
echo "binary_sha12: $(sha256sum "$SRC/ds4-server" | cut -c1-12)" >> "$OUT/run.log"
grep MemAvailable /proc/meminfo >> "$OUT/run.log"
free -b >> "$OUT/run.log"

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
  assert port_free "no listener on $PORT" "listener present" 1
  echo "G1_ABORT port busy"; exit 1
fi
assert port_free "no listener on $PORT" "no listener" 0

TF="$OUT/memwatch.target"; RF="$OUT/memwatch.ready"
rm -f "$TF" "$RF"
"$MEMWATCH" --target-file "$TF" --ready-file "$RF" --threshold-gib "$THRESH_GIB" \
  --interval-sec 2 --log "$OUT/memwatch.log" &
MWPID=$!
note "memwatch launched pid=$MWPID threshold=${THRESH_GIB}GiB"

note "starting ds4-server"
"$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port "$PORT" \
  --ssd-streaming --ssd-streaming-cache-experts 16GB \
  > "$OUT/server.log" 2>&1 &
SPID=$!
SPGID=$(ps -o pgid= -p "$SPID" | tr -d ' ')
STICKS=$(awk '{print $22}' "/proc/$SPID/stat")
echo "$SPID $SPGID $STICKS engine" > "$TF"
note "server pid=$SPID pgid=$SPGID ticks=$STICKS; memwatch ARM REQUESTED (target file written; actual ARM is memwatch.log's ARMED line)"

for i in $(seq 1 30); do [[ "$(awk 'NR==1{print $1}' "$RF" 2>/dev/null)" == ARMED ]] && break; sleep 1; done
if [[ "$(awk 'NR==1{print $1}' "$RF" 2>/dev/null)" == ARMED ]]; then
  assert memwatch_armed "ready-file ARMED $SPID" "$(cat "$RF")" $([[ "$(awk 'NR==1{print $2}' "$RF")" == "$SPID" ]] && echo 0 || echo 1)
else assert memwatch_armed "ready-file reaches ARMED" "$(cat "$RF" 2>/dev/null || echo missing) after 30s" 1; fi

READY=1; T0=$(date +%s)
while (( $(date +%s) - T0 < 1200 )); do
  if ! kill -0 "$SPID" 2>/dev/null; then note "server died during load"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/v1/models" || true)
  if [[ "$code" == 200 ]]; then READY=0; break; fi
  sleep 2
done
ELAPSED=$(( $(date +%s) - T0 ))
assert server_ready "GET /v1/models -> 200 within 1200s" "code=${code:-none} after ${ELAPSED}s (wall-clock)" $READY
if [[ $READY != 0 ]]; then
  note "G1_ABORT server never ready"; tail -40 "$OUT/server.log" >> "$OUT/run.log"
  echo "DISARM $SPID $SPGID $STICKS" > "$TF"; sleep 3
  kill -TERM -- -"$SPGID" 2>/dev/null; sleep 5; kill -KILL -- -"$SPGID" 2>/dev/null
  kill "$MWPID" 2>/dev/null
  echo "G1_DONE result=FAIL"; exit 1
fi
note "server ready ${ELAPSED}s after readiness polling began"

# O_DIRECT proof: every open fd on the GGUF, with fdinfo flags.
# aarch64 O_DIRECT = 0x10000 (octal 0200000).
: > "$OUT/fdinfo.txt"
DIRECT_OK=1
for fd in /proc/$SPID/fd/*; do
  tgt=$(readlink "$fd" 2>/dev/null) || continue
  [[ "$tgt" == "$GGUF" ]] || continue
  n=$(basename "$fd")
  { echo "== fd $n -> $tgt"; cat "/proc/$SPID/fdinfo/$n"; } >> "$OUT/fdinfo.txt"
  flags=$(awk '/^flags:/{print $2}' "/proc/$SPID/fdinfo/$n")
  if (( 8#$flags & 8#0200000 )); then DIRECT_OK=0; echo "== fd $n HAS O_DIRECT (flags octal $flags)" >> "$OUT/fdinfo.txt"; fi
done
assert o_direct_fd "at least one open fd on the GGUF has O_DIRECT (flags octal bit 0200000, aarch64)" "$(grep -c 'HAS O_DIRECT' "$OUT/fdinfo.txt") of $(grep -cE '^== fd [0-9]+ ->' "$OUT/fdinfo.txt") gguf fds; see fdinfo.txt" $DIRECT_OK

cat "/proc/$SPID/io" > "$OUT/io_before.txt"
note "request 1 begin (fixture $FIX)"
C1=$(curl -s -o "$OUT/run1.json" -w '%{http_code}' --max-time 600 -H 'Content-Type: application/json' -d @"$FIX" "http://127.0.0.1:$PORT/v1/completions")
note "request 1 end code=$C1"
cat "/proc/$SPID/io" > "$OUT/io_mid.txt"
note "request 2 begin (same fixture)"
C2=$(curl -s -o "$OUT/run2.json" -w '%{http_code}' --max-time 600 -H 'Content-Type: application/json' -d @"$FIX" "http://127.0.0.1:$PORT/v1/completions")
note "request 2 end code=$C2"
cat "/proc/$SPID/io" > "$OUT/io_after.txt"

python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
for n in ("run1","run2"):
    try:
        j = json.load(open(f"{out}/{n}.json"))
        t = j["choices"][0]["text"]
    except Exception as e:
        t = f"<EXTRACT_ERROR {e}>"
    open(f"{out}/{n}.text","w").write(t)
EOF

assert http_200_run1 "200" "$C1" $([[ "$C1" == 200 ]] && echo 0 || echo 1)
assert http_200_run2 "200" "$C2" $([[ "$C2" == 200 ]] && echo 0 || echo 1)
if cmp -s "$OUT/run1.text" "$OUT/run2.text"; then BI=0; else BI=1; fi
assert byte_identical "run1.text == run2.text" "cmp exit $BI; sha12 run1=$(sha256sum "$OUT/run1.text" | cut -c1-12) run2=$(sha256sum "$OUT/run2.text" | cut -c1-12)" $BI

RB0=$(awk '/^read_bytes/{print $2}' "$OUT/io_before.txt")
RB1=$(awk '/^read_bytes/{print $2}' "$OUT/io_mid.txt")
RB2=$(awk '/^read_bytes/{print $2}' "$OUT/io_after.txt")
D1=$((RB1-RB0)); D2=$((RB2-RB1))
assert read_bytes_growth_run1 ">= 2147483648 (2 GiB read from storage during request 1)" "$D1" $([[ $D1 -ge 2147483648 ]] && echo 0 || echo 1)
echo "read_bytes request 2 delta: $D2 (recorded, not gated)" >> "$A"

STUBS=$(grep -c 'CUDA stub called' "$OUT/server.log")
assert zero_cuda_stubs "0 in UNFILTERED server.log" "$STUBS" $([[ "$STUBS" == 0 ]] && echo 0 || echo 1)

grep -inE 'ssd|stream|o_direct|direct' "$OUT/server.log" | head -60 > "$OUT/streaming_lines.log"
SL=$(wc -l < "$OUT/streaming_lines.log")
assert streaming_engaged "server.log has streaming-mode startup lines" "$SL matching lines (see streaming_lines.log)" $([[ "$SL" -gt 0 ]] && echo 0 || echo 1)

KILLS=$(grep -cE 'BREACH|TERMINAT|KILL' "$OUT/memwatch.log" || true)
MINMEM=$(awk -F'mem_available_gib=' '/mem_available_gib/{print $2}' "$OUT/memwatch.log" | sort -n | head -1)
assert memwatch_no_breach "0 breach lines; min sampled MemAvailable > ${THRESH_GIB} GiB" "breach_lines=$KILLS min_gib=${MINMEM:-na}" $([[ "$KILLS" == 0 ]] && echo 0 || echo 1)

echo "DISARM $SPID $SPGID $STICKS" > "$TF"
sleep 4
kill -TERM "$SPID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
if kill -0 "$SPID" 2>/dev/null; then kill -KILL -- -"$SPGID" 2>/dev/null; assert clean_shutdown "TERM exits <=60s" "needed SIGKILL" 1
else assert clean_shutdown "TERM exits <=60s" "exited" 0; fi
kill "$MWPID" 2>/dev/null
sleep 1
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
  assert port_released "no listener after shutdown" "still listening" 1
else assert port_released "no listener after shutdown" "released" 0; fi

grep MemAvailable /proc/meminfo >> "$OUT/run.log"
note "G1 end"
FAILS=$(grep -c 'result=FAIL' "$A" || true)
chmod -R a+rX "$OUT"
echo "G1_DONE result=$([[ "$FAILS" == 0 ]] && echo PASS || echo "FAIL($FAILS)")"
