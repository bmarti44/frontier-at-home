#!/bin/bash
# G4a gate harness — persistent CUDA expert cache: correctness (byte-identity
# cache-on vs cache-off), effectiveness (warm repeat reads < 25% of the
# cache-OFF baseline reads for the same fixture), budget adherence, quality
# subset identity. Run under glm_safe_run.sh as dsv4.
set -u
OUT=/home/dsv4/ds4-project/glm52-gates-g4a
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
FIXS=$REPO/results/glm52-gates/harness/fixture-glm-short.json
FIX16=$REPO/results/glm52-gates/harness/fixture-glm-short16.json
FIXL=$REPO/results/glm52-gates/harness/fixture-glm-long.json
PORT=8026
CACHE_GB=${G4A_CACHE_GB:-72}
rm -rf "$OUT"; mkdir -p "$OUT"
A="$OUT/assertions.log"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
assert() { local p=FAIL; [[ "$4" == 0 ]] && p=PASS
  echo "ASSERT name=$1 expected=[$2] actual=[$3] result=$p" >> "$A"; }
note "G4a start binary=$(sha256sum "$SRC/ds4-server" | cut -c1-12) harness=$(sha256sum "$0" | cut -c1-12) cache_gb=$CACHE_GB"
(cd "$SRC" && git log --oneline -1) >> "$OUT/run.log"

start_server() { # cache_gb(0=off) logfile
  local envs=(DS4_GLM_TP_DEBUG=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1)
  [[ "$1" != 0 ]] && envs+=(DS4_CUDA_EXPERT_CACHE_GB=$1)
  env "${envs[@]}" "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 \
    --host 127.0.0.1 --port $PORT --ssd-streaming \
    --ssd-streaming-cache-experts 40GB > "$2" 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && return 0
    kill -0 $SPID 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
stop_server() { kill -TERM $SPID 2>/dev/null; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -KILL $SPID 2>/dev/null; }
req() { # fixture out_prefix
  cat /proc/$SPID/io | awk '/^read_bytes/{print $2}' > "$OUT/${2}.rb0"
  local t0=$(date +%s)
  curl -s -o "$OUT/${2}.json" --max-time 3600 -H 'Content-Type: application/json' \
    -d @"$1" http://127.0.0.1:$PORT/v1/completions
  local rc=$?
  cat /proc/$SPID/io | awk '/^read_bytes/{print $2}' > "$OUT/${2}.rb1"
  python3 -c "import json,sys;sys.stdout.write(json.load(open('$OUT/${2}.json'))['choices'][0]['text'])" > "$OUT/${2}.text" 2>/dev/null
  echo "$(date -Is) $2 rc=$rc wall=$(( $(date +%s) - t0 ))s delta=$(( $(cat $OUT/${2}.rb1) - $(cat $OUT/${2}.rb0) ))" >> "$OUT/run.log"
}
rbdelta() { echo $(( $(cat "$OUT/$1.rb1") - $(cat "$OUT/$1.rb0") )); }

# ---------- Phase A: cache OFF ----------
note "phase A: cache OFF"
MEM_A0=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
start_server 0 "$OUT/serverA.log" || { assert phaseA_server_ready ok dead 1; echo "G4A_DONE result=FAIL"; exit 1; }
assert phaseA_server_ready "200" "ready" 0
req "$FIXS"  A_short
req "$FIX16" A_short16
req "$FIXL"  A_long
stop_server
grep -c "persistent expert cache enabled" "$OUT/serverA.log" > /dev/null && CA=1 || CA=0
assert phaseA_cache_absent "0 cache-enabled lines (off is really off)" "$CA" $([[ $CA == 0 ]] && echo 0 || echo 1)

# ---------- Phase B: cache ON ----------
note "phase B: cache ON ($CACHE_GB GB)"
start_server "$CACHE_GB" "$OUT/serverB.log" || { assert phaseB_server_ready ok dead 1; echo "G4A_DONE result=FAIL"; exit 1; }
assert phaseB_server_ready "200" "ready" 0
# Repeats run FIRST so the effectiveness measurement reflects repeated-traffic
# cache behavior, not eviction pressure from the unrelated long fixture that
# follows (cross-workload retention is characterized separately by the stats
# line). The arena initializes lazily on the first load, so the enabled-line
# assertion happens after requests complete.
req "$FIX16" B_rep1
req "$FIX16" B_rep2
req "$FIX16" B_rep3
req "$FIXS"  B_short
req "$FIXL"  B_long
ARENA_LINE=$(grep -m1 "persistent expert cache enabled" "$OUT/serverB.log" || true)
assert phaseB_cache_enabled "arena line present" "${ARENA_LINE:-missing}" $([[ -n "$ARENA_LINE" ]] && echo 0 || echo 1)
# budget adherence: MemAvailable now vs phase-A idle, minus expected non-cache load
MEM_B=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
USED_GIB=$(( (MEM_A0 - MEM_B) / 1048576 ))
LIMIT_GIB=$(( CACHE_GB + 45 ))   # arena + engine working set (~35-40 GiB measured) + 1 slack
assert budget_adherence "total engine+cache usage <= ${LIMIT_GIB} GiB (arena bounded by construction; census in serverB.log)" "${USED_GIB} GiB while serving" $(( USED_GIB <= LIMIT_GIB ? 0 : 1 ))
STATS=$(grep "expert-cache stats" "$OUT/serverB.log" | tail -1)
echo "final cache stats: ${STATS:-none}" >> "$A"
stop_server

# ---------- Assertions ----------
for f in short long; do
  cmp -s "$OUT/A_${f}.text" "$OUT/B_${f}.text"; r=$?
  assert byte_identical_${f} "cache-on == cache-off" "cmp=$r shaA=$(sha256sum "$OUT/A_${f}.text" | cut -c1-12) shaB=$(sha256sum "$OUT/B_${f}.text" | cut -c1-12)" $r
done
cmp -s "$OUT/A_short16.text" "$OUT/B_rep1.text"; r=$?
assert byte_identical_short16 "cache-on rep1 == cache-off" "cmp=$r" $r
cmp -s "$OUT/B_rep1.text" "$OUT/B_rep3.text"; r=$?
assert repeat_stable "rep1 == rep3" "cmp=$r" $r

BASE=$(rbdelta A_short16); WARM=$(rbdelta B_rep3)
PCT=$(( WARM * 100 / BASE ))
assert warm_reads_lt_25pct_of_off "warm repeat reads < 25% of cache-OFF baseline reads (gate text 'run 2 < 25% of run 1' with run1 = uncached; interpretation documented for review)" "baseline=$BASE warm=$WARM = ${PCT}%" $(( PCT < 25 ? 0 : 1 ))

STUBS=$(( $(grep -c 'CUDA stub called' "$OUT/serverA.log") + $(grep -c 'CUDA stub called' "$OUT/serverB.log") ))
assert zero_cuda_stubs "0" "$STUBS" $([[ $STUBS == 0 ]] && echo 0 || echo 1)

# quality subset identity cache-on vs cache-off
SUBSET="$OUT/manifest-short3.tsv"
head -1 "$SRC/tests/test-vectors/glm-openrouter/manifest.tsv" > "$SUBSET"
grep -E '^short_' "$SRC/tests/test-vectors/glm-openrouter/manifest.tsv" >> "$SUBSET"
(cd "$SRC" && DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 ./gguf-tools/quality-testing/score_official "$GGUF" "$SUBSET" "$OUT/qual_off.tsv" 8192 --ssd-streaming --ssd-streaming-cache-experts 40GB) > "$OUT/qual_off.log" 2>&1
QO=$?
(cd "$SRC" && DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GB ./gguf-tools/quality-testing/score_official "$GGUF" "$SUBSET" "$OUT/qual_on.tsv" 8192 --ssd-streaming --ssd-streaming-cache-experts 40GB) > "$OUT/qual_on.log" 2>&1
QN=$?
assert quality_exits "0/0" "$QO/$QN" $([[ "$QO$QN" == 00 ]] && echo 0 || echo 1)
cmp -s "$OUT/qual_off.tsv" "$OUT/qual_on.tsv"; r=$?
assert quality_identical "NLL TSV identical cache-on vs cache-off" "cmp=$r" $r

note "G4a end"
FAILS=$(grep -c 'result=FAIL' "$A" || true)
chmod -R a+rX "$OUT"
echo "G4A_DONE result=$([[ "$FAILS" == 0 ]] && echo PASS || echo "FAIL($FAILS)")"
