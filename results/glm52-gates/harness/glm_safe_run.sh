#!/bin/bash
# glm_safe_run.sh — hardened wrapper for ANY GLM/ds4 engine invocation on the
# Spark, after two whole-box freezes caused by unbounded unified-memory
# allocation in an engine test. Run as dsv4:
#   sudo -u dsv4 bash glm_safe_run.sh [--tag NAME] -- <command...>
#
# Protections (layered, all mandatory):
#   1. ulimit -v hard cap (default 95 GiB): runaway mmap/managed allocations
#      fail with ENOMEM inside the process instead of freezing the kernel.
#   2. 1-second sidecar sampler: MemAvailable + engine VmRSS + read_bytes,
#      appended to a PERSISTENT log and fdatasync'd every sample, so the
#      final seconds survive a hard freeze/power cycle.
#   3. Kill floor (default 18 GiB MemAvailable): sampler SIGKILLs the whole
#      process group the moment available memory crosses the floor.
#   4. Wall-clock timeout (default 2400 s) via timeout(1).
#   5. Start/exit records with command, env, tree commit, binary sha.
# Logs: /home/dsv4/ds4-project/glm52-crashlog/<ts>-<tag>/
set -u
VLIMIT_KB=${GLM_SAFE_VLIMIT_KB:-419430400}  # 400 GiB backstop: engine mmaps the whole GGUF (196.6 GiB VIRTUAL, file-backed, mostly non-resident), so RLIMIT_AS must clear that. Resident-growth protection is the kill-floor sampler below.
KILL_FLOOR_GIB=${GLM_SAFE_KILL_FLOOR_GIB:-18}
MIN_START_GIB=${GLM_SAFE_MIN_START_GIB:-110}
TIMEOUT_S=${GLM_SAFE_TIMEOUT_S:-2400}
TAG=run
if [[ "${1:-}" == --tag ]]; then TAG=$2; shift 2; fi
[[ "${1:-}" == -- ]] && shift

TS=$(date +%Y%m%d-%H%M%S)
DIR=/home/dsv4/ds4-project/glm52-crashlog/$TS-$TAG
mkdir -p "$DIR"
MAIN="$DIR/main.log"; SAMP="$DIR/samples.log"
plog() { echo "$(date -Is) $*" >> "$MAIN"; sync -d "$MAIN" 2>/dev/null || sync; }
WRAP=
PG=

forward_signal() {
  local signal=$1 exit_code=$2
  trap - INT TERM HUP
  plog "SAFE_RUN forwarding $signal as TERM to isolated pgid=${PG:-unavailable}"
  if [[ ${PG:-} =~ ^[0-9]+$ ]] && (( PG > 1 )); then
    kill -TERM -- "-$PG" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$PG" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$PG" 2>/dev/null && kill -KILL -- "-$PG" 2>/dev/null || true
    wait "$WRAP" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 -- "-$PG" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-$PG" 2>/dev/null; then
      plog "FATAL isolated pgid=$PG survived signal escalation"
      exit 125
    fi
  fi
  plog "SAFE_RUN interrupted signal=$signal exit=$exit_code"
  exit "$exit_code"
}

plog "SAFE_RUN start tag=$TAG vlimit_kb=$VLIMIT_KB kill_floor_gib=$KILL_FLOOR_GIB min_start_gib=$MIN_START_GIB timeout_s=$TIMEOUT_S"
plog "cmd: $*"
plog "host: $(hostname) kernel: $(uname -r)"
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
[[ -d $SRC/.git ]] && plog "tree: $(cd $SRC && git log --oneline -1) binary_sha12=$(sha256sum $SRC/ds4-server 2>/dev/null | cut -c1-12)"
grep -E 'MemAvailable|MemTotal' /proc/meminfo >> "$MAIN"; sync -d "$MAIN" 2>/dev/null || true

python3 /home/bmarti44/spark-deepseek-v4-flash/scripts/03_memory_guard.py \
  --required-gib "$MIN_START_GIB" --stable-samples 3 --timeout-seconds 0 \
  >>"$MAIN" || { plog "FATAL insufficient stable memory before launch"; exit 8; }

ulimit -v "$VLIMIT_KB" || { plog "FATAL cannot set ulimit -v"; exit 9; }

setsid timeout --signal=TERM --kill-after=30 "$TIMEOUT_S" "$@" > "$DIR/cmd.log" 2>&1 &
WRAP=$!
PG=$WRAP
trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM
trap 'forward_signal HUP 129' HUP
sleep 0.5
# find the deepest child (the engine) for RSS sampling; fall back to wrapper
ENG=$(pgrep -P "$WRAP" | head -1); ENG=${ENG:-$WRAP}
ENG2=$(pgrep -P "$ENG" 2>/dev/null | head -1); ENG=${ENG2:-$ENG}
actual_pg=$(ps -o pgid= -p "$WRAP" | tr -d ' ')
if [[ $actual_pg != "$PG" ]]; then
  plog "FATAL isolated process-group mismatch expected=$PG actual=${actual_pg:-missing}"
  kill -KILL -- "-$PG" 2>/dev/null || true
  wait "$WRAP" 2>/dev/null || true
  exit 10
fi
plog "wrapper_pid=$WRAP engine_pid=$ENG pgid=$PG (sampler at 1 Hz)"
: > "$SAMP"

KILLED=""
while kill -0 "$WRAP" 2>/dev/null; do
  MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  # sample the largest ds4* process (sol G3 finding 8: child-guess picked the
  # wrong pid); fall back to the original guess
  BIG=$(pgrep -x 'ds4-server|ds4|ds4-bench|ds4-eval' 2>/dev/null | while read -r p2; do
          printf '%s %s\n' "$(awk '/VmRSS/{print $2}' /proc/$p2/status 2>/dev/null || echo 0)" "$p2"
        done | sort -rn | head -1 | awk '{print $2}')
  SPID2=${BIG:-$ENG}
  RSS=$(awk '/VmRSS/{print $2}' "/proc/$SPID2/status" 2>/dev/null || echo 0)
  RB=$(awk '/^read_bytes/{print $2}' "/proc/$SPID2/io" 2>/dev/null || echo 0)
  echo "$(date -Is) mem_avail_kb=$MA eng_rss_kb=$RSS read_bytes=$RB" >> "$SAMP"
  sync -d "$SAMP" 2>/dev/null || true
  if (( MA < KILL_FLOOR_GIB * 1048576 )); then
    plog "KILL_FLOOR breached: MemAvailable=${MA}kB < ${KILL_FLOOR_GIB}GiB — SIGKILL pgid $PG"
    kill -KILL -- -"$PG" 2>/dev/null
    KILLED=floor
    break
  fi
  sleep 1
done
wait "$WRAP" 2>/dev/null; RC=$?
tail -25 "$DIR/cmd.log" >> "$MAIN" 2>/dev/null
plog "SAFE_RUN end rc=$RC killed=${KILLED:-no} (124=timeout, 137=SIGKILL/ENOMEM-adjacent)"
grep MemAvailable /proc/meminfo >> "$MAIN"; sync
echo "SAFE_RUN_DONE rc=$RC killed=${KILLED:-no} dir=$DIR"
exit "$RC"
