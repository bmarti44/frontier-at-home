#!/bin/bash
# glm_safe_run.sh — hardened wrapper for ANY GLM/ds4 engine invocation on the
# Spark, after two whole-box freezes caused by unbounded unified-memory
# allocation in an engine test. Run as dsv4:
#   sudo -u dsv4 bash glm_safe_run.sh [--tag NAME] -- <command...>
#
# Protections (layered, all mandatory):
#   1. ulimit -v hard cap (default 400 GiB): runaway mmap/managed allocations
#      fail with ENOMEM inside the process instead of freezing the kernel.
#   2. 4 Hz sidecar sampler: MemAvailable + engine VmRSS + read_bytes,
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
CANDIDATE_PROVENANCE=${GLM_SAFE_LOG_CANDIDATE_PROVENANCE:-0}
EXPECTED_BINARY_SHA256=${GLM_SAFE_EXPECTED_BINARY_SHA256:-}
TAG=run
config_error() {
  printf 'FATAL invalid %s\n' "$*" >&2
  exit 2
}
for pair in \
  "GLM_SAFE_VLIMIT_KB:$VLIMIT_KB" \
  "GLM_SAFE_KILL_FLOOR_GIB:$KILL_FLOOR_GIB" \
  "GLM_SAFE_MIN_START_GIB:$MIN_START_GIB" \
  "GLM_SAFE_TIMEOUT_S:$TIMEOUT_S"
do
  name=${pair%%:*}
  value=${pair#*:}
  [[ $value =~ ^[0-9]{1,9}$ ]] || config_error "$name"
done
VLIMIT_KB=$((10#$VLIMIT_KB))
KILL_FLOOR_GIB=$((10#$KILL_FLOOR_GIB))
MIN_START_GIB=$((10#$MIN_START_GIB))
TIMEOUT_S=$((10#$TIMEOUT_S))
if (( VLIMIT_KB < 1048576 || VLIMIT_KB > 419430400 )); then
  config_error "GLM_SAFE_VLIMIT_KB"
fi
if (( KILL_FLOOR_GIB < 18 || KILL_FLOOR_GIB > 64 )); then
  config_error "GLM_SAFE_KILL_FLOOR_GIB"
fi
if (( MIN_START_GIB < 110 || MIN_START_GIB > 119 )); then
  config_error "GLM_SAFE_MIN_START_GIB"
fi
if (( TIMEOUT_S < 1 || TIMEOUT_S > 3600 )); then
  config_error "GLM_SAFE_TIMEOUT_S"
fi
if (( MIN_START_GIB <= KILL_FLOOR_GIB )); then
  config_error "memory floors"
fi
[[ $CANDIDATE_PROVENANCE =~ ^[01]$ ]] ||
  config_error "GLM_SAFE_LOG_CANDIDATE_PROVENANCE"
if [[ "${1:-}" == --tag ]]; then
  [[ -n ${2:-} ]] || config_error "tag"
  TAG=$2
  shift 2
fi
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  config_error "tag"
[[ "${1:-}" == -- ]] && shift
(( $# > 0 )) || config_error "command"

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
plog "candidate_provenance_enabled=$CANDIDATE_PROVENANCE"
if [[ $CANDIDATE_PROVENANCE == 1 ]]; then
  APPROVED_SRC_ROOT=/home/dsv4/ds4-project/src
  CANDIDATE_SRC=$(realpath -e -- "${GLM_CANDIDATE_SRC:-}" 2>/dev/null || true)
  [[ $CANDIDATE_SRC == "$APPROVED_SRC_ROOT"/* ]] ||
    config_error "GLM_CANDIDATE_SRC"
  CANDIDATE_BINARY=$(realpath -e -- "$CANDIDATE_SRC/ds4-server" 2>/dev/null || true)
  [[ $CANDIDATE_BINARY == "$CANDIDATE_SRC"/ds4-server ]] ||
    config_error "GLM_CANDIDATE_SRC binary containment"
  [[ -f $CANDIDATE_BINARY && -x $CANDIDATE_BINARY ]] ||
    config_error "GLM_CANDIDATE_SRC binary"
  [[ $EXPECTED_BINARY_SHA256 =~ ^[0-9a-f]{64}$ ]] ||
    config_error "GLM_SAFE_EXPECTED_BINARY_SHA256"
  CANDIDATE_HASH=$(sha256sum -- "$CANDIDATE_BINARY" | awk '{print $1}')
  [[ $CANDIDATE_HASH == "$EXPECTED_BINARY_SHA256" ]] ||
    config_error "GLM_SAFE_EXPECTED_BINARY_SHA256 mismatch"
  CANDIDATE_DEVICE_INODE=$(stat -Lc '%d:%i' -- "$CANDIDATE_BINARY")
  plog "candidate_src=$CANDIDATE_SRC candidate_binary_sha256=$CANDIDATE_HASH candidate_device_inode=$CANDIDATE_DEVICE_INODE"
fi
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
plog "wrapper_pid=$WRAP engine_pid=$ENG pgid=$PG (sampler at 4 Hz)"
: > "$SAMP"

KILLED=""
EXECUTED_CANDIDATE_OBSERVED=0
EXECUTED_PID=""
EXECUTED_START_TICKS=""
PROVENANCE_FAILURE=""
while kill -0 "$WRAP" 2>/dev/null; do
  MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  # sample the largest ds4* process (sol G3 finding 8: child-guess picked the
  # wrong pid); fall back to the original guess
  BIG=$(pgrep -x 'ds4-server|ds4|ds4-bench|ds4-eval' 2>/dev/null | while read -r p2; do
          p2_group=$(ps -o pgid= -p "$p2" 2>/dev/null | tr -d ' ')
          [[ $p2_group == "$PG" ]] || continue
          printf '%s %s\n' "$(awk '/VmRSS/{print $2}' /proc/$p2/status 2>/dev/null || echo 0)" "$p2"
        done | sort -rn | head -1 | awk '{print $2}')
  SPID2=${BIG:-$ENG}
  if [[ $CANDIDATE_PROVENANCE == 1 && -n $BIG && $EXECUTED_CANDIDATE_OBSERVED == 0 ]]; then
    EXECUTED_PATH=$(readlink -f -- "/proc/$SPID2/exe" 2>/dev/null || true)
    EXECUTED_HASH=$(sha256sum -- "/proc/$SPID2/exe" 2>/dev/null | awk '{print $1}')
    EXECUTED_DEVICE_INODE=$(stat -Lc '%d:%i' -- "/proc/$SPID2/exe" 2>/dev/null || true)
    if [[ $EXECUTED_PATH != "$CANDIDATE_BINARY" ||
          $EXECUTED_HASH != "$EXPECTED_BINARY_SHA256" ||
          $EXECUTED_DEVICE_INODE != "$CANDIDATE_DEVICE_INODE" ]]; then
      plog "FATAL executed candidate mismatch pid=$SPID2 path=${EXECUTED_PATH:-missing} executed_binary_sha256=${EXECUTED_HASH:-missing} device_inode=${EXECUTED_DEVICE_INODE:-missing}"
      kill -KILL -- -"$PG" 2>/dev/null || true
      KILLED=provenance
      PROVENANCE_FAILURE=mismatch
      break
    fi
    EXECUTED_CANDIDATE_OBSERVED=1
    EXECUTED_PID=$SPID2
    EXECUTED_START_TICKS=$(awk '{print $22}' "/proc/$EXECUTED_PID/stat" 2>/dev/null || true)
    [[ $EXECUTED_START_TICKS =~ ^[0-9]+$ ]] || {
      plog "FATAL executed candidate start ticks unavailable pid=$EXECUTED_PID"
      kill -KILL -- -"$PG" 2>/dev/null || true
      KILLED=provenance
      PROVENANCE_FAILURE=start-ticks
      break
    }
    plog "executed_candidate_verified pid=$EXECUTED_PID start_ticks=$EXECUTED_START_TICKS path=$EXECUTED_PATH executed_binary_sha256=$EXECUTED_HASH device_inode=$EXECUTED_DEVICE_INODE"
  fi
  if [[ $CANDIDATE_PROVENANCE == 1 && $EXECUTED_CANDIDATE_OBSERVED == 1 ]]; then
    CURRENT_START_TICKS=$(awk '{print $22}' "/proc/$EXECUTED_PID/stat" 2>/dev/null || true)
    if [[ $CURRENT_START_TICKS != "$EXECUTED_START_TICKS" ]]; then
      sleep 0.1
      if ! kill -0 "$WRAP" 2>/dev/null; then
        break
      fi
      plog "FATAL executed candidate identity changed pid=$EXECUTED_PID reason=start-ticks"
      kill -KILL -- -"$PG" 2>/dev/null || true
      KILLED=provenance
      PROVENANCE_FAILURE=continuous-identity
      break
    fi
    CURRENT_GROUP=$(ps -o pgid= -p "$EXECUTED_PID" 2>/dev/null | tr -d ' ')
    CURRENT_PATH=$(readlink -f -- "/proc/$EXECUTED_PID/exe" 2>/dev/null || true)
    CURRENT_HASH=$(sha256sum -- "/proc/$EXECUTED_PID/exe" 2>/dev/null | awk '{print $1}')
    CURRENT_DEVICE_INODE=$(stat -Lc '%d:%i' -- "/proc/$EXECUTED_PID/exe" 2>/dev/null || true)
    if [[ $CURRENT_GROUP != "$PG" ||
          $CURRENT_PATH != "$CANDIDATE_BINARY" ||
          $CURRENT_HASH != "$EXPECTED_BINARY_SHA256" ||
          $CURRENT_DEVICE_INODE != "$CANDIDATE_DEVICE_INODE" ]]; then
      plog "FATAL executed candidate identity changed pid=$EXECUTED_PID start_ticks=$CURRENT_START_TICKS pgid=${CURRENT_GROUP:-missing} path=${CURRENT_PATH:-missing} executed_binary_sha256=${CURRENT_HASH:-missing} device_inode=${CURRENT_DEVICE_INODE:-missing}"
      kill -KILL -- -"$PG" 2>/dev/null || true
      KILLED=provenance
      PROVENANCE_FAILURE=continuous-identity
      break
    fi
  fi
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
  sleep 0.25
done
wait "$WRAP" 2>/dev/null; RC=$?
if [[ $CANDIDATE_PROVENANCE == 1 && $EXECUTED_CANDIDATE_OBSERVED == 0 ]]; then
  plog "FATAL executed candidate binary was not observed"
  RC=11
elif [[ -n $PROVENANCE_FAILURE ]]; then
  RC=11
fi
tail -25 "$DIR/cmd.log" >> "$MAIN" 2>/dev/null
plog "SAFE_RUN end rc=$RC killed=${KILLED:-no} (124=timeout, 137=SIGKILL/ENOMEM-adjacent)"
grep MemAvailable /proc/meminfo >> "$MAIN"; sync
echo "SAFE_RUN_DONE rc=$RC killed=${KILLED:-no} dir=$DIR"
exit "$RC"
