#!/bin/bash
# glm_safe_run.sh — hardened wrapper for ANY GLM/ds4 engine invocation on the
# Spark, after two whole-box freezes caused by unbounded unified-memory
# allocation in an engine test. Normally run as dsv4:
#   sudo -u dsv4 bash glm_safe_run.sh [--tag NAME] -- <command...>
# GLM_SAFE_RUN_AS_CURRENT_USER=1 selects the default-off, sudo-free path used
# by the logged-in benchmark owner under the same user-systemd containment.
#
# Protections (layered, all mandatory):
#   1. ulimit -v hard cap (default 400 GiB): runaway mmap/managed allocations
#      fail with ENOMEM inside the process instead of freezing the kernel.
#   2. Periodic sidecar sampler: MemAvailable + engine VmRSS + read_bytes;
#      actual cadence is recorded in samples.log.
#      appended to a PERSISTENT log and fdatasync'd every sample, so the
#      final seconds survive a hard freeze/power cycle.
#   3. Kill floor (default 18 GiB MemAvailable): sampler SIGKILLs the whole
#      process group the moment available memory crosses the floor.
#   4. Wall-clock timeout (default 2400 s) via timeout(1).
#   5. Start/exit records with command, env, tree commit, binary sha.
# Logs: /home/dsv4/ds4-project/glm52-crashlog/<ts>-<tag>/
set -u
umask 077
VLIMIT_KB=${GLM_SAFE_VLIMIT_KB:-419430400}  # 400 GiB backstop: engine mmaps the whole GGUF (196.6 GiB VIRTUAL, file-backed, mostly non-resident), so RLIMIT_AS must clear that. Resident-growth protection is the kill-floor sampler below.
KILL_FLOOR_GIB=${GLM_SAFE_KILL_FLOOR_GIB:-18}
MIN_START_GIB=${GLM_SAFE_MIN_START_GIB:-110}
TIMEOUT_S=${GLM_SAFE_TIMEOUT_S:-2400}
CANDIDATE_PROVENANCE=${GLM_SAFE_LOG_CANDIDATE_PROVENANCE:-0}
EXPECTED_BINARY_SHA256=${GLM_SAFE_EXPECTED_BINARY_SHA256:-}
PROVENANCE_ENV_ALLOWLIST=${GLM_SAFE_PROVENANCE_ENV_ALLOWLIST:-}
EXPECTED_ENV_SHA256=${GLM_SAFE_EXPECTED_ENV_SHA256:-}
FINAL_ARTIFACTS=${GLM_SAFE_FINAL_ARTIFACTS:-}
CKV_RUN_NONCE=${DS4_GLM_CKV_RUN_NONCE:-}
WITNESS_NONCE=${GLM_SAFE_WITNESS_NONCE:-}
WITNESS_ARTIFACT=${GLM_SAFE_WITNESS_ARTIFACT:-}
REQUIRE_CGROUP=${GLM_SAFE_REQUIRE_CGROUP:-0}
EXPECTED_CGROUP_UNIT=${GLM_SAFE_CGROUP_UNIT:-}
RUN_AS_CURRENT_USER=${GLM_SAFE_RUN_AS_CURRENT_USER:-0}
ROOT_AUTHORITY=${GLM_W1_ROOT_AUTHORITY:-0}
AUTHORITY_CRASH_ROOT=${GLM_SAFE_CRASH_ROOT:-}
MEMORY_GUARD_OVERRIDE=${GLM_SAFE_MEMORY_GUARD_PATH:-}
EXPECTED_MEMORY_GUARD_SHA256=${GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256:-}
KERNEL_GPU_FAULT_RE='NVRM.*Xid|NVRM.*NV_ERR_NO_MEMORY|NVRM.*Out of memory|oom-kill|Out of memory: Killed process'
USERSPACE_GPU_OOM_RE='CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|CUDA.{0,160}(allocation failed|out of memory)'
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
if (( TIMEOUT_S < 1 || TIMEOUT_S > 9000 )); then
  config_error "GLM_SAFE_TIMEOUT_S"
fi
if (( MIN_START_GIB <= KILL_FLOOR_GIB )); then
  config_error "memory floors"
fi
[[ $CANDIDATE_PROVENANCE =~ ^[01]$ ]] ||
  config_error "GLM_SAFE_LOG_CANDIDATE_PROVENANCE"
[[ $REQUIRE_CGROUP =~ ^[01]$ ]] ||
  config_error "GLM_SAFE_REQUIRE_CGROUP"
[[ $RUN_AS_CURRENT_USER =~ ^[01]$ ]] ||
  config_error "GLM_SAFE_RUN_AS_CURRENT_USER"
[[ $ROOT_AUTHORITY =~ ^[01]$ ]] ||
  config_error "GLM_W1_ROOT_AUTHORITY"
if [[ $ROOT_AUTHORITY == 1 ]]; then
  [[ $RUN_AS_CURRENT_USER == 0 && $(id -un) == dsv4 ]] ||
    config_error "root authority identity"
  [[ $AUTHORITY_CRASH_ROOT =~ ^/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}/crashlog$ ]] ||
    config_error "GLM_SAFE_CRASH_ROOT"
fi
if [[ -n $WITNESS_NONCE && ! $WITNESS_NONCE =~ ^[0-9a-f]{64}$ ]]; then
  config_error "GLM_SAFE_WITNESS_NONCE"
fi
if [[ -n $CKV_RUN_NONCE && ! $CKV_RUN_NONCE =~ ^[0-9a-f]{64}$ ]]; then
  config_error "DS4_GLM_CKV_RUN_NONCE"
fi
if [[ -n $WITNESS_NONCE ]]; then
  if [[ $ROOT_AUTHORITY == 1 ]]; then
    [[ $WITNESS_ARTIFACT =~ ^/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}/artifacts/attempt-[0-9]{2}\.tsv$ ]] ||
      config_error "GLM_SAFE_WITNESS_ARTIFACT"
  else
    [[ $WITNESS_ARTIFACT =~ ^/home/bmarti44/\.local/state/glm52-[A-Za-z0-9._/-]+$ ]] ||
      config_error "GLM_SAFE_WITNESS_ARTIFACT"
  fi
  WITNESS_PARENT=$(realpath -e -- "$(dirname -- "$WITNESS_ARTIFACT")" 2>/dev/null || true)
  if [[ $ROOT_AUTHORITY == 1 ]]; then
    [[ $WITNESS_PARENT =~ ^/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}/artifacts$ ]] ||
      config_error "GLM_SAFE_WITNESS_ARTIFACT parent"
  else
    [[ $WITNESS_PARENT == /home/bmarti44/.local/state/glm52-* ]] ||
      config_error "GLM_SAFE_WITNESS_ARTIFACT parent"
  fi
  [[ $WITNESS_ARTIFACT == "$WITNESS_PARENT"/"$(basename -- "$WITNESS_ARTIFACT")" ]] ||
    config_error "GLM_SAFE_WITNESS_ARTIFACT normalization"
fi
ENV_PROVENANCE=0
PROVENANCE_ENV_NAMES=""
if [[ -n $PROVENANCE_ENV_ALLOWLIST || -n $EXPECTED_ENV_SHA256 ]]; then
  [[ $CANDIDATE_PROVENANCE == 1 ]] ||
    config_error "environment provenance requires candidate provenance"
  [[ $EXPECTED_ENV_SHA256 =~ ^[0-9a-f]{64}$ ]] ||
    config_error "GLM_SAFE_EXPECTED_ENV_SHA256"
  [[ $PROVENANCE_ENV_ALLOWLIST =~ ^DS4_[A-Z0-9_]{0,59}(,DS4_[A-Z0-9_]{0,59}){0,31}$ ]] ||
    config_error "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST"
  PROVENANCE_ENV_NAMES=$(
    tr ',' '\n' <<<"$PROVENANCE_ENV_ALLOWLIST" | LC_ALL=C sort -u
  )
  PROVENANCE_ENV_NAME_COUNT=$(wc -l <<<"$PROVENANCE_ENV_NAMES")
  PROVENANCE_ENV_INPUT_COUNT=$(
    awk -F, '{print NF}' <<<"$PROVENANCE_ENV_ALLOWLIST"
  )
  [[ $PROVENANCE_ENV_NAME_COUNT == "$PROVENANCE_ENV_INPUT_COUNT" ]] ||
    config_error "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST duplicates"
  PROVENANCE_ENV_ALLOWLIST=$(paste -sd, <<<"$PROVENANCE_ENV_NAMES")
  ENV_PROVENANCE=1
fi
FINAL_ARTIFACT_PATHS=()
if [[ -n $FINAL_ARTIFACTS ]]; then
  [[ $CANDIDATE_PROVENANCE == 1 ]] ||
    config_error "final artifacts require candidate provenance"
  IFS=, read -r -a FINAL_ARTIFACT_PATHS <<<"$FINAL_ARTIFACTS"
  (( ${#FINAL_ARTIFACT_PATHS[@]} >= 1 && ${#FINAL_ARTIFACT_PATHS[@]} <= 8 )) ||
    config_error "GLM_SAFE_FINAL_ARTIFACTS count"
  declare -A FINAL_ARTIFACT_SEEN=()
  for artifact in "${FINAL_ARTIFACT_PATHS[@]}"; do
    [[ $artifact =~ ^/home/bmarti44/\.local/state/glm52-[A-Za-z0-9._/-]+$ ]] ||
      config_error "GLM_SAFE_FINAL_ARTIFACTS path"
    parent=$(realpath -m -- "$(dirname -- "$artifact")" 2>/dev/null || true)
    [[ $parent == /home/bmarti44/.local/state/glm52-* ]] ||
      config_error "GLM_SAFE_FINAL_ARTIFACTS parent"
    [[ $artifact == "$parent"/"$(basename -- "$artifact")" ]] ||
      config_error "GLM_SAFE_FINAL_ARTIFACTS normalization"
    [[ -z ${FINAL_ARTIFACT_SEEN[$artifact]+x} ]] ||
      config_error "GLM_SAFE_FINAL_ARTIFACTS duplicate"
    FINAL_ARTIFACT_SEEN[$artifact]=1
  done
fi
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
if [[ $ROOT_AUTHORITY == 1 ]]; then
  CRASH_ROOT=$AUTHORITY_CRASH_ROOT
elif [[ $RUN_AS_CURRENT_USER == 1 ]]; then
  [[ $(id -u) != 0 && $(id -un) == bmarti44 ]] ||
    config_error "GLM_SAFE_RUN_AS_CURRENT_USER identity"
  CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
else
  CRASH_ROOT=/home/dsv4/ds4-project/glm52-crashlog
fi
DIR=$CRASH_ROOT/$TS-$TAG
mkdir -p "$DIR"
MAIN="$DIR/main.log"; SAMP="$DIR/samples.log"
KERNEL_LOG="$DIR/kernel.log"
RUN_STARTED_EPOCH=$(date -u +%s)
plog() { echo "$(date -u --iso-8601=ns) $*" >> "$MAIN"; sync -d "$MAIN" 2>/dev/null || sync; }
live_group_pids() {
  ps -eo pid=,pgid=,stat= | awk -v group="$1" \
    '$2 == group && $3 !~ /^Z/ {print $1}'
}
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
if [[ -n $CKV_RUN_NONCE ]]; then
  plog "run_nonce=$CKV_RUN_NONCE"
fi
CGROUP_PATH=""
CGROUP_DIR=""
CGROUP_BASELINE_PIDS=""
declare -A CGROUP_EVENTS_BEFORE=()
if [[ $REQUIRE_CGROUP == 1 ]]; then
  [[ $EXPECTED_CGROUP_UNIT =~ ^glm52-[A-Za-z0-9_-]{1,80}$ ]] ||
    config_error "GLM_SAFE_CGROUP_UNIT"
  CGROUP_PATH=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  [[ $CGROUP_PATH == */"$EXPECTED_CGROUP_UNIT.service" ]] ||
    config_error "GLM_SAFE_CGROUP_UNIT membership"
  CGROUP_DIR=/sys/fs/cgroup$CGROUP_PATH
  [[ -r $CGROUP_DIR/memory.high && -r $CGROUP_DIR/memory.max &&
     -r $CGROUP_DIR/memory.current && -r $CGROUP_DIR/memory.peak &&
     -r $CGROUP_DIR/memory.swap.max &&
     -r $CGROUP_DIR/memory.swap.current &&
     -r $CGROUP_DIR/memory.oom.group &&
     -r $CGROUP_DIR/memory.events.local && -r $CGROUP_DIR/cgroup.procs ]] ||
    config_error "cgroup controls"
  CGROUP_HIGH=$(<"$CGROUP_DIR/memory.high")
  CGROUP_MAX=$(<"$CGROUP_DIR/memory.max")
  CGROUP_SWAP_MAX=$(<"$CGROUP_DIR/memory.swap.max")
  CGROUP_OOM_GROUP=$(<"$CGROUP_DIR/memory.oom.group")
  [[ $CGROUP_HIGH =~ ^[0-9]+$ && $CGROUP_MAX =~ ^[0-9]+$ &&
     $CGROUP_SWAP_MAX == 0 && $CGROUP_OOM_GROUP == 1 ]] ||
    config_error "finite cgroup controls"
  while read -r key value; do
    CGROUP_EVENTS_BEFORE["$key"]=$value
  done < "$CGROUP_DIR/memory.events.local"
  CGROUP_BASELINE_PIDS=$(<"$CGROUP_DIR/cgroup.procs")
  plog "cgroup_verified path=$CGROUP_PATH memory_high=$CGROUP_HIGH memory_max=$CGROUP_MAX memory_swap_max=$CGROUP_SWAP_MAX memory_oom_group=$CGROUP_OOM_GROUP"
fi
if [[ $CANDIDATE_PROVENANCE == 1 ]]; then
  if [[ $ROOT_AUTHORITY == 1 ]]; then
    APPROVED_SRC_ROOT=$(dirname -- "$AUTHORITY_CRASH_ROOT")
  elif [[ $RUN_AS_CURRENT_USER == 1 ]]; then
    APPROVED_SRC_ROOT=/home/bmarti44/.cache
  else
    APPROVED_SRC_ROOT=/home/dsv4/ds4-project/src
  fi
  CANDIDATE_SRC=$(realpath -e -- "${GLM_CANDIDATE_SRC:-}" 2>/dev/null || true)
  if [[ $ROOT_AUTHORITY == 1 ]]; then
    [[ $CANDIDATE_SRC == "$APPROVED_SRC_ROOT"/frozen-scorer ]] ||
      config_error "GLM_CANDIDATE_SRC"
  elif [[ $RUN_AS_CURRENT_USER == 1 ]]; then
    [[ $CANDIDATE_SRC == /home/bmarti44/.cache/glm52-* ]] ||
      config_error "GLM_CANDIDATE_SRC"
  else
    [[ $CANDIDATE_SRC == "$APPROVED_SRC_ROOT"/* ]] ||
      config_error "GLM_CANDIDATE_SRC"
  fi
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

HARNESS_ROOT=$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$(readlink -f -- "$0")")")")")
if [[ -n $MEMORY_GUARD_OVERRIDE ]]; then
  [[ $MEMORY_GUARD_OVERRIDE =~ ^/proc/[0-9]+/fd/[0-9]+$ ]] ||
    config_error "GLM_SAFE_MEMORY_GUARD_PATH"
  [[ $EXPECTED_MEMORY_GUARD_SHA256 =~ ^[0-9a-f]{64}$ ]] ||
    config_error "GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256"
  MEMORY_GUARD=$MEMORY_GUARD_OVERRIDE
  MEMORY_GUARD_SHA256=$(sha256sum -- "$MEMORY_GUARD" 2>/dev/null | awk '{print $1}')
  [[ $MEMORY_GUARD_SHA256 == "$EXPECTED_MEMORY_GUARD_SHA256" ]] ||
    config_error "GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256 mismatch"
  plog "memory_guard_descriptor_path=$MEMORY_GUARD memory_guard_sha256=$MEMORY_GUARD_SHA256"
else
  MEMORY_GUARD=$HARNESS_ROOT/scripts/03_memory_guard.py
fi
python3 "$MEMORY_GUARD" \
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
plog "wrapper_pid=$WRAP engine_pid=$ENG pgid=$PG (periodic sampler; actual cadence in samples.log)"
: > "$SAMP"

KILLED=""
EXECUTED_CANDIDATE_OBSERVED=0
EXECUTED_PID=""
EXECUTED_START_TICKS=""
EXECUTED_CANDIDATE_CLEAN_EXIT=0
EXECUTED_CANDIDATE_EXIT_PENDING=0
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
    if [[ $ENV_PROVENANCE == 1 ]]; then
      EXECUTED_ENV_HASH=$(
        python3 - "/proc/$SPID2/environ" "$PROVENANCE_ENV_ALLOWLIST" <<'PY'
import hashlib
import sys

path, raw_names = sys.argv[1:]
names = raw_names.split(",")
entries = {}
with open(path, "rb") as stream:
    for item in stream.read().split(b"\0"):
        if b"=" not in item:
            continue
        name, value = item.split(b"=", 1)
        entries[name.decode("ascii", errors="strict")] = value
canonical = b"".join(
    name.encode("ascii") + b"=" + entries.get(name, b"<UNSET>") + b"\n"
    for name in names
)
print(hashlib.sha256(canonical).hexdigest())
PY
      )
      if [[ $EXECUTED_ENV_HASH != "$EXPECTED_ENV_SHA256" ]]; then
        plog "FATAL executed candidate environment mismatch pid=$SPID2 executed_environment_sha256=${EXECUTED_ENV_HASH:-missing} expected_environment_sha256=$EXPECTED_ENV_SHA256"
        kill -KILL -- -"$PG" 2>/dev/null || true
        KILLED=provenance
        PROVENANCE_FAILURE=environment
        break
      fi
      plog "executed_environment_allowlist=$PROVENANCE_ENV_ALLOWLIST executed_environment_sha256=$EXECUTED_ENV_HASH"
    fi
    plog "executed_candidate_verified pid=$EXECUTED_PID start_ticks=$EXECUTED_START_TICKS path=$EXECUTED_PATH executed_binary_sha256=$EXECUTED_HASH device_inode=$EXECUTED_DEVICE_INODE"
  fi
  if [[ $CANDIDATE_PROVENANCE == 1 && $EXECUTED_CANDIDATE_OBSERVED == 1 ]]; then
    if [[ $EXECUTED_CANDIDATE_EXIT_PENDING == 1 ]]; then
      REPLACEMENT_PID=""
      REPLACEMENT_PID=$(pgrep -x 'ds4-server|ds4|ds4-bench|ds4-eval' 2>/dev/null |
        while read -r candidate_pid; do
          if [[ $candidate_pid == "$EXECUTED_PID" ]]; then
            REUSED_STAT=""
            IFS= read -r REUSED_STAT <"/proc/$candidate_pid/stat" 2>/dev/null || true
            [[ -n $REUSED_STAT ]] || continue
            REUSED_STAT_REST=${REUSED_STAT##*) }
            read -r -a REUSED_STAT_FIELDS <<<"$REUSED_STAT_REST"
            REUSED_STATE=${REUSED_STAT_FIELDS[0]:-}
            REUSED_START_TICKS=${REUSED_STAT_FIELDS[19]:-}
            if [[ $REUSED_START_TICKS == "$EXECUTED_START_TICKS" &&
                  ($REUSED_STATE == Z || $REUSED_STATE == X) ]]; then
              continue
            fi
          fi
          candidate_group=$(ps -o pgid= -p "$candidate_pid" 2>/dev/null | tr -d ' ')
          [[ $candidate_group == "$PG" ]] && { echo "$candidate_pid"; break; }
        done)
      if [[ -n $REPLACEMENT_PID ]]; then
        plog "FATAL replacement candidate appeared after verified exit pid=$REPLACEMENT_PID"
        kill -KILL -- -"$PG" 2>/dev/null || true
        KILLED=provenance
        PROVENANCE_FAILURE=replacement
        break
      fi
    else
      CURRENT_STATE=""
      CURRENT_START_TICKS=""
      CURRENT_STAT=""
      IFS= read -r CURRENT_STAT <"/proc/$EXECUTED_PID/stat" 2>/dev/null || true
      if [[ -n $CURRENT_STAT ]]; then
        CURRENT_STAT_REST=${CURRENT_STAT##*) }
        read -r -a CURRENT_STAT_FIELDS <<<"$CURRENT_STAT_REST"
        CURRENT_STATE=${CURRENT_STAT_FIELDS[0]:-}
        CURRENT_START_TICKS=${CURRENT_STAT_FIELDS[19]:-}
      fi
      CURRENT_GROUP=$(ps -o pgid= -p "$EXECUTED_PID" 2>/dev/null | tr -d ' ')
      CURRENT_PATH=$(readlink -f -- "/proc/$EXECUTED_PID/exe" 2>/dev/null || true)
      CURRENT_HASH=$(sha256sum -- "/proc/$EXECUTED_PID/exe" 2>/dev/null | awk '{print $1}')
      CURRENT_DEVICE_INODE=$(stat -Lc '%d:%i' -- "/proc/$EXECUTED_PID/exe" 2>/dev/null || true)
      IDENTITY_INCOMPLETE=0
      if [[ -z $CURRENT_START_TICKS || -z $CURRENT_GROUP ||
            -z $CURRENT_PATH ||
            $CURRENT_PATH == "/proc/$EXECUTED_PID/exe" ||
            -z $CURRENT_HASH || -z $CURRENT_DEVICE_INODE ]]; then
        IDENTITY_INCOMPLETE=1
      fi
      if [[ -n $CURRENT_START_TICKS &&
            $CURRENT_START_TICKS != "$EXECUTED_START_TICKS" ]]; then
        plog "FATAL executed candidate identity changed pid=$EXECUTED_PID reason=start-ticks"
        kill -KILL -- -"$PG" 2>/dev/null || true
        KILLED=provenance
        PROVENANCE_FAILURE=continuous-identity
        break
      elif [[ -z $CURRENT_STATE || $CURRENT_STATE == Z || $CURRENT_STATE == X ]]; then
        EXECUTED_CANDIDATE_EXIT_PENDING=1
        plog "executed candidate exited; monitoring controller and process group pid=$EXECUTED_PID"
      elif [[ (-n $CURRENT_GROUP && $CURRENT_GROUP != "$PG") ||
              (-n $CURRENT_PATH &&
               $CURRENT_PATH != "/proc/$EXECUTED_PID/exe" &&
               $CURRENT_PATH != "$CANDIDATE_BINARY") ||
              (-n $CURRENT_HASH && $CURRENT_HASH != "$EXPECTED_BINARY_SHA256") ||
              (-n $CURRENT_DEVICE_INODE &&
               $CURRENT_DEVICE_INODE != "$CANDIDATE_DEVICE_INODE") ]]; then
        plog "FATAL executed candidate identity changed pid=$EXECUTED_PID start_ticks=$CURRENT_START_TICKS pgid=${CURRENT_GROUP:-missing} path=${CURRENT_PATH:-missing} executed_binary_sha256=${CURRENT_HASH:-missing} device_inode=${CURRENT_DEVICE_INODE:-missing}"
        kill -KILL -- -"$PG" 2>/dev/null || true
        KILLED=provenance
        PROVENANCE_FAILURE=continuous-identity
        break
      elif [[ $IDENTITY_INCOMPLETE == 1 ]]; then
        CONFIRM_STATE=""
        CONFIRM_START_TICKS=""
        CONFIRM_STAT=""
        IFS= read -r CONFIRM_STAT <"/proc/$EXECUTED_PID/stat" 2>/dev/null || true
        if [[ -n $CONFIRM_STAT ]]; then
          CONFIRM_STAT_REST=${CONFIRM_STAT##*) }
          read -r -a CONFIRM_STAT_FIELDS <<<"$CONFIRM_STAT_REST"
          CONFIRM_STATE=${CONFIRM_STAT_FIELDS[0]:-}
          CONFIRM_START_TICKS=${CONFIRM_STAT_FIELDS[19]:-}
        fi
        if [[ -n $CONFIRM_START_TICKS &&
              $CONFIRM_START_TICKS != "$EXECUTED_START_TICKS" ]]; then
          plog "FATAL executed candidate identity changed pid=$EXECUTED_PID reason=confirmed-start-ticks"
          kill -KILL -- -"$PG" 2>/dev/null || true
          KILLED=provenance
          PROVENANCE_FAILURE=continuous-identity
          break
        elif [[ -z $CONFIRM_STATE ||
                $CONFIRM_STATE == Z || $CONFIRM_STATE == X ]]; then
          EXECUTED_CANDIDATE_EXIT_PENDING=1
          plog "executed candidate exited during identity sample; monitoring controller and process group pid=$EXECUTED_PID"
        else
          plog "FATAL executed candidate identity unavailable while process remained live pid=$EXECUTED_PID start_ticks=$CONFIRM_START_TICKS state=$CONFIRM_STATE"
          kill -KILL -- -"$PG" 2>/dev/null || true
          KILLED=provenance
          PROVENANCE_FAILURE=continuous-identity
          break
        fi
      fi
    fi
  fi
  RSS=$(awk '/VmRSS/{print $2}' "/proc/$SPID2/status" 2>/dev/null || echo 0)
  RB=$(awk '/^read_bytes/{print $2}' "/proc/$SPID2/io" 2>/dev/null || echo 0)
  [[ $RSS =~ ^[0-9]+$ ]] || RSS=0
  [[ $RB =~ ^[0-9]+$ ]] || RB=0
  CGROUP_CURRENT=na
  CGROUP_PEAK=na
  CGROUP_SWAP_CURRENT=na
  if [[ $REQUIRE_CGROUP == 1 ]]; then
    CGROUP_CURRENT=$(<"$CGROUP_DIR/memory.current")
    CGROUP_PEAK=$(<"$CGROUP_DIR/memory.peak")
    CGROUP_SWAP_CURRENT=$(<"$CGROUP_DIR/memory.swap.current")
    if [[ ! $CGROUP_CURRENT =~ ^[0-9]+$ ||
          ! $CGROUP_PEAK =~ ^[0-9]+$ ||
          ! $CGROUP_SWAP_CURRENT =~ ^[0-9]+$ ]]; then
      plog "FATAL cgroup safety telemetry became unreadable"
      kill -KILL -- -"$PG" 2>/dev/null || true
      KILLED=telemetry
      break
    fi
  fi
  echo "$(date -u --iso-8601=ns) mem_avail_kb=$MA eng_rss_kb=$RSS read_bytes=$RB cgroup_current_bytes=$CGROUP_CURRENT cgroup_peak_bytes=$CGROUP_PEAK cgroup_swap_current_bytes=$CGROUP_SWAP_CURRENT" >> "$SAMP"
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
if [[ $EXECUTED_CANDIDATE_EXIT_PENDING == 1 && $RC != 0 ]]; then
  plog "FATAL wrapper command failed after candidate exit rc=$RC"
fi
SURVIVORS=$(live_group_pids "$PG")
if [[ -n $SURVIVORS ]]; then
  plog "FATAL isolated process group survived command completion pgid=$PG pids=$(tr '\n' ',' <<<"$SURVIVORS" | sed 's/,$//')"
  kill -TERM -- -"$PG" 2>/dev/null || true
  for _ in $(seq 1 30); do
    [[ -z $(live_group_pids "$PG") ]] && break
    sleep 0.1
  done
  [[ -z $(live_group_pids "$PG") ]] || kill -KILL -- -"$PG" 2>/dev/null || true
  for _ in $(seq 1 50); do
    [[ -z $(live_group_pids "$PG") ]] && break
    sleep 0.1
  done
  if [[ -n $(live_group_pids "$PG") ]]; then
    plog "FATAL isolated process-group cleanup failed pgid=$PG"
    RC=125
  else
    RC=12
  fi
fi
if [[ $CANDIDATE_PROVENANCE == 1 && $EXECUTED_CANDIDATE_OBSERVED == 0 ]]; then
  plog "FATAL executed candidate binary was not observed"
  RC=11
elif [[ -n $PROVENANCE_FAILURE ]]; then
  RC=11
fi
if [[ $REQUIRE_CGROUP == 1 ]]; then
  CGROUP_NEW_PIDS=""
  while read -r pid; do
    [[ -n $pid ]] || continue
    if ! grep -qx -- "$pid" <<<"$CGROUP_BASELINE_PIDS"; then
      CGROUP_NEW_PIDS+="${CGROUP_NEW_PIDS:+ }$pid"
    fi
  done < "$CGROUP_DIR/cgroup.procs"
  if [[ -n $CGROUP_NEW_PIDS ]]; then
    plog "FATAL cgroup descendants survived command completion pids=$CGROUP_NEW_PIDS"
    for pid in $CGROUP_NEW_PIDS; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
      remaining=""
      for pid in $CGROUP_NEW_PIDS; do
        kill -0 "$pid" 2>/dev/null && remaining+="${remaining:+ }$pid"
      done
      [[ -z $remaining ]] && break
      sleep 0.1
    done
    for pid in ${remaining:-}; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 50); do
      remaining=""
      for pid in $CGROUP_NEW_PIDS; do
        state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -n $state && $state != Z* ]] && remaining+="${remaining:+ }$pid"
      done
      [[ -z $remaining ]] && break
      sleep 0.1
    done
    if [[ -n $remaining ]]; then
      plog "FATAL cgroup descendant cleanup failed pids=$remaining"
      RC=125
    else
      plog "cgroup descendant cleanup complete"
      RC=15
    fi
  fi
  CGROUP_EVENT_FAILURES=""
  while read -r key value; do
    before=${CGROUP_EVENTS_BEFORE[$key]:-0}
    if [[ $key =~ ^(high|max|oom|oom_kill)$ ]] && (( value > before )); then
      CGROUP_EVENT_FAILURES+="${CGROUP_EVENT_FAILURES:+ }$key:$before->$value"
    fi
  done < "$CGROUP_DIR/memory.events.local"
  if [[ -n $CGROUP_EVENT_FAILURES ]]; then
    plog "FATAL cgroup memory event delta $CGROUP_EVENT_FAILURES"
    RC=14
  fi
  CGROUP_CURRENT_END=$(<"$CGROUP_DIR/memory.current")
  CGROUP_PEAK_END=$(<"$CGROUP_DIR/memory.peak")
  CGROUP_SWAP_CURRENT_END=$(<"$CGROUP_DIR/memory.swap.current")
  plog "cgroup_final current_bytes=$CGROUP_CURRENT_END peak_bytes=$CGROUP_PEAK_END swap_current_bytes=$CGROUP_SWAP_CURRENT_END events=$(tr '\n' ',' <"$CGROUP_DIR/memory.events.local")"
fi
if [[ $RC == 0 && ${#FINAL_ARTIFACT_PATHS[@]} -gt 0 ]]; then
  for artifact in "${FINAL_ARTIFACT_PATHS[@]}"; do
    parent=$(realpath -e -- "$(dirname -- "$artifact")" 2>/dev/null || true)
    if [[ ! -f $artifact || -L $artifact ||
          $artifact != "$parent"/"$(basename -- "$artifact")" ]]; then
      plog "FATAL final artifact is absent or unsafe path=$artifact"
      RC=17
      break
    fi
    artifact_sha256=$(sha256sum -- "$artifact" | awk '{print $1}')
    artifact_identity=$(stat -Lc '%d:%i:%s' -- "$artifact")
    plog "final_artifact_verified path=$artifact sha256=$artifact_sha256 device_inode=$artifact_identity"
  done
fi
tail -25 "$DIR/cmd.log" >> "$MAIN" 2>/dev/null
if grep -Eiq "$USERSPACE_GPU_OOM_RE" "$MAIN" "$DIR/cmd.log"; then
  plog "FATAL CUDA userspace GPU/OOM evidence appeared during run"
  RC=16
fi
RUN_ENDED_EPOCH=$(date -u +%s)
if ! journalctl -k --since "@$RUN_STARTED_EPOCH" \
      --until "@$((RUN_ENDED_EPOCH + 1))" \
      --no-pager -o short-iso-precise >"$KERNEL_LOG" 2>&1; then
  plog "FATAL kernel journal could not be captured for Xid verification"
  RC=16
elif grep -Eiq "$KERNEL_GPU_FAULT_RE" "$KERNEL_LOG"; then
  plog "FATAL kernel GPU/OOM evidence appeared during run"
  RC=16
fi
sync -d "$KERNEL_LOG" 2>/dev/null || true
if [[ $CANDIDATE_PROVENANCE == 1 &&
      $EXECUTED_CANDIDATE_OBSERVED == 1 &&
      -z $PROVENANCE_FAILURE && $RC == 0 ]]; then
  EXECUTED_CANDIDATE_CLEAN_EXIT=1
  plog "executed candidate was verified alive at least once; no identity contradiction observed by the periodic sampler; actual cadence is recorded in samples.log; wrapper and descendant checks clean"
fi
for safety_artifact in "$SAMP" "$KERNEL_LOG"; do
  if [[ ! -f $safety_artifact || -L $safety_artifact ]]; then
    plog "FATAL wrapper safety artifact is absent or unsafe path=$safety_artifact"
    RC=17
    continue
  fi
  safety_sha256=$(sha256sum -- "$safety_artifact" | awk '{print $1}')
  safety_size=$(stat -Lc '%s' -- "$safety_artifact")
  plog "safety_artifact_verified name=$(basename -- "$safety_artifact") sha256=$safety_sha256 size=$safety_size"
done
plog "SAFE_RUN end rc=$RC killed=${KILLED:-no} (124=timeout, 137=SIGKILL/ENOMEM-adjacent)"
if [[ -n $WITNESS_NONCE ]]; then
  if [[ ! -f $WITNESS_ARTIFACT || -L $WITNESS_ARTIFACT ]]; then
    plog "FATAL witness result artifact is absent or unsafe"
    RC=17
    ARTIFACT_SHA256=missing
    ARTIFACT_IDENTITY=missing
  else
    ARTIFACT_SHA256=$(sha256sum -- "$WITNESS_ARTIFACT" | awk '{print $1}')
    ARTIFACT_IDENTITY=$(stat -Lc '%d:%i:%s' -- "$WITNESS_ARTIFACT")
  fi
  CMD_SHA256=$(sha256sum -- "$DIR/cmd.log" | awk '{print $1}')
  SAMPLES_SHA256=$(sha256sum -- "$SAMP" | awk '{print $1}')
  WITNESS_MESSAGE="W1_WITNESS nonce=$WITNESS_NONCE unit=$EXPECTED_CGROUP_UNIT binary=${CANDIDATE_HASH:-missing} environment=${EXECUTED_ENV_HASH:-missing} pid=${EXECUTED_PID:-missing} start_ticks=${EXECUTED_START_TICKS:-missing} rc=$RC killed=${KILLED:-no} cmd_sha256=$CMD_SHA256 samples_sha256=$SAMPLES_SHA256 artifact_sha256=$ARTIFACT_SHA256 artifact_identity=$ARTIFACT_IDENTITY"
  { printf '%s\n' "$WITNESS_MESSAGE"; sleep 1; } |
    /usr/bin/systemd-cat --identifier=glm52-w1-witness --priority=notice
  printf '%s\n' "$WITNESS_MESSAGE"
fi
grep MemAvailable /proc/meminfo >> "$MAIN"; sync
echo "SAFE_RUN_DONE rc=$RC killed=${KILLED:-no} dir=$DIR"
exit "$RC"
