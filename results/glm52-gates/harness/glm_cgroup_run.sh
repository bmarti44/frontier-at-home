#!/bin/bash
# Default-off outer containment for authoritative GLM runs. This must execute
# as the logged-in benchmark owner; the dsv4 account has no delegated cgroup.
set -Eeuo pipefail
umask 077

[[ $# -ge 3 && ${1:-} == --tag && -n ${2:-} ]] || {
  echo "usage: $0 --tag NAME -- command..." >&2
  exit 2
}
TAG=$2
shift 2
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$ ]] || {
  echo "invalid cgroup tag" >&2
  exit 2
}
[[ ${1:-} == -- ]] && shift
(( $# > 0 )) || { echo "missing cgroup command" >&2; exit 2; }
ROOT_AUTHORITY=${GLM_W1_ROOT_AUTHORITY:-0}
[[ $ROOT_AUTHORITY =~ ^[01]$ ]] || exit 2
if [[ $ROOT_AUTHORITY == 1 ]]; then
  [[ $(id -u) == 0 ]] || {
    echo "root W1 authority requires root" >&2
    exit 2
  }
else
  [[ $(id -u) != 0 && $(id -un) != dsv4 ]] || {
    echo "cgroup launcher must run as the logged-in benchmark owner" >&2
    exit 2
  }
fi
RUN_CWD=$(pwd -P)
[[ $RUN_CWD == /* && $RUN_CWD != *$'\n'* && -x $RUN_CWD ]] || {
  echo "invalid launch working directory" >&2
  exit 2
}
PINNED_SAFE_PATH=${GLM_SAFE_PINNED_SAFE_PATH:-}
PINNED_SAFE_SHA256=${GLM_SAFE_PINNED_SAFE_SHA256:-}
if [[ -n $PINNED_SAFE_PATH || -n $PINNED_SAFE_SHA256 ]]; then
  [[ $PINNED_SAFE_PATH =~ ^/proc/[1-9][0-9]*/fd/[0-9]+$ &&
     $PINNED_SAFE_SHA256 =~ ^[0-9a-f]{64}$ && -f $PINNED_SAFE_PATH &&
     $(sha256sum -- "$PINNED_SAFE_PATH" | awk '{print $1}') == "$PINNED_SAFE_SHA256" ]] || {
    echo "invalid pinned safe-run script" >&2
    exit 2
  }
fi

KILL_FLOOR_GIB=${GLM_SAFE_KILL_FLOOR_GIB:-18}
TIMEOUT_S=${GLM_SAFE_TIMEOUT_S:-2400}
EVIDENCE_DIR=${GLM_SAFE_EVIDENCE_DIR:-}
RUN_AS_CURRENT_USER=${GLM_SAFE_RUN_AS_CURRENT_USER:-0}
MEMORY_HIGH_GIB=${GLM_SAFE_MEMORY_HIGH_GIB:-}
[[ $KILL_FLOOR_GIB =~ ^[0-9]{1,2}$ && $TIMEOUT_S =~ ^[0-9]{1,4}$ ]] || {
  echo "invalid cgroup resource configuration" >&2
  exit 2
}
KILL_FLOOR_GIB=$((10#$KILL_FLOOR_GIB))
TIMEOUT_S=$((10#$TIMEOUT_S))
(( KILL_FLOOR_GIB >= 18 && KILL_FLOOR_GIB <= 64 )) || exit 2
if (( TIMEOUT_S < 1 || TIMEOUT_S > 9000 )); then
  echo "invalid GLM_SAFE_TIMEOUT_S" >&2
  exit 2
fi
[[ $RUN_AS_CURRENT_USER =~ ^[01]$ ]] || {
  echo "invalid GLM_SAFE_RUN_AS_CURRENT_USER" >&2
  exit 2
}
if [[ -n $MEMORY_HIGH_GIB ]]; then
  [[ $MEMORY_HIGH_GIB =~ ^[0-9]{2,3}$ ]] || {
    echo "invalid GLM_SAFE_MEMORY_HIGH_GIB" >&2
    exit 2
  }
  MEMORY_HIGH_GIB=$((10#$MEMORY_HIGH_GIB))
  (( MEMORY_HIGH_GIB >= 32 && MEMORY_HIGH_GIB <= 101 )) || {
    echo "invalid GLM_SAFE_MEMORY_HIGH_GIB" >&2
    exit 2
  }
fi
if [[ $RUN_AS_CURRENT_USER == 1 ]]; then
  [[ $(id -un) == bmarti44 ]] || {
    echo "current-user GLM mode requires bmarti44" >&2
    exit 2
  }
  [[ -z $EVIDENCE_DIR ]] || {
    echo "current-user GLM mode manages evidence directly" >&2
    exit 2
  }
fi
if [[ -n $EVIDENCE_DIR ]]; then
  [[ $EVIDENCE_DIR =~ ^/home/dsv4/ds4-project/glm52-confirm-[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ||
     $EVIDENCE_DIR =~ ^/home/dsv4/ds4-project/glm52-decisive-[A-Za-z0-9][A-Za-z0-9._-]{0,79}/[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
    echo "invalid GLM_SAFE_EVIDENCE_DIR" >&2
    exit 2
  }
fi

available_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [[ -n $MEMORY_HIGH_GIB ]]; then
  # The caller derives MemoryHigh from a guarded cache-off RSS measurement,
  # the exact arena allocation, and explicit overhead. MemoryMax adds only a
  # bounded 2 GiB excursion and may never consume the whole-system kill floor.
  high_mib=$((MEMORY_HIGH_GIB * 1024))
  max_mib=$((high_mib + 2048))
  (( max_mib + KILL_FLOOR_GIB * 1024 <= available_mib )) || {
    echo "profile memory envelope exceeds safe host budget" >&2
    exit 8
  }
else
  max_mib=$((available_mib - KILL_FLOOR_GIB * 1024 - 4096))
  high_mib=$((max_mib - 4096))
fi
(( high_mib >= 32768 && max_mib > high_mib )) || {
  echo "insufficient memory for bounded cgroup" >&2
  exit 8
}

UNIT="glm52-${TAG//./-}-$$"
if [[ -n $PINNED_SAFE_PATH ]]; then
  SAFE=$PINNED_SAFE_PATH
elif [[ $ROOT_AUTHORITY == 1 ]]; then
  SAFE=$(dirname -- "$(readlink -f -- "$0")")/glm_safe_run.sh
else
  SAFE=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/glm_safe_run.sh
fi
EVIDENCE_EXPORT=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/glm_evidence_export.py
UNIT_ACTIVE=0
export_evidence() {
  [[ $RUN_AS_CURRENT_USER == 0 ]] || return 0
  [[ -n $EVIDENCE_DIR ]] || return 0
  local evidence_export_rc=0
  local crash_dir
  sudo -n -u dsv4 /usr/bin/python3 "$EVIDENCE_EXPORT" "$EVIDENCE_DIR" ||
    evidence_export_rc=1
  while IFS= read -r crash_dir; do
    [[ -n $crash_dir ]] || continue
    sudo -n -u dsv4 /usr/bin/python3 "$EVIDENCE_EXPORT" "$crash_dir" ||
      evidence_export_rc=1
  done < <(
    sudo -n -u dsv4 find /home/dsv4/ds4-project/glm52-crashlog \
      -mindepth 1 -maxdepth 1 -type d -name "*-$TAG" -print
  )
  if (( evidence_export_rc != 0 )); then
    echo "evidence export failed for $EVIDENCE_DIR" >&2
    return 1
  fi
  return 0
}
stop_unit() {
  trap - INT TERM HUP
  if (( UNIT_ACTIVE )); then
    if [[ $ROOT_AUTHORITY == 1 ]]; then
      systemctl stop "$UNIT.service" >/dev/null 2>&1 || true
    else
      systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
    fi
  fi
}
handle_signal() {
  local signal_rc=$1
  stop_unit
  export_evidence || true
  exit "$signal_rc"
}
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP

if [[ $RUN_AS_CURRENT_USER == 1 ]]; then
  run_home=/home/bmarti44
else
  run_home=/home/dsv4
fi
env_args=(
  "HOME=$run_home"
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  GLM_SAFE_REQUIRE_CGROUP=1
  "GLM_SAFE_CGROUP_UNIT=$UNIT"
  "GLM_SAFE_RUN_AS_CURRENT_USER=$RUN_AS_CURRENT_USER"
)
for name in \
  GLM_CANDIDATE_SRC GLM_PORT GLM_EXPERT_CACHE_GB \
  GLM_REQUIRE_TOKEN_TIMING_LOG DS4_CUDA_IQ2_DOWN_REFERENCE \
  DS4_CUDA_MOE_NO_EXPERT_TILES DS4_CUDA_MOE_NO_ATOMIC_DOWN \
  DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS \
  DS4_LOCK_EXPECTED_DEV_INO DS4_LOCK_FILE \
  DS4_CUDA_EXPERT_CACHE_GB DS4_CUDA_EXPERT_CACHE_PIN \
  DS4_CUDA_EXPERT_CACHE_SLRU DS4_CUDA_FETCH_THREADS \
  DS4_CUDA_EXPERT_SLAB_PATH DS4_CUDA_EXPERT_SLAB_SHA256 \
  DS4_CUDA_EXPERT_SLAB_MODEL_SHA256 DS4_CUDA_EXPERT_SLAB_TRACE \
  DS4_CUDA_EXPERT_SLAB_AUTH_TRACE DS4_CUDA_EXPERT_SLAB_PREFETCH_SHA \
  DS4_CUDA_LOAD_PROFILE DS4_TOKEN_TIMING_LOG DS4_GLM_TP_DEBUG \
  DS4_GLM_PREFETCH DS4_GLM_PREFETCH_SHARED_CORRECTION \
  DS4_GLM_PREFETCH_THREADS \
  DS4_GLM_PREDACC_SHARED \
  DS4_W7_PINNED_HARNESS_SHA256 \
  DS4_W7_CANDIDATE_HASH \
  DS4_W7_SEALED_LIVE_PATH DS4_W7_SEALED_PRIMARY_PATH \
  DS4_CUDA_STABLE_MODEL_REMAP \
  DS4_GLM_SYNC_TRACE DS4_GLM_LOGIT_DUMP DS4_GLM_LOGIT_DUMP_ALL \
  DS4_GLM_W9_CAPTURE_DIR \
  DS4_GLM_RESTORED_FRONTIER_DIAGNOSTIC \
  DS4_GLM_UNION_TRACE_CORPUS \
  DS4_GLM_STREAMING_TOKEN_PREFILL_MAX \
  DS4_JSON_REPLACE_INVALID_UTF8 \
  DS4_METAL_GRAPH_DUMP_PREFIX DS4_METAL_GRAPH_DUMP_NAME \
  DS4_METAL_GRAPH_DUMP_LAYER \
  DS4_GLM_COMPACT_CACHE_F16 DS4_GLM_COMPACT_CACHE_E4M3_FAKE \
  DS4_GLM_COMPACT_CACHE_INT8_FAKE \
  DS4_GLM_COMPACT_CACHE_AFFINE_INT8 \
  DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE \
  DS4_GLM_CKV_NVME DS4_GLM_CKV_NVME_EXACT \
  DS4_GLM_CKV_DIR DS4_GLM_CKV_MODEL_SHA256 \
  DS4_GLM_CKV_RUN_NONCE \
  DS4_GLM_CKV_MAX_GIB DS4_GLM_CKV_TRACE_PATH \
  DS4_GLM_CKV_TRACE_SAMPLE_POSITIONS DS4_GLM_CKV_TRACE_MAX_RECORDS \
  GLM_SAFE_LOG_CANDIDATE_PROVENANCE GLM_SAFE_EXPECTED_BINARY_SHA256 \
  GLM_SAFE_PROVENANCE_ENV_ALLOWLIST GLM_SAFE_EXPECTED_ENV_SHA256 \
  GLM_SAFE_FINAL_ARTIFACTS GLM_SAFE_DONE_DIGESTS \
  GLM_SAFE_WITNESS_NONCE \
  GLM_SAFE_WITNESS_ARTIFACT \
  GLM_W1_ROOT_AUTHORITY GLM_SAFE_CRASH_ROOT \
  GLM_SAFE_VLIMIT_KB GLM_SAFE_KILL_FLOOR_GIB GLM_SAFE_MIN_START_GIB \
  GLM_SAFE_TIMEOUT_S
do
  if [[ -v $name ]]; then
    env_args+=("$name=${!name}")
  fi
done

if [[ $ROOT_AUTHORITY == 1 ]]; then
  # The immutable root submitter owns the global inference lock for the whole
  # campaign. Re-locking it in this separate process would fail every attempt.
  contained_command=(
    /usr/bin/env -i "${env_args[@]}"
    /usr/bin/bash "$SAFE" --tag "$TAG" -- "$@"
  )
elif [[ $RUN_AS_CURRENT_USER == 1 ]]; then
  contained_command=(
    /usr/bin/env -i "${env_args[@]}"
    /usr/bin/flock -n -E 75 /run/lock/frontier-at-home/inference.lock
    /usr/bin/bash "$SAFE" --tag "$TAG" -- "$@"
  )
else
  contained_command=(
    /usr/bin/sudo -n -u dsv4 -- /usr/bin/env -i "${env_args[@]}"
    /usr/bin/flock -n -E 75 /run/lock/frontier-at-home/inference.lock
    /usr/bin/bash "$SAFE" --tag "$TAG" -- "$@"
  )
fi

UNIT_ACTIVE=1
set +e
if [[ $ROOT_AUTHORITY == 1 ]]; then
  systemd-run --wait --collect --pipe --quiet \
    --expand-environment=no \
    --uid=dsv4 --gid=dsv4 \
    --working-directory="$RUN_CWD" \
    --unit="$UNIT" --service-type=exec \
    -p KillMode=control-group \
    -p SendSIGKILL=yes \
    -p TimeoutStopSec=45s \
    -p "RuntimeMaxSec=$((TIMEOUT_S + 60))s" \
    -p MemoryAccounting=yes \
    -p "MemoryHigh=${high_mib}M" \
    -p "MemoryMax=${max_mib}M" \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    -p TasksMax=4096 \
    -p ProtectHome=read-only \
    -p NoNewPrivileges=yes \
    -- "${contained_command[@]}"
else
  systemd-run --user --wait --collect --pipe --quiet \
    --expand-environment=no \
    --working-directory="$RUN_CWD" \
    --unit="$UNIT" --service-type=exec \
    -p KillMode=control-group \
    -p SendSIGKILL=yes \
    -p TimeoutStopSec=45s \
    -p "RuntimeMaxSec=$((TIMEOUT_S + 60))s" \
    -p MemoryAccounting=yes \
    -p "MemoryHigh=${high_mib}M" \
    -p "MemoryMax=${max_mib}M" \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    -p TasksMax=4096 \
    -- "${contained_command[@]}"
fi
command_rc=$?
set -e
set +e
if [[ $ROOT_AUTHORITY == 1 ]]; then
  systemctl stop "$UNIT.service" >/dev/null 2>&1
else
  systemctl --user stop "$UNIT.service" >/dev/null 2>&1
fi
set -e
UNIT_ACTIVE=0

# Export only the exact preregistered confirmation tree and this tag's safety
# logs. Permission changes run after every outcome and never rewrite evidence
# bytes. An export failure must not mask an existing command failure.
if ! export_evidence; then
  (( command_rc != 0 )) || exit 16
fi
exit "$command_rc"
