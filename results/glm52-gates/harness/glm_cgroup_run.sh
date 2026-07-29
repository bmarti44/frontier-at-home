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
[[ $(id -u) != 0 && $(id -un) != dsv4 ]] || {
  echo "cgroup launcher must run as the logged-in benchmark owner" >&2
  exit 2
}

KILL_FLOOR_GIB=${GLM_SAFE_KILL_FLOOR_GIB:-18}
TIMEOUT_S=${GLM_SAFE_TIMEOUT_S:-2400}
EVIDENCE_DIR=${GLM_SAFE_EVIDENCE_DIR:-}
RUN_AS_CURRENT_USER=${GLM_SAFE_RUN_AS_CURRENT_USER:-0}
[[ $KILL_FLOOR_GIB =~ ^[0-9]{1,2}$ && $TIMEOUT_S =~ ^[0-9]{1,4}$ ]] || {
  echo "invalid cgroup resource configuration" >&2
  exit 2
}
KILL_FLOOR_GIB=$((10#$KILL_FLOOR_GIB))
TIMEOUT_S=$((10#$TIMEOUT_S))
(( KILL_FLOOR_GIB >= 18 && KILL_FLOOR_GIB <= 64 )) || exit 2
(( TIMEOUT_S >= 1 && TIMEOUT_S <= 3600 )) || exit 2
[[ $RUN_AS_CURRENT_USER =~ ^[01]$ ]] || {
  echo "invalid GLM_SAFE_RUN_AS_CURRENT_USER" >&2
  exit 2
}
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
max_mib=$((available_mib - KILL_FLOOR_GIB * 1024 - 4096))
high_mib=$((max_mib - 4096))
(( high_mib >= 32768 && max_mib > high_mib )) || {
  echo "insufficient memory for bounded cgroup" >&2
  exit 8
}

UNIT="glm52-${TAG//./-}-$$"
SAFE=/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/glm_safe_run.sh
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
    systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
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
  DS4_LOCK_FILE \
  DS4_CUDA_EXPERT_CACHE_GB DS4_CUDA_EXPERT_CACHE_PIN \
  DS4_CUDA_EXPERT_CACHE_SLRU DS4_CUDA_FETCH_THREADS \
  DS4_GLM_COMPACT_CACHE_F16 DS4_GLM_COMPACT_CACHE_E4M3_FAKE \
  DS4_GLM_COMPACT_CACHE_INT8_FAKE \
  DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE \
  GLM_SAFE_LOG_CANDIDATE_PROVENANCE GLM_SAFE_EXPECTED_BINARY_SHA256 \
  GLM_SAFE_PROVENANCE_ENV_ALLOWLIST GLM_SAFE_EXPECTED_ENV_SHA256 \
  GLM_SAFE_VLIMIT_KB GLM_SAFE_KILL_FLOOR_GIB GLM_SAFE_MIN_START_GIB \
  GLM_SAFE_TIMEOUT_S
do
  if [[ -v $name ]]; then
    env_args+=("$name=${!name}")
  fi
done

if [[ $RUN_AS_CURRENT_USER == 1 ]]; then
  contained_command=(
    /usr/bin/env -i "${env_args[@]}"
    /usr/bin/flock -n -E 75 "/run/user/$UID/glm52-inference.lock"
    /usr/bin/bash "$SAFE" --tag "$TAG" -- "$@"
  )
else
  contained_command=(
    /usr/bin/sudo -n -u dsv4 -- /usr/bin/env -i "${env_args[@]}"
    /usr/bin/flock -n -E 75 /run/dsv4/inference.lock
    /usr/bin/bash "$SAFE" --tag "$TAG" -- "$@"
  )
fi

UNIT_ACTIVE=1
set +e
systemd-run --user --wait --collect --pipe --quiet \
  --expand-environment=no \
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
command_rc=$?
set -e
set +e
systemctl --user stop "$UNIT.service" >/dev/null 2>&1
set -e
UNIT_ACTIVE=0

# Export only the exact preregistered confirmation tree and this tag's safety
# logs. Permission changes run after every outcome and never rewrite evidence
# bytes. An export failure must not mask an existing command failure.
if ! export_evidence; then
  (( command_rc != 0 )) || exit 16
fi
exit "$command_rc"
