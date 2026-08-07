#!/bin/bash
# Reviewed W7 strict/candidate/cold smoke. Every arm starts a fresh server under
# the hardened GLM containment chain. The candidate changes only the exact
# default-off restored-frontier diagnostic.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly INVOKED_SCRIPT=$0
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly MEMORY_GUARD=$REPO/scripts/03_memory_guard.py
readonly TRACE_SCORER=$REPO/scripts/83_score_w7_deployed_trace.py
readonly SCORER=$REPO/scripts/85_score_w7_resume_equivalence.py
readonly BIN=/home/bmarti44/.cache/glm52-w7-7822efd-build-v3/build1-ds4-server
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly LIVE_SOURCE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json
readonly PRIMARY_SOURCE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly POOL=$REPO/results/glm52-gates/harness/w7-production-fixture-pool-v1.json
readonly TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
readonly TOKENIZER_RUNTIME=/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2
readonly TOKENIZER_INIT=$TOKENIZER_RUNTIME/tokenizers/__init__.py
readonly TOKENIZER_NATIVE=$TOKENIZER_RUNTIME/tokenizers/tokenizers.abi3.so
readonly OUT_PARENT=/home/bmarti44/.local/state/glm52-w7-equivalence
readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock
readonly CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
readonly PORT=8097
readonly CACHE_GIB=40
readonly RANDOM_SEED_SHA256=ddc037a1c685854d872a266a00fd5268f518ea5649a38eb79a3c57e45534b415

readonly BINARY_SHA256=2c586aef10c9d9d63827e1141ad2f00ad0d20b259a62e5b53405061eb11c036c
readonly MODEL_BYTES=211075856448
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly LIVE_SHA256=d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a
readonly PRIMARY_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly POOL_SHA256=c71f1c9c90164baae00492befed68765fd9bee40fef3de8c3b291cc06794ecb9
readonly TOKENIZER_SHA256=19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
readonly TOKENIZER_INIT_SHA256=eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4
readonly TOKENIZER_NATIVE_SHA256=fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910
readonly TRACE_SCORER_SHA256=6cec5063906a52c577617b4173a1deed14d0ae2fffebff19bbef6e96442dc985
readonly SCORER_SHA256=0e497e8f425d215db122ff6f3e7cf5ede0872c5e98ab93f69b5758b0595188ae
readonly CGROUP_SHA256=e5a37b35d3ff1e8a7ee08d0f2c1396441b0dbc4abd64220389362ae6c6994c32
readonly SAFE_SHA256=6e4d382bc5e5818787af8c17aae7a0750ca3ab7b36471f21355789d194b2e801
readonly MEMORY_GUARD_SHA256=3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58

verify_file() {
  [[ $# == 2 && -f $1 && ! -L $1 ]] || return 2
  [[ $(sha256sum -- "$1" | awk '{print $1}') == "$2" ]]
}

validate_execution_authority() {
  [[ ${W7_EXECUTED_HARNESS_SHA256:-} =~ ^[0-9a-f]{64}$ &&
     ${W7_FROZEN_CANDIDATE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || return 2
  verify_file "$INVOKED_SCRIPT" "$W7_EXECUTED_HARNESS_SHA256" || return 2
  git -C "$REPO" cat-file -e "$W7_FROZEN_CANDIDATE_COMMIT^{commit}" 2>/dev/null || return 2
  git -C "$REPO" show \
    "$W7_FROZEN_CANDIDATE_COMMIT:results/glm52-gates/harness/w7_resume_equivalence_v1.sh" \
    | sha256sum | awk -v expected="$W7_EXECUTED_HARNESS_SHA256" '$1 == expected {ok=1} END{exit !ok}'
}

verify_dependencies() {
  verify_file "$BIN" "$BINARY_SHA256"
  verify_file "$LIVE_SOURCE" "$LIVE_SHA256"
  verify_file "$PRIMARY_SOURCE" "$PRIMARY_SHA256"
  verify_file "$POOL" "$POOL_SHA256"
  verify_file "$TOKENIZER" "$TOKENIZER_SHA256"
  verify_file "$TOKENIZER_INIT" "$TOKENIZER_INIT_SHA256"
  verify_file "$TOKENIZER_NATIVE" "$TOKENIZER_NATIVE_SHA256"
  verify_file "$TRACE_SCORER" "$TRACE_SCORER_SHA256"
  verify_file "$SCORER" "$SCORER_SHA256"
  verify_file "$CGROUP" "$CGROUP_SHA256"
  verify_file "$SAFE" "$SAFE_SHA256"
  verify_file "$MEMORY_GUARD" "$MEMORY_GUARD_SHA256"
  [[ -f $MODEL && ! -L $MODEL && $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]]
}

stop_server() {
  local pid=${server_pid:-}
  [[ $pid =~ ^[0-9]+$ ]] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
  server_pid=
}

inventory_kv() {
  local dir=$1 output=$2 file name digest
  : >"$output"
  while IFS= read -r -d '' file; do
    name=${file##*/}
    [[ $name =~ ^[0-9a-f]{40}\.kv$ ]] || return 2
    digest=$(sha256sum -- "$file" | awk '{print $1}')
    printf '%s  %s\n' "$digest" "$name" >>"$output"
  done < <(find "$dir" -maxdepth 1 -type f -name '*.kv' -print0 | sort -z)
  [[ -s $output ]]
}

driver() {
  [[ $# == 2 ]] || return 2
  local arm=$1 arm_out=$2 code
  [[ $arm == strict || $arm == candidate || $arm == cold ]] || return 2
  mkdir "$arm_out/kv"
  "$BIN" --cuda -m "$MODEL" -c 8192 --host 127.0.0.1 --port "$PORT" \
    --trace "$arm_out/request.trace" \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$arm_out/kv" --kv-disk-space-mb 4096 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 8 \
    >"$arm_out/server.log" 2>&1 &
  server_pid=$!
  trap stop_server EXIT INT TERM HUP
  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$PORT/v1/models" || true)
    [[ $code == 200 ]] && break
    sleep 2
  done
  [[ ${code:-} == 200 ]] || return 1
  if [[ $arm != cold ]]; then
    curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
      -o "$arm_out/live-response.json" -w '%{http_code}\n' \
      -d @"$arm_out/live-request.json" "http://127.0.0.1:$PORT/v1/completions" \
      >"$arm_out/live-http-status"
    inventory_kv "$arm_out/kv" "$arm_out/kv-before.sha256"
  fi
  curl -sS --fail-with-body --max-time 1200 -H 'Content-Type: application/json' \
    -o "$arm_out/primary-response.json" -w '%{http_code}\n' \
    -d @"$arm_out/primary-request.json" "http://127.0.0.1:$PORT/v1/completions" \
    >"$arm_out/primary-http-status"
  if [[ $arm != cold ]]; then inventory_kv "$arm_out/kv" "$arm_out/kv-after.sha256"; fi
  stop_server
  trap - EXIT INT TERM HUP
}

write_arm_metadata() {
  local arm=$1 arm_out=$2 rc=$3
  /usr/bin/python3 -I -B - "$arm" "$arm_out" "$rc" "$LIVE_SHA256" "$PRIMARY_SHA256" <<'PY'
import json, pathlib, sys
arm, out_raw, rc_raw, live_sha, primary_sha = sys.argv[1:]
requests = {"primary": primary_sha}
if arm != "cold": requests["live"] = live_sha
doc = {"schema_version": 1, "arm": arm, "containment_rc": int(rc_raw), "request_sha256": requests}
pathlib.Path(out_raw, "arm.json").write_text(json.dumps(doc, sort_keys=True) + "\n")
PY
}

score_trace() {
  local arm_out=$1
  /usr/bin/python3 -I -B "$TRACE_SCORER" \
    --trace "$arm_out/request.trace" --pool "$POOL" \
    --live-request "$arm_out/live-request.json" \
    --primary-request "$arm_out/primary-request.json" \
    --tokenizer "$TOKENIZER" --tokenizer-runtime "$TOKENIZER_RUNTIME" \
    >"$arm_out/trace-result.json" 2>"$arm_out/trace-scorer.stderr"
}

run_arm() {
  local arm=$1 root=$2 arm_out=$root/$arm tag=w7eq-${arm}-${root##*-} rc
  mkdir "$arm_out"
  install -m 0600 "$PRIMARY_SOURCE" "$arm_out/primary-request.json"
  if [[ $arm != cold ]]; then install -m 0600 "$LIVE_SOURCE" "$arm_out/live-request.json"; fi
  local candidate_env=()
  if [[ $arm == candidate ]]; then
    candidate_env=(DS4_GLM_RESTORED_FRONTIER_DIAGNOSTIC=1)
  fi
  set +e
  /usr/bin/env -i \
    HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_RUNTIME_DIR=/run/user/1000 \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    GLM_SAFE_RUN_AS_CURRENT_USER=1 GLM_SAFE_MEMORY_HIGH_GIB=78 \
    GLM_SAFE_KILL_FLOOR_GIB=24 GLM_SAFE_MIN_START_GIB=110 GLM_SAFE_TIMEOUT_S=2400 \
    GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 GLM_SAFE_EXPECTED_BINARY_SHA256=$BINARY_SHA256 \
    GLM_CANDIDATE_SRC=/tmp/glm52-w7-build1.ob4Q0O/src \
    DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GIB DS4_CUDA_EXPERT_CACHE_PIN=1 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_GLM_SYNC_TRACE=1 \
    DS4_GLM_LOGIT_DUMP="$arm_out/logits" DS4_GLM_LOGIT_DUMP_ALL=1 \
    "${candidate_env[@]}" \
    DS4_LOCK_FILE=$ENGINE_LOCK DS4_LOCK_EXPECTED_DEV_INO=$engine_lock_identity \
      "$CGROUP" --tag "$tag" -- /usr/bin/env \
        W7_EXECUTED_HARNESS_SHA256="$W7_EXECUTED_HARNESS_SHA256" \
        W7_FROZEN_CANDIDATE_COMMIT="$W7_FROZEN_CANDIDATE_COMMIT" \
        /usr/bin/bash "$INVOKED_SCRIPT" --driver "$arm" "$arm_out" \
      >"$arm_out/containment.stdout" 2>"$arm_out/containment.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$arm_out/containment.rc"
  write_arm_metadata "$arm" "$arm_out" "$rc"
  (( rc == 0 )) || return 2
  if [[ $arm != cold ]]; then score_trace "$arm_out"; fi
}

if [[ ${1:-} == --self-test ]]; then
  verify_dependencies
  python3 -m unittest scripts.tests.test_w7_resume_equivalence_scorer >/dev/null
  echo W7_EQUIVALENCE_SELFTEST_OK
  exit 0
elif [[ ${1:-} == --driver ]]; then
  shift
  validate_execution_authority
  driver "$@"
  exit $?
fi

[[ $# == 0 && $(id -un) == bmarti44 ]] || exit 2
validate_execution_authority
verify_dependencies
[[ -z $(git -C "$REPO" status --porcelain) ]] || exit 2
! pgrep -x ds4-server >/dev/null && ! pgrep -x fio >/dev/null || exit 75
[[ ! -L $ENGINE_LOCK && -f $ENGINE_LOCK &&
   $(stat -Lc '%U:%G:%a:%h' -- "$ENGINE_LOCK") == bmarti44:bmarti44:600:1 ]] || exit 2
/usr/bin/flock -n -E 75 -- "$ENGINE_LOCK" /usr/bin/true || exit 75
readonly engine_lock_identity=$(stat -Lc '%d:%i' -- "$ENGINE_LOCK")
swap_used_kib=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
(( swap_used_kib < 1048576 )) || exit 8
/usr/bin/python3 -I -B "$MEMORY_GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 0 >/dev/null
mkdir -p "$OUT_PARENT" "$CRASH_ROOT"
nonce=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
readonly attempt_out=$OUT_PARENT/attempt-$nonce
mkdir "$attempt_out"
/usr/bin/python3 -I -B - "$RANDOM_SEED_SHA256" "$attempt_out/arm-order" <<'PY'
import hashlib, pathlib, sys
seed = bytes.fromhex(sys.argv[1])
arms = sorted(("strict", "candidate", "cold"), key=lambda arm: hashlib.sha256(seed + arm.encode()).digest())
pathlib.Path(sys.argv[2]).write_text("\n".join(arms) + "\n")
PY
while IFS= read -r arm; do run_arm "$arm" "$attempt_out"; done <"$attempt_out/arm-order"
verify_dependencies
/usr/bin/python3 -I -B "$SCORER" \
  --strict "$attempt_out/strict" --candidate "$attempt_out/candidate" \
  --cold "$attempt_out/cold" --output "$attempt_out/summary.json" \
  >"$attempt_out/scorer.stdout" 2>"$attempt_out/scorer.stderr"
sync -f "$attempt_out"
echo "W7 equivalence evidence: $attempt_out"
