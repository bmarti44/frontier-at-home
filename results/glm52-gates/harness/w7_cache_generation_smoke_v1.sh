#!/usr/bin/env bash
# W7.1 one-arm diagnostic. With no arguments this launcher creates a fresh
# evidence directory and executes its --driver arm through the reviewed GLM
# cgroup + safe-run containment path.
set -Eeuo pipefail
umask 077

readonly ROOT=/home/bmarti44/spark-deepseek-v4-flash
readonly HARNESS="$ROOT/results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh"
readonly CGROUP="$ROOT/results/glm52-gates/harness/glm_cgroup_run.sh"
readonly SAFE="$ROOT/results/glm52-gates/harness/glm_safe_run.sh"
readonly SCORER="$ROOT/scripts/89_score_w7_cache_generation.py"
readonly BIN=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6/ds4-server
readonly CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6
readonly BINARY_SHA256=eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_BYTES=211075856448
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly LIVE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json
readonly LIVE_SHA256=d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a
readonly PRIMARY=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly PRIMARY_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly ENV_SHA256=ea8cc542bf2138646cb5bb3d38c9f7e7d88eef3e5a8fe7faf13074463f5a5e64
readonly PORT=8097
server_pid=
attempt=
out=
candidate=
outer_finalized=0
failure_reason=before-attempt
harness_fd=
cgroup_fd=
safe_fd=
scorer_fd=
harness_fd_path=
cgroup_fd_path=
safe_fd_path=
scorer_fd_path=
harness_sha256=
cgroup_sha256=
safe_sha256=
scorer_sha256=

if [[ $0 == "$HARNESS" && ! -L $0 && $(readlink -f -- "$0") == "$HARNESS" ]]; then
  :
elif [[ $0 =~ ^/proc/[1-9][0-9]*/fd/[0-9]+$ &&
        ${DS4_W7_PINNED_HARNESS_SHA256:-} =~ ^[0-9a-f]{64}$ &&
        $(sha256sum -- "$0" | awk '{print $1}') == "$DS4_W7_PINNED_HARNESS_SHA256" ]]; then
  :
else
  exit 2
fi

verify_file() {
  [[ -f $1 && ! -L $1 && $(sha256sum -- "$1" | awk '{print $1}') == "$2" ]]
}

verify_dependencies_fast() {
  verify_file "$BIN" "$BINARY_SHA256"
  [[ -f $MODEL && ! -L $MODEL && $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]]
  verify_file "$LIVE" "$LIVE_SHA256"
  verify_file "$PRIMARY" "$PRIMARY_SHA256"
  [[ -f $CGROUP && ! -L $CGROUP && -f $SAFE && ! -L $SAFE && -f $SCORER && ! -L $SCORER ]]
}

verify_reviewed_sources() {
  local candidate=$1 path
  [[ $candidate =~ ^[0-9a-f]{40}$ ]]
  git -C "$ROOT" cat-file -e "$candidate^{commit}"
  git -C "$ROOT" merge-base --is-ancestor "$candidate" HEAD
  for path in \
    results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh \
    results/glm52-gates/harness/glm_cgroup_run.sh \
    results/glm52-gates/harness/glm_safe_run.sh \
    scripts/89_score_w7_cache_generation.py
  do
    [[ -f $ROOT/$path && ! -L $ROOT/$path ]]
    git -C "$ROOT" show "$candidate:$path" | cmp -s - "$ROOT/$path"
  done
}

pin_reviewed_scripts() {
  local candidate=$1 path expected
  exec {harness_fd}<"$HARNESS"
  exec {cgroup_fd}<"$CGROUP"
  exec {safe_fd}<"$SAFE"
  exec {scorer_fd}<"$SCORER"
  harness_fd_path="/proc/$$/fd/$harness_fd"
  cgroup_fd_path="/proc/$$/fd/$cgroup_fd"
  safe_fd_path="/proc/$$/fd/$safe_fd"
  scorer_fd_path="/proc/$$/fd/$scorer_fd"
  for path in \
    "results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh:$harness_fd_path:harness_sha256" \
    "results/glm52-gates/harness/glm_cgroup_run.sh:$cgroup_fd_path:cgroup_sha256" \
    "results/glm52-gates/harness/glm_safe_run.sh:$safe_fd_path:safe_sha256" \
    "scripts/89_score_w7_cache_generation.py:$scorer_fd_path:scorer_sha256"
  do
    IFS=: read -r tracked descriptor variable <<<"$path"
    git -C "$ROOT" show "$candidate:$tracked" | cmp -s - "$descriptor"
    expected=$(sha256sum -- "$descriptor" | awk '{print $1}')
    printf -v "$variable" '%s' "$expected"
  done
}

verify_driver_containment() {
  local unit=${GLM_SAFE_CGROUP_UNIT:-} path dir high max swap oom_group
  [[ ${GLM_SAFE_REQUIRE_CGROUP:-} == 1 ]]
  [[ $unit =~ ^glm52-w7-c10-[0-9a-f]{12}-[0-9]+$ ]]
  path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  [[ $path == */"$unit.service" ]]
  dir=/sys/fs/cgroup$path
  [[ -r $dir/memory.high && -r $dir/memory.max && -r $dir/memory.swap.max && -r $dir/memory.oom.group ]]
  read -r high <"$dir/memory.high"
  read -r max <"$dir/memory.max"
  read -r swap <"$dir/memory.swap.max"
  read -r oom_group <"$dir/memory.oom.group"
  [[ $high == 83751862272 && $max == 85899345920 && $swap == 0 && $oom_group == 1 ]]
}

sync_parent() {
  sync -d "$1" 2>/dev/null || sync
  sync -f "$(dirname -- "$1")" 2>/dev/null || sync
}

write_child_exit() {
  local out=$1 exit_status=$2
  python3 - "$out/child-exit.json" "$exit_status" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = {"shutdown_requested": True, "forced_kill": False, "exit_status": int(sys.argv[2])}
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(data, f, sort_keys=True, separators=(",", ":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
dfd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
PY
}

stop_server_gracefully() {
  local out=$1 rc
  [[ ${server_pid:-} =~ ^[0-9]+$ ]] || return 1
  kill -TERM "$server_pid" 2>/dev/null || return 1
  for _ in $(seq 1 300); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$server_pid" 2>/dev/null && return 1
  set +e
  wait "$server_pid"
  rc=$?
  set -e
  server_pid=
  write_child_exit "$out" "$rc"
  [[ $rc == 0 ]]
}

cleanup_driver() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ ${server_pid:-} =~ ^[0-9]+$ ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 300); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 0.1
    done
    wait "$server_pid" 2>/dev/null || true
  fi
  exit "$rc"
}

run_driver() {
  [[ $# == 1 ]]
  local out=$1 code= model_hash before after model_fd_path
  verify_driver_containment
  [[ $out =~ ^/home/bmarti44/\.local/state/glm52-w7-cache-generation/attempt-[0-9a-f]{32}/on$ ]]
  [[ -d $out && ! -L $out && -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]
  [[ ${DS4_CUDA_STABLE_MODEL_REMAP:-} == 1 ]]
  verify_dependencies_fast
  mkdir "$out/kv"

  # Keep one verified model description open while the child opens the same
  # inode via procfs; a path replacement cannot change the bytes served.
  exec {model_fd}<"$MODEL"
  model_fd_path="/proc/$$/fd/$model_fd"
  before=$(stat -Lc '%d:%i:%s' -- "$model_fd_path")
  [[ ${before##*:} == "$MODEL_BYTES" ]]
  model_hash=$(sha256sum -- "$model_fd_path" | awk '{print $1}')
  [[ $model_hash == "$MODEL_SHA256" ]]
  python3 - "$out/model.identity.json" "$before" "$model_hash" "$model_fd_path" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1]); dev, ino, size = sys.argv[2].split(":")
obj = {"bytes": int(size), "device": int(dev), "inode": int(ino), "sha256": sys.argv[3], "executed_path": sys.argv[4]}
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(obj, f, sort_keys=True, separators=(",", ":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
PY

  "$BIN" --cuda -m "$model_fd_path" -c 8192 --host 127.0.0.1 --port "$PORT" \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$out/kv" --kv-disk-space-mb 4096 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 8 \
    >"$out/server.log" 2>&1 &
  server_pid=$!
  trap cleanup_driver EXIT INT TERM HUP

  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$PORT/v1/models" || true)
    [[ $code == 200 ]] && break
    sleep 1
  done
  [[ $code == 200 ]]

  curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
    -o "$out/live-response.json" -w '%{http_code}\n' -d @"$LIVE" \
    "http://127.0.0.1:$PORT/v1/completions" >"$out/live-http-status"
  curl -sS --fail-with-body --max-time 1200 -H 'Content-Type: application/json' \
    -o "$out/primary-response.json" -w '%{http_code}\n' -d @"$PRIMARY" \
    "http://127.0.0.1:$PORT/v1/completions" >"$out/primary-http-status"

  stop_server_gracefully "$out"
  trap - EXIT INT TERM HUP
  after=$(stat -Lc '%d:%i:%s' -- "$model_fd_path")
  [[ $after == "$before" ]]
  sync -f "$out"
}

publish_outer_evidence() {
  local attempt=$1 out=$2 containment_rc=$3 crash_dir=$4 candidate=$5 score_rc execution_head
  execution_head=$(git -C "$ROOT" rev-parse HEAD)
  set +e
  python3 - "$scorer_fd_path" "$attempt" "$out" "$crash_dir" "$candidate" "$execution_head" \
    "$BINARY_SHA256" "$MODEL_SHA256" "$MODEL_BYTES" "$LIVE_SHA256" "$PRIMARY_SHA256" "$ENV_SHA256" \
    "$scorer_sha256" "$harness_sha256" "$cgroup_sha256" "$safe_sha256" <<'PY'
import importlib.machinery, importlib.util, pathlib, sys
scorer_path=pathlib.Path(sys.argv[1]); attempt=pathlib.Path(sys.argv[2]); out=pathlib.Path(sys.argv[3]); crash=pathlib.Path(sys.argv[4])
candidate, execution_head, binary, model, model_bytes, live, primary, environment, scorer_sha, harness_sha, cgroup_sha, safe_sha=sys.argv[5:]
loader=importlib.machinery.SourceFileLoader("w7_scorer",str(scorer_path)); spec=importlib.util.spec_from_loader(loader.name,loader); module=importlib.util.module_from_spec(spec); loader.exec_module(module)
identities={"candidate_hash":candidate,"execution_head":execution_head,"binary_sha256":binary,
 "model_sha256":model,"model_bytes":int(model_bytes),"live_request_sha256":live,
 "primary_request_sha256":primary,"executed_environment_sha256":environment,
 "scorer_sha256":scorer_sha,"harness_sha256":harness_sha,"cgroup_sha256":cgroup_sha,
 "safe_run_sha256":safe_sha,"containment":{"memory_high_gib":78,"memory_max_gib":80,
 "kill_floor_gib":24,"minimum_start_gib":110,"timeout_seconds":2400,"swap_max":0}}
result=module.score_and_publish_bound_attempt(attempt=attempt,out=out,crash_dir=crash,evidence_dir=out/"evidence",identities=identities)
raise SystemExit(0 if result["verdict"] == "PASS" else 1)
PY
  score_rc=$?
  set -e
  [[ -d $out/evidence ]] && outer_finalized=1
  [[ $containment_rc == 0 && $score_rc == 0 ]]
}

finalize_outer() {
  local original_rc=$? final_rc
  trap - EXIT
  [[ $original_rc != 0 ]] || original_rc=1
  if [[ ${outer_finalized:-0} == 0 && -n ${attempt:-} && -n ${out:-} && -d $attempt && ! -L $attempt ]]; then
    set +e
    if [[ ! -e $out ]]; then
      mkdir -m 0700 "$out"
    fi
    if [[ ! -d $out || -L $out ]]; then
      exit 125
    fi
    python3 - "$scorer_fd_path" "$attempt" "$out" "${candidate:-unknown}" "$failure_reason" "$original_rc" \
      "$BINARY_SHA256" "$MODEL_SHA256" "$MODEL_BYTES" "$LIVE_SHA256" "$PRIMARY_SHA256" "$ENV_SHA256" \
      "$scorer_sha256" "$harness_sha256" "$cgroup_sha256" "$safe_sha256" <<'PY'
import hashlib, importlib.machinery, importlib.util, json, pathlib, subprocess, sys
scorer_path, attempt, out = map(pathlib.Path, sys.argv[1:4]); candidate, reason=sys.argv[4:6]; rc=int(sys.argv[6])
binary, model, model_bytes, live, primary, environment, scorer_sha, harness_sha, cgroup_sha, safe_sha=sys.argv[7:]
loader=importlib.machinery.SourceFileLoader("w7_scorer",str(scorer_path)); spec=importlib.util.spec_from_loader(loader.name,loader); module=importlib.util.module_from_spec(spec); loader.exec_module(module)
def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
artifacts={}
rows=[]
for name in ("containment.stdout","containment.stderr","containment.rc"):
    path=attempt/name
    if path.is_file() and not path.is_symlink():
        artifacts[name]=sha(path)
        rows.append({"source":name,"text":path.read_text(encoding="utf-8",errors="strict")})
manifest={"schema":"glm52-w7-runtime-v3","candidate_hash":candidate,
 "execution_head":subprocess.check_output(["git","-C","/home/bmarti44/spark-deepseek-v4-flash","rev-parse","HEAD"],text=True).strip(),
 "binary_sha256":binary,"model_sha256":model,"model_bytes":int(model_bytes),
 "live_request_sha256":live,"primary_request_sha256":primary,
 "executed_environment_sha256":environment,"scorer_sha256":scorer_sha,
 "harness_sha256":harness_sha,"cgroup_sha256":cgroup_sha,"safe_run_sha256":safe_sha,
 "containment":{"memory_high_gib":78,"memory_max_gib":80,"kill_floor_gib":24,
 "minimum_start_gib":110,"timeout_seconds":2400,"swap_max":0},
 "purpose":"failed W7 production-path diagnostic","artifacts":artifacts}
summary={"checks":{"runtime_completed":False,"evidence_finalizer_completed":True},"observed":{"failure_reason":reason,"outer_exit_status":rc},"verdict":"FAIL"}
module.publish_triplet_atomic(out/"evidence",manifest,rows,summary)
PY
    final_rc=$?
    set -e
    [[ $final_rc == 0 ]] || original_rc=125
  fi
  exit "$original_rc"
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]]
  verify_dependencies_fast
  echo W7_CACHE_GENERATION_SMOKE_SELFTEST_OK
  exit 0
fi

if [[ ${1:-} == --driver ]]; then
  [[ $# == 3 && $2 == on ]]
  run_driver "$3"
  exit
fi

[[ $# == 2 && $1 == --candidate && $2 =~ ^[0-9a-f]{40}$ ]] || exit 2
candidate=$2
verify_dependencies_fast
verify_reviewed_sources "$candidate"
pin_reviewed_scripts "$candidate"
[[ $(id -u) != 0 && $(id -un) == bmarti44 ]]
base=/home/bmarti44/.local/state/glm52-w7-cache-generation
mkdir -p "$base"
nonce=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
attempt="$base/attempt-$nonce"
out="$attempt/on"
mkdir -m 0700 "$attempt"
trap finalize_outer EXIT
mkdir -m 0700 "$out"
failure_reason=containment-launch
tag="w7-c10-${nonce:0:12}"
final_artifacts="$out/server.log,$out/live-response.json,$out/live-http-status,$out/primary-response.json,$out/primary-http-status,$out/child-exit.json,$out/model.identity.json"

set +e
GLM_SAFE_MEMORY_HIGH_GIB=78 \
GLM_SAFE_KILL_FLOOR_GIB=24 \
GLM_SAFE_MIN_START_GIB=110 \
GLM_SAFE_TIMEOUT_S=2400 \
GLM_SAFE_RUN_AS_CURRENT_USER=1 \
GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
GLM_SAFE_EXPECTED_BINARY_SHA256="$BINARY_SHA256" \
GLM_SAFE_PROVENANCE_ENV_ALLOWLIST=DS4_CUDA_STABLE_MODEL_REMAP \
GLM_SAFE_EXPECTED_ENV_SHA256="$ENV_SHA256" \
GLM_SAFE_FINAL_ARTIFACTS="$final_artifacts" \
GLM_SAFE_DONE_DIGESTS=1 \
GLM_SAFE_PINNED_SAFE_PATH="$safe_fd_path" \
GLM_SAFE_PINNED_SAFE_SHA256="$safe_sha256" \
GLM_CANDIDATE_SRC="$CANDIDATE_SRC" \
DS4_W7_PINNED_HARNESS_SHA256="$harness_sha256" \
DS4_CUDA_STABLE_MODEL_REMAP=1 \
DS4_CUDA_EXPERT_CACHE_GB=40 \
DS4_CUDA_EXPERT_CACHE_PIN=1 \
DS4_CUDA_EXPERT_CACHE_SLRU=1 \
DS4_CUDA_FETCH_THREADS=6 \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
DS4_GLM_SYNC_TRACE=1 \
/usr/bin/bash "$cgroup_fd_path" --tag "$tag" -- /usr/bin/bash "$harness_fd_path" --driver on "$out" \
  >"$attempt/containment.stdout" 2>"$attempt/containment.stderr"
containment_rc=$?
set -e
printf '%s\n' "$containment_rc" >"$attempt/containment.rc"
sync_parent "$attempt/containment.rc"
failure_reason=containment-result-validation
crash_dir=$(sed -nE 's#^SAFE_RUN_DONE rc=[0-9]+ killed=[a-z]+ dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) main_sha256=[0-9a-f]{64} samples_sha256=[0-9a-f]{64} kernel_sha256=[0-9a-f]{64}$#\1#p' "$attempt/containment.stdout")
[[ $(printf '%s\n' "$crash_dir" | sed '/^$/d' | wc -l) == 1 && -d $crash_dir && ! -L $crash_dir ]]
failure_reason=evidence-binding-and-scoring
publish_outer_evidence "$attempt" "$out" "$containment_rc" "$crash_dir" "$candidate"
trap - EXIT
printf 'W7_CACHE_GENERATION_ATTEMPT=%s\n' "$attempt"
