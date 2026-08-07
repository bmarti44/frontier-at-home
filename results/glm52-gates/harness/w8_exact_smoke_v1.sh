#!/bin/bash
# Matched model-backed correctness smoke for exact W8. This deliberately uses
# the already-qualified 5,066-token W7 fixture because it crosses the indexed
# prefill boundary cheaply. It is not direct-1M or performance evidence.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly SCORER=$REPO/scripts/90_score_w8_exact_smoke.py
readonly ENGINE_PATCH=$REPO/results/glm52-gates/harness/ds4-w8-exact-ckv.patch
readonly REVIEW_RECEIPT=$REPO/results/glm52-gates/W8-exact-smoke-review-r241.json
readonly DRAND_VERIFY=$REPO/scripts/89_verify_drand_receipt.mjs
readonly NODE=/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node
readonly RUNTIME_DIR=/home/bmarti44/.cache/glm52-w8-119996d-runtime
readonly BIN=/home/bmarti44/.cache/glm52-w8-119996d-runtime/ds4-server
readonly SRC=/home/bmarti44/.cache/glm52-w8-exact-current
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly REQUEST=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly OUT_PARENT=/home/bmarti44/.local/state/glm52-w8-exact-smoke
readonly CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
readonly LOCK=/run/user/1000/ds4-engine.lock
readonly PORT=8098

readonly SOURCE_COMMIT=119996d9fa0ea3ffb046ce52c0765c78615de4be
readonly BINARY_SHA256=0c80fda2d7b135dc1d6f763ca0a1b40cecdc9facdf5bc493238860a1aa660091
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly MODEL_BYTES=211075856448
readonly REQUEST_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly ENGINE_PATCH_SHA256=f6468a476411f1d7cc161737f8de404b20a60e22789b34830cdcc3df70fde9a2

sha() { sha256sum -- "$1" | awk '{print $1}'; }

verify_reviewed_components() {
  local commit=$1 relative path expected
  for relative in \
    results/glm52-gates/harness/w8_exact_smoke_v1.sh \
    scripts/90_score_w8_exact_smoke.py \
    results/glm52-gates/harness/glm_cgroup_run.sh \
    results/glm52-gates/harness/glm_safe_run.sh \
    scripts/03_memory_guard.py \
    scripts/89_verify_drand_receipt.mjs \
    results/glm52-gates/harness/ds4-w8-exact-ckv.patch
  do
    path=$REPO/$relative
    [[ -f $path && ! -L $path ]]
    expected=$(git -C "$REPO" show "$commit:$relative" | sha256sum | awk '{print $1}')
    [[ $(sha "$path") == "$expected" ]]
  done
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

driver() {
  [[ $# == 2 && ( $1 == resident || $1 == exact ) ]]
  local arm=$1 out=$2 code=
  mkdir "$out/ckv"
  chmod 0700 "$out/ckv"
  "$BIN" --cuda -m "$MODEL" -c 8192 --host 127.0.0.1 --port "$PORT" \
    --trace "$out/request.trace" --ssd-streaming \
    --ssd-streaming-cache-experts 40GB >"$out/server.log" 2>&1 &
  server_pid=$!
  trap stop_server EXIT INT TERM HUP
  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$PORT/v1/models" || true)
    [[ $code == 200 ]] && break
    sleep 2
  done
  [[ $code == 200 ]]
  curl -sS --fail-with-body --max-time 1800 -H 'Content-Type: application/json' \
    -o "$out/response.json" -w '%{http_code}\n' -d @"$out/request.json" \
    "http://127.0.0.1:$PORT/v1/completions" >"$out/http-status"
  stop_server
  trap - EXIT INT TERM HUP
  [[ $(<"$out/http-status") == 200 ]]
  [[ -z $(find "$out/ckv" -mindepth 1 -maxdepth 1 -print -quit) ]]
}

bind_safety() {
  local arm_out=$1 safe_dir
  safe_dir=$(sed -n 's/^SAFE_RUN_DONE rc=[0-9][0-9]* killed=[^[:space:]]* dir=\([^[:space:]]*\)$/\1/p' \
    "$arm_out/containment.stdout")
  [[ $safe_dir == "$CRASH_ROOT"/* && -d $safe_dir && ! -L $safe_dir ]]
  mkdir -p "$arm_out/safety"
  local name
  for name in main.log samples.log kernel.log; do
    [[ -f $safe_dir/$name && ! -L $safe_dir/$name ]]
    install -m 0600 "$safe_dir/$name" "$arm_out/safety/$name"
  done
}

run_arm() {
  local arm=$1 arm_root=$2 tag rc
  local arm_out=$arm_root/$arm
  mkdir "$arm_out"
  install -m 0600 "$REQUEST" "$arm_out/request.json"
  tag=w8sm-${arm:0:1}-${W8_ATTEMPT_NONCE:0:12}
  local -a exact_env=()
  if [[ $arm == exact ]]; then
    exact_env=(
      DS4_GLM_CKV_NVME_EXACT=1
      DS4_GLM_CKV_DIR=$arm_out/ckv
      DS4_GLM_CKV_MODEL_SHA256=$MODEL_SHA256
      DS4_GLM_CKV_MAX_GIB=192
    )
  fi
  set +e
  /usr/bin/env -i HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_RUNTIME_DIR=/run/user/1000 \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    GLM_SAFE_RUN_AS_CURRENT_USER=1 GLM_SAFE_MEMORY_HIGH_GIB=78 \
    GLM_SAFE_KILL_FLOOR_GIB=24 GLM_SAFE_MIN_START_GIB=110 \
    GLM_SAFE_TIMEOUT_S=3600 GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
    GLM_SAFE_EXPECTED_BINARY_SHA256=$BINARY_SHA256 GLM_CANDIDATE_SRC=$RUNTIME_DIR \
    DS4_CUDA_EXPERT_CACHE_GB=40 DS4_CUDA_EXPERT_CACHE_PIN=1 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_GLM_SYNC_TRACE=1 \
    DS4_GLM_LOGIT_DUMP=$arm_out/logits DS4_GLM_LOGIT_DUMP_ALL=1 \
    DS4_LOCK_FILE=$LOCK DS4_LOCK_EXPECTED_DEV_INO=$W8_LOCK_IDENTITY \
    "${exact_env[@]}" "$CGROUP" --tag "$tag" -- \
      /usr/bin/bash "$0" --driver "$arm" "$arm_out" \
      >"$arm_out/containment.stdout" 2>"$arm_out/containment.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$arm_out/containment.rc"
  bind_safety "$arm_out" || true
  (( rc == 0 )) || return 2
  [[ -f $arm_out/safety/main.log && -f $arm_out/safety/samples.log &&
     -f $arm_out/safety/kernel.log ]]
}

if [[ ${1:-} == --driver ]]; then
  shift
  driver "$@"
  exit $?
fi

[[ $# == 1 && -f $1 && ! -L $1 ]]
readonly randomness_receipt=$1
[[ $(id -un) == bmarti44 && -z $(git -C "$REPO" status --porcelain) ]]
[[ -f $REVIEW_RECEIPT && ! -L $REVIEW_RECEIPT ]]
read -r reviewed_commit drand_min_round < <(python3 -I -B - "$REVIEW_RECEIPT" <<'PY'
import json,re,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
commit=d.get("candidate_hash",""); floor=d.get("drand_min_round")
if (d.get("schema")!="glm52-w8-exact-smoke-review-v1" or
    d.get("verdict")!="PASS_RUNTIME_ALLOWED" or
    d.get("critical")!=[] or d.get("high")!=[] or
    not isinstance(commit,str) or not re.fullmatch(r"[0-9a-f]{40}",commit) or
    not isinstance(floor,int) or floor<1): raise SystemExit(2)
print(commit,floor)
PY
)
git -C "$REPO" merge-base --is-ancestor "$reviewed_commit" HEAD
[[ $(git -C "$REPO" show "HEAD:results/glm52-gates/W8-exact-smoke-review-r241.json" | sha256sum | awk '{print $1}') == $(sha "$REVIEW_RECEIPT") ]]
verify_reviewed_components "$reviewed_commit"
[[ -x $BIN && $(sha "$BIN") == "$BINARY_SHA256" ]]
[[ $(realpath -e -- "$RUNTIME_DIR/ds4-server") == $(realpath -e -- "$BIN") ]]
[[ $(git -C "$SRC" rev-parse HEAD) == "$SOURCE_COMMIT" && -z $(git -C "$SRC" status --porcelain) ]]
[[ -f $MODEL && ! -L $MODEL && $(stat -Lc '%s' "$MODEL") == "$MODEL_BYTES" ]]
[[ -f $REQUEST && ! -L $REQUEST && $(sha "$REQUEST") == "$REQUEST_SHA256" ]]
[[ -f $ENGINE_PATCH && ! -L $ENGINE_PATCH && $(sha "$ENGINE_PATCH") == "$ENGINE_PATCH_SHA256" ]]
[[ -r $SAFE && ! -L $SAFE && -x $CGROUP && -x $SCORER && -x $NODE ]]
! pgrep -x ds4-server >/dev/null && ! pgrep -x fio >/dev/null
[[ -f $LOCK && ! -L $LOCK && $(stat -Lc '%U:%G:%a:%h' "$LOCK") == bmarti44:bmarti44:600:1 ]]
/usr/bin/flock -n -E 75 -- "$LOCK" /usr/bin/true
readonly W8_LOCK_IDENTITY=$(stat -Lc '%d:%i' "$LOCK")
export W8_LOCK_IDENTITY
swap_used_kib=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
(( swap_used_kib < 1048576 ))
python3 -I -B "$GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 0 >/dev/null

read -r round randomness signature previous < <(python3 -I -B - "$randomness_receipt" <<'PY'
import json, re, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
values=(str(d.get("round","")),d.get("randomness",""),d.get("signature",""),d.get("previous_signature",""))
if not re.fullmatch(r"[1-9][0-9]*",values[0]) or any(not isinstance(v,str) for v in values): raise SystemExit(2)
print(*values)
PY
)
"$NODE" "$DRAND_VERIFY" "$round" "$randomness" "$signature" "$previous" >/dev/null
(( round > drand_min_round ))
readonly W8_ATTEMPT_NONCE=$(printf '%s' "$randomness$SOURCE_COMMIT" | sha256sum | awk '{print $1}')
export W8_ATTEMPT_NONCE
if (( 16#${W8_ATTEMPT_NONCE:0:2} % 2 )); then
  arms=(exact resident)
else
  arms=(resident exact)
fi

mkdir -p "$OUT_PARENT" "$CRASH_ROOT"
readonly root=$OUT_PARENT/attempt-${W8_ATTEMPT_NONCE:0:32}
mkdir "$root"
install -m 0600 "$randomness_receipt" "$root/randomness-receipt.json"
python3 -I -B - "$root/manifest.json" "$randomness_receipt" "${arms[*]}" \
  "$SOURCE_COMMIT" "$BINARY_SHA256" "$MODEL_SHA256" "$REQUEST_SHA256" \
  "$(sha "$SCORER")" "$(sha "$0")" "$ENGINE_PATCH_SHA256" \
  "$reviewed_commit" "$round" <<'PY'
import hashlib,json,pathlib,sys
out,receipt,order,source,binary,model,request,scorer,harness,patch,reviewed,round_s=sys.argv[1:]
doc={"schema":"glm52-w8-exact-smoke-manifest-v1","source_commit":source,
     "binary_sha256":binary,"model_sha256":model,"request_sha256":request,
     "scorer_sha256":scorer,"harness_sha256":harness,"engine_patch_sha256":patch,
     "reviewed_candidate_commit":reviewed,"drand_round":int(round_s),
     "randomness_receipt_sha256":hashlib.sha256(pathlib.Path(receipt).read_bytes()).hexdigest(),
     "arm_order":order.split(),"arms":{}}
for arm in ("resident","exact"):
    doc["arms"][arm]={"binary_sha256":binary,"model_sha256":model,
      "request_sha256":request,"context":8192,
      "ckv_mode":"resident" if arm=="resident" else "exact-f32-nvme"}
pathlib.Path(out).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
PY

stage=arms
finalize_attempt() {
  local rc=$1
  [[ ${attempt_terminal:-0} == 1 ]] && return 0
  set +e
  python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
    --raw "$root/raw.jsonl" --summary "$root/summary.json" \
    --failure-reason "harness_exit_rc=$rc stage=${stage:-unknown}" >/dev/null 2>&1
  set -e
}
attempt_terminal=0
trap 'finalize_attempt $?' EXIT
for arm in "${arms[@]}"; do
  stage=arm-$arm
  run_arm "$arm" "$root"
done
stage=scorer
python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
  --raw "$root/raw.jsonl" --summary "$root/summary.json"
python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
  --raw "$root/raw.jsonl" --summary "$root/summary.json" --verify-terminal
attempt_terminal=1
trap - EXIT
printf 'W8_EXACT_SMOKE_DONE %s\n' "$root"
