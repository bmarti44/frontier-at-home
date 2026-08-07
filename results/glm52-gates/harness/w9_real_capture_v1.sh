#!/bin/bash
# Matched capture-OFF/capture-ON correctness run for W9. This produces the
# real 512-wide tensors for the offline falsifier; it is not a perf or 1M run.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly SCORER=$REPO/scripts/91_score_w9_real_capture.py
readonly PROMPT_BUILDER=$REPO/scripts/92_build_w9_prompt.py
readonly DRAND_VERIFY=$REPO/scripts/89_verify_drand_receipt.mjs
readonly REVIEW_RECEIPT=$REPO/results/glm52-gates/W9-real-capture-review-r248.json
readonly NODE=/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node
readonly RUNTIME_DIR=/home/bmarti44/.cache/glm52-w9-9ebc0f2-runtime
readonly BIN=$RUNTIME_DIR/ds4-server
readonly ENGINE_SRC=/home/bmarti44/.cache/glm52-w9-real-capture
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
readonly OUT_PARENT=/home/bmarti44/.local/state/glm52-w9-real-capture
readonly CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock

readonly ENGINE_COMMIT=9ebc0f2879c126db095ecd25be0567166868d52c
readonly BINARY_SHA256=1e233b07cbeb13f0802bb94d598328a77e6fbe7b65f8ddd650bb516539913476
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly MODEL_BYTES=211075856448
readonly TOKENIZER_SHA256=19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d

sha() { sha256sum -- "$1" | awk '{print $1}'; }

verify_reviewed_components() {
  local commit=$1 relative expected
  for relative in \
    results/glm52-gates/harness/w9_real_capture_v1.sh \
    results/glm52-gates/harness/glm_safe_run.sh \
    results/glm52-gates/harness/glm_cgroup_run.sh \
    scripts/03_memory_guard.py \
    scripts/89_verify_drand_receipt.mjs \
    scripts/91_score_w9_real_capture.py \
    scripts/92_build_w9_prompt.py
  do
    expected=$(git -C "$REPO" show "$commit:$relative" | sha256sum | awk '{print $1}')
    [[ -f $REPO/$relative && ! -L $REPO/$relative && $(sha "$REPO/$relative") == "$expected" ]]
  done
}

driver() {
  [[ $# == 2 && ( $1 == off || $1 == on ) ]]
  local arm=$1 arm_out=$2
  local -a publications=()
  "$BIN" --cuda -m "$MODEL" --raw-prompt --prompt-file "$arm_out/prompt.txt" \
    -c 8193 --temp 0 --dump-logits "$arm_out/next-logits.json" --ssd-streaming \
    --ssd-streaming-cache-experts 40GB \
    >"$arm_out/cli.stdout" 2>"$arm_out/cli.stderr"
  ! grep -Fq 'prefill logits dump failed' "$arm_out/cli.stderr"
  mapfile -t publications < <(sed -n 's/^ds4: prefill logits dumped to //p' \
    "$arm_out/cli.stderr")
  [[ ${#publications[@]} == 1 && ${publications[0]} == "$arm_out"/logits.sync* &&
     -f ${publications[0]} && ! -L ${publications[0]} ]]
}

arm_output_path() {
  [[ $# == 2 && ( $1 == off || $1 == on ) && $2 == /* ]]
  printf '%s/%s\n' "$2" "$1"
}

bind_safety() {
  local arm_out=$1 safe_dir name
  safe_dir=$(sed -n 's/^SAFE_RUN_DONE rc=[0-9][0-9]* killed=[^[:space:]]* dir=\([^[:space:]]*\)$/\1/p' \
    "$arm_out/containment.stdout")
  [[ $safe_dir == "$CRASH_ROOT"/* && -d $safe_dir && ! -L $safe_dir ]]
  mkdir "$arm_out/safety"
  for name in main.log samples.log kernel.log; do
    [[ -f $safe_dir/$name && ! -L $safe_dir/$name ]]
    install -m 0600 "$safe_dir/$name" "$arm_out/safety/$name"
  done
}

run_arm() {
  local arm=$1 arm_root=$2 arm_out rc tag
  arm_out=$(arm_output_path "$arm" "$arm_root")
  mkdir "$arm_out"
  install -m 0600 "$arm_root/prompt.txt" "$arm_out/prompt.txt"
  tag=w9-${arm}-${W9_NONCE:0:12}
  local -a capture_env=()
  if [[ $arm == on ]]; then
    capture_env=(DS4_GLM_W9_CAPTURE_DIR=$arm_out/capture)
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
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
    DS4_GLM_LOGIT_DUMP=$arm_out/logits DS4_GLM_LOGIT_DUMP_ALL=1 \
    DS4_LOCK_FILE=$ENGINE_LOCK DS4_LOCK_EXPECTED_DEV_INO=$W9_LOCK_IDENTITY \
    "${capture_env[@]}" "$CGROUP" --tag "$tag" -- \
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
c=d.get("candidate_hash",""); floor=d.get("drand_min_round")
if (d.get("schema")!="glm52-w9-real-capture-review-v1" or
    d.get("verdict")!="PASS_RUNTIME_ALLOWED" or d.get("critical")!=[] or
    d.get("high")!=[] or not re.fullmatch(r"[0-9a-f]{40}",c) or
    not isinstance(floor,int) or floor<1): raise SystemExit(2)
print(c,floor)
PY
)
git -C "$REPO" merge-base --is-ancestor "$reviewed_commit" HEAD
[[ $(git -C "$REPO" show "HEAD:results/glm52-gates/W9-real-capture-review-r248.json" | sha256sum | awk '{print $1}') == $(sha "$REVIEW_RECEIPT") ]]
verify_reviewed_components "$reviewed_commit"
[[ -x $BIN && ! -L $BIN && $(sha "$BIN") == "$BINARY_SHA256" ]]
[[ $(git -C "$ENGINE_SRC" rev-parse HEAD) == "$ENGINE_COMMIT" && -z $(git -C "$ENGINE_SRC" status --porcelain) ]]
[[ -f $MODEL && ! -L $MODEL &&
   $(stat -Lc '%U:%G:%a:%h:%s' "$MODEL") == "dsv4:dsv4:664:1:$MODEL_BYTES" ]]
[[ $(sha "$MODEL") == "$MODEL_SHA256" ]]
readonly MODEL_IDENTITY=$(stat -Lc '%d:%i:%s:%y:%z' "$MODEL")
[[ -f $TOKENIZER && ! -L $TOKENIZER && $(sha "$TOKENIZER") == "$TOKENIZER_SHA256" ]]
[[ -r $SAFE && ! -L $SAFE && -x $CGROUP && -x $SCORER && -x $PROMPT_BUILDER && -x $NODE ]]
! pgrep -x ds4-server >/dev/null && ! pgrep -x ds4 >/dev/null && ! pgrep -x fio >/dev/null
[[ -f $ENGINE_LOCK && ! -L $ENGINE_LOCK && $(stat -Lc '%U:%G:%a:%h' "$ENGINE_LOCK") == bmarti44:bmarti44:600:1 ]]
/usr/bin/flock -n -E 75 -- "$ENGINE_LOCK" /usr/bin/true
readonly W9_LOCK_IDENTITY=$(stat -Lc '%d:%i' "$ENGINE_LOCK")
export W9_LOCK_IDENTITY
verify_model_identity() {
  [[ -f $MODEL && ! -L $MODEL &&
     $(stat -Lc '%U:%G:%a:%h' "$MODEL") == dsv4:dsv4:664:1 &&
     $(stat -Lc '%d:%i:%s:%y:%z' "$MODEL") == "$MODEL_IDENTITY" ]]
}
swap_used_kib=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
(( swap_used_kib < 1048576 ))
python3 -I -B "$GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 0 >/dev/null

read -r round randomness signature previous < <(python3 -I -B - "$randomness_receipt" <<'PY'
import json,re,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
v=(str(d.get("round","")),d.get("randomness",""),d.get("signature",""),d.get("previous_signature",""))
if not re.fullmatch(r"[1-9][0-9]*",v[0]) or any(not isinstance(x,str) for x in v): raise SystemExit(2)
print(*v)
PY
)
"$NODE" "$DRAND_VERIFY" "$round" "$randomness" "$signature" "$previous" >/dev/null
(( round > drand_min_round ))
readonly W9_NONCE=$(printf '%s' "$randomness$ENGINE_COMMIT" | sha256sum | awk '{print $1}')
export W9_NONCE
if (( 16#${W9_NONCE:0:2} % 2 )); then arms=(on off); else arms=(off on); fi

mkdir -p "$OUT_PARENT" "$CRASH_ROOT"
readonly root=$OUT_PARENT/attempt-${W9_NONCE:0:32}
mkdir "$root"
install -m 0600 "$randomness_receipt" "$root/randomness-receipt.json"
/usr/bin/env -i HOME=/home/bmarti44 PATH=/usr/bin:/bin \
  PYTHONPATH=/home/bmarti44/.local/lib/python3.12/site-packages \
  /usr/bin/python3 -B "$PROMPT_BUILDER" --tokenizer "$TOKENIZER" \
  --randomness "$randomness" --output "$root/prompt.txt" >"$root/prompt-build.txt"
readonly prompt_sha=$(sha "$root/prompt.txt")
readonly base_config_sha=$(printf '%s\n' \
  'ctx=8193' 'tokens=0' 'temperature=0' 'ssd_streaming=1' \
  'dump_logits_mode=1' \
  'expert_cache_gb=40' 'expert_cache_pin=1' 'expert_cache_slru=1' \
  'fetch_threads=6' 'moe_no_atomic_down=1' | sha256sum | awk '{print $1}')

python3 -I -B - "$root/manifest.json" "${arms[*]}" "$reviewed_commit" \
  "$ENGINE_COMMIT" "$BINARY_SHA256" "$MODEL_SHA256" "$TOKENIZER_SHA256" \
  "$prompt_sha" "$base_config_sha" "$(sha "$SCORER")" "$(sha "$0")" \
  "$(sha "$PROMPT_BUILDER")" "$(sha "$randomness_receipt")" "$round" <<'PY'
import json,pathlib,sys
(out,order,reviewed,engine,binary,model,tokenizer,prompt,config,scorer,harness,
 builder,receipt,round_s)=sys.argv[1:]
d={"schema":"glm52-w9-real-capture-manifest-v1","reviewed_candidate_commit":reviewed,
   "engine_commit":engine,"binary_sha256":binary,"model_sha256":model,
   "tokenizer_sha256":tokenizer,"prompt_sha256":prompt,"configuration_sha256":config,
   "scorer_sha256":scorer,"harness_sha256":harness,"prompt_builder_sha256":builder,
   "randomness_receipt_sha256":receipt,"drand_round":int(round_s),
   "arm_order":order.split(),"arms":{}}
for arm in ("off","on"):
    d["arms"][arm]={"binary_sha256":binary,"model_sha256":model,
      "tokenizer_sha256":tokenizer,"prompt_sha256":prompt,
      "configuration_sha256":config,"context":8193,"capture":arm=="on"}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY

stage=arms
terminal=0
finalize() {
  local rc=$1
  (( terminal == 0 )) || return 0
  set +e
  python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
    --raw "$root/raw.jsonl" --summary "$root/summary.json" \
    --failure-reason "harness_exit_rc=$rc stage=${stage:-unknown}" >/dev/null 2>&1
  set -e
}
trap 'finalize $?' EXIT
for arm in "${arms[@]}"; do
  stage=arm-$arm
  run_arm "$arm" "$root"
  verify_model_identity
done
stage=scorer
python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
  --raw "$root/raw.jsonl" --summary "$root/summary.json"
python3 -I -B "$SCORER" --root "$root" --manifest "$root/manifest.json" \
  --raw "$root/raw.jsonl" --summary "$root/summary.json" --verify-terminal
terminal=1
trap - EXIT
printf 'W9_REAL_CAPTURE_DONE %s\n' "$root"
