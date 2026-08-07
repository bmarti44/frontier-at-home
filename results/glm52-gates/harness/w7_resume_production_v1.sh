#!/bin/bash
# Reviewed W7 production strict/candidate/cold equivalence. Every arm starts a
# fresh server under the hardened GLM containment chain. The candidate proves
# automatic restored-frontier behavior without a diagnostic opt-in.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly INVOKED_SCRIPT=$0
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly MEMORY_GUARD=$REPO/scripts/03_memory_guard.py
readonly ENGINE_FREEZE=$REPO/results/glm52-gates/W7-resume-production-freeze-v1.json
readonly TRACE_SCORER=$REPO/scripts/83_score_w7_deployed_trace.py
readonly SCORER=$REPO/scripts/87_score_w7_resume_production.py
readonly BIN=/home/bmarti44/.cache/glm52-w7-3ba062e-runtime/ds4-server
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly LIVE_SOURCE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json
readonly PRIMARY_SOURCE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly POOL=$REPO/results/glm52-gates/harness/w7-production-fixture-pool-v1.json
readonly TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
readonly TOKENIZER_RUNTIME=/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2
readonly TOKENIZER_INIT=$TOKENIZER_RUNTIME/tokenizers/__init__.py
readonly TOKENIZER_NATIVE=$TOKENIZER_RUNTIME/tokenizers/tokenizers.abi3.so
readonly OUT_PARENT=/home/bmarti44/.local/state/glm52-w7-production-equivalence
readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock
readonly CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
readonly PORT=8097
readonly CACHE_GIB=40

readonly BINARY_SHA256=c8b08e4ebd59f35f5dba7bcc0943b5d6f377cd15cecb46fc5bdb22dccfd6a51a
readonly MODEL_BYTES=211075856448
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly LIVE_SHA256=d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a
readonly PRIMARY_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly POOL_SHA256=c71f1c9c90164baae00492befed68765fd9bee40fef3de8c3b291cc06794ecb9
readonly TOKENIZER_SHA256=19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
readonly TOKENIZER_INIT_SHA256=eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4
readonly TOKENIZER_NATIVE_SHA256=fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910
readonly TRACE_SCORER_SHA256=6cec5063906a52c577617b4173a1deed14d0ae2fffebff19bbef6e96442dc985
readonly SCORER_SHA256=a44e5a80265ccab46db44dab2ab7fe1174f8282a41bda7588b18f39ff6337d67
readonly CGROUP_SHA256=e5a37b35d3ff1e8a7ee08d0f2c1396441b0dbc4abd64220389362ae6c6994c32
readonly SAFE_SHA256=6e4d382bc5e5818787af8c17aae7a0750ca3ab7b36471f21355789d194b2e801
readonly MEMORY_GUARD_SHA256=3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58
readonly ENGINE_FREEZE_SHA256=2a97c273e713cd18045d1c38ec671f5941cdb71287ecbbbcffc136c01ffd68d6
readonly CONFIGURATION_SHA256=7f78db9b848fe13d11a02059ae3a53d0abc6e84c4b1e551b3d84c1b8e8b752b8
readonly ENGINE_SOURCE_COMMIT=3ba062e5433e56df7c6da70b58cc9757e7777d54

verify_file() {
  [[ $# == 2 && -f $1 && ! -L $1 ]] || return 2
  [[ $(sha256sum -- "$1" | awk '{print $1}') == "$2" ]]
}

verify_sealed_fd() {
  [[ $# == 2 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9a-f]{64}$ ]] || return 2
  /usr/bin/python3 -I -B - "$1" "$2" <<'PY'
import fcntl, hashlib, os, sys
fd = int(sys.argv[1])
expected = sys.argv[2]
required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != required:
    raise SystemExit("input descriptor is not completely sealed")
os.lseek(fd, 0, os.SEEK_SET)
with os.fdopen(os.dup(fd), "rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
if digest != expected:
    raise SystemExit("sealed input digest mismatch")
PY
}

validate_execution_authority() {
  [[ ${W7_EXECUTED_HARNESS_SHA256:-} =~ ^[0-9a-f]{64}$ &&
     ${W7_FROZEN_CANDIDATE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || return 2
  [[ -r $INVOKED_SCRIPT &&
     $(sha256sum -- "$INVOKED_SCRIPT" | awk '{print $1}') == "$W7_EXECUTED_HARNESS_SHA256" ]] || return 2
  git -C "$REPO" cat-file -e "$W7_FROZEN_CANDIDATE_COMMIT^{commit}" 2>/dev/null || return 2
  git -C "$REPO" show \
    "$W7_FROZEN_CANDIDATE_COMMIT:results/glm52-gates/harness/w7_resume_production_v1.sh" \
    | sha256sum | awk -v expected="$W7_EXECUTED_HARNESS_SHA256" '$1 == expected {ok=1} END{exit !ok}'
}

arm_tag() {
  [[ $# == 2 && $2 =~ ^[0-9a-f]{32}$ ]] || return 2
  case "$1" in
    strict) printf 'w7eq-s-%.12s\n' "$2" ;;
    candidate) printf 'w7eq-c-%.12s\n' "$2" ;;
    cold) printf 'w7eq-o-%.12s\n' "$2" ;;
    *) return 2 ;;
  esac
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
  verify_file "$ENGINE_FREEZE" "$ENGINE_FREEZE_SHA256"
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
  local dir=$1 output=$2
  /usr/bin/python3 -I -B - "$dir" "$output" <<'PY'
import hashlib, pathlib, re, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for path in sorted(root.glob("*.kv")):
    if path.is_symlink() or re.fullmatch(r"[0-9a-f]{40}\.kv", path.name) is None:
        raise SystemExit("unsafe KV inventory path")
    full_hash, normalized_hash = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as handle:
        header = handle.read(64)
        if len(header) != 64:
            raise SystemExit("short KV file")
        full_hash.update(header)
        normalized = bytearray(header)
        normalized[12:16] = b"\0" * 4
        normalized[24:40] = b"\0" * 16
        normalized_hash.update(normalized)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            full_hash.update(chunk)
            normalized_hash.update(chunk)
    rows.append(
        f"{full_hash.hexdigest()}  {normalized_hash.hexdigest()}  {path.name}\n"
    )
if not rows:
    raise SystemExit("empty KV inventory")
output.write_text("".join(rows), encoding="utf-8")
PY
}

driver() {
  [[ $# == 2 ]] || return 2
  local arm=$1 arm_out=$2 code
  local boundary_trim=8
  [[ $arm == strict || $arm == candidate || $arm == cold ]] || return 2
  [[ $arm == cold ]] && boundary_trim=20
  mkdir "$arm_out/kv"
  "$BIN" --cuda -m "$MODEL" -c 8192 --host 127.0.0.1 --port "$PORT" \
    --trace "$arm_out/request.trace" \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$arm_out/kv" --kv-disk-space-mb 4096 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens "$boundary_trim" \
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
import hashlib, json, pathlib, sys
arm, out_raw, rc_raw, live_sha, primary_sha = sys.argv[1:]
out = pathlib.Path(out_raw)
requests = {"primary": hashlib.sha256((out / "primary-request.json").read_bytes()).hexdigest()}
if arm != "cold": requests["live"] = hashlib.sha256((out / "live-request.json").read_bytes()).hexdigest()
if requests["primary"] != primary_sha or (arm != "cold" and requests["live"] != live_sha):
    raise SystemExit("actual arm request digest mismatch")
doc = {"schema_version": 1, "arm": arm, "containment_rc": int(rc_raw), "request_sha256": requests}
(out / "arm.json").write_text(json.dumps(doc, sort_keys=True) + "\n")
PY
}

bind_safety_evidence() {
  local arm_out=$1 safe_dir
  safe_dir=$(sed -n 's/^SAFE_RUN_DONE rc=0 killed=no dir=\(\/home\/bmarti44\/.local\/state\/glm52-crashlog\/[^[:space:]]*\)$/\1/p' \
    "$arm_out/containment.stdout")
  [[ -n $safe_dir && $safe_dir == "$CRASH_ROOT"/* && -d $safe_dir && ! -L $safe_dir ]] || return 2
  mkdir "$arm_out/safety"
  local name
  for name in main.log samples.log kernel.log; do
    [[ -f $safe_dir/$name && ! -L $safe_dir/$name ]] || return 2
    install -m 0600 "$safe_dir/$name" "$arm_out/safety/$name"
  done
}

score_trace() {
  local arm_out=$1
  /usr/bin/python3 -I -B "/proc/$$/fd/$trace_scorer_fd" \
    --trace "$arm_out/request.trace" --pool "$POOL" \
    --live-request "$arm_out/live-request.json" \
    --primary-request "$arm_out/primary-request.json" \
    --tokenizer "$TOKENIZER" --tokenizer-runtime "$TOKENIZER_RUNTIME" \
    >"$arm_out/trace-result.json" 2>"$arm_out/trace-scorer.stderr"
}

run_arm() {
  local arm=$1 root=$2
  local arm_out=$root/$arm
  local tag rc
  tag=$(arm_tag "$arm" "${root##*-}")
  mkdir "$arm_out"
  install -m 0600 "$PRIMARY_SOURCE" "$arm_out/primary-request.json"
  if [[ $arm != cold ]]; then install -m 0600 "$LIVE_SOURCE" "$arm_out/live-request.json"; fi
  set +e
  /usr/bin/env -i \
    HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_RUNTIME_DIR=/run/user/1000 \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    GLM_SAFE_RUN_AS_CURRENT_USER=1 GLM_SAFE_MEMORY_HIGH_GIB=78 \
    GLM_SAFE_KILL_FLOOR_GIB=24 GLM_SAFE_MIN_START_GIB=110 GLM_SAFE_TIMEOUT_S=2400 \
    GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 GLM_SAFE_EXPECTED_BINARY_SHA256=$BINARY_SHA256 \
    GLM_CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-3ba062e-runtime \
    DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GIB DS4_CUDA_EXPERT_CACHE_PIN=1 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_GLM_SYNC_TRACE=1 \
    DS4_GLM_LOGIT_DUMP="$arm_out/logits" DS4_GLM_LOGIT_DUMP_ALL=1 \
    DS4_LOCK_FILE=$ENGINE_LOCK DS4_LOCK_EXPECTED_DEV_INO=$engine_lock_identity \
      "$CGROUP" --tag "$tag" -- /usr/bin/env \
        W7_EXECUTED_HARNESS_SHA256="$W7_EXECUTED_HARNESS_SHA256" \
        W7_FROZEN_CANDIDATE_COMMIT="$W7_FROZEN_CANDIDATE_COMMIT" \
        /usr/bin/bash "/proc/$$/fd/$harness_fd" --driver "$arm" "$arm_out" \
      >"$arm_out/containment.stdout" 2>"$arm_out/containment.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$arm_out/containment.rc"
  if (( rc == 0 )); then bind_safety_evidence "$arm_out"; fi
  write_arm_metadata "$arm" "$arm_out" "$rc"
  (( rc == 0 )) || return 2
  if [[ $arm != cold ]]; then score_trace "$arm_out"; fi
}

if [[ ${1:-} == --validate-sealed-runtime ]]; then
  [[ $# == 1 ]]
  validate_execution_authority
  [[ ${W7_RANDOM_SEED_SHA256:-} =~ ^[0-9a-f]{64}$ ]]
  verify_sealed_fd "${W7_SEALED_HARNESS_FD:-}" "$W7_EXECUTED_HARNESS_SHA256"
  verify_sealed_fd "${W7_SEALED_SCORER_FD:-}" "$SCORER_SHA256"
  verify_sealed_fd "${W7_SEALED_TRACE_SCORER_FD:-}" "$TRACE_SCORER_SHA256"
  echo W7_SEALED_RUNTIME_OK
  exit 0
elif [[ ${1:-} == --self-test ]]; then
  verify_dependencies
  python3 -m unittest scripts.tests.test_w7_resume_production_scorer >/dev/null
  echo W7_PRODUCTION_EQUIVALENCE_SELFTEST_OK
  exit 0
elif [[ ${1:-} == --validate-tag ]]; then
  [[ $# == 3 ]]
  tag=$(arm_tag "$2" "$3")
  set +e
  validator_output=$("$CGROUP" --tag "$tag" -- 2>&1)
  validator_rc=$?
  set -e
  [[ $validator_rc == 2 && $validator_output == "missing cgroup command" ]]
  printf '%s\n' "$tag"
  exit 0
elif [[ ${1:-} == --verify-model ]]; then
  [[ $# == 3 ]]
  verify_file "$2" "$3"
  exit $?
elif [[ ${1:-} == --driver ]]; then
  shift
  validate_execution_authority
  driver "$@"
  exit $?
fi

[[ $# == 0 && $(id -un) == bmarti44 ]] || exit 2
validate_execution_authority
verify_dependencies
verify_file "$MODEL" "$MODEL_SHA256"
[[ ${W7_RANDOM_SEED_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ -n ${W7_RANDOMNESS_RECEIPT_JSON:-} ]] || exit 2
readonly runtime_seed_sha256=$W7_RANDOM_SEED_SHA256
readonly harness_fd=${W7_SEALED_HARNESS_FD:-}
readonly scorer_fd=${W7_SEALED_SCORER_FD:-}
readonly trace_scorer_fd=${W7_SEALED_TRACE_SCORER_FD:-}
verify_sealed_fd "$harness_fd" "$W7_EXECUTED_HARNESS_SHA256"
verify_sealed_fd "$scorer_fd" "$SCORER_SHA256"
verify_sealed_fd "$trace_scorer_fd" "$TRACE_SCORER_SHA256"
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
/usr/bin/python3 -I -B - "$runtime_seed_sha256" \
  "$W7_FROZEN_CANDIDATE_COMMIT" "$W7_RANDOMNESS_RECEIPT_JSON" \
  "$attempt_out/randomness.json" <<'PY'
import hashlib, json, os, pathlib, sys
seed, candidate, raw, output = sys.argv[1:]
def strict(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate randomness key")
        out[key] = value
    return out
doc = json.loads(raw, object_pairs_hook=strict)
required = {
    "schema_version", "source", "freeze_floor_round", "round", "randomness",
    "signature", "previous_signature", "relay_agreement",
}
if (set(doc) != required or type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
        or doc["source"] != "drand-default-preregistered-three-relay"
        or type(doc["freeze_floor_round"]) is not int
        or type(doc["round"]) is not int
        or doc["round"] <= doc["freeze_floor_round"]
        or doc["relay_agreement"] != ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"]):
    raise SystemExit("invalid randomness receipt")
for key, length in (("randomness", 64), ("signature", 192), ("previous_signature", 192)):
    value = doc[key]
    if not isinstance(value, str) or len(value) != length:
        raise SystemExit("invalid randomness field")
    bytes.fromhex(value)
if hashlib.sha256(bytes.fromhex(doc["signature"])).hexdigest() != doc["randomness"]:
    raise SystemExit("randomness signature derivation mismatch")
material = (b"GLM52-W7-ARM-ORDER-V1\0" + candidate.encode() + b"\0"
            + str(doc["round"]).encode() + b"\0" + doc["randomness"].encode())
if hashlib.sha256(material).hexdigest() != seed:
    raise SystemExit("randomness seed derivation mismatch")
encoded = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(fd, encoded)
    os.fsync(fd)
finally:
    os.close(fd)
PY
readonly randomness_receipt_sha256=$(sha256sum -- "$attempt_out/randomness.json" | awk '{print $1}')
/usr/bin/python3 -I -B - "$runtime_seed_sha256" "$attempt_out/arm-order" <<'PY'
import hashlib, pathlib, sys
seed = bytes.fromhex(sys.argv[1])
arms = sorted(("strict", "candidate", "cold"), key=lambda arm: hashlib.sha256(seed + arm.encode()).digest())
pathlib.Path(sys.argv[2]).write_text("\n".join(arms) + "\n")
PY
while IFS= read -r arm; do run_arm "$arm" "$attempt_out"; done <"$attempt_out/arm-order"
verify_dependencies
/usr/bin/python3 -I -B "/proc/$$/fd/$scorer_fd" \
  --strict "$attempt_out/strict" --candidate "$attempt_out/candidate" \
  --cold "$attempt_out/cold" --output "$attempt_out/summary.json" \
  --trace-scorer "/proc/$$/fd/$trace_scorer_fd" --pool "$POOL" \
  --tokenizer "$TOKENIZER" --tokenizer-runtime "$TOKENIZER_RUNTIME" \
  --harness-sha256 "$W7_EXECUTED_HARNESS_SHA256" \
  --binary-sha256 "$BINARY_SHA256" --model-sha256 "$MODEL_SHA256" \
  --scorer-sha256 "$SCORER_SHA256" --seed-sha256 "$runtime_seed_sha256" \
  --trace-scorer-sha256 "$TRACE_SCORER_SHA256" --fixture-sha256 "$POOL_SHA256" \
  --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --tokenizer-init-sha256 "$TOKENIZER_INIT_SHA256" \
  --tokenizer-native-sha256 "$TOKENIZER_NATIVE_SHA256" \
  --cgroup-sha256 "$CGROUP_SHA256" --safe-sha256 "$SAFE_SHA256" \
  --memory-guard-sha256 "$MEMORY_GUARD_SHA256" \
  --engine-freeze-sha256 "$ENGINE_FREEZE_SHA256" \
  --configuration-sha256 "$CONFIGURATION_SHA256" \
  --engine-source-commit "$ENGINE_SOURCE_COMMIT" \
  --randomness-receipt-sha256 "$randomness_receipt_sha256" \
  --binary-path "$BIN" --binary-device-inode "$(stat -Lc '%d:%i' -- "$BIN")" \
  --arm-order "$attempt_out/arm-order" \
  >"$attempt_out/scorer.stdout" 2>"$attempt_out/scorer.stderr"
[[ -s $attempt_out/manifest.json && -s $attempt_out/raw.jsonl && -s $attempt_out/summary.json ]]
sync -f "$attempt_out"
echo "W7 production equivalence evidence: $attempt_out"
