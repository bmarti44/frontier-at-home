#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/bmarti44/spark-deepseek-v4-flash
TAG=${MATCHED_TAG:?MATCHED_TAG is required}
SEED=${MATCHED_SEED:?MATCHED_SEED is required}
BLOCKS=${MATCHED_BLOCKS:-5}
PORT=${MATCHED_PORT:-8021}
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,19}$ ]] || { echo "invalid MATCHED_TAG" >&2; exit 2; }
[[ $SEED =~ ^[0-9]+$ ]] || { echo "MATCHED_SEED must be a non-negative integer" >&2; exit 2; }
[[ $BLOCKS =~ ^[1-5]$ ]] || { echo "MATCHED_BLOCKS must be 1-5" >&2; exit 2; }
[[ $PORT =~ ^[0-9]{4,5}$ ]] || { echo "invalid MATCHED_PORT" >&2; exit 2; }
PORT=$((10#$PORT))
(( PORT >= 1024 && PORT <= 65535 )) || { echo "invalid MATCHED_PORT" >&2; exit 2; }

OUT=/home/bmarti44/.local/state/glm52-decisive-$TAG
CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
GLM_ARM=$REPO/results/glm52-goal/harness/glm_decisive_arm.sh
DSV4_ARM=$REPO/results/glm52-goal/harness/dsv4_decisive_arm.sh
GLM_PROFILE=$REPO/configs/glm52-lossless-plateau-profile.json
DSV4_PROFILE=$REPO/configs/dsv4-matched-32k-profile.json
CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
GLM_CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6
GLM_BINARY_SHA256=eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2
FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'
GLM_ENV_ALLOWLIST=DS4_CUDA_EXPERT_CACHE_GB,DS4_CUDA_EXPERT_CACHE_PIN,DS4_CUDA_EXPERT_CACHE_SLRU,DS4_CUDA_FETCH_THREADS,DS4_CUDA_IQ2_DOWN_REFERENCE,DS4_CUDA_MOE_NO_ATOMIC_DOWN,DS4_CUDA_STABLE_MODEL_REMAP,DS4_TOKEN_TIMING_LOG

wait_full_release() {
    python3 "$REPO/scripts/03_memory_guard.py" \
        --required-gib 110 --stable-samples 3 --timeout-seconds 180
}

assert_idle() {
    local label=$1
    for process_name in ds4-server llama-server fio; do
        if pgrep -x "$process_name" >/dev/null; then
            echo "$label: unexpected process remains: $process_name" >&2
            return 1
        fi
    done
    if ss -H -ltn "sport = :$PORT" | grep -q .; then
        echo "$label: selected listener remains on port $PORT" >&2
        return 1
    fi
    if systemctl --user list-units 'glm52-*' --state=active --no-legend |
            grep -q .; then
        echo "$label: a GLM campaign supervisor remains active" >&2
        return 1
    fi
    return 0
}

kernel_cursor() {
    journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p'
}

assert_no_kernel_faults_since() {
    local cursor=$1 output=$2
    [[ -n $cursor ]] || { echo "kernel cursor is missing" >&2; return 1; }
    journalctl -k --after-cursor "$cursor" --no-pager >"$output"
    ! grep -Eiq "$FAULT_PATTERN" "$output"
}

verify_campaign_artifacts() {
    python3 - "$REPO" "$GLM_PROFILE" "$DSV4_PROFILE" "$OUT/campaign-preflight.json" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve()
glm_profile_path = pathlib.Path(sys.argv[2])
dsv_profile_path = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])

def digest(path, *, evict=False):
    value = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while True:
            chunk = stream.read(16 * 1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
        before = os.fstat(stream.fileno())
        if evict:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise SystemExit(f"artifact changed while hashing: {path}")
    return value.hexdigest(), before

def load_profile(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)

def verify_repo_artifacts(profile):
    bindings = profile.get("artifact_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise SystemExit("profile has no artifact bindings")
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise SystemExit(f"artifact escapes repository: {relative}")
        actual, _ = digest(path)
        if actual != expected:
            raise SystemExit(f"campaign artifact digest mismatch: {relative}")

glm = load_profile(glm_profile_path)
dsv = load_profile(dsv_profile_path)
verify_repo_artifacts(glm)
verify_repo_artifacts(dsv)
if pathlib.Path(glm["model_path"]).is_symlink() or os.access(glm["model_path"], os.W_OK):
    raise SystemExit("GLM model must be a non-symlink not writable by the campaign owner")
glm_digest, glm_info = digest(glm["model_path"], evict=True)
if glm_digest != glm["model_sha256"] or glm_info.st_size != glm["model_bytes"]:
    raise SystemExit("GLM model identity mismatch")
for path, expected in dsv["runtime_closure_sha256"].items():
    actual, _ = digest(path)
    if actual != expected:
        raise SystemExit(f"DeepSeek runtime closure mismatch: {path}")
with open(dsv["weights_manifest_path"], encoding="utf-8") as stream:
    weights = json.load(stream)
manifest_digest, _ = digest(dsv["weights_manifest_path"])
if manifest_digest != dsv["weights_manifest_sha256"]:
    raise SystemExit("DeepSeek weights manifest mismatch")
for entry in weights["files"]:
    path = root / "weights" / "unsloth-ud-q2_k_xl" / entry["name"]
    if path.is_symlink() or os.access(path, os.W_OK):
        raise SystemExit(f"DeepSeek model shard is writable or a symlink: {path}")
    actual, info = digest(path, evict=True)
    if actual != entry["sha256"] or info.st_size != entry["bytes"]:
        raise SystemExit(f"DeepSeek model shard mismatch: {path}")
environment = glm["runtime"]["engine_environment"]
canonical = "".join(f"{name}={environment[name]}\n" for name in sorted(environment))
record = {
    "dsv4_binary": dsv["binary_path"],
    "dsv4_binary_sha256": dsv["binary_sha256"],
    "glm_environment_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    "glm_model_device_inode_size": f"{glm_info.st_dev}:{glm_info.st_ino}:{glm_info.st_size}",
    "glm_model_sha256": glm_digest,
}
with open(output, "x", encoding="ascii") as stream:
    json.dump(record, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

copy_safety_evidence() {
    local safe_tag=$1 arm_out=$2
    local -a matches=()
    mapfile -t matches < <(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 \
        -type d -name "*-$safe_tag" -print)
    (( ${#matches[@]} == 1 )) || {
        echo "$safe_tag: expected exactly one safety evidence directory" >&2
        return 1
    }
    cp -- "${matches[0]}/samples.log" "$arm_out/samples.log"
    cp -- "${matches[0]}/main.log" "$arm_out/safety.main.log"
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    assert_idle cleanup || rc=1
    wait_full_release >/dev/null 2>&1 || rc=1
    exit "$rc"
}
trap cleanup EXIT

[[ ! -e $OUT ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p -- "$OUT"
assert_idle initial
wait_full_release >"$OUT/initial-memory.json"
verify_campaign_artifacts
GLM_MODEL_IDENTITY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["glm_model_device_inode_size"])' "$OUT/campaign-preflight.json")
GLM_ENV_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["glm_environment_sha256"])' "$OUT/campaign-preflight.json")
DSV4_BINARY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dsv4_binary"])' "$OUT/campaign-preflight.json")
DSV4_BINARY_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dsv4_binary_sha256"])' "$OUT/campaign-preflight.json")

run_glm() {
    local label=$1 arm_out=$OUT/$label safe_tag="$TAG-$label" cursor rc=0
    [[ ! -e $arm_out ]] || return 1
    [[ -z $(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$safe_tag" -print -quit) ]] || return 1
    mkdir -p -- "$arm_out"
    wait_full_release >/dev/null
    cursor=$(kernel_cursor)
    set +e
    env GLM_SAFE_RUN_AS_CURRENT_USER=1 \
        GLM_CANDIDATE_SRC="$GLM_CANDIDATE_SRC" \
        GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
        GLM_SAFE_EXPECTED_BINARY_SHA256="$GLM_BINARY_SHA256" \
        GLM_SAFE_PROVENANCE_ENV_ALLOWLIST="$GLM_ENV_ALLOWLIST" \
        GLM_SAFE_EXPECTED_ENV_SHA256="$GLM_ENV_SHA256" \
        GLM_SAFE_KILL_FLOOR_GIB=40 GLM_SAFE_MIN_START_GIB=110 \
        GLM_SAFE_TIMEOUT_S=5400 GLM_PORT="$PORT" \
        GLM_VERIFIED_MODEL_DEVICE_INODE_SIZE="$GLM_MODEL_IDENTITY" \
        "$CGROUP" --tag "$safe_tag" -- \
        bash "$GLM_ARM" "$arm_out" "$label" "$SEED"
    rc=$?
    set -e
    copy_safety_evidence "$safe_tag" "$arm_out" || rc=1
    assert_no_kernel_faults_since "$cursor" "$arm_out/kernel.log" || rc=1
    assert_idle "$label" || rc=1
    wait_full_release >/dev/null || rc=1
    return "$rc"
}

run_dsv4() {
    local label=$1 arm_out=$OUT/$label safe_tag="$TAG-$label" cursor rc=0
    [[ ! -e $arm_out ]] || return 1
    [[ -z $(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$safe_tag" -print -quit) ]] || return 1
    mkdir -p -- "$arm_out"
    wait_full_release >/dev/null
    cursor=$(kernel_cursor)
    set +e
    env GLM_SAFE_RUN_AS_CURRENT_USER=1 GLM_SAFE_KILL_FLOOR_GIB=18 \
        GLM_SAFE_MIN_START_GIB=110 GLM_SAFE_TIMEOUT_S=5400 \
        DSV4_MATCHED_BINARY="$DSV4_BINARY" \
        DSV4_MATCHED_BINARY_SHA256="$DSV4_BINARY_SHA256" \
        MATCHED_PORT="$PORT" \
        "$CGROUP" --tag "$safe_tag" -- \
        bash "$DSV4_ARM" "$arm_out" "$label" "$SEED"
    rc=$?
    set -e
    copy_safety_evidence "$safe_tag" "$arm_out" || rc=1
    assert_no_kernel_faults_since "$cursor" "$arm_out/kernel.log" || rc=1
    assert_idle "$label" || rc=1
    wait_full_release >/dev/null || rc=1
    return "$rc"
}

run_arm() {
    local arm=$1 label=$2
    if (( SEED % 2 == 1 )); then
        if [[ $arm == A ]]; then run_dsv4 "$label"; else run_glm "$label"; fi
    else
        if [[ $arm == A ]]; then run_glm "$label"; else run_dsv4 "$label"; fi
    fi
}

for ((block=0; block<BLOCKS; block++)); do
    (( block % 2 == 0 )) && order=ABBA || order=BAAB
    for sequence in 0 1 2 3; do
        arm=${order:sequence:1}
        run_arm "$arm" "block${block}-seq${sequence}-arm${arm}"
    done
done

assert_idle terminal
wait_full_release >"$OUT/terminal-memory.json"
trap - EXIT
echo "DECISIVE_MATCHED_DONE out=$OUT"
