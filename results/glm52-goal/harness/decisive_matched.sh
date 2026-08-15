#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/bmarti44/spark-deepseek-v4-flash
TAG=${MATCHED_TAG:?MATCHED_TAG is required}
[[ ! -v MATCHED_SEED ]] || {
    echo "MATCHED_SEED is prohibited; the public receipt derives it" >&2
    exit 2
}
BLOCKS=${MATCHED_BLOCKS:-5}
PORT=${MATCHED_PORT:-8021}
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,19}$ ]] || { echo "invalid MATCHED_TAG" >&2; exit 2; }
[[ $BLOCKS =~ ^[1-5]$ ]] || { echo "MATCHED_BLOCKS must be 1-5" >&2; exit 2; }
[[ $PORT =~ ^[0-9]{4,5}$ ]] || { echo "invalid MATCHED_PORT" >&2; exit 2; }
PORT=$((10#$PORT))
(( PORT >= 1024 && PORT <= 65535 )) || { echo "invalid MATCHED_PORT" >&2; exit 2; }

OUT=/home/bmarti44/.local/state/glm52-decisive-$TAG
PYTHON=/usr/bin/python3.12
PYTHON_SHA256=a7d56a8a764faf7bbf5c164055a48fd072be52287bdeb523a9e07b2042f4e7e1
FREEZE_RECEIPT=results/glm52-gates/lossless-plateau-candidate13-preaudit.json
RANDOMNESS_RELATIVE=results/glm52-gates/lossless-plateau-candidate13-randomness.json
if [[ ${MATCHED_RETAINED_RUNTIME:-0} == 1 ]]; then
    RANDOMNESS_INPUT=${MATCHED_RANDOMNESS_RECEIPT:?retained randomness receipt is required}
    [[ $RANDOMNESS_INPUT == "$OUT/retained/randomness-receipt.json" ]] || {
        echo "retained randomness receipt path is not canonical" >&2
        exit 2
    }
else
    RANDOMNESS_INPUT=${MATCHED_RANDOMNESS_RECEIPT:?MATCHED_RANDOMNESS_RECEIPT is required}
    [[ $RANDOMNESS_INPUT == "$REPO/$RANDOMNESS_RELATIVE" ]] || {
        echo "randomness receipt path is not the candidate-13 canonical path" >&2
        exit 2
    }
fi
[[ ! -v PYTHONPATH && ! -v PYTHONHOME && ! -v PYTHONSTARTUP ]] || {
    echo "Python environment injection is prohibited" >&2
    exit 2
}
[[ $(sha256sum "$PYTHON" | awk '{print $1}') == "$PYTHON_SHA256" ]] || {
    echo "bound Python executable changed" >&2
    exit 2
}
if [[ ${MATCHED_RETAINED_RUNTIME:-0} != 1 ]]; then
    [[ ! -e $OUT ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
    head=$(git -C "$REPO" rev-parse HEAD)
    git -C "$REPO" show "$head:results/glm52-goal/harness/decisive_matched.sh" |
        cmp -s -- - "${BASH_SOURCE[0]}" || {
            echo "campaign entrypoint differs from committed HEAD" >&2
            exit 1
        }
    mkdir -p -- "$OUT/retained"
    env -i HOME=/home/bmarti44 PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        "$PYTHON" -I -B -S - "$REPO" "$head" "$OUT/retained" \
        "$OUT/retained-manifest.json" "$FREEZE_RECEIPT" \
        "$RANDOMNESS_RELATIVE" "$RANDOMNESS_INPUT" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys

repo = pathlib.Path(sys.argv[1])
head = sys.argv[2]
retained = pathlib.Path(sys.argv[3])
manifest_path = pathlib.Path(sys.argv[4])
receipt_path = sys.argv[5]
randomness_relative = sys.argv[6]
randomness_input = pathlib.Path(sys.argv[7])
profiles = (
    "configs/glm52-lossless-plateau-profile.json",
    "configs/dsv4-matched-32k-profile.json",
)

def git_bytes(relative, commit=head):
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout

receipt = json.loads(git_bytes(receipt_path))
reviewed_commit = receipt.get("candidate_commit")
if not isinstance(reviewed_commit, str) or len(reviewed_commit) != 40:
    raise SystemExit("reviewed runtime commit is missing")
subprocess.run(
    ["git", "-C", str(repo), "merge-base", "--is-ancestor", reviewed_commit, head],
    check=True,
)
freeze_commit = subprocess.run(
    [
        "git", "-C", str(repo), "log", "-1", "--diff-filter=A",
        "--format=%H", "--", receipt_path,
    ],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
).stdout.strip()
if len(freeze_commit) != 40:
    raise SystemExit("candidate freeze commit is missing")
subprocess.run(
    ["git", "-C", str(repo), "merge-base", "--is-ancestor", reviewed_commit, freeze_commit],
    check=True,
)
subprocess.run(
    ["git", "-C", str(repo), "merge-base", "--is-ancestor", freeze_commit, head],
    check=True,
)
profile_docs = [json.loads(git_bytes(path)) for path in profiles]
python_runtimes = [profile.get("python_runtime") for profile in profile_docs]
if not isinstance(python_runtimes[0], dict) or python_runtimes[0] != python_runtimes[1]:
    raise SystemExit("matched profiles disagree on the Python runtime")
python_runtime = python_runtimes[0]
paths = set(profiles)
for profile in profile_docs:
    bindings = profile.get("artifact_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise SystemExit("retained profile has no artifact bindings")
    paths.update(bindings)
required = {
    "results/glm52-goal/harness/decisive_matched.sh",
    "results/glm52-goal/harness/glm_decisive_arm.sh",
    "results/glm52-goal/harness/dsv4_decisive_arm.sh",
    "results/glm52-gates/harness/glm_cgroup_run.sh",
    "results/glm52-gates/harness/glm_safe_run.sh",
    "results/glm52-gates/harness/dsv4_matched_cgroup_run.sh",
    "scripts/03_memory_guard.py",
    "scripts/30_bench_speed.py",
    "scripts/56_collect_matched_evidence.py",
    "scripts/89_verify_drand_receipt.mjs",
    "scripts/glm52_goal.py",
    "fixtures/ctx-32k.txt",
    "vendor/official-encoding/tokenizer.json",
}
if not required <= paths:
    raise SystemExit(f"profiles omit required retained closure: {sorted(required - paths)}")
digests = {}
for relative in sorted(paths):
    if relative.startswith("/") or ".." in pathlib.PurePosixPath(relative).parts:
        raise SystemExit(f"unsafe retained path: {relative}")
    raw = git_bytes(relative)
    if raw != git_bytes(relative, reviewed_commit):
        raise SystemExit(f"path differs from reviewed runtime commit: {relative}")
    expected_values = {
        profile["artifact_sha256"].get(relative)
        for profile in profile_docs
        if relative in profile["artifact_sha256"]
    }
    actual = hashlib.sha256(raw).hexdigest()
    if expected_values and expected_values != {actual}:
        raise SystemExit(f"profile digest mismatch at committed HEAD: {relative}")
    destination = retained / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    digests[relative] = actual

expected_randomness_path = repo / randomness_relative
if (
    randomness_input != expected_randomness_path
    or randomness_input.is_symlink()
    or not randomness_input.is_file()
):
    raise SystemExit("randomness receipt input path is absent or unsafe")
randomness_raw = git_bytes(randomness_relative)
if randomness_input.read_bytes() != randomness_raw:
    raise SystemExit("randomness receipt differs from committed HEAD")
randomness_destination = retained / "randomness-receipt.json"
fd = os.open(randomness_destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "wb") as stream:
    stream.write(randomness_raw)
    stream.flush()
    os.fsync(stream.fileno())
randomness_receipt_sha256 = hashlib.sha256(randomness_raw).hexdigest()
digests["randomness-receipt.json"] = randomness_receipt_sha256

freeze_receipt_raw = git_bytes(receipt_path)
freeze_destination = retained / "freeze-receipt.json"
fd = os.open(freeze_destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "wb") as stream:
    stream.write(freeze_receipt_raw)
    stream.flush()
    os.fsync(stream.fileno())
freeze_receipt_sha256 = hashlib.sha256(freeze_receipt_raw).hexdigest()
digests["freeze-receipt.json"] = freeze_receipt_sha256

def sha256_path(path):
    value = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def stdlib_tree(path):
    root = pathlib.Path(path)
    value = hashlib.sha256()
    for item in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        relative = item.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode):
            kind = b"L"
            payload = item.readlink().as_posix().encode()
        elif stat.S_ISREG(info.st_mode):
            kind = b"F"
            payload = bytes.fromhex(sha256_path(item))
        else:
            continue
        name = relative.as_posix().encode()
        value.update(kind + len(name).to_bytes(4, "big") + name)
        value.update(info.st_size.to_bytes(8, "big") + payload)
    return value.hexdigest()

for path_key, digest_key in (
    ("executable_path", "executable_sha256"),
    ("libpython_path", "libpython_sha256"),
    ("tokenizer_native_path", "tokenizer_native_sha256"),
):
    if sha256_path(python_runtime[path_key]) != python_runtime[digest_key]:
        raise SystemExit(f"Python runtime changed: {path_key}")
if stdlib_tree(python_runtime["stdlib_path"]) != python_runtime["stdlib_tree_sha256"]:
    raise SystemExit("Python standard-library tree changed")
native_raw = pathlib.Path(python_runtime["tokenizer_native_path"]).read_bytes()
native_destination = retained / "runtime/tokenizers.abi3.so"
native_destination.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(native_destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
with os.fdopen(fd, "wb") as stream:
    stream.write(native_raw)
    stream.flush()
    os.fsync(stream.fileno())
digests["runtime/tokenizers.abi3.so"] = hashlib.sha256(native_raw).hexdigest()
manifest = {
    "schema": "matched-retained-closure-v1",
    "git_head": head,
    "reviewed_runtime_commit": reviewed_commit,
    "freeze_commit": freeze_commit,
    "freeze_receipt_sha256": freeze_receipt_sha256,
    "randomness_receipt_sha256": randomness_receipt_sha256,
    "python_runtime": python_runtime,
    "sha256": digests,
}
with open(manifest_path, "x", encoding="ascii") as stream:
    json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(retained, 0o500)
PY
    exec env -i HOME=/home/bmarti44 XDG_RUNTIME_DIR=/run/user/1000 \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        LANG=C.UTF-8 LC_ALL=C.UTF-8 MATCHED_RETAINED_RUNTIME=1 \
        MATCHED_TAG="$TAG" \
        MATCHED_RANDOMNESS_RECEIPT="$OUT/retained/randomness-receipt.json" \
        MATCHED_BLOCKS="$BLOCKS" MATCHED_PORT="$PORT" \
        /usr/bin/bash "$OUT/retained/results/glm52-goal/harness/decisive_matched.sh"
fi

CODE_ROOT=$OUT/retained
CGROUP=$CODE_ROOT/results/glm52-gates/harness/glm_cgroup_run.sh
DSV4_CGROUP=$CODE_ROOT/results/glm52-gates/harness/dsv4_matched_cgroup_run.sh
GLM_ARM=$CODE_ROOT/results/glm52-goal/harness/glm_decisive_arm.sh
DSV4_ARM=$CODE_ROOT/results/glm52-goal/harness/dsv4_decisive_arm.sh
GLM_PROFILE=$CODE_ROOT/configs/glm52-lossless-plateau-profile.json
DSV4_PROFILE=$CODE_ROOT/configs/dsv4-matched-32k-profile.json
BENCH=$CODE_ROOT/scripts/30_bench_speed.py
COLLECTOR=$CODE_ROOT/scripts/56_collect_matched_evidence.py
DRAND_VERIFIER=$CODE_ROOT/scripts/89_verify_drand_receipt.mjs
TOKENIZER_NATIVE=$CODE_ROOT/runtime/tokenizers.abi3.so
TOKENIZER_NATIVE_SHA256=3c7e64a6cf423a4b675d535b0e56667382a02fa71f86380719d2a442ad98c1c7
CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
GLM_CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6
GLM_BINARY_SHA256=eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2
FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'
GLM_ENV_ALLOWLIST=DS4_CUDA_EXPERT_CACHE_GB,DS4_CUDA_EXPERT_CACHE_PIN,DS4_CUDA_EXPERT_CACHE_SLRU,DS4_CUDA_FETCH_THREADS,DS4_CUDA_IQ2_DOWN_REFERENCE,DS4_CUDA_MOE_NO_ATOMIC_DOWN,DS4_CUDA_STABLE_MODEL_REMAP,DS4_TOKEN_TIMING_LOG

readarray -t RANDOMNESS_BINDING < <(
    "$PYTHON" -I -B -S - "$COLLECTOR" "$RANDOMNESS_INPUT" \
        "$OUT/retained-manifest.json" "$DRAND_VERIFIER" "$REPO" <<'PY'
import importlib.util
import json
import pathlib
import sys

collector_path, receipt_path, manifest_path, verifier_path, source_repo = sys.argv[1:]
spec = importlib.util.spec_from_file_location("matched_evidence", collector_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load retained matched collector")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with open(manifest_path, encoding="ascii") as stream:
    manifest = json.load(stream)
candidate = manifest.get("reviewed_runtime_commit")
freeze = manifest.get("freeze_commit")
seed = module.verify_randomness_receipt(
    pathlib.Path(receipt_path),
    candidate_hash=candidate,
    freeze_commit=freeze,
    drand_verifier=pathlib.Path(verifier_path),
    retained_manifest=pathlib.Path(manifest_path),
    source_repo=pathlib.Path(source_repo),
)
with open(receipt_path, encoding="ascii") as stream:
    round_value = json.load(stream)["receipt"]["round"]
print(seed)
print(round_value)
print(candidate)
print(freeze)
PY
)
(( ${#RANDOMNESS_BINDING[@]} == 4 )) || {
    echo "randomness derivation did not produce an exact binding" >&2
    exit 2
}
MATCHED_DERIVED_SEED=${RANDOMNESS_BINDING[0]}
DRAND_ROUND=${RANDOMNESS_BINDING[1]}
CANDIDATE_HASH=${RANDOMNESS_BINDING[2]}
FREEZE_COMMIT=${RANDOMNESS_BINDING[3]}
[[ $MATCHED_DERIVED_SEED =~ ^[0-9]+$ && $DRAND_ROUND =~ ^[1-9][0-9]*$ &&
   $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ && $FREEZE_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
    echo "randomness binding output is malformed" >&2
    exit 2
}
[[ $TAG == "p13-r$DRAND_ROUND" ]] || {
    echo "campaign tag does not bind the verified drand round" >&2
    exit 2
}
SEED=$MATCHED_DERIVED_SEED

wait_full_release() {
    "$PYTHON" -I -B -S "$CODE_ROOT/scripts/03_memory_guard.py" \
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
    if systemctl --user list-units 'glm52-*' 'dsv4-matched-*' --state=active --no-legend |
            grep -q .; then
        echo "$label: a matched campaign supervisor remains active" >&2
        return 1
    fi
    return 0
}

stop_dsv_units_for_tag() {
    local tag=${1//./-} unit
    while read -r unit _; do
        [[ $unit == dsv4-matched-${tag}-*.service ]] || continue
        systemctl --user stop "$unit" >/dev/null 2>&1 || true
    done < <(
        systemctl --user list-units "dsv4-matched-${tag}-*" --all --no-legend \
            2>/dev/null || true
    )
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
    "$PYTHON" -I -B -S - "$CODE_ROOT" "$REPO" "$GLM_PROFILE" "$DSV4_PROFILE" \
        "$OUT/models" "$OUT/campaign-preflight.json" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve()
source_repo = pathlib.Path(sys.argv[2]).resolve()
glm_profile_path = pathlib.Path(sys.argv[3])
dsv_profile_path = pathlib.Path(sys.argv[4])
model_root = pathlib.Path(sys.argv[5]).resolve()
output = pathlib.Path(sys.argv[6])

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
shards = []
for entry in weights["files"]:
    source = source_repo / "weights" / "unsloth-ud-q2_k_xl" / entry["name"]
    path = model_root / entry["name"]
    if path.is_symlink() or os.access(path, os.W_OK):
        raise SystemExit(f"DeepSeek model shard is writable or a symlink: {path}")
    actual, info = digest(path, evict=True)
    source_info = source.stat()
    if (
        actual != entry["sha256"]
        or info.st_size != entry["bytes"]
        or (info.st_dev, info.st_ino) != (source_info.st_dev, source_info.st_ino)
    ):
        raise SystemExit(f"DeepSeek model shard mismatch: {path}")
    shards.append({
        "path": str(path),
        "name": entry["name"],
        "sha256": actual,
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
    })
environment = glm["runtime"]["engine_environment"]
canonical = "".join(f"{name}={environment[name]}\n" for name in sorted(environment))
record = {
    "dsv4_binary": dsv["binary_path"],
    "dsv4_binary_sha256": dsv["binary_sha256"],
    "glm_environment_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    "glm_model_device_inode_size": f"{glm_info.st_dev}:{glm_info.st_ino}:{glm_info.st_size}",
    "glm_model_sha256": glm_digest,
    "dsv4_shards": shards,
}
with open(output, "x", encoding="ascii") as stream:
    json.dump(record, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

prepare_dsv_model_links() {
    mkdir -p -- "$OUT/models"
    "$PYTHON" -I -B -S - "$DSV4_PROFILE" "$REPO" "$OUT/models" <<'PY'
import json
import os
import pathlib
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
repo = pathlib.Path(sys.argv[2])
target = pathlib.Path(sys.argv[3])
weights = json.load(open(profile["weights_manifest_path"], encoding="utf-8"))
for entry in weights["files"]:
    source = repo / "weights" / "unsloth-ud-q2_k_xl" / entry["name"]
    destination = target / entry["name"]
    os.link(source, destination)
PY
}

verify_terminal_closure() {
    "$PYTHON" -I -B -S - "$OUT/retained-manifest.json" "$CODE_ROOT" \
        "$OUT/campaign-preflight.json" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

manifest = json.load(open(sys.argv[1], encoding="ascii"))
root = pathlib.Path(sys.argv[2])
preflight = json.load(open(sys.argv[3], encoding="ascii"))
observed = set()
for path in root.rglob("*"):
    if path.is_file():
        relative = path.relative_to(root).as_posix()
        observed.add(relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["sha256"].get(relative) != actual:
            raise SystemExit(f"retained closure changed: {relative}")
if observed != set(manifest["sha256"]):
    raise SystemExit("retained closure membership changed")

def sha256_path(path):
    value = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def stdlib_tree(path):
    base = pathlib.Path(path)
    value = hashlib.sha256()
    for item in sorted(base.rglob("*"), key=lambda p: p.relative_to(base).as_posix()):
        relative = item.relative_to(base)
        if "__pycache__" in relative.parts:
            continue
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode):
            kind, payload = b"L", item.readlink().as_posix().encode()
        elif stat.S_ISREG(info.st_mode):
            kind, payload = b"F", bytes.fromhex(sha256_path(item))
        else:
            continue
        name = relative.as_posix().encode()
        value.update(kind + len(name).to_bytes(4, "big") + name)
        value.update(info.st_size.to_bytes(8, "big") + payload)
    return value.hexdigest()

runtime = manifest["python_runtime"]
for path_key, digest_key in (
    ("executable_path", "executable_sha256"),
    ("libpython_path", "libpython_sha256"),
):
    if sha256_path(runtime[path_key]) != runtime[digest_key]:
        raise SystemExit(f"executed Python runtime changed: {path_key}")
if stdlib_tree(runtime["stdlib_path"]) != runtime["stdlib_tree_sha256"]:
    raise SystemExit("executed Python standard-library tree changed")
for shard in preflight["dsv4_shards"]:
    path = pathlib.Path(shard["path"])
    info = path.stat()
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if (
        (info.st_dev, info.st_ino, info.st_size)
        != (shard["device"], shard["inode"], shard["bytes"])
        or digest.hexdigest() != shard["sha256"]
    ):
        raise SystemExit(f"DeepSeek shard changed during campaign: {path}")
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
    cp -- "${matches[0]}/kernel.log" "$arm_out/safety.kernel.log"
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    stop_dsv_units_for_tag "$TAG"
    assert_idle cleanup || rc=1
    wait_full_release >/dev/null 2>&1 || rc=1
    if [[ ${PREFLIGHT_DONE:-0} == 1 ]]; then
        verify_terminal_closure || rc=1
    fi
    exit "$rc"
}
trap cleanup EXIT

[[ -d $OUT && -d $CODE_ROOT && -f $OUT/retained-manifest.json ]] || {
    echo "retained campaign closure is missing" >&2
    exit 1
}
exec {INFERENCE_LOCK_FD}<>/run/lock/frontier-at-home/inference.lock
flock -n -E 75 "$INFERENCE_LOCK_FD"
PARENT_LOCK_PID=$$
PARENT_LOCK_START_TICKS=$(awk '{print $22}' "/proc/$$/stat")
PARENT_LOCK_DEV_INO=$(stat -Lc '%d:%i' "/proc/$$/fd/$INFERENCE_LOCK_FD")
PARENT_LOCK_KERNEL_KEY=$(awk '$1 == "lock:" && $3 == "FLOCK" && $5 == "WRITE" {print $7}' \
    "/proc/$$/fdinfo/$INFERENCE_LOCK_FD")
[[ $PARENT_LOCK_KERNEL_KEY =~ ^[0-9a-f]+:[0-9a-f]+:[0-9]+$ ]] || {
    echo "global inference-lock kernel binding is missing" >&2
    exit 1
}
exec {SAFE_FD}<"$CODE_ROOT/results/glm52-gates/harness/glm_safe_run.sh"
SAFE_FD_SHA256=$(sha256sum "/proc/$$/fd/$SAFE_FD" | awk '{print $1}')
assert_idle initial
wait_full_release >"$OUT/initial-memory.json"
prepare_dsv_model_links
verify_campaign_artifacts
PREFLIGHT_DONE=1
GLM_MODEL_IDENTITY=$("$PYTHON" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["glm_model_device_inode_size"])' "$OUT/campaign-preflight.json")
GLM_ENV_SHA256=$("$PYTHON" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["glm_environment_sha256"])' "$OUT/campaign-preflight.json")
DSV4_BINARY=$("$PYTHON" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["dsv4_binary"])' "$OUT/campaign-preflight.json")
DSV4_BINARY_SHA256=$("$PYTHON" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["dsv4_binary_sha256"])' "$OUT/campaign-preflight.json")
DSV4_MODEL_FIRST=$("$PYTHON" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["dsv4_shards"][0]["path"])' "$OUT/campaign-preflight.json")

run_glm() {
    local label=$1 arm_out=$OUT/$label safe_tag="$TAG-$label" cursor rc=0
    [[ ! -e $arm_out ]] || return 1
    [[ -z $(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$safe_tag" -print -quit) ]] || return 1
    mkdir -p -- "$arm_out"
    wait_full_release >/dev/null
    cursor=$(kernel_cursor)
    set +e
    env GLM_SAFE_RUN_AS_CURRENT_USER=1 \
        GLM_SAFE_PARENT_LOCK_PID="$PARENT_LOCK_PID" \
        GLM_SAFE_PARENT_LOCK_START_TICKS="$PARENT_LOCK_START_TICKS" \
        GLM_SAFE_PARENT_LOCK_FD="$INFERENCE_LOCK_FD" \
        GLM_SAFE_PARENT_LOCK_DEV_INO="$PARENT_LOCK_DEV_INO" \
        GLM_SAFE_PARENT_LOCK_KERNEL_KEY="$PARENT_LOCK_KERNEL_KEY" \
        GLM_SAFE_PINNED_SAFE_PATH="/proc/$$/fd/$SAFE_FD" \
        GLM_SAFE_PINNED_SAFE_SHA256="$SAFE_FD_SHA256" \
        GLM_CANDIDATE_SRC="$GLM_CANDIDATE_SRC" \
        GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
        GLM_SAFE_EXPECTED_BINARY_SHA256="$GLM_BINARY_SHA256" \
        GLM_SAFE_PROVENANCE_ENV_ALLOWLIST="$GLM_ENV_ALLOWLIST" \
        GLM_SAFE_EXPECTED_ENV_SHA256="$GLM_ENV_SHA256" \
        GLM_SAFE_KILL_FLOOR_GIB=40 GLM_SAFE_MIN_START_GIB=110 \
        GLM_SAFE_TIMEOUT_S=5400 GLM_SAFE_DONE_DIGESTS=1 GLM_PORT="$PORT" \
        MATCHED_BENCH_PATH="$BENCH" \
        MATCHED_PYTHON_PATH="$PYTHON" \
        MATCHED_TOKENIZER_NATIVE_PATH="$TOKENIZER_NATIVE" \
        MATCHED_TOKENIZER_NATIVE_SHA256="$TOKENIZER_NATIVE_SHA256" \
        GLM_VERIFIED_MODEL_DEVICE_INODE_SIZE="$GLM_MODEL_IDENTITY" \
        "$CGROUP" --tag "$safe_tag" -- \
        bash "$GLM_ARM" "$arm_out" "$label" "$SEED" \
        >"$arm_out/safety.wrapper.out" 2>"$arm_out/safety.wrapper.err"
    rc=$?
    set -e
    stop_dsv_units_for_tag "$safe_tag"
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
    env DSV4_MATCHED_KILL_FLOOR_GIB=8 DSV4_MATCHED_MIN_START_GIB=110 \
        DSV4_MATCHED_MEMORY_HIGH_GIB=100 DSV4_MATCHED_MEMORY_MAX_GIB=102 \
        DSV4_MATCHED_TIMEOUT_S=5400 \
        GLM_SAFE_PARENT_LOCK_PID="$PARENT_LOCK_PID" \
        GLM_SAFE_PARENT_LOCK_START_TICKS="$PARENT_LOCK_START_TICKS" \
        GLM_SAFE_PARENT_LOCK_FD="$INFERENCE_LOCK_FD" \
        GLM_SAFE_PARENT_LOCK_DEV_INO="$PARENT_LOCK_DEV_INO" \
        GLM_SAFE_PARENT_LOCK_KERNEL_KEY="$PARENT_LOCK_KERNEL_KEY" \
        DSV4_MATCHED_BINARY="$DSV4_BINARY" \
        DSV4_MATCHED_BINARY_SHA256="$DSV4_BINARY_SHA256" \
        DSV4_MATCHED_MODEL_FIRST="$DSV4_MODEL_FIRST" \
        DSV4_MATCHED_SHARDS_JSON="$OUT/campaign-preflight.json" \
        MATCHED_BENCH_PATH="$BENCH" \
        MATCHED_PYTHON_PATH="$PYTHON" \
        MATCHED_TOKENIZER_NATIVE_PATH="$TOKENIZER_NATIVE" \
        MATCHED_TOKENIZER_NATIVE_SHA256="$TOKENIZER_NATIVE_SHA256" \
        MATCHED_PORT="$PORT" \
        "$DSV4_CGROUP" --tag "$safe_tag" -- \
        bash "$DSV4_ARM" "$arm_out" "$label" "$SEED" \
        >"$arm_out/safety.wrapper.out" 2>"$arm_out/safety.wrapper.err"
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
verify_terminal_closure
"$PYTHON" -I -B -S "$COLLECTOR" "$OUT" \
    --fixture "$CODE_ROOT/fixtures/ctx-32k.txt" \
    --dsv4-profile "$DSV4_PROFILE" --glm-profile "$GLM_PROFILE" \
    --serving-manifest "$REPO/weights/unsloth-ud-q2_k_xl/manifest.json" \
    --randomness-receipt "$RANDOMNESS_INPUT" \
    --candidate-hash "$CANDIDATE_HASH" --freeze-commit "$FREEZE_COMMIT" \
    --drand-verifier "$DRAND_VERIFIER" --source-repo "$REPO" \
    --out "$OUT/raw.jsonl"
verify_terminal_closure
PREFLIGHT_DONE=0
trap - EXIT
echo "DECISIVE_MATCHED_DONE out=$OUT"
