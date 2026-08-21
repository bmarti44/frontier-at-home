#!/usr/bin/env bash
# Build the pinned poolsideai Laguna llama.cpp revision for the GB10 Spark.
set -Eeuo pipefail
umask 077

readonly LLAMACPP_REPOSITORY=https://github.com/poolsideai/llama.cpp
readonly LLAMACPP_BRANCH=laguna
readonly LLAMACPP_COMMIT=06f8cebd7fe728687be3d19f8bdedb70d75883af
readonly CACHE_ROOT=${HOME:?HOME must be set}/.cache/llamacpp-laguna-06f8cebd
readonly SOURCE_DIR=$CACHE_ROOT/src
readonly BUILD_DIR=$SOURCE_DIR/build

usage() {
    cat <<'EOF'
Usage: 13_build_laguna_llamacpp.sh [--help]

Clone and clean-build pinned poolsideai llama.cpp branch laguna for CUDA sm_121
using at most two build jobs. The script does not download model weights or
start a server.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case ${1:-} in
    '') ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) \
    || die 'cannot resolve script directory'
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) \
    || die 'cannot resolve repository root'
MANIFEST=$REPO_ROOT/configs/build-manifests/llamacpp-laguna-06f8cebd.json

for command_name in awk cmake flock git python3 uname; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required command not found: $command_name"
done

cmake_flags=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES=121
    -DCMAKE_BUILD_TYPE=Release
)

mkdir -p -- "$CACHE_ROOT"
if [[ -e $CACHE_ROOT/build.lock ]]; then
    exec 9<"$CACHE_ROOT/build.lock"
else
    exec 9>"$CACHE_ROOT/build.lock"
fi
flock -n 9 || die 'another Laguna llama.cpp build holds the build lock'

SERVER_BINARY=$BUILD_DIR/bin/llama-server
CLI_BINARY=$BUILD_DIR/bin/llama-cli
if [[ -x $SERVER_BINARY && -f $MANIFEST ]] && python3 - \
    "$MANIFEST" "$LLAMACPP_REPOSITORY" "$LLAMACPP_BRANCH" \
    "$LLAMACPP_COMMIT" "$SERVER_BINARY" "$CLI_BINARY" "${cmake_flags[@]}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

(
    manifest_name,
    repository,
    branch,
    commit,
    server_name,
    cli_name,
    *cmake_flags,
) = sys.argv[1:]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


try:
    with open(manifest_name, encoding="utf-8") as stream:
        manifest = json.load(stream)
    server = manifest["binaries"]["llama-server"]
    cli = manifest["binaries"]["llama-cli"]
    binary_directory = pathlib.Path(server_name).parent
    libraries = manifest["shared_libraries"]
    source_directory = manifest["source_directory"]
    matches = (
        manifest["repository"] == repository
        and manifest["branch"] == branch
        and manifest["commit"] == commit
        and manifest["cmake_flags"] == cmake_flags
        and server["path"] == server_name
        and server["sha256"] == digest(pathlib.Path(server_name))
        and cli["path"] == cli_name
        and cli["sha256"] == digest(pathlib.Path(cli_name))
        and isinstance(libraries, dict)
        and bool(libraries)
        and all(
            pathlib.Path(name).name == name
            and name.endswith(".so")
            and entry["sha256"] == digest(binary_directory / name)
            for name, entry in libraries.items()
        )
        and subprocess.run(
            ["git", "-C", source_directory, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip() == commit
        and subprocess.run(
            ["git", "-C", source_directory, "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout == ""
    )
except (KeyError, OSError, subprocess.SubprocessError, TypeError, ValueError):
    matches = False
raise SystemExit(0 if matches else 1)
PY
then
    printf 'Pinned Laguna llama.cpp build already exists and is verified.\nllama_server=%s\nmanifest=%s\n' \
        "$SERVER_BINARY" "$MANIFEST"
    exit 0
fi

[[ $(uname -m) == aarch64 ]] || die 'this build requires aarch64'
[[ -x /usr/local/cuda/bin/nvcc ]] || die 'CUDA nvcc is missing: /usr/local/cuda/bin/nvcc'
nvcc_version=$(/usr/local/cuda/bin/nvcc --version) || die 'nvcc --version failed'
[[ $nvcc_version =~ release[[:space:]]13\.0 ]] \
    || die 'this build requires CUDA toolkit release 13.0'

available_kib=$(awk '$1 == "MemAvailable:" {print $2; found=1; exit} END {if (!found) exit 1}' /proc/meminfo) \
    || die 'cannot read MemAvailable'
(( available_kib >= 11 * 1048576 )) ||
    die 'less than 11 GiB is available for the build'
# Below the production-stopped floor, a build is permitted only when this
# process is already inside the required cgroup-v2 memory containment.
if (( available_kib < 110 * 1048576 )); then
    cgroup_path=$(awk -F: '$1 == "0" && $2 == "" {print $3; found=1; exit} END {if (!found) exit 1}' \
        /proc/self/cgroup) || die 'cannot resolve this process cgroup'
    [[ $cgroup_path == /* && $cgroup_path != *'/../'* && $cgroup_path != */.. ]] \
        || die "unsafe cgroup path reported for this process: $cgroup_path"
    cgroup_dir=/sys/fs/cgroup${cgroup_path%/}
    [[ -r $cgroup_dir/memory.max && -r $cgroup_dir/memory.swap.max ]] \
        || die 'build requires cgroup-v2 memory.max and memory.swap.max containment'
    read -r memory_max <"$cgroup_dir/memory.max" \
        || die 'cannot read this process memory.max'
    read -r memory_swap_max <"$cgroup_dir/memory.swap.max" \
        || die 'cannot read this process memory.swap.max'
    [[ $memory_max =~ ^[0-9]+$ && $memory_swap_max == 0 ]] \
        || die 'resident-production build requires numeric memory.max and memory.swap.max=0'
    (( memory_max <= 11 * 1024 * 1024 * 1024 )) \
        || die 'resident-production build requires memory.max no greater than 11 GiB'
fi

printf '[1/6] Preparing isolated Laguna llama.cpp source cache: %s\n' "$SOURCE_DIR" >&2
fresh_clone=false
if [[ ! -e $SOURCE_DIR ]]; then
    git clone --no-checkout --branch "$LLAMACPP_BRANCH" -- \
        "$LLAMACPP_REPOSITORY" "$SOURCE_DIR" >&2 \
        || die 'Laguna llama.cpp clone failed'
    fresh_clone=true
fi
[[ -d $SOURCE_DIR/.git ]] || die "source cache is not a Git clone: $SOURCE_DIR"
origin_url=$(git -C "$SOURCE_DIR" remote get-url origin) \
    || die 'cannot read Laguna llama.cpp origin URL'
case $origin_url in
    "$LLAMACPP_REPOSITORY"|"$LLAMACPP_REPOSITORY.git") ;;
    *) die "unexpected Laguna llama.cpp origin: $origin_url" ;;
esac
# A fresh --no-checkout clone has an empty worktree (status shows all files
# deleted), so the tamper check only applies to a pre-existing cache.
if [[ $fresh_clone == false ]]; then
    [[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
        || die "Laguna llama.cpp source cache is dirty: $SOURCE_DIR"
fi

printf '[2/6] Fetching branch %s and checking out pinned commit %s...\n' \
    "$LLAMACPP_BRANCH" "$LLAMACPP_COMMIT" >&2
git -C "$SOURCE_DIR" fetch --force origin \
    "refs/heads/$LLAMACPP_BRANCH:refs/remotes/origin/$LLAMACPP_BRANCH" >&2 \
    || die "failed to fetch Laguna llama.cpp branch $LLAMACPP_BRANCH"
git -C "$SOURCE_DIR" cat-file -e "$LLAMACPP_COMMIT^{commit}" \
    || die "pinned commit is missing from branch fetch: $LLAMACPP_COMMIT"
git -C "$SOURCE_DIR" merge-base --is-ancestor \
    "$LLAMACPP_COMMIT" "refs/remotes/origin/$LLAMACPP_BRANCH" \
    || die "pinned commit is not on origin/$LLAMACPP_BRANCH"
git -C "$SOURCE_DIR" checkout --detach "$LLAMACPP_COMMIT" >&2 \
    || die 'failed to check out pinned Laguna llama.cpp commit'
actual_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD) \
    || die 'cannot read checked-out Laguna llama.cpp commit'
[[ $actual_commit == "$LLAMACPP_COMMIT" ]] \
    || die "checked-out commit mismatch: $actual_commit"
[[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
    || die 'Laguna llama.cpp source became dirty after checkout'

printf '[3/6] Removing the prior CMake build tree...\n' >&2
[[ $BUILD_DIR == "$CACHE_ROOT/src/build" ]] || die 'refusing unsafe build-directory removal'
build_parent=$(cd -- "$(dirname -- "$BUILD_DIR")" && pwd -P) \
    || die 'cannot resolve physical build-directory parent'
physical_build_dir=$build_parent/$(basename -- "$BUILD_DIR")
expected_build_dir=${HOME:?HOME must be set}/.cache/llamacpp-laguna-06f8cebd/src/build
[[ $physical_build_dir == "$expected_build_dir" ]] \
    || die "refusing build-directory removal through symlinked parent: $physical_build_dir"
cmake -E remove_directory "$BUILD_DIR" \
    || die 'failed to remove prior CMake build tree'

printf '[4/6] Configuring Release CUDA build for sm_121...\n' >&2
PATH=/usr/local/cuda/bin:$PATH cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    "${cmake_flags[@]}" >&2 || die 'CMake configure failed'

printf '[5/6] Building llama-server and llama-cli with two jobs...\n' >&2
PATH=/usr/local/cuda/bin:$PATH cmake --build "$BUILD_DIR" --config Release \
    -j2 --target llama-server llama-cli >&2 || die 'CMake build failed'

[[ -x $SERVER_BINARY ]] || die "missing executable: $SERVER_BINARY"
[[ -x $CLI_BINARY ]] || die "missing executable: $CLI_BINARY"
actual_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD) \
    || die 'cannot re-read Laguna llama.cpp commit after build'
[[ $actual_commit == "$LLAMACPP_COMMIT" ]] \
    || die "Laguna llama.cpp commit changed during the build: $actual_commit"
[[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
    || die 'Laguna llama.cpp source became dirty during the build; refusing manifest publication'

printf '[6/6] Recording pinned build manifest: %s\n' "$MANIFEST" >&2
mkdir -p -- "$(dirname -- "$MANIFEST")"
manifest_tmp=$MANIFEST.tmp.$$
python3 - "$manifest_tmp" "$MANIFEST" "$LLAMACPP_REPOSITORY" \
    "$LLAMACPP_BRANCH" "$LLAMACPP_COMMIT" "$SOURCE_DIR" "$BUILD_DIR" \
    "$SERVER_BINARY" "$CLI_BINARY" "${cmake_flags[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import shlex
import sys

(
    temporary_name,
    output_name,
    repository,
    branch,
    commit,
    source_name,
    build_name,
    server_name,
    cli_name,
    *cmake_flags,
) = sys.argv[1:]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


build = pathlib.Path(build_name)
libraries = {}
for path in sorted((build / "bin").glob("*.so")):
    libraries[path.name] = {"sha256": digest(path)}
if not libraries:
    raise SystemExit("build produced no shared libraries to record")

manifest = {
    "schema_version": 1,
    "repository": repository,
    "branch": branch,
    "commit": commit,
    "source_directory": source_name,
    "build_directory": build_name,
    "cmake_flags": cmake_flags,
    "cmake_configure_command": shlex.join(
        ["cmake", "-S", source_name, "-B", build_name, *cmake_flags]
    ),
    "cmake_build_command": shlex.join([
        "cmake", "--build", build_name, "--config", "Release", "-j2",
        "--target", "llama-server", "llama-cli",
    ]),
    "jobs": 2,
    "binaries": {
        "llama-server": {"path": server_name, "sha256": digest(server_name)},
        "llama-cli": {"path": cli_name, "sha256": digest(cli_name)},
    },
    "shared_libraries": libraries,
}
output = pathlib.Path(output_name)
temporary = pathlib.Path(temporary_name)
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY

printf 'Build complete.\ncommit=%s\nllama_server=%s\nmanifest=%s\n' \
    "$LLAMACPP_COMMIT" "$SERVER_BINARY" "$MANIFEST"
