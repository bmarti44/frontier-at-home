#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: 13_build_llamacpp.sh [--host-class CLASS] [--cuda-arch N] [--rocm-arch gfxNNNN] [--help]

Clone and build the pinned llama.cpp revision, verify the resulting
binaries, and write a build manifest. The default host class cuda-spark
builds CUDA for sm_121 with every historical Spark assertion intact;
cuda-generic|metal|rocm|cpu build for other consumer hardware
(scripts/lib/build_host_class.sh, configs/hardware-matrix.json).
EOF
}

die_build() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

die_env() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac
# shellcheck source=lib/build_host_class.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/build_host_class.sh"
build_host_class_parse "$@"
(( ${#BUILD_HOST_CLASS_ARGS[@]} == 0 )) || { usage >&2; exit 2; }

[[ -n "${HOME:-}" ]] || die_env 'HOME is not set'
LLAMACPP_HOME="${LLAMACPP_HOME:-$HOME/llamacpp-project}"
SRC_DIR="$LLAMACPP_HOME/src/llama.cpp"
BUILD_DIR="$SRC_DIR/build"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" \
    || die_env 'cannot resolve script directory'
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)" \
    || die_env 'cannot resolve repository root'
PIN_FILE="$REPO_ROOT/configs/versions.lock"

for command_name in python3 git mkdir sha256sum gcc uname date head; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die_env "required command not found: $command_name"
done
[[ -r "$PIN_FILE" ]] || die_env "version lock is not readable: $PIN_FILE"

pin_values="$(python3 - "$PIN_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        pin = json.load(stream)["pins"]["llama.cpp"]
    repo = pin["repo"]
    commit = pin["commit"]
    if not isinstance(repo, str) or not repo or "\n" in repo:
        raise ValueError("llama.cpp repo must be a non-empty, single-line string")
    if not isinstance(commit, str):
        raise ValueError("llama.cpp commit must be a string")
    print(repo)
    print(commit)
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"invalid version lock: {error}", file=sys.stderr)
    sys.exit(2)
PY
)" || die_env "failed to parse version lock: $PIN_FILE"
llamacpp_repo="${pin_values%%$'\n'*}"
llamacpp_commit="${pin_values#*$'\n'}"
[[ "$llamacpp_commit" =~ ^[0-9a-f]{40}$ ]] \
    || die_env 'invalid llama.cpp commit pin'

mkdir -p -- "$LLAMACPP_HOME/src" \
    || die_build "cannot create source parent directory: $LLAMACPP_HOME/src"
if [[ ! -e "$SRC_DIR" ]]; then
    printf 'Cloning pinned llama.cpp repository...\n' >&2
    git clone -- "$llamacpp_repo" "$SRC_DIR" >&2 \
        || die_build "failed to clone llama.cpp into $SRC_DIR"
fi
[[ -d "$SRC_DIR" ]] || die_build "source path is not a directory: $SRC_DIR"

printf 'Fetching llama.cpp revisions...\n' >&2
git -C "$SRC_DIR" fetch --all --tags --prune >&2 \
    || die_build "failed to fetch llama.cpp repository: $SRC_DIR"
git -C "$SRC_DIR" checkout --detach "$llamacpp_commit" >&2 \
    || die_build "failed to check out pinned llama.cpp commit: $llamacpp_commit"
actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null)" \
    || die_build "cannot read llama.cpp HEAD: $SRC_DIR"
[[ "$actual_commit" == "$llamacpp_commit" ]] \
    || die_build "llama.cpp HEAD mismatch: expected $llamacpp_commit, got $actual_commit"
[[ -z $(git -C "$SRC_DIR" status --porcelain) ]] \
    || die_build "llama.cpp worktree is dirty: $SRC_DIR"
source_describe=$(git -C "$SRC_DIR" describe --always --dirty) \
    || die_build "cannot describe llama.cpp worktree: $SRC_DIR"
[[ $source_describe != *-dirty ]] \
    || die_build "llama.cpp worktree describe reports dirty: $source_describe"

build_host_class_require_platform
command -v cmake >/dev/null 2>&1 \
    || die_env 'cmake is required to build llama.cpp but was not found in PATH'

nvcc_version="n/a"
if [[ $BUILD_HOST_CLASS == cuda-spark || $BUILD_HOST_CLASS == cuda-generic ]]; then
    nvcc_version="$(nvcc --version)" || die_env 'nvcc --version failed'
fi
gcc_version="$(gcc --version)" || die_env 'gcc --version failed'
gcc_version=${gcc_version%%$'\n'*}
cmake_version="$(cmake --version)" || die_env 'cmake --version failed'
cmake_version=${cmake_version%%$'\n'*}
parallelism="$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
    || die_env 'cannot determine build parallelism'

mapfile -t accelerator_flags < <(build_host_class_cmake_flags)
configure_args=(
    cmake -S "$SRC_DIR" -B "$BUILD_DIR"
    "${accelerator_flags[@]}"
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_CURL=OFF
)
build_args=(
    cmake --build "$BUILD_DIR" --config Release -j"$parallelism"
    --target llama-server llama-cli llama-bench
)

printf 'Configuring pinned llama.cpp build...\n' >&2
"${configure_args[@]}" >&2 || die_build 'llama.cpp CMake configure failed'
printf 'Building pinned llama.cpp targets...\n' >&2
"${build_args[@]}" >&2 || die_build 'llama.cpp build failed'

binaries=(llama-server llama-cli llama-bench)
for binary in "${binaries[@]}"; do
    [[ -f "$BUILD_DIR/bin/$binary" && -x "$BUILD_DIR/bin/$binary" ]] \
        || die_build "required executable is missing: $BUILD_DIR/bin/$binary"
done

# Accelerator code lives in the shared ggml libraries, not the executable
# (llama.cpp default builds ggml as shared libraries); the per-class
# assertion inspects them and reports the observed architecture.
arch_observed=$(build_host_class_assert_artifacts "$BUILD_DIR/bin") \
    || die_build 'per-host-class artifact assertion failed'
"$BUILD_DIR/bin/llama-server" --version >/dev/null 2>&1 \
    || die_build 'llama-server --version smoke test failed'

hashes=()
for binary in "${binaries[@]}"; do
    digest="$(sha256sum -- "$BUILD_DIR/bin/$binary")" \
        || die_build "cannot hash binary: $binary"
    hashes+=("${digest%% *}")
done

built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    || die_env 'cannot obtain current UTC time'
manifest="$LLAMACPP_HOME/build-manifest.json"
manifest_tmp="$manifest.partial"
python3 - "$manifest_tmp" "$manifest" "$llamacpp_commit" "$source_describe" \
    "$SRC_DIR" "$BUILD_DIR" "$parallelism" "$nvcc_version" \
    "$gcc_version" "$cmake_version" "$built_at" \
    "${hashes[0]}" "${hashes[1]}" "${hashes[2]}" \
    "$BUILD_HOST_CLASS" "$(build_host_class_backend)" "$arch_observed" \
    "${accelerator_flags[@]}" <<'PY' \
    || die_build 'failed to write build manifest'
import hashlib
import json
import os
import shlex
import sys

(temporary, output, commit, describe, source, build, parallelism, nvcc, gcc, cmake,
 built_at, server_hash, cli_hash, bench_hash, host_class, backend,
 arch_observed, *accelerator_flags) = sys.argv[1:]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Record every shared library alongside the thin binaries. The engine code (incl.
# the CUDA fatbinary in libggml-cuda.so) lives here, and the serve-time integrity
# check requires this map so a library-only rebuild cannot slip unverified code
# past an unchanged llama-server.
bin_dir = os.path.join(build, "bin")
shared_libraries = {
    name: {"sha256": sha256_file(os.path.join(bin_dir, name))}
    for name in sorted(os.listdir(bin_dir))
    if name.endswith((".so", ".dylib"))
}
if not shared_libraries:
    raise SystemExit("build produced no shared libraries to record")
configure = (
    ["cmake", "-S", source, "-B", build]
    + accelerator_flags
    + ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
)
build_command = [
    "cmake", "--build", build, "--config", "Release", f"-j{parallelism}",
    "--target", "llama-server", "llama-cli", "llama-bench",
]
manifest = {
    "commit": commit,
    "source_describe": describe,
    "cmake_configure_command": shlex.join(configure),
    "cmake_build_command": shlex.join(build_command),
    "nvcc_version": nvcc,
    "gcc_version": gcc,
    "cmake_version": cmake,
    "binaries": {
        "llama-server": {"sha256": server_hash},
        "llama-cli": {"sha256": cli_hash},
        "llama-bench": {"sha256": bench_hash},
    },
    "shared_libraries": shared_libraries,
    "backend": backend,
    "host_class": host_class,
    "arch_assertion": {"observed": arch_observed},
    "built_at": built_at,
}
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, sort_keys=True, indent=2)
    stream.write("\n")
os.replace(temporary, output)
PY

printf '{"ok":true}\n'
