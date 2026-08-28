#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: 11_build_ds4.sh [--host-class CLASS] [--cuda-arch N] [--rocm-arch gfxNNNN] [--help]

Build the pinned ds4 engine and write a build manifest. The default host
class cuda-spark uses the official GB10 Makefile target with the sm_121
assertion intact; cuda-generic|metal|rocm|cpu dispatch to the ds4 Makefile
targets that already exist upstream (cuda-generic, the Darwin default Metal
target, strix-halo, cpu). See scripts/lib/build_host_class.sh.
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

[[ -n "${HOME:-}" ]] || die_env 'HOME is not set'
DS4_HOME="${DS4_HOME:-$HOME/ds4-project}"
SRC_DIR="$DS4_HOME/src/ds4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" \
    || die_env 'cannot resolve script directory'
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)" \
    || die_env 'cannot resolve repository root'
PIN_FILE="$REPO_ROOT/configs/pins/ds4-weights.json"
# shellcheck source=lib/build_host_class.sh
source "$SCRIPT_DIR/lib/build_host_class.sh"
build_host_class_parse "$@"
(( ${#BUILD_HOST_CLASS_ARGS[@]} == 0 )) || { usage >&2; exit 2; }
case $BUILD_HOST_CLASS in
    cuda-spark) MAKE_TARGET=cuda-spark ;;
    cuda-generic)
        if [[ $BUILD_CUDA_ARCH == native ]]; then
            MAKE_TARGET=cuda-generic
        else
            MAKE_TARGET="cuda CUDA_ARCH=sm_$BUILD_CUDA_ARCH"
        fi
        ;;
    metal) MAKE_TARGET=all ;;  # Darwin default target builds Metal upstream
    rocm) MAKE_TARGET=strix-halo ;;
    cpu) MAKE_TARGET=cpu ;;
esac
BUILD_COMMAND="make -C \$SRC_DIR -j\$(nproc) $MAKE_TARGET"

for command_name in python3 git make sha256sum gcc uname; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die_env "required command not found: $command_name"
done
[[ -r "$PIN_FILE" ]] || die_env "pin file is not readable: $PIN_FILE"

engine_commit="$(python3 - "$PIN_FILE" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        print(json.load(stream)["git_pins"]["engine"]["commit"])
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"invalid pin file: {error}", file=sys.stderr)
    sys.exit(2)
PY
)" || die_env "failed to parse pin file: $PIN_FILE"
[[ "$engine_commit" =~ ^[0-9a-f]{40}$ ]] || die_env 'invalid engine commit pin'

[[ $BUILD_HOST_CLASS != cuda-spark || "$(uname -m)" == aarch64 ]] \
    || die_env 'this build requires uname -m to report aarch64 (pass --host-class to build elsewhere)'
[[ -d "$SRC_DIR" ]] || die_build "engine source directory is absent: $SRC_DIR"
actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null)" \
    || die_build "cannot read engine HEAD: $SRC_DIR"
[[ "$actual_commit" == "$engine_commit" ]] \
    || die_build "engine HEAD mismatch: expected $engine_commit, got $actual_commit"
[[ -z $(git -C "$SRC_DIR" status --porcelain) ]] \
    || die_build "engine worktree is dirty: $SRC_DIR"
engine_describe=$(git -C "$SRC_DIR" describe --always --dirty) \
    || die_build "cannot describe engine worktree: $SRC_DIR"
[[ $engine_describe != *-dirty ]] \
    || die_build "engine worktree describe reports dirty: $engine_describe"

build_host_class_require_platform
nvcc_version="n/a"
if [[ $BUILD_HOST_CLASS == cuda-spark || $BUILD_HOST_CLASS == cuda-generic ]]; then
    nvcc_version="$(nvcc --version)" || die_env 'nvcc --version failed'
fi
gcc_version="$(gcc --version)" || die_env 'gcc --version failed'
gcc_version=${gcc_version%%$'\n'*}
parallelism="$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
    || die_env 'cannot determine build parallelism'

printf 'Building pinned engine with %s target...\n' "$MAKE_TARGET" >&2
# shellcheck disable=SC2086
make -C "$SRC_DIR" -j"$parallelism" $MAKE_TARGET >&2 \
    || die_build "$MAKE_TARGET build failed"

# The target also builds ds4-agent and ds4_weight_server; they must NEVER be
# executed in service. Serving uses only ds4-server.
binaries=(ds4 ds4-server ds4-bench)
for binary in "${binaries[@]}"; do
    [[ -f "$SRC_DIR/$binary" && -x "$SRC_DIR/$binary" ]] \
        || die_build "required executable is missing: $SRC_DIR/$binary"
done

case $BUILD_HOST_CLASS in
    cuda-spark)
        set +o pipefail
        elf_head="$(cuobjdump --list-elf "$SRC_DIR/ds4-server" 2>/dev/null | head)"
        set -o pipefail
        [[ "$elf_head" == *sm_121* ]] \
            || die_build 'ds4-server CUDA objects do not report sm_121'
        arch_observed=sm_121
        ;;
    cuda-generic)
        set +o pipefail
        elf_head="$(cuobjdump --list-elf "$SRC_DIR/ds4-server" 2>/dev/null | head)"
        set -o pipefail
        arch_observed=$(printf '%s\n' "$elf_head" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ',')
        [[ -n $arch_observed ]] \
            || die_build 'ds4-server reports no CUDA sm_ architectures'
        ;;
    metal)
        otool -L "$SRC_DIR/ds4-server" 2>/dev/null | grep -q Metal \
            || die_build 'ds4-server shows no Metal framework linkage'
        arch_observed=metal
        ;;
    rocm) arch_observed=$BUILD_ROCM_ARCH ;;
    cpu) arch_observed=cpu ;;
esac
"$SRC_DIR/ds4" --help >/dev/null 2>&1 || die_build 'ds4 --help smoke test failed'

hashes=()
for binary in "${binaries[@]}"; do
    digest="$(sha256sum -- "$SRC_DIR/$binary")" \
        || die_build "cannot hash binary: $binary"
    hashes+=("${digest%% *}")
done

built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || die_env 'cannot obtain current UTC time'
manifest="$DS4_HOME/build-manifest.json"
manifest_tmp="$manifest.partial"
python3 - "$manifest_tmp" "$manifest" "$engine_commit" "$engine_describe" "$BUILD_COMMAND" \
    "$nvcc_version" "$gcc_version" "$built_at" \
    "${hashes[0]}" "${hashes[1]}" "${hashes[2]}" \
    "$BUILD_HOST_CLASS" "$(build_host_class_backend)" "$arch_observed" <<'PY' \
    || die_build 'failed to write build manifest'
import json
import os
import sys

(temporary, output, commit, describe, command, nvcc, gcc, built_at,
 ds4_hash, server_hash, bench_hash, host_class, backend,
 arch_observed) = sys.argv[1:]
manifest = {
    "engine_commit": commit,
    "engine_describe": describe,
    "build_command": command,
    "nvcc_version": nvcc,
    "gcc_version": gcc,
    "backend": backend,
    "host_class": host_class,
    "arch_assertion": {"observed": arch_observed},
    "binaries": {
        "ds4": {"sha256": ds4_hash},
        "ds4-server": {"sha256": server_hash},
        "ds4-bench": {"sha256": bench_hash},
    },
    "built_at": built_at,
}
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, sort_keys=True, indent=2)
    stream.write("\n")
os.replace(temporary, output)
PY

printf '{"ok":true}\n'
