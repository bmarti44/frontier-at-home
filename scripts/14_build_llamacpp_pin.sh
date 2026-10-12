#!/usr/bin/env bash
# Build one registered llama.cpp pin (configs/pins/llamacpp.json) for one
# host class, into $HOME/.cache/<pin>/src/build where profile engines look
# for it ({cache_root}/<pin>/src/build/bin/llama-server).
set -Eeuo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: 14_build_llamacpp_pin.sh --pin <id> [--host-class CLASS]
           [--cuda-arch N] [--rocm-arch gfxNNNN] [--dry-run]

Clone, verify, and clean-build a pinned llama.cpp revision with at most two
build jobs, then record configs/build-manifests/<pin>[-<class>].json (the
cuda-spark build keeps the unsuffixed name). --dry-run prints the resolved
plan as JSON and touches nothing.

Classes: cuda-spark (default) | cuda-generic | metal | rocm | vulkan | cpu
Pins:    configs/pins/llamacpp.json ("portable" is the off-Spark default)
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) \
    || die 'cannot resolve script directory'
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) \
    || die 'cannot resolve repository root'
REGISTRY=$REPO_ROOT/configs/pins/llamacpp.json
die_env() { die "$@"; }
die_build() { die "$@"; }
# shellcheck source=lib/build_host_class.sh
source "$SCRIPT_DIR/lib/build_host_class.sh"
build_host_class_parse "$@"

PIN=
DRY_RUN=false
set -- "${BUILD_HOST_CLASS_ARGS[@]+"${BUILD_HOST_CLASS_ARGS[@]}"}"
while (( $# > 0 )); do
    case $1 in
        --pin) PIN=${2:-}; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $PIN ]] || { usage >&2; exit 2; }

mapfile -t pin_fields < <(python3 - "$REGISTRY" "$PIN" <<'PY'
import json, sys
pins = json.load(open(sys.argv[1], encoding="utf-8"))["pins"]
pin = pins.get(sys.argv[2])
if pin is None:
    sys.exit(3)
for key in ("repository", "tag", "branch", "commit"):
    print(pin.get(key) or "")
PY
) || die "unknown pin $PIN (not in configs/pins/llamacpp.json)"
(( ${#pin_fields[@]} == 4 )) || die "unknown pin $PIN (not in configs/pins/llamacpp.json)"
readonly REPOSITORY=${pin_fields[0]} TAG=${pin_fields[1]} BRANCH=${pin_fields[2]} COMMIT=${pin_fields[3]}
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || die "pin $PIN has no full commit"
[[ -n $TAG || -n $BRANCH ]] || die "pin $PIN names neither a tag nor a branch"

readonly CACHE_ROOT=${HOME:?HOME must be set}/.cache/$PIN
readonly SOURCE_DIR=$CACHE_ROOT/src
readonly BUILD_DIR=$SOURCE_DIR/build
if [[ $BUILD_HOST_CLASS == cuda-spark ]]; then
    MANIFEST=$REPO_ROOT/configs/build-manifests/$PIN.json
else
    MANIFEST=$REPO_ROOT/configs/build-manifests/$PIN-$BUILD_HOST_CLASS.json
fi
mapfile -t accelerator_flags < <(build_host_class_cmake_flags)
cmake_flags=("${accelerator_flags[@]}" -DCMAKE_BUILD_TYPE=Release)

if "$DRY_RUN"; then
    python3 - "$PIN" "$REPOSITORY" "$TAG" "$BRANCH" "$COMMIT" "$BUILD_HOST_CLASS" \
        "$(build_host_class_backend)" "$SOURCE_DIR" "$BUILD_DIR" "$MANIFEST" \
        "${cmake_flags[@]}" <<'PY'
import json, sys
keys = ("pin", "repository", "tag", "branch", "commit", "host_class", "backend",
        "source_directory", "build_directory", "manifest")
plan = {key: value or None for key, value in zip(keys, sys.argv[1:11])}
plan["cmake_flags"] = sys.argv[11:]
print(json.dumps(plan, indent=1))
PY
    exit 0
fi

for command_name in awk cmake flock git python3 uname; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required command not found: $command_name"
done
build_host_class_require_platform

if [[ -r /proc/meminfo ]]; then
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
fi
# No procfs (macOS): the build-memory gate is Linux-only; the serve-time
# watchdog remains the real guard (docs/BACKEND-CONTRACT.md section 3).

printf '[1/6] Preparing isolated source cache: %s\n' "$SOURCE_DIR" >&2
mkdir -p -- "$CACHE_ROOT"
exec 9>"$CACHE_ROOT/build.lock"
flock -n 9 || die "another build of $PIN holds the build lock"
fresh_clone=false
if [[ ! -e $SOURCE_DIR ]]; then
    git clone --no-checkout -- "$REPOSITORY" "$SOURCE_DIR" >&2 \
        || die 'llama.cpp clone failed'
    fresh_clone=true
fi
[[ -d $SOURCE_DIR/.git ]] || die "source cache is not a Git clone: $SOURCE_DIR"
origin_url=$(git -C "$SOURCE_DIR" remote get-url origin) \
    || die 'cannot read origin URL'
case $origin_url in
    "$REPOSITORY"|"$REPOSITORY.git") ;;
    *) die "unexpected origin: $origin_url" ;;
esac
# A fresh --no-checkout clone has an empty worktree (status shows all files
# deleted), so the tamper check only applies to a pre-existing cache.
if [[ $fresh_clone == false ]]; then
    [[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
        || die "source cache is dirty: $SOURCE_DIR"
fi

if [[ -n $TAG ]]; then
    printf '[2/6] Fetching pinned tag %s...\n' "$TAG" >&2
    git -C "$SOURCE_DIR" fetch --force --tags origin >&2 \
        || die 'failed to fetch tags'
    ref_commit=$(git -C "$SOURCE_DIR" rev-parse "refs/tags/$TAG^{commit}") \
        || die "tag is missing: $TAG"
    [[ $ref_commit == "$COMMIT" ]] \
        || die "tag $TAG resolves to $ref_commit, expected $COMMIT"
else
    printf '[2/6] Fetching pinned branch %s...\n' "$BRANCH" >&2
    git -C "$SOURCE_DIR" fetch --force origin \
        "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" >&2 \
        || die "failed to fetch branch $BRANCH"
    git -C "$SOURCE_DIR" merge-base --is-ancestor "$COMMIT" "origin/$BRANCH" \
        || die "commit $COMMIT is not on origin/$BRANCH"
fi
git -C "$SOURCE_DIR" checkout --detach "$COMMIT" >&2 \
    || die 'failed to check out the pinned commit'
actual_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD) \
    || die 'cannot read the checked-out commit'
[[ $actual_commit == "$COMMIT" ]] || die "checked-out commit mismatch: $actual_commit"
[[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
    || die 'source became dirty after checkout'

printf '[3/6] Removing the prior CMake build tree...\n' >&2
[[ $BUILD_DIR == "$CACHE_ROOT/src/build" ]] || die 'refusing unsafe build-directory removal'
cmake -E remove_directory "$BUILD_DIR" || die 'failed to remove prior CMake build tree'

printf '[4/6] Configuring Release build (%s)...\n' "$BUILD_HOST_CLASS" >&2
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" "${cmake_flags[@]}" >&2 \
    || die 'CMake configure failed'

printf '[5/6] Building llama-server and llama-cli with two jobs...\n' >&2
cmake --build "$BUILD_DIR" --config Release -j2 --target llama-server llama-cli >&2 \
    || die 'CMake build failed'

SERVER_BINARY=$BUILD_DIR/bin/llama-server
CLI_BINARY=$BUILD_DIR/bin/llama-cli
[[ -x $SERVER_BINARY ]] || die "missing executable: $SERVER_BINARY"
[[ -x $CLI_BINARY ]] || die "missing executable: $CLI_BINARY"
observed_arch=$(build_host_class_assert_artifacts "$BUILD_DIR/bin")
[[ -z $(git -C "$SOURCE_DIR" status --porcelain) ]] \
    || die 'source became dirty during the build; refusing manifest publication'

printf '[6/6] Recording build manifest: %s\n' "$MANIFEST" >&2
mkdir -p -- "$(dirname -- "$MANIFEST")"
python3 - "$MANIFEST" "$PIN" "$REPOSITORY" "$TAG" "$BRANCH" "$COMMIT" \
    "$SOURCE_DIR" "$BUILD_DIR" "$SERVER_BINARY" "$CLI_BINARY" \
    "$BUILD_HOST_CLASS" "$(build_host_class_backend)" "$observed_arch" \
    "${cmake_flags[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import shlex
import sys

(output_name, pin, repository, tag, branch, commit, source_name, build_name,
 server_name, cli_name, host_class, backend, observed_arch,
 *cmake_flags) = sys.argv[1:]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


build = pathlib.Path(build_name)
libraries = {}
for suffix in ("*.so", "*.dylib"):
    for path in sorted((build / "bin").glob(suffix)):
        libraries[path.name] = {"sha256": digest(path)}
if not libraries:
    raise SystemExit("build produced no shared libraries to record")

manifest = {
    "schema_version": 1,
    "pin": pin,
    "repository": repository,
    "tag": tag or None,
    "branch": branch or None,
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
    "backend": backend,
    "host_class": host_class,
    "observed_arch": observed_arch,
    "binaries": {
        "llama-server": {"path": server_name, "sha256": digest(server_name)},
        "llama-cli": {"path": cli_name, "sha256": digest(cli_name)},
    },
    "shared_libraries": libraries,
}
output = pathlib.Path(output_name)
temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY

printf 'Build complete.\npin=%s\ncommit=%s\nllama_server=%s\nmanifest=%s\n' \
    "$PIN" "$COMMIT" "$SERVER_BINARY" "$MANIFEST"
