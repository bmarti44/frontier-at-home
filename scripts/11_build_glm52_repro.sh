#!/bin/bash
# Build the frozen GLM-capable ds4 server twice from fresh checkouts and require
# byte identity. No model is loaded and compiler parallelism is capped at two.
set -Eeuo pipefail
umask 077

readonly CC_PATH=/usr/bin/cc
readonly MAKE_PATH=/usr/bin/make
readonly CUDA_HOME_PATH=/usr/local/cuda
readonly NVCC_PATH=/usr/local/cuda/bin/nvcc

clean_git() {
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
        /usr/bin/git "$@"
}

verify_no_symlink_components() {
    /usr/bin/python3 - "$WORK_ROOT" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
for component in reversed((path, *path.parents)):
    metadata = os.lstat(component)
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"symlinked canonical path component: {component}")
if not path.is_dir() or os.lstat(path).st_uid != os.getuid():
    raise SystemExit("canonical work root owner or type is invalid")
PY
}

[[ $# == 4 ]] || {
    echo "usage: $0 SOURCE_REPOSITORY COMMIT WORK_ROOT OUTPUT_DIRECTORY" >&2
    exit 2
}
SOURCE_REPOSITORY=$1
COMMIT=$2
WORK_ROOT=$3
OUTPUT_DIRECTORY=$4
CANONICAL_WORK_ROOT=/home/bmarti44/.cache/glm52-ds4-repro-v1
for path in "$SOURCE_REPOSITORY" "$WORK_ROOT" "$OUTPUT_DIRECTORY"; do
    [[ $path == /* ]] || {
        echo "all paths must be absolute" >&2
        exit 2
    }
done
[[ $WORK_ROOT == "$CANONICAL_WORK_ROOT" ]] || {
    echo "WORK_ROOT must be the canonical CUDA build path" >&2
    exit 2
}
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || {
    echo "commit must be a full lowercase Git object ID" >&2
    exit 2
}
[[ -d $SOURCE_REPOSITORY/.git || -f $SOURCE_REPOSITORY/.git ]] || {
    echo "source repository is not a Git worktree" >&2
    exit 2
}
[[ ! -e $OUTPUT_DIRECTORY ]] || {
    echo "output path must be absent" >&2
    exit 2
}
clean_git -C "$SOURCE_REPOSITORY" cat-file -e "$COMMIT^{commit}"
available_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
(( available_kib >= 4 * 1048576 )) || {
    echo "less than 4 GiB is available; refusing compiler start" >&2
    exit 8
}

mkdir -p -- "$(dirname "$WORK_ROOT")"
exec 9>"$WORK_ROOT.lock"
flock -n 9 || {
    echo "another reproducible build owns the canonical path" >&2
    exit 75
}
mkdir -p -- "$WORK_ROOT" "$OUTPUT_DIRECTORY"
chmod 0700 -- "$WORK_ROOT"
verify_no_symlink_components
exec 8<"$WORK_ROOT"
WORK_ROOT_IDENTITY=$(/usr/bin/stat -Lc '%d:%i' "/proc/$$/fd/8")
verify_work_root() {
    verify_no_symlink_components
    [[ $(/usr/bin/stat -Lc '%d:%i' "$WORK_ROOT") == "$WORK_ROOT_IDENTITY" ]]
    [[ $(/usr/bin/stat -Lc '%d:%i' "/proc/$$/fd/8") == "$WORK_ROOT_IDENTITY" ]]
}
verify_work_root
WORKTREE=$WORK_ROOT/src
KEEP_DIR=$WORK_ROOT/nvcc-keep
mkdir -p -- "$KEEP_DIR"
SOURCE_DATE_EPOCH=$(
    clean_git -C "$SOURCE_REPOSITORY" show -s --format=%ct "$COMMIT"
)
[[ $SOURCE_DATE_EPOCH =~ ^[0-9]{10}$ ]] || {
    echo "invalid source commit epoch" >&2
    exit 2
}
if [[ -e $WORKTREE ]]; then
    [[ -f $WORKTREE/.git ]]
    [[ $(clean_git -C "$WORKTREE" rev-parse HEAD) == "$COMMIT" ]]
    [[ -z $(clean_git -C "$WORKTREE" status --short) ]]
    clean_git -C "$SOURCE_REPOSITORY" worktree remove "$WORKTREE"
fi

build_one() {
    local number=$1
    local destination=$OUTPUT_DIRECTORY/build${number}-ds4-server
    local cflags nvccflags
    verify_work_root
    clean_git -C "$SOURCE_REPOSITORY" worktree add --detach "$WORKTREE" "$COMMIT"
    (
        cd "$WORKTREE"
        clean_git ls-files -z |
            xargs -0 -r touch --date="@$SOURCE_DATE_EPOCH" --
    )
    [[ $KEEP_DIR == "$WORK_ROOT/nvcc-keep" ]]
    find "$KEEP_DIR" -mindepth 1 -delete
    cflags="-O3 -ffast-math -g -march=native -Wall -Wextra -std=c99"
    cflags+=" -D_GNU_SOURCE -fno-finite-math-only"
    cflags+=" -ffile-prefix-map=$WORKTREE=/usr/src/ds4"
    cflags+=" -fdebug-prefix-map=$WORKTREE=/usr/src/ds4"
    cflags+=" -fmacro-prefix-map=$WORKTREE=/usr/src/ds4"
    nvccflags="-O3 -g -lineinfo --use_fast_math -arch=native"
    nvccflags+=" --frandom-seed=${COMMIT:0:8} --keep --keep-dir=$KEEP_DIR"
    nvccflags+=" -Xcompiler -march=native -Xcompiler -pthread"
    nvccflags+=" -Xcompiler -ffile-prefix-map=$WORKTREE=/usr/src/ds4"
    nvccflags+=" -Xcompiler -fdebug-prefix-map=$WORKTREE=/usr/src/ds4"
    nvccflags+=" -Xcompiler -fmacro-prefix-map=$WORKTREE=/usr/src/ds4"
    env -i PATH=/usr/bin:/bin HOME=/nonexistent \
        SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" TZ=UTC LC_ALL=C \
        /usr/bin/time -v -o "$OUTPUT_DIRECTORY/build${number}.time" \
        "$MAKE_PATH" -C "$WORKTREE" -j2 CUDA_ARCH=native \
        CC="$CC_PATH" NVCC="$NVCC_PATH" CUDA_HOME="$CUDA_HOME_PATH" \
        DS4_LINK="$NVCC_PATH $nvccflags" \
        CFLAGS="$cflags" NVCCFLAGS="$nvccflags" ds4-server \
        >"$OUTPUT_DIRECTORY/build${number}.log" 2>&1
    [[ -z $(clean_git -C "$WORKTREE" status --short) ]] || {
        echo "build changed tracked source content" >&2
        exit 9
    }
    install -m 0500 -- "$WORKTREE/ds4-server" "$destination"
    verify_work_root
}

build_one 1
clean_git -C "$SOURCE_REPOSITORY" worktree remove "$WORKTREE"
build_one 2
verify_work_root
cmp -s "$OUTPUT_DIRECTORY/build1-ds4-server" \
    "$OUTPUT_DIRECTORY/build2-ds4-server" || {
    echo "independent builds are not byte-identical" >&2
    exit 10
}

/usr/bin/python3 - "$OUTPUT_DIRECTORY" "$COMMIT" "$SOURCE_DATE_EPOCH" \
    "$CC_PATH" "$NVCC_PATH" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

output = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
source_epoch = int(sys.argv[3])
cc_path = sys.argv[4]
nvcc_path = sys.argv[5]
binary = output / "build1-ds4-server"
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
cc = subprocess.run(
    [cc_path, "--version"], text=True, stdout=subprocess.PIPE, check=True
).stdout.splitlines()[0]
nvcc = subprocess.run(
    [nvcc_path, "--version"],
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.splitlines()[-1]
value = {
    "schema_version": 1,
    "source_commit": commit,
    "source_date_epoch": source_epoch,
    "binary_sha256": digest,
    "binary_bytes": binary.stat().st_size,
    "builds": 2,
    "byte_identical": True,
    "jobs": 2,
    "cc_version": cc,
    "nvcc_version": nvcc,
}
temporary = output / "manifest.json.tmp"
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output / "manifest.json")
PY
chmod a+r "$OUTPUT_DIRECTORY"/manifest.json "$OUTPUT_DIRECTORY"/*.log \
    "$OUTPUT_DIRECTORY"/*.time
echo "reproducible_binary=$OUTPUT_DIRECTORY/build1-ds4-server"
