#!/bin/bash
# Build the frozen GLM-capable ds4 server twice from fresh checkouts and require
# byte identity. No model is loaded and compiler parallelism is capped at two.
set -Eeuo pipefail
umask 077

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
git -C "$SOURCE_REPOSITORY" cat-file -e "$COMMIT^{commit}"
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
WORKTREE=$WORK_ROOT/src
KEEP_DIR=$WORK_ROOT/nvcc-keep
mkdir -p -- "$KEEP_DIR"
SOURCE_DATE_EPOCH=$(
    git -C "$SOURCE_REPOSITORY" show -s --format=%ct "$COMMIT"
)
[[ $SOURCE_DATE_EPOCH =~ ^[0-9]{10}$ ]] || {
    echo "invalid source commit epoch" >&2
    exit 2
}
if [[ -e $WORKTREE ]]; then
    [[ -f $WORKTREE/.git ]]
    [[ $(git -C "$WORKTREE" rev-parse HEAD) == "$COMMIT" ]]
    [[ -z $(git -C "$WORKTREE" status --short) ]]
    git -C "$SOURCE_REPOSITORY" worktree remove "$WORKTREE"
fi

build_one() {
    local number=$1
    local destination=$OUTPUT_DIRECTORY/build${number}-ds4-server
    local cflags nvccflags
    git -C "$SOURCE_REPOSITORY" worktree add --detach "$WORKTREE" "$COMMIT"
    (
        cd "$WORKTREE"
        git ls-files -z |
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
    env SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" TZ=UTC LC_ALL=C \
        /usr/bin/time -v -o "$OUTPUT_DIRECTORY/build${number}.time" \
        make -C "$WORKTREE" -j2 CUDA_ARCH=native \
        CFLAGS="$cflags" NVCCFLAGS="$nvccflags" ds4-server \
        >"$OUTPUT_DIRECTORY/build${number}.log" 2>&1
    [[ -z $(git -C "$WORKTREE" status --short) ]] || {
        echo "build changed tracked source content" >&2
        exit 9
    }
    install -m 0500 -- "$WORKTREE/ds4-server" "$destination"
}

build_one 1
git -C "$SOURCE_REPOSITORY" worktree remove "$WORKTREE"
build_one 2
cmp -s "$OUTPUT_DIRECTORY/build1-ds4-server" \
    "$OUTPUT_DIRECTORY/build2-ds4-server" || {
    echo "independent builds are not byte-identical" >&2
    exit 10
}

/usr/bin/python3 - "$OUTPUT_DIRECTORY" "$COMMIT" "$SOURCE_DATE_EPOCH" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

output = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
source_epoch = int(sys.argv[3])
binary = output / "build1-ds4-server"
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
cc = subprocess.run(
    ["cc", "--version"], text=True, stdout=subprocess.PIPE, check=True
).stdout.splitlines()[0]
nvcc = subprocess.run(
    ["/usr/local/cuda/bin/nvcc", "--version"],
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
