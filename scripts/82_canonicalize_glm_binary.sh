#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || {
  echo "usage: $0 INPUT_BINARY OUTPUT_BINARY" >&2
  exit 2
}

input=$(realpath -e -- "$1")
output=$2
[[ -f $input && -x $input && ! -e $output && ! -L $output ]] || {
  echo "canonicalize_glm_binary: invalid input or existing output" >&2
  exit 2
}
parent=$(realpath -e -- "$(dirname -- "$output")")
output=$parent/$(basename -- "$output")

cp --preserve=mode,timestamps -- "$input" "$output"
objcopy --strip-debug --remove-section=.note.gnu.build-id -- "$output"
python3 - "$output" "$parent" <<'PY'
import os
import sys

file_descriptor = os.open(sys.argv[1], os.O_RDONLY | os.O_CLOEXEC)
try:
    os.fsync(file_descriptor)
finally:
    os.close(file_descriptor)
directory_descriptor = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
sha256sum -- "$output"
