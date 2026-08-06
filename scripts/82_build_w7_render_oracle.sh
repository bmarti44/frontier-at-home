#!/bin/bash
# Reproduce the frozen W7 C-parser oracle from the reviewed production source.
set -Eeuo pipefail
umask 077

[[ $# == 2 ]] || {
  echo "usage: $0 FROZEN_ENGINE_SOURCE_DIR OUTPUT" >&2
  exit 2
}
readonly ENGINE_DIR=$(readlink -f -- "$1")
readonly OUTPUT=$2
readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE=$REPO/results/glm52-gates/harness/w7_completion_render_oracle.c
readonly CC=/usr/bin/cc
readonly CC_SHA256=a20520ee21543f243d40636a9181a142c45ecd989de31ab86b99a8ea5ada870d
readonly SERVER_SHA256=d48d748edb56220727875d705f8487406c0f4f5b64b4d28ec0b829eb5ce87f07
readonly SOURCE_SHA256=9590b8eaa238e311ca0468e6983280b798cbb94c3d727920f5e839ac8ee20539
readonly OUTPUT_SHA256=6bd6896581db71bdb76a9afdb59a9254b151ade22017e17f111fd3345fb5ad66

[[ $(sha256sum -- "$CC" | awk '{print $1}') == "$CC_SHA256" ]] || exit 2
[[ $(sha256sum -- "$ENGINE_DIR/ds4_server.c" | awk '{print $1}') == "$SERVER_SHA256" ]] || exit 2
[[ $(sha256sum -- "$SOURCE" | awk '{print $1}') == "$SOURCE_SHA256" ]] || exit 2
[[ ! -e $OUTPUT ]] || {
  echo "refusing to replace existing output: $OUTPUT" >&2
  exit 2
}

"$CC" -O2 -std=c99 -D_GNU_SOURCE -ffunction-sections -fdata-sections \
  -ffile-prefix-map="$ENGINE_DIR"=/usr/src/ds4 \
  -ffile-prefix-map="$REPO"=/usr/src/frontier-at-home \
  -I"$ENGINE_DIR" "$SOURCE" -Wl,--gc-sections -pthread -lm -o "$OUTPUT"
[[ $(sha256sum -- "$OUTPUT" | awk '{print $1}') == "$OUTPUT_SHA256" ]] || {
  echo "oracle output SHA-256 mismatch" >&2
  exit 2
}
echo "$OUTPUT_SHA256  $OUTPUT"
