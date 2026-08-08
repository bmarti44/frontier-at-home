#!/bin/bash
# Frozen CPU-only launcher for the W9 real-tensor FP4 falsifier.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SCRIPT=$REPO/scripts/93_score_w9_fp4_falsifier.py
readonly CAPTURE=/home/bmarti44/.local/state/glm52-w9-real-capture/attempt-73838408ccb1d126ade7b67c8d86fa00/on/capture

[[ $# == 2 && -f $1 && ! -L $1 && $2 == /* && ! -e $2 && ! -L $2 ]]
exec /usr/bin/env -i \
  HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
  PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONPATH=/home/bmarti44/.local/lib/python3.12/site-packages \
  /usr/bin/python3 -B "$SCRIPT" \
    --capture-root "$CAPTURE" --randomness-receipt "$1" --output "$2"
