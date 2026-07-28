#!/usr/bin/env bash
# Restore the proven 8K DeepSeek profile in a persistent user-systemd cgroup.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly LAUNCHER=$REPO/scripts/21_serve_llamacpp.sh

[[ $EUID == 1000 && $(id -un) == bmarti44 ]] || {
    echo "61_restore_dsv4_user.sh: must run as bmarti44" >&2
    exit 2
}
sudo -n -u dsv4 true || {
    echo "61_restore_dsv4_user.sh: dsv4 delegation is unavailable" >&2
    exit 2
}

sudo -n -u dsv4 -- env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
    DSV4_PORT=8013 \
    DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
    DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
    DSV4_MEM_FLOOR_GIB=18 DSV4_WATCHDOG_FLOOR_GIB=18 \
    DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=0 \
    DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=0 \
    DSV4_ALLOW_RETRY_AFTER_FAILED_START=1 \
    DSV4_START_HOLD_FILE=/run/dsv4/context-worker.start-hold \
    DSV4_UBATCH=512 DSV4_BATCH=2048 DSV4_UBATCH_LARGE=0 \
    CTX=8192 DSV4_PARALLEL=1 DSV4_NO_MMAP=1 \
    DSV4_SPEC_TYPE=ngram-map-k4v \
    "$LAUNCHER" start
