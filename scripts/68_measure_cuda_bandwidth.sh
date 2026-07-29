#!/usr/bin/env bash
# Root-side bounded CUDA read-bandwidth measurement for the foundation gate.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE=$REPO/scripts/68_cuda_read_bandwidth.cu
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly INFERENCE_LOCK=/run/lock/frontier-at-home/inference.lock
readonly FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'

die() {
    printf '68_measure_cuda_bandwidth.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# == 1 ]] || die "usage: $0 ABSOLUTE_OUTPUT_JSON"
OUTPUT=$1
(( EUID == 0 )) || die "must run inside the delegated foundation authority"
[[ $OUTPUT == /* && ! -e $OUTPUT && -d ${OUTPUT%/*} ]] ||
    die "output must be a new file in an existing absolute directory"
[[ -f $SOURCE && ! -L $SOURCE ]] || die "probe source is absent or unsafe"
[[ -f $INFERENCE_LOCK && ! -L $INFERENCE_LOCK ]] ||
    die "inference lock is absent or unsafe"

exec 9<>"$INFERENCE_LOCK"
flock -n 9 || die "another inference or measurement owner is active"

/usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
    --interval-seconds 1 --timeout-seconds 60 >/dev/null

PROBE_TMP=$(/usr/bin/mktemp -d /tmp/frontier-bandwidth.XXXXXX)
cleanup() {
    local rc=$?
    trap - EXIT
    [[ $PROBE_TMP == /tmp/frontier-bandwidth.* && -d $PROBE_TMP ]] &&
        /usr/bin/rm -rf -- "$PROBE_TMP"
    exit "$rc"
}
trap cleanup EXIT

KERNEL_CURSOR=$(
    /usr/bin/journalctl -k -n 0 --show-cursor --no-pager |
        /usr/bin/sed -n 's/^-- cursor: //p'
)
[[ -n $KERNEL_CURSOR ]] || die "cannot freeze kernel journal cursor"

/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=native \
    "$SOURCE" -o "$PROBE_TMP/read-bandwidth"

# The parent retains the inference lock while the probe runs in a hard 4 GiB
# cgroup with MemorySwapMax=0. The probe itself allocates exactly 2 GiB.
/usr/bin/systemd-run --quiet --pipe --wait --collect \
    --unit="frontier-bandwidth-$$" \
    --property=Type=exec \
    --property=KillMode=control-group \
    --property=OOMPolicy=kill \
    --property=MemoryHigh=3G \
    --property=MemoryMax=4G \
    --property=MemorySwapMax=0 \
    --property=RuntimeMaxSec=300 \
    "$PROBE_TMP/read-bandwidth" >"$PROBE_TMP/result.json"

/usr/bin/python3 - "$PROBE_TMP/result.json" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if set(value) != {
    "schema_version", "bytes", "samples", "bandwidth_gb_s",
    "elapsed_ms", "checksum",
}:
    raise SystemExit("bandwidth result schema is invalid")
if (
    value["schema_version"] != 1
    or value["bytes"] != 2 * 1024 * 1024 * 1024
    or value["samples"] != 5
    or len(value["bandwidth_gb_s"]) != 5
    or len(value["elapsed_ms"]) != 5
    or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or item <= 0
        for field in ("bandwidth_gb_s", "elapsed_ms")
        for item in value[field]
    )
    or not isinstance(value["checksum"], (int, float))
    or isinstance(value["checksum"], bool)
    or not math.isfinite(value["checksum"])
    or value["checksum"] == 0
):
    raise SystemExit("bandwidth result values are invalid")
PY

/usr/bin/journalctl -k --after-cursor "$KERNEL_CURSOR" --no-pager \
    >"$PROBE_TMP/kernel.log"
if /usr/bin/grep -Eiq "$FAULT_PATTERN" "$PROBE_TMP/kernel.log"; then
    die "kernel OOM/Xid evidence appeared during the bandwidth probe"
fi

/usr/bin/install -m 0444 "$PROBE_TMP/result.json" "$OUTPUT"
