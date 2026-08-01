#!/bin/bash
# One-time, narrow deployment of benchmark-owner access to the shared lock.
set -Eeuo pipefail
umask 022

[[ $(id -u) == 0 ]] || {
    echo "71_install_glm_benchmark_lock_acl.sh: run as root" >&2
    exit 2
}

readonly ROOT=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE=$ROOT/configs/tmpfiles/frontier-at-home-glm-benchmark.conf
readonly TARGET=/etc/tmpfiles.d/frontier-at-home-glm-benchmark.conf
readonly LOCK=/run/lock/frontier-at-home/inference.lock

[[ -f $SOURCE && ! -L $SOURCE && -f $LOCK && ! -L $LOCK ]] || {
    echo "71_install_glm_benchmark_lock_acl.sh: source or shared lock is unsafe" >&2
    exit 2
}
/usr/bin/install -o root -g root -m 0644 -- "$SOURCE" "$TARGET"
/usr/bin/systemd-tmpfiles --create frontier-at-home-glm-benchmark.conf
/usr/sbin/runuser -u bmarti44 -- /usr/bin/python3 - "$LOCK" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise SystemExit("shared inference lock is not a stable regular file")
finally:
    os.close(descriptor)
PY
echo "Installed persistent benchmark-owner access to $LOCK; no service control was granted."
