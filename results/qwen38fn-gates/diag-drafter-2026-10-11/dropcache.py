#!/usr/bin/env python3
"""DIAGNOSTIC helper for run-diag.sh arm C: evict the checkpoint's page cache.

Arm B showed the kernel's NVRM NV_ERR_NO_MEMORY lines land whenever a large
GPU allocation meets MemFree near 1-3 GiB with ~23 GiB of checkpoint page
cache still to reclaim. Once a second this calls
posix_fadvise(POSIX_FADV_DONTNEED) on every *.safetensors file. That needs
no privilege and skips pages a process has mapped, so the engine's in-place
PLE mapping and the loader's live mmap stay intact. Runs until killed.

Usage: dropcache.py <model dir> <out log>
"""
import glob
import os
import sys
import time

model_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
log = open(out, "a", buffering=1)
stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cached_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


log.write(f"{stamp()} dropcache start: {len(files)} files in {model_dir}\n")
passes = 0
while True:
    before = cached_gib()
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    passes += 1
    if passes % 10 == 1:
        log.write(f"{stamp()} pass {passes}: Cached {before:.2f} -> {cached_gib():.2f} GiB\n")
    time.sleep(1)
