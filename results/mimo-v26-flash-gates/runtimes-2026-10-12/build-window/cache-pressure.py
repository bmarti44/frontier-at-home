#!/usr/bin/env python3
"""Evict stale block buffers and slab by streaming large files through page cache.

Reads files under the given roots (no mmap) until Buffers and SReclaimable are
small, then drops the page cache of everything it read. Page cache counts as
available memory, so MemAvailable stays high throughout; it stops early if
MemAvailable falls below 20 GiB or swap grows by more than 0.25 GiB.
Usage: cache-pressure.py <repo-root> <dir>...
"""
import os
import sys
import time

repo, roots = sys.argv[1], sys.argv[2:]
sys.path.insert(0, os.path.join(repo, "scripts/lib"))
from checkpoint_cache import release_tree  # noqa: E402

GiB = 1 << 30


def meminfo():
    out = {}
    with open("/proc/meminfo") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) * 1024 / GiB
    return out


def show(tag, m):
    print(f"{time.strftime('%H:%M:%S')} {tag} MemFree={m['MemFree']:.2f} MemAvailable={m['MemAvailable']:.2f} "
          f"Cached={m['Cached']:.2f} Buffers={m['Buffers']:.2f} SReclaimable={m['SReclaimable']:.2f} "
          f"SwapFree={m['SwapFree']:.2f}", flush=True)


start = meminfo()
show("start", start)
read = 0
next_check = 2 * GiB
done = False
buf = bytearray(16 << 20)
for root in roots:
    for directory, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            with open(path, "rb", buffering=0) as stream:
                while (n := stream.readinto(buf)):
                    read += n
                    if read >= next_check:
                        next_check += 2 * GiB
                        m = meminfo()
                        if m["MemAvailable"] < 20 or start["SwapFree"] - m["SwapFree"] > 0.25:
                            show("SAFETY STOP", m)
                            done = True
                        elif m["Buffers"] < 1.5 and m["SReclaimable"] < 2.0:
                            show("target reached", m)
                            done = True
                        elif read % (10 * GiB) < 2 * GiB:
                            show(f"read {read / GiB:.0f} GiB", m)
                    if done:
                        break
            if done:
                break
        if done:
            break
    if done:
        break
show(f"read {read / GiB:.1f} GiB, releasing", meminfo())
for root in roots:
    print(root, release_tree(root), flush=True)
show("end", meminfo())
