#!/usr/bin/env python3
"""DIAGNOSTIC watcher for run-diag.sh.

After the drafter's weights load (the second "Loading weights took" line),
send SIGUSR1 to EngineCore at +1/+2/+3 s and when MemAvailable first drops
below each of 24/21/18/16 GiB. SIGKILL the scope below 15 GiB at any time.
Exit when the server is ready or the scope is gone.

Usage: watch.py <server log> <offset> <out log> <scope unit>
"""
import os
import re
import signal
import subprocess
import sys
import time

server_log, offset, out, unit = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
THRESHOLDS, STOP_GIB, AFTER_LOAD_S = (24.0, 21.0, 18.0, 16.0), 15.0, (1.0, 2.0, 3.0)


def avail_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable")


def server_text():
    try:
        with open(server_log, "rb") as f:
            f.seek(offset)
            return f.read()
    except OSError:
        return b""


def scope_active():
    return subprocess.run(["systemctl", "--user", "is-active", "-q", f"{unit}.scope"]).returncode == 0


log = open(out, "a", buffering=1)
stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
fired, loaded_at, seen_scope = set(), None, False
while True:
    a, text = avail_gib(), server_text()
    pids = re.findall(rb"\(EngineCore pid=(\d+)\)", text)
    pid = int(pids[-1]) if pids else None
    active = scope_active()
    seen_scope |= active
    if a < STOP_GIB:
        log.write(f"{stamp()} STOP {a:.2f} GiB: SIGKILL scope\n")
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"])
        sys.exit(0)
    if loaded_at is None and text.count(b"Loading weights took") >= 2:
        loaded_at = time.monotonic()
        log.write(f"{stamp()} drafter weights loaded at {a:.2f} GiB\n")
    if loaded_at is not None and pid:
        due = [f"t+{s}" for s in AFTER_LOAD_S if time.monotonic() - loaded_at >= s]
        due += [f"<{t}" for t in THRESHOLDS if a < t]
        for key in due:
            if key not in fired:
                fired.add(key)
                try:
                    os.kill(pid, signal.SIGUSR1)
                    log.write(f"{stamp()} dump {key} at {a:.2f} GiB pid {pid}\n")
                except ProcessLookupError:
                    log.write(f"{stamp()} engine gone at {a:.2f} GiB\n")
    if b"Starting vLLM server on" in text:
        log.write(f"{stamp()} READY at {a:.2f} GiB\n")
        sys.exit(0)
    if seen_scope and not active:
        log.write(f"{stamp()} scope gone at {a:.2f} GiB\n")
        sys.exit(0)
    time.sleep(0.25)
