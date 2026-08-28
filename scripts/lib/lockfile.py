#!/usr/bin/env python3
"""Portable single-residency lock (macOS has no flock(1) binary).

Wraps fcntl.flock so serve lifecycles on any POSIX platform can hold the
inference residency lock (docs/BACKEND-CONTRACT.md section 3, guarantee 2).
The Linux production path keeps using /usr/bin/flock; this module is the
non-Linux equivalent and a library for Python callers.

CLI:
  lockfile.py check <path>       exit 0 when the lock is free, 1 when held
  lockfile.py run <path> -- CMD  hold the lock (non-blocking) while CMD runs
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys


class LockHeld(RuntimeError):
    pass


def acquire(path: str):
    """Non-blocking exclusive lock; returns the open descriptor holder."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise LockHeld(f"lock is held: {path}") from error
    return handle


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    verb, path = sys.argv[1], sys.argv[2]
    if verb == "check":
        try:
            acquire(path).close()
        except LockHeld:
            return 1
        return 0
    if verb == "run":
        rest = sys.argv[3:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        if not rest:
            print("lockfile.py run: missing command", file=sys.stderr)
            return 2
        try:
            holder = acquire(path)
        except LockHeld as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        try:
            return subprocess.run(rest, check=False).returncode
        finally:
            holder.close()
    print(f"lockfile.py: unknown verb {verb}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
