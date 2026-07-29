#!/usr/bin/env python3
"""Atomically provision and bridge the legacy/current inference locks."""

from __future__ import annotations

import fcntl
import grp
import os
import stat
import sys
from pathlib import Path


RUNTIME = Path("/run/dsv4")
LEGACY = RUNTIME / "inference.lock"
CURRENT_ROOT = Path("/run/lock/frontier-at-home")
CURRENT = CURRENT_ROOT / "inference.lock"


def validate_visible_identity(path: Path, descriptor: int, gid: int) -> None:
    visible = path.lstat()
    opened = os.fstat(descriptor)
    if (
        stat.S_ISLNK(visible.st_mode)
        or visible.st_dev != opened.st_dev
        or visible.st_ino != opened.st_ino
        or visible.st_uid != 0
        or visible.st_gid != gid
        or visible.st_nlink != 1
    ):
        raise RuntimeError(f"inference lock changed during provisioning: {path}")


def open_private(path: Path, gid: int) -> int:
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError(f"unsafe inference lock: {path}")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"inference lock is occupied: {path}") from None
    os.fchown(descriptor, 0, gid)
    os.fchmod(descriptor, 0o660)
    try:
        validate_visible_identity(path, descriptor, gid)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def main() -> int:
    if os.geteuid() != 0 or sys.argv[1:] != []:
        raise RuntimeError("runtime lock provisioner requires root and no arguments")
    gid = grp.getgrnam("dsv4").gr_gid
    RUNTIME.mkdir(mode=0o1770, parents=True, exist_ok=True)
    os.chown(RUNTIME, 0, gid)
    os.chmod(RUNTIME, 0o1770)
    legacy = open_private(LEGACY, gid)
    current = -1
    try:
        CURRENT_ROOT.mkdir(mode=0o750, parents=True, exist_ok=True)
        os.chown(CURRENT_ROOT, 0, gid)
        os.chmod(CURRENT_ROOT, 0o750)
        current = open_private(CURRENT, gid)
    finally:
        if current >= 0:
            fcntl.flock(current, fcntl.LOCK_UN)
            os.close(current)
        fcntl.flock(legacy, fcntl.LOCK_UN)
        os.close(legacy)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"68_provision_runtime_locks.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
