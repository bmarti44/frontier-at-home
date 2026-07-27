#!/usr/bin/python3
"""Make one evidence tree reviewer-readable without following links."""

from __future__ import annotations

import os
import stat
import sys


def fail(message: str) -> None:
    raise RuntimeError(message)


def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def open_checked(name: str, parent_fd: int, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if stat.S_ISDIR(expected.st_mode):
        flags |= os.O_DIRECTORY
    fd = os.open(name, flags, dir_fd=parent_fd)
    if not same_identity(expected, os.fstat(fd)):
        os.close(fd)
        fail("entry identity changed during export")
    return fd


def make_readable(fd: int, item: os.stat_result, is_directory: bool) -> None:
    mode = stat.S_IMODE(item.st_mode) | 0o444
    if is_directory or mode & 0o111:
        mode |= 0o111
    os.fchmod(fd, mode)


def walk_directory(fd: int, root_device: int) -> None:
    for name in os.listdir(fd):
        item = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if item.st_dev != root_device:
            fail("evidence tree crosses a filesystem boundary")
        if stat.S_ISLNK(item.st_mode):
            fail("evidence tree contains a symbolic link")
        if stat.S_ISDIR(item.st_mode):
            child_fd = open_checked(name, fd, item)
            try:
                walk_directory(child_fd, root_device)
                make_readable(child_fd, os.fstat(child_fd), True)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(item.st_mode):
            if item.st_nlink != 1:
                fail("evidence tree contains a multiply-linked file")
            child_fd = open_checked(name, fd, item)
            try:
                make_readable(child_fd, os.fstat(child_fd), False)
            finally:
                os.close(child_fd)
        else:
            fail("evidence tree contains a special file")


def main() -> int:
    if len(sys.argv) != 2 or not os.path.isabs(sys.argv[1]):
        print("usage: glm_evidence_export.py ABSOLUTE_DIRECTORY", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            fail("evidence root is not a physical directory")
        root_fd = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        try:
            opened = os.fstat(root_fd)
            if not same_identity(before, opened):
                fail("evidence root identity changed before export")
            walk_directory(root_fd, opened.st_dev)
            make_readable(root_fd, os.fstat(root_fd), True)
            after = os.stat(path, follow_symlinks=False)
            if not same_identity(opened, after):
                fail("evidence root identity changed during export")
        finally:
            os.close(root_fd)
    except (OSError, RuntimeError) as error:
        print(f"secure evidence export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
