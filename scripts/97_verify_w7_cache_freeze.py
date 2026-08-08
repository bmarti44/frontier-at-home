#!/usr/bin/env python3
"""Verify the deployed binary fields in a W7 cache-generation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


MAX_EXECUTABLE_BYTES = 1024 * 1024 * 1024


def open_and_measure(path: Path, expected_bytes: int) -> tuple[os.stat_result, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise ValueError("artifact must be a regular file of the declared size")
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact ended before its declared size")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("artifact grew beyond its declared size")
        metadata_after = os.fstat(descriptor)
        if (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_mode,
            metadata_after.st_size,
        ) != (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size):
            raise ValueError("artifact identity changed while hashing")
    finally:
        os.close(descriptor)
    return metadata, digest.hexdigest()


def verify(path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    binary = freeze["binary"]
    artifact = Path(binary["path"])
    expected_bytes = binary.get("bytes")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or expected_bytes > MAX_EXECUTABLE_BYTES
    ):
        raise ValueError("invalid declared executable size")
    try:
        metadata, artifact_sha256 = open_and_measure(artifact, expected_bytes)
    except (OSError, ValueError):
        return {
            "checks": {
                "regular_non_symlink": False,
                "bytes_match": False,
                "mode_match": False,
                "sha256_match": False,
            },
            "verdict": "FAIL",
        }
    checks = {
        "regular_non_symlink": stat.S_ISREG(metadata.st_mode),
        "bytes_match": binary["bytes"] == metadata.st_size,
        "mode_match": type(binary.get("mode")) is int
        and binary["mode"] == stat.S_IMODE(metadata.st_mode),
        "sha256_match": isinstance(binary.get("sha256"), str)
        and binary["sha256"] == artifact_sha256,
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.freeze)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        result = {"checks": {}, "verdict": "FAIL"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
