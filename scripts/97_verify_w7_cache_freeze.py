#!/usr/bin/env python3
"""Verify the deployed binary fields in a W7 cache-generation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def open_and_measure(path: Path) -> tuple[os.stat_result, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return metadata, digest.hexdigest()


def verify(path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    binary = freeze["binary"]
    artifact = Path(binary["path"])
    try:
        metadata, artifact_sha256 = open_and_measure(artifact)
    except OSError:
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
        "bytes_match": type(binary.get("bytes")) is int
        and binary["bytes"] == metadata.st_size,
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
