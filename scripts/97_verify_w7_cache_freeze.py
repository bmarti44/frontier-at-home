#!/usr/bin/env python3
"""Verify the deployed binary fields in a W7 cache-generation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    binary = freeze["binary"]
    artifact = Path(binary["path"])
    metadata = artifact.lstat()
    checks = {
        "regular_non_symlink": stat.S_ISREG(metadata.st_mode) and not artifact.is_symlink(),
        "bytes_match": type(binary.get("bytes")) is int
        and binary["bytes"] == metadata.st_size,
        "mode_match": type(binary.get("mode")) is int
        and binary["mode"] == stat.S_IMODE(metadata.st_mode),
        "sha256_match": isinstance(binary.get("sha256"), str)
        and binary["sha256"] == sha256(artifact),
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
