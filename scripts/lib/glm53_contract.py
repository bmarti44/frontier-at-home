"""Fail-closed GLM-5.3 artifact and serving contract (no model imports)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

MODEL = "glm-5.3-flash"
ALIAS = "glm53-flash"
PROFILE = "cuda-spark-128g-1m.json"


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"nonfinite JSON value: {value}")
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid inventory path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("inventory path escapes root or is not canonical")
    return Path(*path.parts)


def verify_inventory(root: Path, manifest: dict) -> dict[str, tuple[int, int, int, int, int]]:
    """Verify a nonempty closed tree; return identities for later revalidation.

    Caller must bind manifest bytes to the frozen candidate before using them.
    Symlinks/special files are forbidden, including symlinked root components.
    Overlay packs should use hard links or copies, not escaping symlinks.
    """
    root = Path(os.path.abspath(root))
    for component in (root, *root.parents):
        if component.is_symlink():
            raise ValueError(f"symlink inventory root: {component}")
    if not root.is_dir():
        raise ValueError(f"inventory root is not a directory: {root}")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ValueError("invalid inventory schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported inventory schema")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("inventory must contain files")
    declared = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("invalid inventory entry")
        path = _relative_path(entry["path"])
        if entry["path"] in declared:
            raise ValueError("duplicate inventory path")
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("invalid inventory size")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("invalid inventory digest")
        declared[path.as_posix()] = entry
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    def snapshot():
        files, identities = set(), {}
        for path in (root, *root.rglob("*")):
            observed = path.lstat()
            mode = observed.st_mode
            name = path.relative_to(root).as_posix()
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"inventory contains symlink or non-regular file: {path}")
            identities[name] = identity(observed)
            if stat.S_ISREG(mode):
                files.add(name)
        return files, identities

    actual, before_tree = snapshot()
    if actual != set(declared):
        raise ValueError(f"inventory coverage mismatch: missing={sorted(set(declared)-actual)}, unlisted={sorted(actual-set(declared))}")
    identities = {}
    for name, entry in declared.items():
        path = root / name
        before = path.stat(follow_symlinks=False)
        if before.st_size != entry["size_bytes"]:
            raise ValueError(f"inventory size mismatch: {name}")
        digest = sha256_file(path)
        after = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(after.st_mode) or identity(before) != identity(after):
            raise ValueError(f"inventory changed during verification: {name}")
        if digest != entry["sha256"]:
            raise ValueError(f"inventory digest mismatch: {name}")
        identities[name] = identity(after)
    after_files, after_tree = snapshot()
    if after_files != actual or after_tree != before_tree:
        raise ValueError("inventory coverage or identity changed during verification")
    return identities


def validate_serving(profile: dict) -> None:
    """The profile cap is aggregate; vLLM's max-model-len is per request."""
    serving = profile.get("serving")
    fields = {"parallel_slots", "request_context_cap", "max_images", "max_videos", "video_frames"}
    if not isinstance(serving, dict) or set(serving) != fields:
        raise ValueError("invalid serving topology schema")
    if any(type(serving[k]) is not int or serving[k] <= 0 for k in fields):
        raise ValueError("serving topology values must be positive integers")
    if serving["parallel_slots"] * serving["request_context_cap"] != profile["context_cap"]:
        raise ValueError("aggregate context topology mismatch")
    argv = profile["launch"].get("args", [])
    for flag, expected in (("--max-model-len", serving["request_context_cap"]),
                           ("--max-num-seqs", serving["parallel_slots"])):
        if argv.count(flag) != 1 or any(arg.startswith(flag + "=") for arg in argv):
            raise ValueError(f"serving topology requires one {flag}")
        index = argv.index(flag)
        if index + 1 == len(argv) or argv[index + 1] != str(expected):
            raise ValueError(f"serving topology disagrees with {flag}")


def require_qualified(profile: dict) -> None:
    # This admission gate deliberately has no environment override. Evidence
    # runs use the separately contained campaign path, never production switch.
    if profile.get("status", {}).get("state") != "qualified":
        raise ValueError("GLM-5.3-Flash is not qualified; use the hardened GLM-5.3 lifecycle for evidence")
    # A status edit alone cannot authorize a Python runtime with unbound files.
    raise ValueError("GLM-5.3-Flash qualification bundle and lifecycle are not yet installed")
