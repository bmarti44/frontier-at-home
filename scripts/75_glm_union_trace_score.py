#!/usr/bin/env python3
"""Validate one immutable GLM union-trace attempt."""

from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import re
from typing import Any


N_EMBD = 7168
N_EXPERT = 256
N_EXPERT_USED = 8
EVENT_RE = re.compile(r"^GLM_UNION_TRACE_OK layer=(\d+) pos=(\d+) rows=(\d+)$")
FILE_RE = re.compile(
    r"^(?P<prefix>.+)_glm_indexed_(?P<kind>ffn_norm|router_logits|router_selected)-"
    r"(?P<layer>\d+)_pos(?P<pos>\d+)\.(?P<ext>f32|i32)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_f32(path: Path) -> bool:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return all(math.isfinite(value) for value in values)


def _valid_selected(path: Path, rows: int) -> bool:
    values = array("i")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if len(values) != rows * N_EXPERT_USED:
        return False
    for offset in range(0, len(values), N_EXPERT_USED):
        row = values[offset:offset + N_EXPERT_USED]
        if any(value < 0 or value >= N_EXPERT for value in row):
            return False
        if len(set(row)) != N_EXPERT_USED:
            return False
    return True


def score_trace(directory: Path, server_log: Path, *, max_bytes: int) -> dict[str, Any]:
    """Return a fail-closed result for one trace attempt."""
    checks: dict[str, bool] = {}
    result: dict[str, Any] = {"verdict": "FAIL", "checks": checks}
    if not directory.is_dir() or not server_log.is_file() or max_bytes <= 0:
        checks["inputs"] = False
        return result
    checks["inputs"] = True

    log_lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    checks["no_trace_errors"] = not any("GLM_UNION_TRACE_ERROR" in line for line in log_lines)
    expected: dict[tuple[int, int], int] = {}
    duplicate_log_key = False
    for line in log_lines:
        match = EVENT_RE.fullmatch(line.strip())
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in expected:
            duplicate_log_key = True
        expected[key] = int(match.group(3))
    checks["unique_nonempty_log_events"] = bool(expected) and not duplicate_log_key

    files: dict[tuple[int, int], dict[str, Path]] = {}
    prefixes: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    total_bytes = 0
    regular = True
    recognized = True
    duplicate_file_key = False
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            regular = False
            continue
        match = FILE_RE.fullmatch(path.name)
        if not match:
            recognized = False
            continue
        layer, pos = int(match.group("layer")), int(match.group("pos"))
        key = (layer, pos)
        kind = match.group("kind")
        if kind in files.setdefault(key, {}):
            duplicate_file_key = True
        files[key][kind] = path
        prefixes.add(match.group("prefix"))
        size = path.stat().st_size
        total_bytes += size
        artifacts.append({"name": path.name, "bytes": size, "sha256": _sha256(path)})
    checks["regular_files_only"] = regular
    checks["recognized_files_only"] = recognized
    checks["one_prefix"] = len(prefixes) == 1
    checks["unique_file_keys"] = not duplicate_file_key
    checks["byte_budget"] = total_bytes <= max_bytes
    checks["event_keys_match"] = set(files) == set(expected)

    shapes_ok = True
    values_ok = True
    wanted = {"ffn_norm", "router_logits", "router_selected"}
    for key, rows in expected.items():
        group = files.get(key, {})
        if set(group) != wanted or rows <= 0:
            shapes_ok = False
            continue
        expected_sizes = {
            "ffn_norm": rows * N_EMBD * 4,
            "router_logits": rows * N_EXPERT * 4,
            "router_selected": rows * N_EXPERT_USED * 4,
        }
        if any(group[kind].stat().st_size != size
               for kind, size in expected_sizes.items()):
            shapes_ok = False
            continue
        if not _finite_f32(group["ffn_norm"]) or not _finite_f32(group["router_logits"]):
            values_ok = False
        if not _valid_selected(group["router_selected"], rows):
            values_ok = False
    checks["exact_triplet_shapes"] = shapes_ok
    checks["finite_values_and_valid_ids"] = values_ok

    result.update({
        "events": len(expected),
        "total_bytes": total_bytes,
        "artifacts": artifacts,
    })
    result["verdict"] = "PASS" if checks and all(checks.values()) else "FAIL"
    return result
