#!/usr/bin/env python3
"""Validate one immutable GLM union-trace attempt."""

from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import re
import struct
from typing import Any


N_EMBD = 7168
N_EXPERT = 256
N_EXPERT_USED = 8
EVENT_RE = re.compile(
    r"^GLM_UNION_TRACE_OK path=full_indexed_batch_ffn "
    r"layer=(\d+) pos=(\d+) rows=(\d+)$"
)
FILE_RE = re.compile(
    r"^(?P<prefix>.+)_glm_indexed_(?P<kind>ffn_norm|router_logits|router_probs|router_selected|router_bias)-"
    r"(?P<layer>\d+)_pos(?P<pos>\d+)\.(?P<ext>f32|i32)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_variable_f32(path: Path) -> bool:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return bool(values) and all(math.isfinite(value) for value in values) and min(values) < max(values)


def _read_f32(path: Path) -> array:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return values


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _selected_rows(path: Path, rows: int) -> list[tuple[int, ...]] | None:
    values = array("i")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if len(values) != rows * N_EXPERT_USED:
        return None
    result: list[tuple[int, ...]] = []
    for offset in range(0, len(values), N_EXPERT_USED):
        row = tuple(values[offset:offset + N_EXPERT_USED])
        if any(value < 0 or value >= N_EXPERT for value in row):
            return None
        if len(set(row)) != N_EXPERT_USED:
            return None
        result.append(row)
    return result


def score_trace(
    directory: Path,
    server_log: Path,
    *,
    max_bytes: int,
    expected_layers: set[int],
    expected_chunks: list[tuple[int, int]],
) -> dict[str, Any]:
    """Return a fail-closed result for one trace attempt."""
    checks: dict[str, bool] = {}
    result: dict[str, Any] = {"verdict": "FAIL", "checks": checks}
    valid_layers = (
        isinstance(expected_layers, set) and bool(expected_layers) and
        all(isinstance(layer, int) and not isinstance(layer, bool) and layer >= 4
            for layer in expected_layers)
    )
    valid_chunks = bool(expected_chunks)
    previous_end: int | None = None
    for chunk in expected_chunks:
        if (not isinstance(chunk, tuple) or len(chunk) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool) for value in chunk)):
            valid_chunks = False
            break
        pos, rows = chunk
        if pos < 0 or rows <= 0 or (previous_end is not None and pos != previous_end):
            valid_chunks = False
            break
        previous_end = pos + rows
    if (not directory.is_dir() or not server_log.is_file() or max_bytes <= 0 or
            not valid_layers or not valid_chunks):
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
    expected_keys = {
        (layer, pos): rows
        for layer in expected_layers
        for pos, rows in expected_chunks
    }
    checks["exact_indexed_chunk_coverage"] = expected == expected_keys

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
        expected_extension = "i32" if kind == "router_selected" else "f32"
        if match.group("ext") != expected_extension:
            recognized = False
            continue
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
    all_selected_rows: list[tuple[int, ...]] = []
    wanted = {"ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias"}
    formula_ok = True
    probs_match_logits = True
    for key, rows in expected.items():
        group = files.get(key, {})
        if set(group) != wanted or rows <= 0:
            shapes_ok = False
            continue
        expected_sizes = {
            "ffn_norm": rows * N_EMBD * 4,
            "router_logits": rows * N_EXPERT * 4,
            "router_selected": rows * N_EXPERT_USED * 4,
            "router_probs": rows * N_EXPERT * 4,
            "router_bias": N_EXPERT * 4,
        }
        if any(group[kind].stat().st_size != size
               for kind, size in expected_sizes.items()):
            shapes_ok = False
            continue
        if (not _finite_variable_f32(group["ffn_norm"]) or
                not _finite_variable_f32(group["router_logits"]) or
                not _finite_variable_f32(group["router_probs"])):
            values_ok = False
        selected_rows = _selected_rows(group["router_selected"], rows)
        if selected_rows is None:
            values_ok = False
        else:
            all_selected_rows.extend(selected_rows)
            logits = _read_f32(group["router_logits"])
            probs = _read_f32(group["router_probs"])
            bias = _read_f32(group["router_bias"])
            if len(bias) != N_EXPERT or not all(math.isfinite(value) for value in bias):
                formula_ok = False
            else:
                for row_index, observed in enumerate(selected_rows):
                    start = row_index * N_EXPERT
                    for expert in range(N_EXPERT):
                        probability = float(probs[start + expert])
                        if (probability < 0.0 or probability > 1.0 or
                                abs(probability - _sigmoid(float(logits[start + expert]))) > 1e-4):
                            probs_match_logits = False
                    scores = [
                        _f32(float(probs[start + expert]) + float(bias[expert]))
                        for expert in range(N_EXPERT)
                    ]
                    expected_selected = tuple(sorted(
                        range(N_EXPERT), key=lambda expert: (-scores[expert], expert)
                    )[:N_EXPERT_USED])
                    if observed != expected_selected:
                        formula_ok = False
    if len(all_selected_rows) < 2 or len(set(all_selected_rows)) < 2:
        values_ok = False
    checks["exact_triplet_shapes"] = shapes_ok
    checks["finite_values_and_valid_ids"] = values_ok
    checks["selected_matches_router_formula"] = formula_ok
    checks["router_probs_match_logits"] = probs_match_logits

    result.update({
        "events": len(expected),
        "total_rows": sum(expected.values()),
        "total_bytes": total_bytes,
        "artifacts": artifacts,
    })
    result["verdict"] = "PASS" if checks and all(checks.values()) else "FAIL"
    return result
