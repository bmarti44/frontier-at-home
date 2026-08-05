#!/usr/bin/env python3
"""Validate and score frozen GLM held-out expert-address baselines."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
import types
import zipfile

import numpy as np


K_VALUES = (2, 4, 8)
BUDGETS = (16, 32, 64)
N_EXPERT = 256
N_SELECTED = 8
FIRST_LAYER = 4
LAST_LAYER = 77
CAPTURE_LAYERS = 79
ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts/78_glm_union_probe.py"
CV_PATH = ROOT / "scripts/79_glm_union_probe_cv.py"
PRECISION_PATH = ROOT / "scripts/80_glm_union_probe_precision.py"
FREEZE_PATH = ROOT / "results/glm52-gates/R0c-union-probe-p1-baseline-freeze.json"
PRECISION_RECEIPT = ROOT / "results/glm52-gates/R0c-union-probe-p1-precision-pass-a416101.json"
TEST_DIRECTORY = Path("/home/bmarti44/.local/state/glm52-p1-splits-r127-76faed9/test")
MODEL_DIRECTORY = Path("/home/bmarti44/.local/state/glm52-p1-precision-r139-a416101")
FROZEN_MODULE_HASHES = {
    PROBE_PATH: "4d17c536de2edf4b91f662a6e44b99bb7fc4f663c6ea5de8a6fa0517d4519373",
    CV_PATH: "0637c558fcde244a84e360191b43759c438e7fa08b4dde3271ff5f6824b82329",
    PRECISION_PATH: "df59dad48db200fb191d5033c82db8b44a9307f673c2bf6ce54d22c556d5dad6",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _execute_module_snapshot(
    name: str,
    path: Path,
    payload: bytes,
    *,
    injected: dict[str, object] | None = None,
    substitutions: tuple[tuple[bytes, bytes], ...] = (),
):
    executable = payload
    for original, replacement in substitutions:
        if executable.count(original) != 1:
            raise ValueError(f"authenticated module dependency edge differs: {path.name}")
        executable = executable.replace(original, replacement, 1)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    if injected:
        module.__dict__.update(injected)
    exec(compile(executable, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _load_frozen_module_graph():
    payloads = {path: _snapshot_regular(path) for path in FROZEN_MODULE_HASHES}
    for path, expected in FROZEN_MODULE_HASHES.items():
        if _sha256_bytes(payloads[path]) != expected:
            raise ValueError(f"frozen module differs: {path.name}")

    probe = _execute_module_snapshot(
        "glm_union_probe_for_baseline", PROBE_PATH, payloads[PROBE_PATH],
    )
    cv = _execute_module_snapshot(
        "glm_union_cv_for_baseline",
        CV_PATH,
        payloads[CV_PATH],
        injected={"__authenticated_probe__": probe},
        substitutions=((
            b"PROBE = _load_probe_module()\n",
            b"PROBE = __authenticated_probe__\n",
        ),),
    )
    precision = _execute_module_snapshot(
        "glm_union_precision_for_baseline",
        PRECISION_PATH,
        payloads[PRECISION_PATH],
        injected={"__authenticated_cv__": cv},
        substitutions=((
            b'CV = _load_module("glm_union_probe_cv_for_precision", CV_PATH)\n',
            b"CV = __authenticated_cv__\n",
        ),),
    )
    if cv.PROBE is not probe or precision.CV is not cv or precision.PROBE is not probe:
        raise ValueError("authenticated module dependency graph identity differs")
    return probe, cv, precision


def structural_baseline_rows(
    request_index: np.ndarray,
    layer: np.ndarray,
    token_position: np.ndarray,
) -> np.ndarray:
    """Select the frozen last eight K=8-complete positions in every group."""
    arrays = (request_index, layer, token_position)
    if (
        any(not isinstance(value, np.ndarray) or value.ndim != 1 for value in arrays) or
        request_index.size == 0 or layer.size != request_index.size or
        token_position.size != request_index.size or
        any(not np.issubdtype(value.dtype, np.integer) for value in arrays) or
        np.any(request_index <= 0) or np.any(layer < 0) or np.any(token_position < 0)
    ):
        raise ValueError("structural baseline input schema is invalid")
    selected: list[int] = []
    positions_by_request: dict[int, np.ndarray] = {}
    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    start = 0
    for stop in range(request_index.size + 1):
        current = None if stop == request_index.size else (
            int(request_index[stop]), int(layer[stop]),
        )
        if current == previous:
            continue
        if previous is not None:
            if previous in seen or (current is not None and current <= previous):
                raise ValueError("structural baseline groups are repeated or reordered")
            positions = token_position[start:stop].astype(np.int64, copy=False)
            if (
                positions.size < 16 or
                not np.array_equal(
                    positions, np.arange(positions[0], positions[0] + positions.size),
                )
            ):
                raise ValueError("structural baseline group lacks eight K8-complete rows")
            chosen = positions[-16:-8]
            request = previous[0]
            if request in positions_by_request and not np.array_equal(
                positions_by_request[request], chosen,
            ):
                raise ValueError("structural baseline positions differ across layers")
            positions_by_request.setdefault(request, chosen.copy())
            selected.extend(range(stop - 16, stop - 8))
            seen.add(previous)
        start = stop
        previous = current
    rows = np.asarray(selected, dtype=np.int64)
    if rows.size != len(seen) * 8 or not positions_by_request:
        raise ValueError("structural baseline coverage is incomplete")
    return rows


def captured_router_rankings(
    scores: np.ndarray,
    selected_ids: np.ndarray,
) -> np.ndarray:
    """Rank exact captured F32 scores and verify production top-8 selection."""
    if (
        not isinstance(scores, np.ndarray) or scores.ndim != 2 or
        scores.shape[0] == 0 or scores.shape[1] != N_EXPERT or
        scores.dtype != np.float32 or not np.isfinite(scores).all() or
        not isinstance(selected_ids, np.ndarray) or
        selected_ids.shape != (scores.shape[0], N_SELECTED) or
        not np.issubdtype(selected_ids.dtype, np.integer) or
        np.any(selected_ids < 0) or np.any(selected_ids >= N_EXPERT) or
        any(np.unique(row).size != N_SELECTED for row in selected_ids)
    ):
        raise ValueError("captured router score schema is invalid")
    expert_ids = np.arange(N_EXPERT, dtype=np.int64)
    rankings = np.stack([
        np.lexsort((expert_ids, -row.astype(np.float64))).astype(np.uint16)
        for row in scores
    ])
    if not np.array_equal(rankings[:, :N_SELECTED], selected_ids.astype(np.uint16)):
        raise ValueError("captured scores disagree with production top-8")
    return rankings


def mtp_prefix_rankings(
    scores: np.ndarray,
    selected_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return stable rankings of cumulative expert scores over nested MTP prefixes."""
    if (
        not isinstance(scores, np.ndarray) or scores.ndim != 3 or
        scores.shape[0] == 0 or scores.shape[1:] != (8, N_EXPERT) or
        scores.dtype != np.float32 or not np.isfinite(scores).all() or
        not isinstance(selected_ids, np.ndarray) or
        selected_ids.shape != (scores.shape[0], 8, N_SELECTED) or
        not np.issubdtype(selected_ids.dtype, np.integer)
    ):
        raise ValueError("MTP prefix score schema is invalid")
    captured_router_rankings(
        scores.reshape(-1, N_EXPERT), selected_ids.reshape(-1, N_SELECTED),
    )
    expert_ids = np.arange(N_EXPERT, dtype=np.int64)
    output: dict[int, np.ndarray] = {}
    running = np.full((scores.shape[0], N_EXPERT), -np.inf, dtype=np.float32)
    for step in range(8):
        running = np.maximum(running, scores[:, step])
        k = step + 1
        if k in K_VALUES:
            output[k] = np.stack([
                np.lexsort((expert_ids, -row.astype(np.float64))).astype(np.uint16)
                for row in running
            ])
    if set(output) != set(K_VALUES):
        raise ValueError("MTP prefix ranking coverage is incomplete")
    return output


def score_baseline_table(
    requests: np.ndarray,
    targets: dict[int, np.ndarray],
    rankings: dict[str, dict[int, np.ndarray]],
    *,
    bootstrap_seed: int = 20260805,
    bootstrap_resamples: int = 10000,
) -> dict[str, object]:
    """Score identical event keys for every frozen method/K/budget cell."""
    methods = {"frequency", "gate_replay", "shared_correction", "mtp", "probe"}
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(targets, dict) or set(targets) != set(K_VALUES) or
        not isinstance(rankings, dict) or set(rankings) != methods or
        any(not isinstance(value, dict) or set(value) != set(K_VALUES)
            for value in rankings.values()) or
        not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool) or
        not isinstance(bootstrap_resamples, int) or isinstance(bootstrap_resamples, bool) or
        bootstrap_resamples <= 0
    ):
        raise ValueError("baseline table schema is invalid")
    for k in K_VALUES:
        target = targets[k]
        if (
            not isinstance(target, np.ndarray) or target.shape != (requests.size, N_EXPERT) or
            target.dtype != np.bool_ or np.any(target.sum(axis=1) <= 0)
        ):
            raise ValueError("baseline target coverage is invalid")
        for method in methods:
            ranking = rankings[method][k]
            if (
                not isinstance(ranking, np.ndarray) or
                ranking.shape != (requests.size, N_EXPERT) or
                not np.issubdtype(ranking.dtype, np.integer) or
                np.any(ranking < 0) or np.any(ranking >= N_EXPERT) or
                any(np.unique(row).size != N_EXPERT for row in ranking)
            ):
                raise ValueError(f"baseline ranking is malformed: {method}/K{k}")

    unique_requests, counts = np.unique(requests.astype(np.int64), return_counts=True)
    if unique_requests.size == 0 or np.any(counts != counts[0]):
        raise ValueError("baseline request event coverage differs")
    event_rows = np.arange(requests.size)[:, None]
    metrics: dict[str, dict[str, dict[str, object]]] = {method: {} for method in methods}
    request_recall: dict[tuple[str, int, int], np.ndarray] = {}
    for method in sorted(methods):
        for k in K_VALUES:
            target = targets[k]
            target_size = target.sum(axis=1).astype(np.float64)
            metrics[method][str(k)] = {}
            for budget in BUDGETS:
                chosen = rankings[method][k][:, :budget]
                hits = target[event_rows, chosen].sum(axis=1).astype(np.float64)
                recall = hits / target_size
                precision = hits / budget
                wasted = budget - hits
                coverage = (hits == target_size).astype(np.float64)
                per_request = np.asarray([
                    float(np.mean(recall[requests == request])) for request in unique_requests
                ], dtype=np.float64)
                request_recall[(method, k, budget)] = per_request
                metrics[method][str(k)][str(budget)] = {
                    "requests": int(unique_requests.size),
                    "events": int(requests.size),
                    "macro_request_recall": float(np.mean(per_request)),
                    "macro_request_precision": float(np.mean([
                        np.mean(precision[requests == request]) for request in unique_requests
                    ])),
                    "macro_request_wasted_experts": float(np.mean([
                        np.mean(wasted[requests == request]) for request in unique_requests
                    ])),
                    "macro_request_full_set_coverage": float(np.mean([
                        np.mean(coverage[requests == request]) for request in unique_requests
                    ])),
                    "event_weighted_recall": float(np.mean(recall)),
                    "event_weighted_precision": float(np.mean(precision)),
                    "event_weighted_wasted_experts": float(np.mean(wasted)),
                    "event_weighted_full_set_coverage": float(np.mean(coverage)),
                }

    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0, unique_requests.size,
        size=(bootstrap_resamples, unique_requests.size),
        endpoint=False,
    )
    paired: dict[str, dict[str, dict[str, float]]] = {}
    all_probe_points_higher = True
    for k in K_VALUES:
        paired[str(k)] = {}
        for budget in BUDGETS:
            difference = (
                request_recall[("probe", k, budget)] -
                request_recall[("frequency", k, budget)]
            )
            sampled = difference[sample_indices].mean(axis=1)
            point = float(np.mean(difference))
            paired[str(k)][str(budget)] = {
                "probe_minus_frequency_point": point,
                "one_sided_95_lower": float(np.quantile(sampled, 0.05)),
                "two_sided_95_lower": float(np.quantile(sampled, 0.025)),
                "two_sided_95_upper": float(np.quantile(sampled, 0.975)),
            }
            all_probe_points_higher = all_probe_points_higher and point > 0.0
    primary = paired["4"]["32"]
    continue_p2 = all_probe_points_higher and primary["one_sided_95_lower"] > 0.0
    return {
        "methods": metrics,
        "paired_probe_minus_frequency": paired,
        "decision": {
            "probe_point_higher_all_nine_cells": all_probe_points_higher,
            "primary_one_sided_95_lower_positive": primary["one_sided_95_lower"] > 0.0,
            "continue_probe_to_P2": continue_p2,
            "verdict": "PASS" if continue_p2 else "STOP_PROBE",
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "unit": "request",
        },
    }


def _snapshot_regular(path: Path, expected_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            expected_bytes is not None and before.st_size != expected_bytes
        ):
            raise ValueError(f"capture artifact size or type differs: {path}")
        payload = bytearray()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"capture artifact ended early: {path}")
            payload.extend(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"capture artifact grew during read: {path}")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size
        ):
            raise ValueError(f"capture artifact identity changed: {path}")
    finally:
        os.close(descriptor)
    return bytes(payload)


def load_capture_source(directory: Path, source_position: int) -> dict[str, object]:
    """Read one source bundle through stable descriptors and validate its schema."""
    if not isinstance(source_position, int) or isinstance(source_position, bool) or source_position < 0:
        raise ValueError("source position is invalid")
    base = directory / f"source-{source_position:08d}"
    metadata_path = base.with_suffix(".json")
    metadata_bytes = _snapshot_regular(metadata_path)
    artifact_bindings = {
        metadata_path.name: {
            "bytes": len(metadata_bytes),
            "sha256": _sha256_bytes(metadata_bytes),
        }
    }
    metadata = _strict_json(metadata_bytes, "capture source metadata")
    expected = {
        "format": "glm52-p1-baseline-source-v1",
        "source_position": source_position,
        "mtp_min_position": source_position,
        "layers_total": CAPTURE_LAYERS,
        "layers_first": FIRST_LAYER,
        "layers_last": LAST_LAYER,
        "experts": N_EXPERT,
        "selected": N_SELECTED,
        "K": 8,
    }
    expected_keys = set(expected) | {
        "prompt_tokens", "vocab", "source_ready_ms", "mtp_ms", "target_ms",
        "cumulative_ms", "elapsed_ms", "predicted_tokens",
    }
    if set(metadata) != expected_keys or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("capture metadata differs")
    if (
        not isinstance(metadata.get("prompt_tokens"), int) or metadata["prompt_tokens"] < 16 or
        not isinstance(metadata.get("vocab"), int) or isinstance(metadata.get("vocab"), bool) or
        metadata["vocab"] <= 0 or
        not isinstance(metadata.get("source_ready_ms"), (int, float)) or
        isinstance(metadata.get("source_ready_ms"), bool) or
        not np.isfinite(metadata["source_ready_ms"]) or metadata["source_ready_ms"] <= 0 or
        not isinstance(metadata.get("elapsed_ms"), (int, float)) or
        isinstance(metadata.get("elapsed_ms"), bool) or
        not np.isfinite(metadata["elapsed_ms"]) or metadata["elapsed_ms"] <= 0 or
        not isinstance(metadata.get("predicted_tokens"), list) or
        len(metadata["predicted_tokens"]) != 8 or
        any(not isinstance(value, int) or isinstance(value, bool) or
            value < 0 or value >= metadata["vocab"]
            for value in metadata["predicted_tokens"])
    ):
        raise ValueError("capture metadata values are invalid")
    for name, length in (("mtp_ms", 7), ("target_ms", 8), ("cumulative_ms", 8)):
        values = metadata.get(name)
        if (
            not isinstance(values, list) or len(values) != length or
            any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                not np.isfinite(value) or value <= 0 for value in values)
        ):
            raise ValueError(f"capture {name} is invalid")
    cumulative = np.asarray(metadata["cumulative_ms"], dtype=np.float64)
    if np.any(np.diff(cumulative) <= 0) or cumulative[-1] > metadata["elapsed_ms"]:
        raise ValueError("capture cumulative timing is inconsistent")

    def array(suffix: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        size = int(np.prod(shape)) * dtype.itemsize
        path = Path(f"{base}{suffix}")
        payload = _snapshot_regular(path, size)
        artifact_bindings[path.name] = {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()

    gate_scores = array("-gate-scores.f32", np.dtype("<f4"), (CAPTURE_LAYERS, N_EXPERT))
    gate_selected = array(
        "-gate-selected.i32", np.dtype("<i4"), (CAPTURE_LAYERS, N_SELECTED),
    )
    shared_scores = array(
        "-shared-scores.f32", np.dtype("<f4"), (CAPTURE_LAYERS, N_EXPERT),
    )
    shared_selected = array(
        "-shared-selected.i32", np.dtype("<i4"), (CAPTURE_LAYERS, N_SELECTED),
    )
    mtp_scores = array(
        "-mtp-scores.f32", np.dtype("<f4"), (8, CAPTURE_LAYERS, N_EXPERT),
    )
    mtp_selected = array(
        "-mtp-selected.i32", np.dtype("<i4"), (8, CAPTURE_LAYERS, N_SELECTED),
    )
    predicted = array("-predicted.i32", np.dtype("<i4"), (8,))
    common = slice(FIRST_LAYER, LAST_LAYER + 1)
    captured_router_rankings(gate_scores[common], gate_selected[common])
    captured_router_rankings(shared_scores[common], shared_selected[common])
    mtp_prefix_rankings(
        np.transpose(mtp_scores[:, common], (1, 0, 2)),
        np.transpose(mtp_selected[:, common], (1, 0, 2)),
    )
    if not np.array_equal(predicted.astype(np.int64), metadata["predicted_tokens"]):
        raise ValueError("capture predicted-token artifacts differ")
    return {
        "metadata": metadata,
        "metadata_sha256": _sha256_bytes(metadata_bytes),
        "artifact_bindings": artifact_bindings,
        "gate_scores": gate_scores,
        "gate_selected": gate_selected,
        "shared_scores": shared_scores,
        "shared_selected": shared_selected,
        "mtp_scores": mtp_scores,
        "mtp_selected": mtp_selected,
        "predicted": predicted,
    }


def _strict_json(payload: bytes, label: str) -> dict[str, object]:
    def pairs(values):
        output = {}
        for key, value in values:
            if key in output:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _tracked_snapshot(path: Path) -> bytes:
    relative = path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    working = _snapshot_regular(path)
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT,
        stdin=subprocess.DEVNULL, capture_output=True, check=True,
    )
    if completed.stdout != working:
        raise ValueError(f"tracked artifact differs from HEAD: {relative}")
    return working


def _load_test_archive(cv_module) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    freeze = _strict_json(_tracked_snapshot(FREEZE_PATH), "baseline freeze")
    binding = freeze.get("test_split")
    if not isinstance(binding, dict) or Path(str(binding.get("directory"))) != TEST_DIRECTORY:
        raise ValueError("frozen test binding is malformed")
    manifest_payload = _snapshot_regular(TEST_DIRECTORY / "manifest.json")
    if _sha256_bytes(manifest_payload) != binding.get("manifest_sha256"):
        raise ValueError("test manifest differs from the freeze")
    manifest = _strict_json(manifest_payload, "test manifest")
    schema = manifest.get("array_schema")
    if (
        manifest.get("format") != "glm52-union-p1-split-npz-v1" or
        manifest.get("split") != "test" or manifest.get("requests") != 20 or
        not isinstance(schema, dict) or
        (TEST_DIRECTORY / "records.npz").stat().st_size != binding.get("output_bytes")
    ):
        raise ValueError("test manifest schema differs")
    records_payload = _snapshot_regular(
        TEST_DIRECTORY / "records.npz", int(binding["output_bytes"]),
    )
    if _sha256_bytes(records_payload) != binding.get("output_sha256"):
        raise ValueError("test records differ from the freeze")
    expected_members = {f"{name}.npy" for name in schema}
    with zipfile.ZipFile(io.BytesIO(records_payload), "r") as archive:
        names = archive.namelist()
        if len(names) != len(expected_members) or set(names) != expected_members:
            raise ValueError("test records ZIP member set differs")
        if archive.testzip() is not None:
            raise ValueError("test records ZIP checksum failed")
    arrays = {}
    with np.load(io.BytesIO(records_payload), allow_pickle=False) as archive:
        if set(archive.files) != set(schema):
            raise ValueError("test records array set differs")
        for name, expected_schema in schema.items():
            value = np.ascontiguousarray(archive[name])
            if value.dtype.hasobject or cv_module._array_schema(value) != expected_schema:
                raise ValueError(f"test records array binding differs: {name}")
            arrays[name] = value.copy()
    required = {
        "request_index", "layer", "token_position", "selected_ids",
        "hidden_q4", "hidden_scale",
    }
    if not required.issubset(arrays):
        raise ValueError("test records omit scorer inputs")
    return arrays, manifest, freeze


def _load_capture_set(
    capture_root: Path,
    test_manifest: dict[str, object],
    test_binding: dict[str, object],
    expected_positions: dict[int, list[int]],
) -> tuple[dict[tuple[int, int], dict[str, object]], dict[str, object]]:
    root = capture_root.resolve(strict=True)
    manifest_payload = _snapshot_regular(root / "manifest.json")
    manifest = _strict_json(manifest_payload, "capture-set manifest")
    entries = manifest.get("requests")
    if (
        manifest.get("schema_version") != 1 or
        manifest.get("format") != "glm52-p1-baseline-capture-set-v1" or
        manifest.get("test_manifest_sha256") != test_binding.get("manifest_sha256") or
        manifest.get("test_output_sha256") != test_binding.get("output_sha256") or
        not isinstance(entries, list) or len(entries) != 20
    ):
        raise ValueError("capture-set manifest differs from held-out binding")
    for name, width in (("engine_commit", 40), ("binary_sha256", 64),
                        ("model_sha256", 64), ("tokenizer_sha256", 64)):
        value = manifest.get(name)
        if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{width}}}", value):
            raise ValueError(f"capture-set {name} is malformed")
    metadata_rows = test_manifest.get("request_metadata")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != 20:
        raise ValueError("test request metadata is malformed")
    request_ids = {
        int(row["request_index"]): str(row["request_id"]) for row in metadata_rows
    }
    captures: dict[tuple[int, int], dict[str, object]] = {}
    seen_requests: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("capture request entry is malformed")
        request = entry.get("request_index")
        relative = entry.get("directory")
        artifacts = entry.get("artifacts")
        positions = entry.get("source_positions")
        if (
            not isinstance(request, int) or isinstance(request, bool) or
            request in seen_requests or request_ids.get(request) != entry.get("request_id") or
            not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
            not isinstance(artifacts, dict) or
            positions != expected_positions.get(request)
        ):
            raise ValueError("capture request identity or source positions differ")
        directory = (root / relative).resolve(strict=True)
        if root not in directory.parents:
            raise ValueError("capture request directory escapes its root")
        for position in positions:
            loaded = load_capture_source(directory, position)
            if loaded["metadata"]["prompt_tokens"] != entry.get("prompt_tokens"):
                raise ValueError("capture prompt length differs")
            observed = loaded["artifact_bindings"]
            expected_names = {
                name: value for name, value in artifacts.items()
                if name.startswith(f"source-{position:08d}")
            }
            if observed != expected_names:
                raise ValueError("capture source artifacts differ from manifest")
            captures[(request, position)] = loaded
        seen_requests.add(request)
    if seen_requests != set(expected_positions) or len(captures) != 160:
        raise ValueError("capture-set request/source coverage is incomplete")
    return captures, {
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "engine_commit": manifest.get("engine_commit"),
        "binary_sha256": manifest.get("binary_sha256"),
        "model_sha256": manifest.get("model_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
    }


def _load_probe_states(precision_module) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    receipt = _strict_json(_tracked_snapshot(PRECISION_RECEIPT), "precision receipt")
    artifacts = receipt.get("artifacts")
    manifest_payload = _snapshot_regular(MODEL_DIRECTORY / "manifest.json")
    model_manifest_payload = _snapshot_regular(MODEL_DIRECTORY / "model-manifest.json")
    if (
        not isinstance(artifacts, dict) or
        _sha256_bytes(manifest_payload) != artifacts.get("manifest_sha256") or
        _sha256_bytes(model_manifest_payload) != artifacts.get("model_manifest_sha256")
    ):
        raise ValueError("selected probe artifacts differ from their receipt")
    manifest = _strict_json(manifest_payload, "selected probe manifest")
    model_manifest = _strict_json(model_manifest_payload, "probe model manifest")
    layer_records = model_manifest.get("layers")
    if (
        manifest.get("selected_rank") != 32 or
        manifest.get("probe_sha256") != FROZEN_MODULE_HASHES[PROBE_PATH] or
        manifest.get("cv_driver_sha256") != FROZEN_MODULE_HASHES[CV_PATH] or
        not isinstance(layer_records, dict)
    ):
        raise ValueError("selected probe manifest differs")
    states = {}
    for layer in range(FIRST_LAYER, LAST_LAYER + 1):
        record = layer_records.get(str(layer))
        if not isinstance(record, dict):
            raise ValueError(f"probe state binding missing layer {layer}")
        path = MODEL_DIRECTORY / str(record.get("file"))
        if (
            _sha256_bytes(_snapshot_regular(path, int(record.get("bytes", 0)))) !=
            record.get("sha256")
        ):
            raise ValueError(f"probe state differs at layer {layer}")
        states[layer] = precision_module._state_from_file(path, record["schema"])
    return states, manifest["training_source_binding"]


def _stable_logit_rankings(logits: np.ndarray) -> np.ndarray:
    if (
        not isinstance(logits, np.ndarray) or logits.ndim != 2 or
        logits.shape[1] != N_EXPERT or not np.isfinite(logits).all()
    ):
        raise ValueError("probe logits are malformed")
    ids = np.arange(N_EXPERT, dtype=np.int64)
    return np.stack([
        np.lexsort((ids, -row.astype(np.float64))).astype(np.uint16)
        for row in logits
    ])


def score_heldout(capture_root: Path, device: str = "cuda") -> dict[str, object]:
    """Authorized one-shot opener and fixed held-out baseline scorer."""
    if capture_root.is_symlink():
        raise ValueError("capture root may not be a symlink")
    capture_root = capture_root.resolve(strict=True)
    if not capture_root.is_dir():
        raise ValueError("capture root is not a directory")
    probe, cv, precision = _load_frozen_module_graph()
    arrays, test_manifest, freeze = _load_test_archive(cv)
    common_mask = (arrays["layer"] >= FIRST_LAYER) & (arrays["layer"] <= LAST_LAYER)
    common = {name: value[common_mask] for name, value in arrays.items()
              if value.ndim > 0 and value.shape[0] == arrays["layer"].shape[0]}
    structural = structural_baseline_rows(
        common["request_index"], common["layer"], common["token_position"],
    )
    if structural.size != 20 * 74 * 8:
        raise ValueError("held-out structural event count differs")
    event_requests = common["request_index"][structural]
    event_layers = common["layer"][structural]
    event_positions = common["token_position"][structural]
    expected_positions = {
        int(request): sorted(set(int(value) for value in event_positions[event_requests == request]))
        for request in np.unique(event_requests)
    }
    if any(len(value) != 8 for value in expected_positions.values()):
        raise ValueError("held-out structural position coverage differs")
    captures, capture_binding = _load_capture_set(
        capture_root, test_manifest, freeze["test_split"], expected_positions,
    )
    states, training_binding = _load_probe_states(precision)
    training_sources, _groups = cv._load_authorized_sources(training_binding)
    frequency_counts = {
        layer: np.zeros(N_EXPERT, dtype=np.int64)
        for layer in range(FIRST_LAYER, LAST_LAYER + 1)
    }
    for source in training_sources:
        for layer in frequency_counts:
            selected = source["selected_ids"][source["layer"] == layer]
            if selected.size == 0:
                raise ValueError(f"training frequency source lacks layer {layer}")
            frequency_counts[layer] += np.bincount(
                selected.reshape(-1), minlength=N_EXPERT,
            )
    ids = np.arange(N_EXPERT, dtype=np.int64)
    frequency_by_layer = {
        layer: np.lexsort((ids, -counts)).astype(np.uint16)
        for layer, counts in frequency_counts.items()
    }
    frequency_events = np.stack([
        frequency_by_layer[int(layer)] for layer in event_layers
    ])

    targets: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        rows, all_targets = probe.future_union_targets(
            common["request_index"], common["layer"], common["token_position"],
            common["selected_ids"], k,
        )
        lookup = {int(row): index for index, row in enumerate(rows)}
        if any(int(row) not in lookup for row in structural):
            raise ValueError(f"structural K{k} target coverage differs")
        targets[k] = np.stack([all_targets[lookup[int(row)]] for row in structural])

    gate_scores = np.stack([
        captures[(int(request), int(position))]["gate_scores"][int(layer)]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ]).astype(np.float32, copy=False)
    gate_selected = np.stack([
        captures[(int(request), int(position))]["gate_selected"][int(layer)]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ])
    shared_scores = np.stack([
        captures[(int(request), int(position))]["shared_scores"][int(layer)]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ]).astype(np.float32, copy=False)
    shared_selected = np.stack([
        captures[(int(request), int(position))]["shared_selected"][int(layer)]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ])
    mtp_scores = np.stack([
        captures[(int(request), int(position))]["mtp_scores"][:, int(layer), :]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ]).astype(np.float32, copy=False)
    mtp_selected = np.stack([
        captures[(int(request), int(position))]["mtp_selected"][:, int(layer), :]
        for request, layer, position in zip(event_requests, event_layers, event_positions)
    ])
    gate_ranking = captured_router_rankings(gate_scores, gate_selected)
    shared_ranking = captured_router_rankings(shared_scores, shared_selected)
    mtp_rankings = mtp_prefix_rankings(mtp_scores, mtp_selected)

    probe_rankings = {
        k: np.empty((structural.size, N_EXPERT), dtype=np.uint16) for k in K_VALUES
    }
    for layer in range(FIRST_LAYER, LAST_LAYER + 1):
        layer_rows = np.flatnonzero(common["layer"] == layer)
        layer_events = np.flatnonzero(event_layers == layer)
        if layer_events.size != 20 * 8:
            raise ValueError(f"held-out event coverage differs at layer {layer}")
        features = np.concatenate([
            probe.unpack_probe_hidden(
                common["hidden_q4"][layer_rows], common["hidden_scale"][layer_rows],
            ),
            probe.causal_expert_history(
                common["request_index"][layer_rows], common["layer"][layer_rows],
                common["token_position"][layer_rows], common["selected_ids"][layer_rows],
            ),
        ], axis=1)
        logits = probe.predict_probe_head(features, states[layer], 32, device=device)
        local_by_key = {
            (int(common["request_index"][row]), int(common["token_position"][row])): index
            for index, row in enumerate(layer_rows)
        }
        local_events = np.asarray([
            local_by_key[(int(event_requests[index]), int(event_positions[index]))]
            for index in layer_events
        ], dtype=np.int64)
        for k_index, k in enumerate(K_VALUES):
            probe_rankings[k][layer_events] = _stable_logit_rankings(
                logits[local_events, k_index],
            )

    rankings = {
        "frequency": {k: frequency_events.copy() for k in K_VALUES},
        "gate_replay": {k: gate_ranking.copy() for k in K_VALUES},
        "shared_correction": {k: shared_ranking.copy() for k in K_VALUES},
        "mtp": mtp_rankings,
        "probe": probe_rankings,
    }
    scored = score_baseline_table(event_requests, targets, rankings)
    scored.update({
        "schema_version": 1,
        "classification": "P1_HELD_OUT_BASELINE_SCORE",
        "capture_binding": capture_binding,
        "coverage": {
            "requests": 20,
            "layers": 74,
            "source_positions_per_request": 8,
            "events": int(structural.size),
        },
        "claim_limit": freeze["claim_limit"],
    })
    return scored


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    parent = path.parent.resolve(strict=True)
    requested = parent / path.name
    temporary = parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, requested)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, value: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def _run_atomic_lifecycle(
    output_root: Path,
    preflight,
    open_heldout,
    capture,
    score,
) -> dict[str, object]:
    """Run one preflight/open/capture/score attempt and seal every opened outcome."""
    requested = output_root.absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(requested)
    parent = requested.parent.resolve(strict=True)
    output_root = parent / requested.name
    preflight_binding = preflight()
    if not isinstance(preflight_binding, dict):
        raise ValueError("atomic lifecycle preflight binding is malformed")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp.", dir=parent))
    attempt = {
        "schema_version": 1,
        "classification": "P1_HELD_OUT_ATOMIC_ATTEMPT",
        "status": "STARTED",
        "started_epoch_ns": time.time_ns(),
        "preflight": preflight_binding,
    }
    _write_json_replace(staging / "attempt.json", attempt)
    try:
        opened = open_heldout(staging)
        captured = capture(opened, staging)
        result = score(opened, captured, staging)
        if not isinstance(result, dict):
            raise ValueError("atomic lifecycle scorer result is malformed")
        _write_json_exclusive(staging / "summary.json", result)
        attempt.update({"status": "COMPLETE", "completed_epoch_ns": time.time_ns()})
        _write_json_replace(staging / "attempt.json", attempt)
    except BaseException as error:
        attempt.update({
            "status": "FAILED",
            "completed_epoch_ns": time.time_ns(),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        _write_json_replace(staging / "attempt.json", attempt)
        _rename_noreplace(staging, output_root)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise
    _rename_noreplace(staging, output_root)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-capture")
    validate.add_argument("--capture-directory", required=True, type=Path)
    validate.add_argument("--source-position", required=True, type=int)
    score = commands.add_parser("score")
    score.add_argument("--capture-root", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-capture":
        loaded = load_capture_source(args.capture_directory, args.source_position)
        print(json.dumps({"metadata": loaded["metadata"], "verdict": "PASS"}, sort_keys=True))
        return 0
    scored = score_heldout(args.capture_root, device=args.device)
    _write_json_exclusive(args.output, scored)
    print(json.dumps({
        "output": str(args.output),
        "verdict": scored["decision"]["verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
