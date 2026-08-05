#!/usr/bin/env python3
"""Validate and score frozen GLM held-out expert-address baselines."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import tempfile
import time
import types
import zipfile

import numpy as np
from tokenizers import Tokenizer


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
QUALITY_SPLIT_PLAN = ROOT / "results/glm52-gates/R0b-union-p0-split-plan.json"
QUALITY_RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-quality-runtime-randomness.json"
QUALITY_FIXTURE_RELATIVE = Path("gguf-tools/quality-testing/data/glm52-openrouter-100")
TOKENIZER_PATH = Path("/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json")
SAFE_RUN_PATH = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
MEMORY_GUARD_PATH = ROOT / "scripts/03_memory_guard.py"
FROZEN_BINARY_SHA256 = "49e728056d18c9eacd6986c6ca70290a2eb6ec46374ae94436555adc0fcc522b"
FROZEN_MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
FROZEN_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
FROZEN_FIXTURE_SHA256 = "49483fb172f700357d14167cfd9a69c686caa4e3b7889a41754bb4ba00584b0a"
FROZEN_ENGINE_COMMIT = "b8a152f29bb68197796b89ba755afb4aefe45dee"
FROZEN_MODEL_STAT = (66306, 679227, 211075856448, 1784912383428586016, 1784912515922318687)
FROZEN_SCRIPT_HASHES = {
    SAFE_RUN_PATH: "7d8bb58e526a5cbdd1980597506079fd2dadac294e255e577bb9fba9f6fdfd1f",
    MEMORY_GUARD_PATH: "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58",
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
    prompt_token_ids = array(
        "-prompt-tokens.i32", np.dtype("<i4"), (metadata["prompt_tokens"],),
    )
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
        "prompt_token_ids": prompt_token_ids,
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


def _render_quality_prompt(prompt: str) -> str:
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: High"
        "<|system|>You are a helpful assistant"
        f"<|user|>{prompt}<|assistant|><think>"
    )


def _quality_wire_body(prompt: str, seed: int) -> bytes:
    return json.dumps({
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, separators=(",", ":")).encode("utf-8")


def _manifest_rows(payload: bytes) -> list[tuple[str, str, str, str]]:
    text = payload.decode("utf-8", errors="strict")
    rows = [
        tuple(row) for row in csv.reader(
            (line for line in text.splitlines() if not line.startswith("#")),
            delimiter="\t",
        ) if row
    ]
    if any(len(row) != 4 for row in rows):
        raise ValueError("quality fixture manifest is malformed")
    return rows  # type: ignore[return-value]


def _load_authorized_test_cases(
    fixture_root: Path,
    test_manifest: dict[str, object],
) -> list[dict[str, object]]:
    """Open the frozen fixture mapping only inside the authorized lifecycle."""
    plan_payload = _tracked_snapshot(QUALITY_SPLIT_PLAN)
    plan = _strict_json(plan_payload, "quality split plan")
    randomness = _strict_json(_tracked_snapshot(QUALITY_RANDOMNESS), "quality randomness")
    seed = randomness.get("seed")
    splits = plan.get("splits")
    if (
        plan.get("schema_version") != 2 or
        plan.get("classification") != "PREREGISTERED" or
        plan.get("scope") != "glm52_union_probe_case_grouped_splits" or
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or
        not isinstance(splits, dict)
    ):
        raise ValueError("quality fixture authority is malformed")
    fixture_root = fixture_root.resolve(strict=True)
    fixture = (fixture_root / QUALITY_FIXTURE_RELATIVE).resolve(strict=True)
    if fixture_root not in fixture.parents or not fixture.is_dir():
        raise ValueError("quality fixture root is invalid")
    tokenizer_payload = _snapshot_regular(TOKENIZER_PATH)
    if _sha256_bytes(tokenizer_payload) != FROZEN_TOKENIZER_SHA256:
        raise ValueError("quality tokenizer differs")
    tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8", errors="strict"))
    content_digest = hashlib.sha256()
    cases: list[dict[str, object]] = []
    seen: set[str] = set()
    for split in ("train", "calibration", "test"):
        split_record = splits.get(split)
        if not isinstance(split_record, dict):
            raise ValueError("quality split record is malformed")
        manifests = split_record.get("block_manifests")
        if not isinstance(manifests, list) or not manifests:
            raise ValueError("quality split block manifests are malformed")
        observed = 0
        for binding in manifests:
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ValueError("quality block binding is malformed")
            relative = binding["path"]
            if not isinstance(relative, str):
                raise ValueError("quality block path is malformed")
            manifest_payload = _tracked_snapshot(ROOT / relative)
            if _sha256_bytes(manifest_payload) != binding["sha256"]:
                raise ValueError("quality block manifest differs")
            relative_bytes = relative.encode("utf-8")
            content_digest.update(len(relative_bytes).to_bytes(8, "big"))
            content_digest.update(relative_bytes)
            content_digest.update(len(manifest_payload).to_bytes(8, "big"))
            content_digest.update(manifest_payload)
            for case_id, prompt_relative, continuation_relative, response_relative in _manifest_rows(manifest_payload):
                if case_id in seen or not re.fullmatch(r"case_[0-9]{3}", case_id):
                    raise ValueError("quality case identity differs")
                seen.add(case_id)
                observed += 1
                raw_by_kind: dict[str, bytes] = {}
                for kind, relative_fixture in (
                    ("prompt", prompt_relative),
                    ("continuation", continuation_relative),
                    ("response", response_relative),
                ):
                    path = (fixture_root / relative_fixture).resolve(strict=True)
                    if fixture_root not in path.parents or path.is_symlink():
                        raise ValueError("quality fixture path escapes its root")
                    raw = _snapshot_regular(path)
                    raw_by_kind[kind] = raw
                    relative_fixture_bytes = relative_fixture.encode("utf-8")
                    content_digest.update(len(relative_fixture_bytes).to_bytes(8, "big"))
                    content_digest.update(relative_fixture_bytes)
                    content_digest.update(len(raw).to_bytes(8, "big"))
                    content_digest.update(raw)
                prompt = raw_by_kind["prompt"].decode("utf-8", errors="strict")
                token_ids = tokenizer.encode(
                    _render_quality_prompt(prompt), add_special_tokens=False,
                ).ids
                case_seed = (seed + int(case_id.removeprefix("case_"))) % 2147483647
                cases.append({
                    "case_id": case_id,
                    "split": split,
                    "seed": case_seed,
                    "prompt": prompt,
                    "rendered_prompt": _render_quality_prompt(prompt),
                    "token_ids": token_ids,
                    "prompt_tokens": len(token_ids),
                    "request_id": _sha256_bytes(_quality_wire_body(prompt, case_seed)),
                })
        if observed != split_record.get("quality_cases"):
            raise ValueError("quality split case count differs")
    if len(cases) != 100 or len(seen) != 100 or content_digest.hexdigest() != FROZEN_FIXTURE_SHA256:
        raise ValueError("quality fixture content differs")
    random.Random(seed).shuffle(cases)
    for request_index, case in enumerate(cases, 1):
        case["request_index"] = request_index
    selected = [case for case in cases if case["split"] == "test"]
    metadata = test_manifest.get("request_metadata")
    expected = [{
        "request_index": case["request_index"],
        "request_id": case["request_id"],
        "case_id": case["case_id"],
        "group_id": case["case_id"],
        "split": "test",
        "seed": case["seed"],
        "prompt_tokens": case["prompt_tokens"],
    } for case in selected]
    if metadata != expected or len(selected) != 20:
        raise ValueError("held-out request mapping differs from frozen fixture lineage")
    return selected


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
    expected_prompt_token_ids: dict[int, np.ndarray],
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
        manifest.get("engine_commit") != FROZEN_ENGINE_COMMIT or
        manifest.get("binary_sha256") != FROZEN_BINARY_SHA256 or
        manifest.get("model_sha256") != FROZEN_MODEL_SHA256 or
        manifest.get("tokenizer_sha256") != FROZEN_TOKENIZER_SHA256 or
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
        if set(entry) != {
            "request_index", "request_id", "prompt_tokens", "token_ids_sha256",
            "source_token_ids", "prefix_sha256", "source_positions", "directory",
            "artifacts",
        }:
            raise ValueError("capture request entry fields differ")
        request = entry.get("request_index")
        relative = entry.get("directory")
        artifacts = entry.get("artifacts")
        positions = entry.get("source_positions")
        expected_tokens = expected_prompt_token_ids.get(request) if (
            isinstance(request, int) and not isinstance(request, bool)
        ) else None
        if (
            not isinstance(request, int) or isinstance(request, bool) or
            request in seen_requests or request_ids.get(request) != entry.get("request_id") or
            not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
            not isinstance(artifacts, dict) or
            positions != expected_positions.get(request) or
            not isinstance(expected_tokens, np.ndarray) or
            entry.get("token_ids_sha256") != _sha256_bytes(
                expected_tokens.astype("<i4", copy=False).tobytes()
            ) or
            entry.get("source_token_ids") != [int(expected_tokens[pos]) for pos in positions] or
            entry.get("prefix_sha256") != [
                _sha256_bytes(expected_tokens[:pos + 1].astype("<i4", copy=False).tobytes())
                for pos in positions
            ]
        ):
            raise ValueError("capture request identity or source positions differ")
        requested_directory = root / relative
        if requested_directory.is_symlink():
            raise ValueError("capture request directory may not be a symlink")
        directory = requested_directory.resolve(strict=True)
        if root not in directory.parents:
            raise ValueError("capture request directory escapes its root")
        expected_artifact_names: set[str] = set()
        for position in positions:
            loaded = load_capture_source(directory, position)
            if loaded["metadata"]["prompt_tokens"] != entry.get("prompt_tokens"):
                raise ValueError("capture prompt length differs")
            if (
                not isinstance(expected_tokens, np.ndarray) or
                not np.array_equal(loaded["prompt_token_ids"], expected_tokens)
            ):
                raise ValueError("capture exact prompt-token lineage differs")
            observed = loaded["artifact_bindings"]
            expected_names = {
                name: value for name, value in artifacts.items()
                if name.startswith(f"source-{position:08d}")
            }
            expected_artifact_names.update(expected_names)
            if observed != expected_names:
                raise ValueError("capture source artifacts differ from manifest")
            captures[(request, position)] = loaded
        if set(artifacts) != expected_artifact_names:
            raise ValueError("capture request manifest has unrelated artifacts")
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
    """Reject the retired split capture/score path before held-out access."""
    if capture_root.is_symlink():
        raise ValueError("capture root may not be a symlink")
    capture_root = capture_root.resolve(strict=True)
    if not capture_root.is_dir():
        raise ValueError("capture root is not a directory")
    raise RuntimeError("standalone held-out scoring is disabled; use the atomic run command")


def _score_opened_heldout(
    capture_root: Path,
    device: str,
    probe,
    cv,
    precision,
    arrays: dict[str, np.ndarray],
    test_manifest: dict[str, object],
    freeze: dict[str, object],
    expected_prompt_token_ids: dict[int, np.ndarray],
) -> dict[str, object]:
    """Score one already-authorized in-memory opening without reopening it."""
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
        expected_prompt_token_ids,
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
        (staging / "summary.json").unlink(missing_ok=True)
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


def _hash_regular(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"identity input is not regular: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ValueError(f"identity input changed while hashing: {path}")
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _preflight_authorized_gate(configuration: dict[str, object]) -> dict[str, object]:
    expected_keys = {"output_root", "device", "candidate_root", "model", "fixture_root"}
    if set(configuration) != expected_keys or configuration.get("device") not in ("cpu", "cuda"):
        raise ValueError("authorized gate configuration is malformed")
    output_requested = Path(configuration["output_root"]).absolute()
    candidate_requested = Path(configuration["candidate_root"]).absolute()
    binary_requested = candidate_requested / "ds4-server"
    model_requested = Path(configuration["model"]).absolute()
    fixture_requested = Path(configuration["fixture_root"]).absolute()
    if any(path.is_symlink() for path in (
        candidate_requested, binary_requested, model_requested, fixture_requested,
    )):
        raise ValueError("authorized gate paths may not be symlinks")
    candidate_root = candidate_requested.resolve(strict=True)
    binary = binary_requested.resolve(strict=True)
    model = model_requested.resolve(strict=True)
    fixture_root = fixture_requested.resolve(strict=True)
    output_parent = output_requested.parent.resolve(strict=True)
    if (
        output_parent != Path("/home/bmarti44/.local/state") or
        not output_requested.name.startswith("glm52-") or
        candidate_root != candidate_requested or
        candidate_root.parent != Path("/home/bmarti44/.cache") or
        not candidate_root.name.startswith("glm52-") or
        binary.parent != candidate_root or
        model != model_requested or fixture_root != fixture_requested
    ):
        raise ValueError("authorized gate path authority differs")
    processes = subprocess.run(
        ["pgrep", "-a", "-x", "ds4|ds4-server"], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False,
    )
    if processes.returncode not in (0, 1) or processes.stdout.strip():
        raise RuntimeError("a DS4 engine process is already active")
    binary_sha256, binary_stat = _hash_regular(binary)
    if binary_sha256 != FROZEN_BINARY_SHA256:
        raise ValueError("authorized gate binary differs")
    model_sha256, model_stat = _hash_regular(model)
    if model_sha256 != FROZEN_MODEL_SHA256 or model_stat != FROZEN_MODEL_STAT:
        raise ValueError("authorized gate model differs")
    tokenizer_sha256, tokenizer_stat = _hash_regular(TOKENIZER_PATH)
    if tokenizer_sha256 != FROZEN_TOKENIZER_SHA256:
        raise ValueError("authorized gate tokenizer differs")
    for path, expected in FROZEN_SCRIPT_HASHES.items():
        if _sha256_bytes(_tracked_snapshot(path)) != expected:
            raise ValueError(f"authorized gate runtime differs: {path.name}")
    fixture = fixture_root / QUALITY_FIXTURE_RELATIVE
    if not fixture.is_dir() or fixture.is_symlink():
        raise ValueError("authorized gate fixture root differs")
    available_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available_kib < 110 * 1024 * 1024:
        raise MemoryError("authorized gate lacks 110 GiB available memory")
    return {
        "engine_commit": FROZEN_ENGINE_COMMIT,
        "binary_sha256": binary_sha256,
        "binary_stat": list(binary_stat),
        "model_sha256": model_sha256,
        "model_stat": list(model_stat),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_stat": list(tokenizer_stat),
        "fixture_content_sha256": FROZEN_FIXTURE_SHA256,
        "candidate_root": str(candidate_root),
        "model": str(model),
        "fixture_root": str(fixture_root),
        "mem_available_kib": available_kib,
    }


def _open_authorized_heldout(
    configuration: dict[str, object],
    preflight: dict[str, object],
    _staging: Path,
) -> dict[str, object]:
    probe, cv, precision = _load_frozen_module_graph()
    arrays, test_manifest, freeze = _load_test_archive(cv)
    cases = _load_authorized_test_cases(Path(str(preflight["fixture_root"])), test_manifest)
    return {
        "probe": probe,
        "cv": cv,
        "precision": precision,
        "arrays": arrays,
        "test_manifest": test_manifest,
        "freeze": freeze,
        "cases": cases,
    }


def _environment_sha256(values: dict[str, str], names: list[str]) -> str:
    return _sha256_bytes(b"".join(
        name.encode("ascii") + b"=" + values.get(name, "<UNSET>").encode("utf-8") + b"\n"
        for name in sorted(names)
    ))


def _file_binding(path: Path) -> dict[str, object]:
    digest, identity = _hash_regular(path)
    return {"bytes": identity[2], "sha256": digest}


def _validate_safe_run_artifacts(
    main_payload: bytes,
    kernel_payload: bytes,
    stdout_payload: bytes,
    expected_environment_sha256: str,
) -> None:
    required_markers = (
        f"candidate_binary_sha256={FROZEN_BINARY_SHA256}".encode("ascii"),
        f"executed_environment_sha256={expected_environment_sha256}".encode("ascii"),
        b"executed candidate was verified alive at least once",
        b"SAFE_RUN end rc=0 killed=no",
    )
    if (
        b"FATAL" in main_payload or
        any(marker not in main_payload for marker in required_markers) or
        kernel_payload not in (b"NO_KERNEL_FAULTS\n", b"-- No entries --\n") or
        b"SAFE_RUN_DONE rc=0 killed=no dir=" not in stdout_payload
    ):
        raise RuntimeError("contained baseline runtime attestation differs")


def _capture_authorized_set(
    configuration: dict[str, object],
    preflight: dict[str, object],
    opened: dict[str, object],
    staging: Path,
) -> dict[str, object]:
    arrays = opened["arrays"]
    test_manifest = opened["test_manifest"]
    freeze = opened["freeze"]
    if not isinstance(arrays, dict) or not isinstance(test_manifest, dict) or not isinstance(freeze, dict):
        raise ValueError("authorized opening is malformed")
    common_mask = (arrays["layer"] >= FIRST_LAYER) & (arrays["layer"] <= LAST_LAYER)
    common_request = arrays["request_index"][common_mask]
    common_layer = arrays["layer"][common_mask]
    common_position = arrays["token_position"][common_mask]
    structural = structural_baseline_rows(common_request, common_layer, common_position)
    expected_positions = {
        int(request): sorted(set(int(value) for value in common_position[structural][common_request[structural] == request]))
        for request in np.unique(common_request[structural])
    }
    capture_root = staging / "capture"
    capture_root.mkdir()
    prompt_root = staging / "prompts"
    prompt_root.mkdir()
    crash_root = Path("/home/bmarti44/.local/state/glm52-crashlog")
    crash_root.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(str(preflight["candidate_root"]))
    binary = candidate_root / "ds4-server"
    model = Path(str(preflight["model"]))
    entries = []
    runtime_logs = []
    cases = opened.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("authorized test case coverage differs")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("authorized test case is malformed")
        request = int(case["request_index"])
        request_dir = capture_root / f"request-{request:08d}"
        prompt_path = prompt_root / f"request-{request:08d}.txt"
        with prompt_path.open("x", encoding="utf-8") as handle:
            handle.write(str(case["rendered_prompt"]))
            handle.flush()
            os.fsync(handle.fileno())
        ds4_values = {
            "DS4_CUDA_FETCH_THREADS": "8",
            "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
            "DS4_GLM_BASELINE_CAPTURE_DIR": str(request_dir),
            "DS4_LOCK_FILE": "/run/dsv4/inference.lock",
        }
        names = sorted(ds4_values)
        environment = {
            **os.environ,
            **ds4_values,
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_TIMEOUT_S": "2400",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": FROZEN_BINARY_SHA256,
            "GLM_CANDIDATE_SRC": str(candidate_root),
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(names),
            "GLM_SAFE_EXPECTED_ENV_SHA256": _environment_sha256(ds4_values, names),
        }
        tag = f"p1-baseline-r{request:03d}-{os.getpid()}"
        before_logs = set(crash_root.glob(f"*-{tag}"))
        command = [
            "bash", str(SAFE_RUN_PATH), "--tag", tag, "--", str(binary),
            "--cuda", "-m", str(model), "-c", "1024", "--ssd-streaming",
            "--ssd-streaming-cache-experts", "6GB", "--glm-mtp",
            "--prompt-file", str(prompt_path), "--raw-prompt", "-n", "1",
            "--temp", "0",
        ]
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=2500, check=False,
        )
        after_logs = set(crash_root.glob(f"*-{tag}")) - before_logs
        if completed.returncode != 0 or len(after_logs) != 1:
            raise RuntimeError(f"contained baseline capture failed for request {request}")
        run_log = after_logs.pop()
        if run_log.is_symlink() or not run_log.is_dir():
            raise ValueError("contained baseline runtime log is invalid")
        log_files = {path.name: path for path in run_log.iterdir() if path.is_file()}
        if set(log_files) != {"cmd.log", "kernel.log", "main.log", "samples.log"}:
            raise ValueError("contained baseline runtime log set differs")
        main_payload = _snapshot_regular(log_files["main.log"])
        kernel_payload = _snapshot_regular(log_files["kernel.log"])
        stdout_payload = completed.stdout.encode("utf-8")
        expected_environment = environment["GLM_SAFE_EXPECTED_ENV_SHA256"]
        _validate_safe_run_artifacts(
            main_payload, kernel_payload, stdout_payload, expected_environment,
        )
        positions = expected_positions.get(request)
        if not isinstance(positions, list) or len(positions) != 8:
            raise ValueError("authorized structural source coverage differs")
        artifacts: dict[str, object] = {}
        for position in positions:
            loaded = load_capture_source(request_dir, position)
            expected_token_ids = np.asarray(case["token_ids"], dtype=np.int32)
            if (
                loaded["metadata"]["prompt_tokens"] != case["prompt_tokens"] or
                not np.array_equal(loaded["prompt_token_ids"], expected_token_ids)
            ):
                raise ValueError("captured prompt-token lineage differs")
            artifacts.update(loaded["artifact_bindings"])
        observed_files = sorted(path.name for path in request_dir.iterdir())
        if observed_files != sorted(artifacts):
            raise ValueError("capture request artifact set differs")
        entries.append({
            "request_index": request,
            "request_id": case["request_id"],
            "prompt_tokens": case["prompt_tokens"],
            "token_ids_sha256": _sha256_bytes(expected_token_ids.astype("<i4").tobytes()),
            "source_token_ids": [int(expected_token_ids[position]) for position in positions],
            "prefix_sha256": [
                _sha256_bytes(expected_token_ids[:position + 1].astype("<i4").tobytes())
                for position in positions
            ],
            "source_positions": positions,
            "directory": request_dir.relative_to(capture_root).as_posix(),
            "artifacts": artifacts,
        })
        log_bindings = {
            path.name: _file_binding(path) for path in sorted(run_log.iterdir()) if path.is_file()
        }
        runtime_logs.append({
            "request_index": request,
            "directory": str(run_log),
            "artifacts": log_bindings,
            "wrapper_exit_code": completed.returncode,
            "wrapper_stdout_sha256": _sha256_bytes(stdout_payload),
            "wrapper_stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        })
        prompt_path.unlink()
    manifest = {
        "schema_version": 1,
        "format": "glm52-p1-baseline-capture-set-v1",
        "test_manifest_sha256": freeze["test_split"]["manifest_sha256"],
        "test_output_sha256": freeze["test_split"]["output_sha256"],
        "engine_commit": FROZEN_ENGINE_COMMIT,
        "binary_sha256": preflight["binary_sha256"],
        "model_sha256": preflight["model_sha256"],
        "tokenizer_sha256": preflight["tokenizer_sha256"],
        "requests": entries,
    }
    _write_json_exclusive(capture_root / "manifest.json", manifest)
    final_model_sha256, final_model_stat = _hash_regular(model)
    if (
        final_model_sha256 != preflight["model_sha256"] or
        list(final_model_stat) != preflight["model_stat"]
    ):
        raise ValueError("authorized model changed during capture")
    binding = {
        "manifest": _file_binding(capture_root / "manifest.json"),
        "runtime_logs": runtime_logs,
    }
    _write_json_exclusive(staging / "raw.json", binding)
    return {"capture_root": capture_root, "binding": binding}


def _score_authorized_gate(
    configuration: dict[str, object],
    _preflight: dict[str, object],
    opened: dict[str, object],
    captured: dict[str, object],
    _staging: Path,
) -> dict[str, object]:
    cases = opened["cases"]
    expected_prompt_token_ids = {
        int(case["request_index"]): np.asarray(case["token_ids"], dtype=np.int32)
        for case in cases
    }
    return _score_opened_heldout(
        Path(captured["capture_root"]), str(configuration["device"]),
        opened["probe"], opened["cv"], opened["precision"], opened["arrays"],
        opened["test_manifest"], opened["freeze"], expected_prompt_token_ids,
    )


def _run_authorized_gate(configuration: dict[str, object]) -> dict[str, object]:
    state: dict[str, object] = {}

    def preflight():
        state["preflight"] = _preflight_authorized_gate(configuration)
        return state["preflight"]

    def opener(staging):
        return _open_authorized_heldout(configuration, state["preflight"], staging)

    def capture(opened, staging):
        return _capture_authorized_set(
            configuration, state["preflight"], opened, staging,
        )

    def score(opened, captured, staging):
        return _score_authorized_gate(
            configuration, state["preflight"], opened, captured, staging,
        )

    return _run_atomic_lifecycle(
        Path(configuration["output_root"]), preflight, opener, capture, score,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-capture")
    validate.add_argument("--capture-directory", required=True, type=Path)
    validate.add_argument("--source-position", required=True, type=int)
    run = commands.add_parser("run")
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument(
        "--candidate-root", type=Path,
        default=Path("/home/bmarti44/.cache/glm52-baseline-b8a152f-canonical"),
    )
    run.add_argument(
        "--model", type=Path,
        default=Path(
            "/home/dsv4/ds4-project/gguf-glm/"
            "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
        ),
    )
    run.add_argument("--fixture-root", type=Path, default=Path("/tmp/glm52-utf8-engine"))
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-capture":
        loaded = load_capture_source(args.capture_directory, args.source_position)
        print(json.dumps({"metadata": loaded["metadata"], "verdict": "PASS"}, sort_keys=True))
        return 0
    if args.command == "run":
        scored = _run_authorized_gate({
            "output_root": args.output_root,
            "candidate_root": args.candidate_root,
            "model": args.model,
            "fixture_root": args.fixture_root,
            "device": args.device,
        })
        print(json.dumps({
            "output_root": str(args.output_root),
            "verdict": scored["decision"]["verdict"],
        }, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
