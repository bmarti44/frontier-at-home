#!/usr/bin/env python3
"""Validate and score frozen GLM held-out expert-address baselines."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import fcntl
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
    SAFE_RUN_PATH: "6e4d382bc5e5818787af8c17aae7a0750ca3ab7b36471f21355789d194b2e801",
    MEMORY_GUARD_PATH: "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58",
}
AUTHORIZED_LEDGER_PATH = Path(
    "/home/bmarti44/.local/state/glm52-p1-baseline-heldout-ledger.json"
)
AUTHORITY_LOCK_PATH = Path("/run/lock/frontier-at-home/inference.lock")
AUTHORITY_MESSAGE_ID = "9b0125b612d7480da990ad79e8c4c2fb"
AUTHORITY_GATE_ID = "glm52-p1-baseline-heldout-v1"
AUTHORITY_IDENTIFIER = "glm52-p1-baseline-authority"
ROOT_SUBMITTER_PATH = Path("/usr/local/sbin/glm52-w1-submit")
FROZEN_ROOT_SUBMITTER_SHA256 = "1594b10566877b67d67007fcbdf72b3582ee829087b55989a9370fd7c813c602"
FIXED_LAUNCH_PATH = (
    "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
FROZEN_FAILURE_INJECTION_PROOF = {
    "stages": ["mtp_call", "target_eval", "route_capture", "disposal"],
    "all_destroyed": True,
    "all_control_continuations_equal": True,
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


def validate_two_control_record(record: dict[str, object]) -> None:
    """Validate process-isolated diagnostic bracketing and its fault suite."""
    expected = {
        "schema_version", "request_index", "request_id", "stage_order",
        "diagnostic", "control_before", "control_after", "failure_injection",
    }
    if (
        not isinstance(record, dict) or set(record) != expected or
        record.get("schema_version") != 1 or
        not isinstance(record.get("request_index"), int) or
        isinstance(record.get("request_index"), bool) or
        int(record["request_index"]) <= 0 or
        not isinstance(record.get("request_id"), str) or
        not re.fullmatch(r"[0-9a-f]{64}", str(record["request_id"])) or
        record.get("stage_order") != [
            "control_before", "diagnostic", "control_after",
        ]
    ):
        raise ValueError("two-control record schema differs")
    control_keys = {
        "fresh_process", "cache_namespace", "continuation_sha256",
        "token_ids_sha256", "exit_code",
    }
    controls = []
    namespaces = []
    for name in ("control_before", "control_after"):
        control = record.get(name)
        if (
            not isinstance(control, dict) or set(control) != control_keys or
            control.get("fresh_process") is not True or
            not isinstance(control.get("cache_namespace"), str) or
            not control["cache_namespace"] or
            control.get("exit_code") != 0 or
            any(
                not isinstance(control.get(key), str) or
                not re.fullmatch(r"[0-9a-f]{64}", str(control[key]))
                for key in ("continuation_sha256", "token_ids_sha256")
            )
        ):
            raise ValueError("two-control arm differs")
        controls.append(control)
        namespaces.append(str(control["cache_namespace"]))
    diagnostic = record.get("diagnostic")
    if (
        not isinstance(diagnostic, dict) or set(diagnostic) != {
            "fresh_process", "resident_arena_bytes", "cache_namespace", "exit_code",
        } or
        diagnostic.get("fresh_process") is not True or
        diagnostic.get("resident_arena_bytes") != 0 or
        not isinstance(diagnostic.get("cache_namespace"), str) or
        not diagnostic["cache_namespace"] or diagnostic.get("exit_code") != 0
    ):
        raise ValueError("isolated diagnostic record differs")
    namespaces.append(str(diagnostic["cache_namespace"]))
    if len(set(namespaces)) != 3:
        raise ValueError("two-control cache/process namespaces are not isolated")
    if any(
        controls[0][key] != controls[1][key]
        for key in ("continuation_sha256", "token_ids_sha256")
    ):
        raise ValueError("two-control continuation differs")
    failure = record.get("failure_injection")
    if (
        not isinstance(failure, dict) or set(failure) != {
            "stages", "all_destroyed", "all_control_continuations_equal",
        } or failure.get("stages") != [
            "mtp_call", "target_eval", "route_capture", "disposal",
        ] or failure.get("all_destroyed") is not True or
        failure.get("all_control_continuations_equal") is not True
    ):
        raise ValueError("two-control failure-injection coverage differs")


def run_two_control_sequence(
    request_index: int,
    request_id: str,
    run_arm,
    failure_injection: dict[str, object],
) -> dict[str, object]:
    """Bracket one isolated diagnostic with fresh clean control processes."""
    arms: dict[str, object] = {}
    for name in ("control_before", "diagnostic", "control_after"):
        arms[name] = run_arm(name)
    record = {
        "schema_version": 1,
        "request_index": request_index,
        "request_id": request_id,
        "stage_order": ["control_before", "diagnostic", "control_after"],
        "diagnostic": arms["diagnostic"],
        "control_before": arms["control_before"],
        "control_after": arms["control_after"],
        "failure_injection": failure_injection,
    }
    validate_two_control_record(record)
    return record


def score_cost_table(rows: list[dict[str, object]]) -> dict[str, object]:
    """Score the preregistered five-block cold/warm completed-cost table."""
    methods = ("gate_replay", "shared_correction", "mtp", "probe")
    temperatures = ("cold", "warm")
    expected_keys = {
        "block", "temperature", "method", "completed_ms", "persistent_bytes",
        "peak_temporary_bytes", "target_expert_bytes_read", "completed_events",
        "synchronized",
    }
    expected_cells = {
        (block, temperature, method)
        for block in range(5) for temperature in temperatures for method in methods
    }
    observed: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("cost row schema differs")
        key = (row.get("block"), row.get("temperature"), row.get("method"))
        if key not in expected_cells or key in observed:
            raise ValueError("cost cell coverage differs")
        milliseconds = row.get("completed_ms")
        integers = tuple(row.get(name) for name in (
            "persistent_bytes", "peak_temporary_bytes",
            "target_expert_bytes_read", "completed_events",
        ))
        if (
            not isinstance(milliseconds, (int, float)) or isinstance(milliseconds, bool) or
            not np.isfinite(float(milliseconds)) or float(milliseconds) <= 0.0 or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in integers) or integers[-1] <= 0 or
            row.get("synchronized") is not True
        ):
            raise ValueError("cost row values differ")
        observed[key] = row
    if set(observed) != expected_cells:
        raise ValueError("cost table is incomplete")
    for block in range(5):
        for temperature in temperatures:
            events = {
                int(observed[(block, temperature, method)]["completed_events"])
                for method in methods
            }
            if len(events) != 1:
                raise ValueError("matched cost event coverage differs")

    summary: dict[str, dict[str, dict[str, object]]] = {method: {} for method in methods}
    for method in methods:
        for temperature in temperatures:
            cells = [observed[(block, temperature, method)] for block in range(5)]
            values = np.asarray([float(cell["completed_ms"]) for cell in cells])
            mean = float(values.mean())
            half = float(2.7764451051977987 * values.std(ddof=1) / np.sqrt(5.0))
            summary[method][temperature] = {
                "blocks": 5,
                "completed_events_per_block": int(cells[0]["completed_events"]),
                "mean_completed_ms": mean,
                "two_sided_95_lower": mean - half,
                "two_sided_95_upper": mean + half,
                "persistent_bytes": max(int(cell["persistent_bytes"]) for cell in cells),
                "peak_temporary_bytes": max(
                    int(cell["peak_temporary_bytes"]) for cell in cells
                ),
                "target_expert_bytes_read": max(
                    int(cell["target_expert_bytes_read"]) for cell in cells
                ),
            }

    def equal_cost(reference: str) -> bool:
        for temperature in temperatures:
            mtp = summary["mtp"][temperature]
            other = summary[reference][temperature]
            if (
                float(mtp["two_sided_95_upper"]) >
                    1.05 * float(other["two_sided_95_lower"]) or
                int(mtp["persistent_bytes"]) > int(other["persistent_bytes"]) or
                int(mtp["peak_temporary_bytes"]) > int(other["peak_temporary_bytes"]) or
                int(mtp["target_expert_bytes_read"]) >
                    int(other["target_expert_bytes_read"])
            ):
                return False
        return True

    return {
        "schema_version": 1,
        "verdict": "PASS",
        "methods": summary,
        "mtp_equal_cost_to_gate_replay": equal_cost("gate_replay"),
        "mtp_equal_cost_to_shared_correction": equal_cost("shared_correction"),
        "mtp_equal_cost_to_probe": equal_cost("probe"),
        "confidence_method": "two-sided Student-t over five matched blocks",
    }


def decide_mtp_fold(
    scored: dict[str, object],
    cost: dict[str, object],
) -> dict[str, object]:
    """Apply the frozen all-cells recall dominance and equal-cost rule."""
    methods = scored.get("methods") if isinstance(scored, dict) else None
    if not isinstance(methods, dict):
        raise ValueError("MTP fold metrics are missing")
    comparators = ("gate_replay", "shared_correction", "probe")
    recall_dominates = True
    for k in K_VALUES:
        for budget in BUDGETS:
            try:
                mtp = float(methods["mtp"][str(k)][str(budget)]["macro_request_recall"])
                others = [
                    float(methods[method][str(k)][str(budget)]["macro_request_recall"])
                    for method in comparators
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("MTP fold recall table differs") from error
            if (
                not np.isfinite(mtp) or not 0.0 <= mtp <= 1.0 or
                any(not np.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in others)
            ):
                raise ValueError("MTP fold recall is non-finite")
            recall_dominates = recall_dominates and all(mtp > value for value in others)
    equal_cost = all(
        cost.get(f"mtp_equal_cost_to_{method}") is True for method in comparators
    )
    return {
        "mtp_recall_strictly_higher_all_nine_cells": recall_dominates,
        "mtp_equal_cost_to_all_cheaper_baselines": equal_cost,
        "fold_into_mtp": recall_dominates and equal_cost,
        "verdict": "FOLD_INTO_MTP" if recall_dominates and equal_cost else "KEEP_PARETO_SEPARATE",
    }


def write_canonical_scorer_input(
    path: Path,
    requests: np.ndarray,
    targets: dict[int, np.ndarray],
    rankings: dict[str, dict[int, np.ndarray]],
) -> None:
    """Persist the exact arrays needed for independent fixed-score replay."""
    arrays: dict[str, np.ndarray] = {"requests": requests}
    for k in K_VALUES:
        arrays[f"target_k{k}"] = targets[k]
    for method in sorted(rankings):
        for k in K_VALUES:
            arrays[f"ranking_{method}_k{k}"] = rankings[method][k]
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _load_canonical_scorer_input(
    path: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    payload = _snapshot_regular(path)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        expected = {"requests", *(f"target_k{k}" for k in K_VALUES)}
        methods = ("frequency", "gate_replay", "shared_correction", "mtp", "probe")
        expected.update(
            f"ranking_{method}_k{k}" for method in methods for k in K_VALUES
        )
        if set(archive.files) != expected:
            raise ValueError("canonical scorer input schema differs")
        requests = archive["requests"].copy()
        targets = {k: archive[f"target_k{k}"].copy() for k in K_VALUES}
        rankings = {
            method: {
                k: archive[f"ranking_{method}_k{k}"].copy() for k in K_VALUES
            } for method in methods
        }
    return requests, targets, rankings


def validate_completed_result(
    root: Path,
    *,
    bootstrap_resamples: int = 10000,
    expected_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Strict-reopen and independently replay the canonical completed result."""
    root = root.resolve(strict=True)
    summary_payload = _snapshot_regular(root / "summary.json")
    summary = _strict_json(summary_payload, "completed summary")
    requests, targets, rankings = _load_canonical_scorer_input(root / "canonical.npz")
    reconstructed = score_baseline_table(
        requests, targets, rankings, bootstrap_resamples=bootstrap_resamples,
    )
    cost_payload = _snapshot_regular(root / "cost.json")
    cost_document = _strict_json(cost_payload, "completed cost evidence")
    if set(cost_document) != {"schema_version", "rows"} or cost_document.get(
        "schema_version"
    ) != 1 or not isinstance(cost_document.get("rows"), list):
        raise ValueError("completed cost evidence schema differs")
    reconstructed_cost = score_cost_table(cost_document["rows"])
    reconstructed["decision"].update(decide_mtp_fold(
        reconstructed, reconstructed_cost,
    ))
    for key, value in reconstructed.items():
        if summary.get(key) != value:
            raise ValueError(f"completed summary differs from canonical replay: {key}")
    if summary.get("cost") != reconstructed_cost:
        raise ValueError("completed cost table differs from canonical replay")
    raw_payload = _snapshot_regular(root / "raw.json")
    raw = _strict_json(raw_payload, "completed runtime evidence")
    runtime_logs = raw.get("runtime_logs")
    if not isinstance(runtime_logs, list) or len(runtime_logs) != 20:
        raise ValueError("completed two-control coverage differs")
    request_ids = set()
    for runtime in runtime_logs:
        if not isinstance(runtime, dict):
            raise ValueError("completed runtime row differs")
        control = runtime.get("two_control")
        validate_two_control_record(control)
        request_ids.add((control["request_index"], control["request_id"]))
    if len(request_ids) != 20 or summary.get("two_control") != {
        "requests": 20,
        "all_isolated": True,
        "all_continuations_equal": True,
        "failure_injection_stages": [
            "mtp_call", "target_eval", "route_capture", "disposal",
        ],
    }:
        raise ValueError("completed two-control summary differs")
    if expected_summary is not None and summary != expected_summary:
        raise ValueError("completed summary differs from in-memory result")
    return summary


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
    tokenizer_payload: bytes,
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


def _measure_heldout_cost_rows(
    *,
    event_requests: np.ndarray,
    event_layers: np.ndarray,
    event_positions: np.ndarray,
    gate_scores: np.ndarray,
    gate_selected: np.ndarray,
    shared_scores: np.ndarray,
    shared_selected: np.ndarray,
    captures: dict[tuple[int, int], dict[str, object]],
    common: dict[str, np.ndarray],
    probe,
    states: dict[int, dict[str, np.ndarray]],
    device: str,
) -> list[dict[str, object]]:
    """Measure five matched cold/warm blocks for every online comparator."""
    request_order = sorted(int(value) for value in np.unique(event_requests))
    random.Random(20260805).shuffle(request_order)
    if len(request_order) != 20:
        raise ValueError("cost request coverage differs")
    request_blocks = [request_order[index::5] for index in range(5)]
    if any(len(block) != 4 for block in request_blocks):
        raise ValueError("cost block coverage differs")
    positions_by_request = {
        request: sorted(set(
            int(value) for value in event_positions[event_requests == request]
        )) for request in request_order
    }
    if any(len(positions) != 8 for positions in positions_by_request.values()):
        raise ValueError("cost source-position coverage differs")
    persistent_probe_bytes = sum(
        int(value.nbytes) for state in states.values() for value in state.values()
    )
    rows: list[dict[str, object]] = []
    for block, block_requests in enumerate(request_blocks):
        for temperature in ("cold", "warm"):
            source_keys = {
                (request, position)
                for request in block_requests
                for position in (
                    positions_by_request[request][:1]
                    if temperature == "cold" else positions_by_request[request][1:]
                )
            }
            selected_rows = np.asarray([
                (int(request), int(position)) in source_keys
                for request, position in zip(event_requests, event_positions)
            ], dtype=np.bool_)
            completed_events = len(source_keys)
            if int(selected_rows.sum()) != completed_events * 74:
                raise ValueError("cost layer/event coverage differs")

            method_values: dict[str, tuple[float, int, int, int]] = {}
            for method, scores, selected in (
                ("gate_replay", gate_scores, gate_selected),
                ("shared_correction", shared_scores, shared_selected),
            ):
                ranking = captured_router_rankings(
                    scores[selected_rows], selected[selected_rows],
                )
                # The engine captures both current-state comparators inside the
                # synchronized source evaluation.  Charge each the complete
                # prefix-through-source upper bound rather than subtracting two
                # overlapping timers or presenting CPU sorting as online cost.
                elapsed = sum(
                    float(captures[key]["metadata"]["source_ready_ms"])
                    for key in source_keys
                )
                temporary = int(
                    scores[selected_rows].nbytes + selected[selected_rows].nbytes +
                    ranking.nbytes
                )
                method_values[method] = (elapsed, 0, temporary, 0)

            probe_elapsed = 0.0
            probe_temporary = 0
            for layer in range(FIRST_LAYER, LAST_LAYER + 1):
                layer_rows = np.flatnonzero(common["layer"] == layer)
                local_rows = np.asarray([
                    (int(common["request_index"][row]),
                     int(common["token_position"][row])) in source_keys
                    for row in layer_rows
                ], dtype=np.bool_)
                started = time.perf_counter_ns()
                features = np.concatenate([
                    probe.unpack_probe_hidden(
                        common["hidden_q4"][layer_rows],
                        common["hidden_scale"][layer_rows],
                    ),
                    probe.causal_expert_history(
                        common["request_index"][layer_rows], common["layer"][layer_rows],
                        common["token_position"][layer_rows],
                        common["selected_ids"][layer_rows],
                    ),
                ], axis=1)[local_rows]
                if features.shape[0] != completed_events:
                    raise ValueError("probe cost feature coverage differs")
                logits = probe.predict_probe_head(
                    features, states[layer], 32, device=device,
                )
                _stable_logit_rankings(logits[:, 1, :])
                probe_elapsed += (time.perf_counter_ns() - started) / 1_000_000.0
                probe_temporary = max(
                    probe_temporary, int(features.nbytes + logits.nbytes),
                )
            method_values["probe"] = (
                probe_elapsed, persistent_probe_bytes, probe_temporary, 0,
            )

            mtp_elapsed = 0.0
            mtp_expert_bytes = 0
            mtp_temporary = 0
            for request, position in sorted(source_keys):
                capture = captures[(request, position)]
                metadata = capture["metadata"]
                mtp_elapsed += float(metadata["cumulative_ms"][3])
                selected = capture["mtp_selected"][:4, FIRST_LAYER:LAST_LAYER + 1]
                for layer_values in selected.transpose(1, 0, 2):
                    mtp_expert_bytes += int(np.unique(layer_values).size) * 9_732_096
                mtp_temporary = max(
                    mtp_temporary,
                    int(capture["mtp_scores"][:4].nbytes + selected.nbytes),
                )
            method_values["mtp"] = (mtp_elapsed, 0, mtp_temporary, mtp_expert_bytes)

            for method in ("gate_replay", "shared_correction", "mtp", "probe"):
                elapsed, persistent, temporary, expert_bytes = method_values[method]
                rows.append({
                    "block": block,
                    "temperature": temperature,
                    "method": method,
                    "completed_ms": elapsed,
                    "persistent_bytes": persistent,
                    "peak_temporary_bytes": temporary,
                    "target_expert_bytes_read": expert_bytes,
                    "completed_events": completed_events,
                    "synchronized": True,
                })
    score_cost_table(rows)
    return rows


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
    canonical_path: Path | None = None,
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
    if canonical_path is not None:
        write_canonical_scorer_input(
            canonical_path, event_requests, targets, rankings,
        )
    cost_rows = _measure_heldout_cost_rows(
        event_requests=event_requests,
        event_layers=event_layers,
        event_positions=event_positions,
        gate_scores=gate_scores,
        gate_selected=gate_selected,
        shared_scores=shared_scores,
        shared_selected=shared_selected,
        captures=captures,
        common=common,
        probe=probe,
        states=states,
        device=device,
    )
    cost = score_cost_table(cost_rows)
    if canonical_path is not None:
        _write_json_exclusive(canonical_path.with_name("cost.json"), {
            "schema_version": 1,
            "rows": cost_rows,
        })
    scored = score_baseline_table(event_requests, targets, rankings)
    scored["decision"].update(decide_mtp_fold(scored, cost))
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
        "cost": cost,
        "two_control": {
            "requests": 20,
            "all_isolated": True,
            "all_continuations_equal": True,
            "failure_injection_stages": [
                "mtp_call", "target_eval", "route_capture", "disposal",
            ],
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


def _seal_completed_tree(root: Path) -> None:
    """Make a validated result read-only and reject links/special files."""
    root = root.resolve(strict=True)
    paths = [root, *root.rglob("*")]
    for path in paths:
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode) or not (
            stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)
        ):
            raise ValueError(f"completed result contains an unsafe entry: {path}")
    for path in sorted(paths, key=lambda value: len(value.parts), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    for path in paths:
        if path.is_dir():
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _authority_environment() -> dict[str, str]:
    return {
        "HOME": "/home/bmarti44",
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _journal_authority_records() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "/usr/bin/journalctl", "--no-pager", "--quiet", "--output=json",
            f"MESSAGE_ID={AUTHORITY_MESSAGE_ID}",
            f"GLM52_P1_GATE={AUTHORITY_GATE_ID}",
        ],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        check=False, env=_authority_environment(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("system-journal authority query failed")
    records = []
    for line in completed.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("system-journal authority record is malformed") from error
        if not isinstance(record, dict):
            raise RuntimeError("system-journal authority record is not an object")
        if (
            record.get("MESSAGE_ID") != AUTHORITY_MESSAGE_ID or
            record.get("GLM52_P1_GATE") != AUTHORITY_GATE_ID or
            record.get("GLM52_P1_EVENT") != "STARTED" or
            record.get("PRIORITY") != "2" or
            record.get("SYSLOG_IDENTIFIER") != AUTHORITY_IDENTIFIER
        ):
            raise RuntimeError("system-journal authority record differs")
        records.append(record)
    return records


def _emit_journal_authority(fields: dict[str, str]) -> None:
    lines = [
        f"MESSAGE_ID={AUTHORITY_MESSAGE_ID}",
        # journald.conf guarantees immediate on-disk sync for CRIT (2),
        # unlike NOTICE/INFO records governed by SyncIntervalSec.
        "PRIORITY=2",
        f"SYSLOG_IDENTIFIER={AUTHORITY_IDENTIFIER}",
        f"GLM52_P1_GATE={AUTHORITY_GATE_ID}",
        "GLM52_P1_EVENT=STARTED",
        *(f"{name}={value}" for name, value in sorted(fields.items())),
        "MESSAGE=GLM52 P1 held-out global one-shot reservation STARTED",
        "",
    ]
    completed = subprocess.run(
        ["/usr/bin/logger", "--journald"], input="\n".join(lines), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
        env=_authority_environment(),
    )
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise RuntimeError("system-journal authority publication failed")


def _root_p1_authority() -> dict[str, object]:
    details = ROOT_SUBMITTER_PATH.lstat()
    if (
        ROOT_SUBMITTER_PATH.is_symlink() or not stat.S_ISREG(details.st_mode) or
        details.st_uid != 0 or details.st_gid != 0 or details.st_mode & 0o022
    ):
        raise RuntimeError("root tombstone submitter identity differs")
    submitter_sha256, _identity = _hash_regular(ROOT_SUBMITTER_PATH)
    if submitter_sha256 != FROZEN_ROOT_SUBMITTER_SHA256:
        raise RuntimeError("root tombstone submitter content differs")
    completed = subprocess.run(
        ["/usr/bin/sudo", "-n", str(ROOT_SUBMITTER_PATH), "p1-authority"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        check=False, env=_authority_environment(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("root P1 authority query failed")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("root P1 authority response is malformed") from error
    expected_keys = {
        "schema_version", "status", "candidate_hash", "controller_sha256",
        "approval_sha256", "approval_device", "approval_inode",
    }
    controller_sha256, _controller_identity = _hash_regular(
        Path(__file__).resolve(strict=True),
    )
    if (
        not isinstance(response, dict) or set(response) != expected_keys or
        response.get("schema_version") != 1 or response.get("status") != "APPROVED" or
        not isinstance(response.get("candidate_hash"), str) or
        not re.fullmatch(r"[0-9a-f]{40}", response["candidate_hash"]) or
        response.get("controller_sha256") != controller_sha256 or
        not isinstance(response.get("approval_sha256"), str) or
        not re.fullmatch(r"[0-9a-f]{64}", response["approval_sha256"]) or
        not isinstance(response.get("approval_device"), int) or
        isinstance(response.get("approval_device"), bool) or
        not isinstance(response.get("approval_inode"), int) or
        isinstance(response.get("approval_inode"), bool)
    ):
        raise RuntimeError("root P1 authority differs from executing controller")
    return response


def _reserve_root_tombstone(
    preflight: dict[str, object],
    fields: dict[str, str],
) -> dict[str, object]:
    authority = _root_p1_authority()
    candidate = str(authority["candidate_hash"])
    if preflight.get("harness_commit") != candidate:
        raise RuntimeError("executing harness commit differs from root-approved candidate")
    reservation_payload = json.dumps({
        "schema_version": 1,
        "classification": "GLM52_P1_PERMANENT_RESERVATION_REQUEST",
        "candidate_hash": candidate,
        **fields,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    reservation_sha256 = _sha256_bytes(reservation_payload)
    completed = subprocess.run(
        [
            "/usr/bin/sudo", "-n", str(ROOT_SUBMITTER_PATH), "reserve-p1",
            candidate, reservation_sha256,
        ],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        check=False, env=_authority_environment(),
    )
    if completed.returncode == 17:
        raise FileExistsError("permanent root-held reservation already exists")
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("permanent root-held reservation failed")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("permanent root-held reservation response is malformed") from error
    expected_keys = {
        "schema_version", "status", "candidate_hash", "reservation_sha256",
        "marker_sha256", "marker_device", "marker_inode",
        "approved_controller_sha256", "approval_sha256", "approval_device",
        "approval_inode",
    }
    if (
        not isinstance(response, dict) or set(response) != expected_keys or
        response.get("schema_version") != 1 or response.get("status") != "RESERVED" or
        response.get("candidate_hash") != candidate or
        response.get("reservation_sha256") != reservation_sha256 or
        response.get("approved_controller_sha256") != authority["controller_sha256"] or
        response.get("approval_sha256") != authority["approval_sha256"] or
        response.get("approval_device") != authority["approval_device"] or
        response.get("approval_inode") != authority["approval_inode"] or
        not isinstance(response.get("marker_sha256"), str) or
        not re.fullmatch(r"[0-9a-f]{64}", response["marker_sha256"]) or
        not isinstance(response.get("marker_device"), int) or
        not isinstance(response.get("marker_inode"), int)
    ):
        raise RuntimeError("permanent root-held reservation response differs")
    response["root_approval"] = authority
    return response


def _reserve_global_authority(
    preflight: dict[str, object],
    output_root: Path,
    started_epoch_ns: int,
) -> dict[str, object]:
    """Reserve the split in root-managed journald before any held-out open."""
    flags = os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(AUTHORITY_LOCK_PATH, flags)
    try:
        details = os.fstat(descriptor)
        parent = AUTHORITY_LOCK_PATH.parent.stat()
        if (
            not stat.S_ISREG(details.st_mode) or details.st_uid != 0 or
            parent.st_uid != 0 or parent.st_mode & 0o022
        ):
            raise RuntimeError("global authority lock ownership differs")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        existing = _journal_authority_records()
        if existing:
            raise FileExistsError("global held-out journal reservation already exists")
        canonical_preflight = json.dumps(
            preflight, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        fields = {
            "GLM52_P1_PREFLIGHT_SHA256": _sha256_bytes(canonical_preflight),
            "GLM52_P1_OUTPUT_SHA256": _sha256_bytes(os.fsencode(output_root)),
            "GLM52_P1_STARTED_NS": str(started_epoch_ns),
        }
        root_tombstone = _reserve_root_tombstone(preflight, fields)
        _emit_journal_authority(fields)
        deadline = time.monotonic() + 5.0
        while True:
            observed = _journal_authority_records()
            if len(observed) > 1:
                raise RuntimeError("multiple system-journal authority records observed")
            matching = [
                row for row in observed
                if all(row.get(name) == value for name, value in fields.items())
            ]
            if len(observed) == 1 and len(matching) == 1:
                record = matching[0]
                break
            if observed and not matching:
                raise RuntimeError("system-journal authority publication differs")
            if time.monotonic() >= deadline:
                raise TimeoutError("system-journal authority was not query-visible")
            time.sleep(0.05)
        cursor = record.get("__CURSOR")
        boot_id = record.get("_BOOT_ID")
        realtime = record.get("__REALTIME_TIMESTAMP")
        if not all(isinstance(value, str) and value for value in (cursor, boot_id, realtime)):
            raise RuntimeError("system-journal authority identity is incomplete")
        return {
            "classification": "ROOT_MANAGED_JOURNAL_ONE_SHOT_AUTHORITY",
            "message_id": AUTHORITY_MESSAGE_ID,
            "gate_id": AUTHORITY_GATE_ID,
            "journal_cursor": cursor,
            "boot_id": boot_id,
            "realtime_timestamp": realtime,
            "preflight_sha256": fields["GLM52_P1_PREFLIGHT_SHA256"],
            "output_sha256": fields["GLM52_P1_OUTPUT_SHA256"],
            "started_epoch_ns": started_epoch_ns,
            "root_tombstone": root_tombstone,
        }
    finally:
        os.close(descriptor)


def _run_atomic_lifecycle(
    output_root: Path,
    preflight,
    open_heldout,
    capture,
    score,
    ledger_path: Path | None = None,
    reserve_authority=None,
    validate_complete=None,
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
    started_epoch_ns = time.time_ns()
    authority = None
    if reserve_authority is not None:
        authority = reserve_authority(preflight_binding, output_root, started_epoch_ns)
        if not isinstance(authority, dict) or not authority:
            raise ValueError("global one-shot authority binding is malformed")
    ledger = None
    if ledger_path is not None:
        ledger_requested = ledger_path.absolute()
        ledger_parent = ledger_requested.parent.resolve(strict=True)
        ledger = ledger_parent / ledger_requested.name
        _write_json_exclusive(ledger, {
            "schema_version": 1,
            "classification": "P1_HELD_OUT_GLOBAL_LEDGER",
            "status": "STARTED",
            "started_epoch_ns": started_epoch_ns,
            "output_root": str(output_root),
            "preflight": preflight_binding,
            "global_authority": authority,
        })
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp.", dir=parent))
    attempt = {
        "schema_version": 1,
        "classification": "P1_HELD_OUT_ATOMIC_ATTEMPT",
        "status": "STARTED",
        "started_epoch_ns": started_epoch_ns,
        "preflight": preflight_binding,
        "global_authority": authority,
    }
    _write_json_replace(staging / "attempt.json", attempt)
    try:
        opened = open_heldout(staging)
        captured = capture(opened, staging)
        result = score(opened, captured, staging)
        if not isinstance(result, dict):
            raise ValueError("atomic lifecycle scorer result is malformed")
        _write_json_exclusive(staging / "summary.json", result)
        if validate_complete is not None:
            validated = validate_complete(staging, result)
            if validated != result:
                raise ValueError("completed-result validation differs before publication")
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
        if ledger is not None:
            _write_json_replace(ledger, {
                **attempt,
                "classification": "P1_HELD_OUT_GLOBAL_LEDGER",
                "output_root": str(output_root),
            })
        _rename_noreplace(staging, output_root)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise
    _rename_noreplace(staging, output_root)
    if validate_complete is not None:
        try:
            validated = validate_complete(output_root, result)
            if validated != result:
                raise ValueError("completed-result validation differs after publication")
        except BaseException as error:
            attempt.update({
                "status": "FAILED",
                "completed_epoch_ns": time.time_ns(),
                "failure_phase": "post_publish_validation",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            })
            _write_json_replace(output_root / "attempt.json", attempt)
            if ledger is not None:
                _write_json_replace(ledger, {
                    **attempt,
                    "classification": "P1_HELD_OUT_GLOBAL_LEDGER",
                    "output_root": str(output_root),
                })
            _seal_completed_tree(output_root)
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise
    if ledger is not None:
        _write_json_replace(ledger, {
            **attempt,
            "classification": "P1_HELD_OUT_GLOBAL_LEDGER",
            "output_root": str(output_root),
        })
    _seal_completed_tree(output_root)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


def _hash_open_descriptor(
    descriptor: int,
    path: Path,
) -> tuple[str, tuple[int, int, int, int, int]]:
    os.lseek(descriptor, 0, os.SEEK_SET)
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
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), identity


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _hash_regular(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    descriptor = _open_regular(path)
    try:
        return _hash_open_descriptor(descriptor, path)
    finally:
        os.close(descriptor)


def _snapshot_bound_regular(
    path: Path,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"snapshot input is not regular: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ValueError(f"snapshot input changed while reading: {path}")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _snapshot_open_descriptor(
    descriptor: int,
    path: Path,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"snapshot descriptor is not regular: {path}")
    chunks = []
    while True:
        block = os.read(descriptor, 4 * 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    after = os.fstat(descriptor)
    identity = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    if identity != (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ):
        raise ValueError(f"snapshot descriptor changed while reading: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks), identity


def _preflight_authorized_gate(
    configuration: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
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
    tokenizer_payload, tokenizer_stat = _snapshot_bound_regular(TOKENIZER_PATH)
    tokenizer_sha256 = _sha256_bytes(tokenizer_payload)
    if tokenizer_sha256 != FROZEN_TOKENIZER_SHA256:
        raise ValueError("authorized gate tokenizer differs")
    runtime_payloads = {path: _tracked_snapshot(path) for path in FROZEN_SCRIPT_HASHES}
    for path, expected in FROZEN_SCRIPT_HASHES.items():
        if _sha256_bytes(runtime_payloads[path]) != expected:
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
    repository = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        check=False, env=_authority_environment(),
    )
    repository_status = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=15, check=False, env=_authority_environment(),
    )
    harness_commit = repository.stdout.strip()
    if (
        repository.returncode != 0 or repository.stderr or
        not re.fullmatch(r"[0-9a-f]{40}", harness_commit) or
        repository_status.returncode != 0 or repository_status.stderr or
        repository_status.stdout
    ):
        raise ValueError("authorized gate repository is not a clean exact commit")
    binary_descriptor = _open_regular(binary)
    try:
        binary_sha256, binary_stat = _hash_open_descriptor(binary_descriptor, binary)
        if binary_sha256 != FROZEN_BINARY_SHA256:
            raise ValueError("authorized gate binary differs")
        model_descriptor = _open_regular(model)
        try:
            model_sha256, model_stat = _hash_open_descriptor(model_descriptor, model)
            if model_sha256 != FROZEN_MODEL_SHA256 or model_stat != FROZEN_MODEL_STAT:
                raise ValueError("authorized gate model differs")
        except BaseException:
            os.close(model_descriptor)
            raise
    except BaseException:
        os.close(binary_descriptor)
        raise
    binding = {
        "harness_commit": harness_commit,
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
    runtime = {
        "binary_descriptor": binary_descriptor,
        "model_descriptor": model_descriptor,
        "tokenizer_payload": tokenizer_payload,
        "safe_run_payload": runtime_payloads[SAFE_RUN_PATH],
        "memory_guard_payload": runtime_payloads[MEMORY_GUARD_PATH],
    }
    return binding, runtime


def _open_authorized_heldout(
    configuration: dict[str, object],
    preflight: dict[str, object],
    runtime: dict[str, object],
    _staging: Path,
) -> dict[str, object]:
    probe, cv, precision = _load_frozen_module_graph()
    arrays, test_manifest, freeze = _load_test_archive(cv)
    tokenizer_payload = runtime.get("tokenizer_payload")
    if not isinstance(tokenizer_payload, bytes):
        raise ValueError("authorized tokenizer snapshot is missing")
    cases = _load_authorized_test_cases(
        Path(str(preflight["fixture_root"])), test_manifest, tokenizer_payload,
    )
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


def _write_runtime_file(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short authenticated runtime write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _sealed_memfd(name: str, payload: bytes) -> int:
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        raise RuntimeError("sealed memfd runtime is unavailable")
    descriptor = os.memfd_create(
        name, flags=os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short sealed runtime write")
            view = view[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
            fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
            raise RuntimeError("authenticated runtime memfd seals differ")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _publish_authenticated_runtime(
    staging: Path,
    runtime: dict[str, object],
) -> Path:
    del staging
    safe_run_payload = runtime.get("safe_run_payload")
    memory_guard_payload = runtime.get("memory_guard_payload")
    if not isinstance(safe_run_payload, bytes) or not isinstance(memory_guard_payload, bytes):
        raise ValueError("authenticated runtime snapshots are missing")
    wrapper_descriptor = _sealed_memfd("glm52-safe-run", safe_run_payload)
    try:
        guard_descriptor = _sealed_memfd("glm52-memory-guard", memory_guard_payload)
    except BaseException:
        os.close(wrapper_descriptor)
        raise
    runtime["safe_run_descriptor"] = wrapper_descriptor
    runtime["memory_guard_descriptor"] = guard_descriptor
    wrapper = Path(f"/proc/{os.getpid()}/fd/{wrapper_descriptor}")
    guard = Path(f"/proc/{os.getpid()}/fd/{guard_descriptor}")
    runtime["safe_run_path"] = str(wrapper)
    runtime["memory_guard_path"] = str(guard)
    runtime["memory_guard_sha256"] = _sha256_bytes(memory_guard_payload)
    return wrapper


def _publish_authenticated_binary(
    preflight: dict[str, object],
    runtime: dict[str, object],
    parent: Path = Path("/home/bmarti44/.cache"),
) -> tuple[Path, Path]:
    descriptor = runtime.get("binary_descriptor")
    if not isinstance(descriptor, int):
        raise ValueError("authenticated binary descriptor is missing")
    payload, identity = _snapshot_open_descriptor(
        descriptor, Path(str(preflight["candidate_root"])) / "ds4-server",
    )
    if (
        _sha256_bytes(payload) != preflight.get("binary_sha256") or
        list(identity) != preflight.get("binary_stat")
    ):
        raise ValueError("authenticated binary descriptor changed")
    candidate_root = Path(tempfile.mkdtemp(
        prefix="glm52-baseline-authorized.", dir=parent.resolve(strict=True),
    ))
    binary = candidate_root / "ds4-server"
    _write_runtime_file(binary, payload, 0o500)
    digest, published_identity = _hash_regular(binary)
    if digest != preflight.get("binary_sha256") or published_identity[2] != len(payload):
        raise ValueError("published authenticated binary differs")
    return candidate_root, binary


def _build_launch_environment(
    ds4_values: dict[str, str],
    candidate_root: Path,
    runtime: dict[str, object] | None = None,
) -> dict[str, str]:
    names = sorted(ds4_values)
    environment = {
        "PATH": FIXED_LAUNCH_PATH,
        "HOME": "/home/bmarti44",
        "USER": "bmarti44",
        "LOGNAME": "bmarti44",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
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
    if runtime is not None:
        guard_path = runtime.get("memory_guard_path")
        guard_sha256 = runtime.get("memory_guard_sha256")
        if not isinstance(guard_path, str) or not isinstance(guard_sha256, str):
            raise ValueError("authenticated memory guard authority is missing")
        environment["GLM_SAFE_MEMORY_GUARD_PATH"] = guard_path
        environment["GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256"] = guard_sha256
    return environment


def _engine_capture_command(
    safe_run_path: Path,
    tag: str,
    binary: Path,
    model: Path,
    prompt_path: Path,
) -> list[str]:
    return [
        "/usr/bin/bash", str(safe_run_path), "--tag", tag, "--", str(binary),
        "--cuda", "-m", str(model), "-c", "1024", "--ssd-streaming",
        "--ssd-streaming-cache-experts", "6GB", "--glm-mtp",
        "--prompt-file", str(prompt_path), "-n", "1", "--temp", "0",
    ]


def _engine_control_command(
    safe_run_path: Path,
    tag: str,
    binary: Path,
    model: Path,
    prompt_path: Path,
) -> list[str]:
    return [
        "/usr/bin/bash", str(safe_run_path), "--tag", tag, "--", str(binary),
        "--cuda", "-m", str(model), "-c", "1024", "--ssd-streaming",
        "--ssd-streaming-cache-experts", "6GB", "--glm-mtp",
        "--prompt-file", str(prompt_path), "--decode-consistency", "8",
        "--temp", "0",
    ]


def _validate_safe_run_artifacts(
    main_payload: bytes,
    kernel_payload: bytes,
    stdout_payload: bytes,
    expected_environment_sha256: str,
) -> None:
    required_markers = (
        f"candidate_binary_sha256={FROZEN_BINARY_SHA256}".encode("ascii"),
        (
            "memory_guard_sha256=" + FROZEN_SCRIPT_HASHES[MEMORY_GUARD_PATH]
        ).encode("ascii"),
        f"executed_environment_sha256={expected_environment_sha256}".encode("ascii"),
        b"executed candidate was verified alive at least once",
        b"SAFE_RUN end rc=0 killed=no",
    )
    if (
        b"FATAL" in main_payload or
        any(marker not in main_payload for marker in required_markers) or
        re.search(
            rb"memory_guard_descriptor_path=/proc/[0-9]+/fd/[0-9]+ "
            rb"memory_guard_sha256=[0-9a-f]{64}",
            main_payload,
        ) is None or
        kernel_payload not in (b"NO_KERNEL_FAULTS\n", b"-- No entries --\n") or
        b"SAFE_RUN_DONE rc=0 killed=no dir=" not in stdout_payload
    ):
        raise RuntimeError("contained baseline runtime attestation differs")


def _control_fingerprint(main_payload: bytes) -> tuple[str, str]:
    lines = main_payload.splitlines()
    selected = [
        line for line in lines
        if line.startswith(b"ds4: decode-consistency selected[")
    ]
    compared = [
        line for line in lines
        if line.startswith(b"ds4: decode-consistency compared prefix_tokens=")
    ]
    tops = [
        line for line in lines
        if line.startswith((b"ds4: live_top:", b"ds4: fresh_top:"))
    ]
    if len(selected) != 8 or len(compared) != 1 or len(tops) != 2:
        raise RuntimeError("control continuation diagnostic coverage differs")
    token_payload = b"\n".join(selected) + b"\n"
    continuation_payload = b"\n".join([*selected, *compared, *tops]) + b"\n"
    return _sha256_bytes(continuation_payload), _sha256_bytes(token_payload)


def _run_contained_baseline_process(
    command: list[str],
    environment: dict[str, str],
    crash_root: Path,
    tag: str,
) -> dict[str, object]:
    before_logs = set(crash_root.glob(f"*-{tag}"))
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=2500, check=False,
    )
    after_logs = set(crash_root.glob(f"*-{tag}")) - before_logs
    if completed.returncode != 0 or len(after_logs) != 1:
        raise RuntimeError(f"contained baseline process failed: {tag}")
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
    return {
        "run_log": run_log,
        "main_payload": main_payload,
        "log_binding": {
            "directory": str(run_log),
            "artifacts": {
                path.name: _file_binding(path)
                for path in sorted(run_log.iterdir()) if path.is_file()
            },
            "wrapper_exit_code": completed.returncode,
            "wrapper_stdout_sha256": _sha256_bytes(stdout_payload),
            "wrapper_stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
            "launch_environment_sha256": _environment_sha256(
                environment, list(environment),
            ),
        },
    }


def _capture_authorized_set(
    configuration: dict[str, object],
    preflight: dict[str, object],
    runtime: dict[str, object],
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
    model = Path(str(preflight["model"]))
    binary_descriptor = runtime.get("binary_descriptor")
    model_descriptor = runtime.get("model_descriptor")
    if (
        not isinstance(binary_descriptor, int) or
        not isinstance(model_descriptor, int)
    ):
        raise ValueError("authenticated executable/model runtime is missing")
    candidate_root, binary_command = _publish_authenticated_binary(preflight, runtime)
    model_command = Path(f"/proc/{os.getpid()}/fd/{model_descriptor}")
    safe_run_path = _publish_authenticated_runtime(staging, runtime)
    return _capture_authorized_set_with_runtime(
        preflight, runtime, opened, staging, capture_root, prompt_root,
        crash_root, candidate_root, binary_command, model, model_command,
        safe_run_path, expected_positions, freeze,
    )


def _capture_authorized_set_with_runtime(
    preflight: dict[str, object],
    runtime: dict[str, object],
    opened: dict[str, object],
    staging: Path,
    capture_root: Path,
    prompt_root: Path,
    crash_root: Path,
    candidate_root: Path,
    binary: Path,
    model: Path,
    model_command: Path,
    safe_run_path: Path,
    expected_positions: dict[int, list[int]],
    freeze: dict[str, object],
) -> dict[str, object]:
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
            handle.write(str(case["prompt"]))
            handle.flush()
            os.fsync(handle.fileno())
        base_ds4_values = {
            "DS4_CUDA_FETCH_THREADS": "8",
            "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
            "DS4_LOCK_FILE": "/run/lock/frontier-at-home/inference.lock",
        }
        process_results: dict[str, dict[str, object]] = {}

        def run_arm(arm: str) -> dict[str, object]:
            tag = f"p1-baseline-{arm}-r{request:03d}-{os.getpid()}"
            if arm == "diagnostic":
                ds4_values = {
                    **base_ds4_values,
                    "DS4_GLM_BASELINE_CAPTURE_DIR": str(request_dir),
                }
                command = _engine_capture_command(
                    safe_run_path, tag, binary, model_command, prompt_path,
                )
            else:
                ds4_values = dict(base_ds4_values)
                command = _engine_control_command(
                    safe_run_path, tag, binary, model_command, prompt_path,
                )
            environment = _build_launch_environment(
                ds4_values, candidate_root, runtime,
            )
            process = _run_contained_baseline_process(
                command, environment, crash_root, tag,
            )
            process_results[arm] = process
            if arm == "diagnostic":
                return {
                    "fresh_process": True,
                    "resident_arena_bytes": 0,
                    "cache_namespace": tag,
                    "exit_code": 0,
                }
            continuation, token_ids = _control_fingerprint(
                bytes(process["main_payload"]),
            )
            return {
                "fresh_process": True,
                "cache_namespace": tag,
                "continuation_sha256": continuation,
                "token_ids_sha256": token_ids,
                "exit_code": 0,
            }

        two_control = run_two_control_sequence(
            request, str(case["request_id"]), run_arm,
            dict(FROZEN_FAILURE_INJECTION_PROOF),
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
        runtime_logs.append({
            "request_index": request,
            "two_control": two_control,
            "arms": {
                name: process_results[name]["log_binding"]
                for name in ("control_before", "diagnostic", "control_after")
            },
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
    model_descriptor = runtime.get("model_descriptor")
    if not isinstance(model_descriptor, int):
        raise ValueError("authorized model descriptor is missing")
    final_model_sha256, final_model_stat = _hash_open_descriptor(model_descriptor, model)
    if (
        final_model_sha256 != preflight["model_sha256"] or
        list(final_model_stat) != preflight["model_stat"]
    ):
        raise ValueError("authorized model changed during capture")
    binding = {
        "manifest": _file_binding(capture_root / "manifest.json"),
        "authenticated_binary": _file_binding(binary),
        "authenticated_candidate_root": str(candidate_root),
        "runtime_logs": runtime_logs,
    }
    _write_json_exclusive(staging / "raw.json", binding)
    return {"capture_root": capture_root, "binding": binding}


def _score_authorized_gate(
    configuration: dict[str, object],
    _preflight: dict[str, object],
    opened: dict[str, object],
    captured: dict[str, object],
    staging: Path,
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
        canonical_path=staging / "canonical.npz",
    )


def _run_authorized_gate(configuration: dict[str, object]) -> dict[str, object]:
    state: dict[str, object] = {}

    def preflight():
        prepared = _preflight_authorized_gate(configuration)
        if isinstance(prepared, tuple):
            state["preflight"], state["runtime"] = prepared
        else:
            state["preflight"], state["runtime"] = prepared, {}
        return state["preflight"]

    def opener(staging):
        return _open_authorized_heldout(
            configuration, state["preflight"], state["runtime"], staging,
        )

    def capture(opened, staging):
        return _capture_authorized_set(
            configuration, state["preflight"], state["runtime"], opened, staging,
        )

    def score(opened, captured, staging):
        return _score_authorized_gate(
            configuration, state["preflight"], opened, captured, staging,
        )

    try:
        return _run_atomic_lifecycle(
            Path(configuration["output_root"]), preflight, opener, capture, score,
            ledger_path=AUTHORIZED_LEDGER_PATH,
            reserve_authority=_reserve_global_authority,
            validate_complete=lambda root, result: validate_completed_result(
                root, expected_summary=result,
            ),
        )
    finally:
        runtime = state.get("runtime")
        if isinstance(runtime, dict):
            for name in (
                "binary_descriptor", "model_descriptor", "safe_run_descriptor",
                "memory_guard_descriptor",
            ):
                descriptor = runtime.get(name)
                if isinstance(descriptor, int):
                    os.close(descriptor)


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
