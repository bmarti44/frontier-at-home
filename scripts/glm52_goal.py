#!/usr/bin/env python3
"""Fail-closed controller and fixed acceptance formulas for the GLM-5.2 goal.

The controller deliberately does not infer PASS from prose or engine logs.  A
gate can advance only through a registered runner which writes a manifest,
raw.jsonl and summary.json under the immutable attempt directory.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
import re
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / "results" / "glm52-goal"
W1_AUTHORITY_RECEIPT_ROOT = Path("/var/lib/glm52-w1/by-composite")
W1_AUTHORITY_ATTEMPT_ROOT = Path("/var/lib/glm52-w1/controller-attempts")
STATUSES = frozenset({"PENDING", "RED_CONFIRMED", "PASS", "FAIL", "NO_RESULT"})
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "NO_RESULT"})
W1_PACKED_RETRIEVAL_REASON = (
    "real packed storage, memory, and fidelity passed; "
    "deterministic retrieval qualification remains unfinished"
)
# Default-tree runners are executable authority. Keep the registry explicit
# and content-addressed; an untracked or stale executable must never become a
# controller action merely because it has a familiar filename.
DEFAULT_RUNNER_SHA256: dict[str, str] = {}
GATE_ORDER = (
    "foundation",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8",
    "W9",
    "W10",
    "W11",
    "switch",
    "parity",
    "review",
)


def _git_env() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }

# One-sided 95% Student-t critical values, indexed by degrees of freedom.
_T95 = {
    1: 6.3138,
    2: 2.9200,
    3: 2.3534,
    4: 2.1318,
    5: 2.0150,
    6: 1.9432,
    7: 1.8946,
    8: 1.8595,
    9: 1.8331,
    10: 1.8125,
    11: 1.7959,
    12: 1.7823,
    13: 1.7709,
    14: 1.7613,
    15: 1.7531,
    16: 1.7459,
    17: 1.7396,
    18: 1.7341,
    19: 1.7291,
    20: 1.7247,
    21: 1.7207,
    22: 1.7171,
    23: 1.7139,
    24: 1.7109,
    25: 1.7081,
    26: 1.7056,
    27: 1.7033,
    28: 1.7011,
    29: 1.6991,
    30: 1.6973,
}


class GoalError(RuntimeError):
    """A fail-closed controller or evidence validation error."""


def _finite_positive(values: Iterable[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} is empty")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{label} contains non-positive or non-finite values")
    return result


def decode_tokens_per_second(token_timestamps: Iterable[float]) -> float:
    """Return (N-1)/(tN-t1), requiring at least 128 emitted tokens."""
    timestamps = []
    for value in token_timestamps:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("token timestamps must be exact numeric values")
        try:
            timestamp = float(value)
        except OverflowError as exc:
            raise ValueError("token timestamp is outside the numeric range") from exc
        timestamps.append(timestamp)
    if len(timestamps) < 128:
        raise ValueError("decode requires at least 128 token timestamps")
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("token timestamps must be finite")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("token timestamps must be strictly increasing")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        raise ValueError("decode interval must be positive")
    return (len(timestamps) - 1) / elapsed


def _t95(df: int) -> float:
    if df < 1:
        raise ValueError("at least two paired samples are required")
    return _T95.get(min(df, 30), 1.6449)


def paired_ratio_bound(
    candidate: Iterable[float], reference: Iterable[float], *, side: str
) -> float:
    """One-sided 95% bound of the geometric mean paired ratio.

    Ratios are analyzed in log space. Fixtures must be paired in execution
    order; callers are responsible for ABBA/BAAB block validation.
    """
    left = _finite_positive(candidate, "candidate samples")
    right = _finite_positive(reference, "reference samples")
    if len(left) != len(right):
        raise ValueError("paired sample counts differ")
    if len(left) < 2:
        raise ValueError("at least two paired samples are required")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    logs = [math.log(a / b) for a, b in zip(left, right)]
    mean = statistics.fmean(logs)
    sem = statistics.stdev(logs) / math.sqrt(len(logs))
    delta = _t95(len(logs) - 1) * sem
    return math.exp(mean - delta if side == "lower" else mean + delta)


def performance_verdict(samples: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered matched-performance acceptance formula."""
    required = (
        "decode_glm",
        "decode_dsv4",
        "prefill_glm",
        "prefill_dsv4",
        "prefill_time_glm",
        "prefill_time_dsv4",
        "warm_ttft_glm",
        "warm_ttft_dsv4",
        "cold_ttft_glm",
        "cold_ttft_dsv4",
    )
    missing = [name for name in required if name not in samples]
    if missing:
        raise ValueError(f"missing performance samples: {', '.join(missing)}")
    metrics = {
        "decode_ratio_lower_95": paired_ratio_bound(
            samples["decode_glm"], samples["decode_dsv4"], side="lower"
        ),
        "prefill_ratio_lower_95": paired_ratio_bound(
            samples["prefill_glm"], samples["prefill_dsv4"], side="lower"
        ),
        "prefill_time_ratio_upper_95": paired_ratio_bound(
            samples["prefill_time_glm"],
            samples["prefill_time_dsv4"],
            side="upper",
        ),
        "warm_ttft_ratio_upper_95": paired_ratio_bound(
            samples["warm_ttft_glm"], samples["warm_ttft_dsv4"], side="upper"
        ),
        "cold_ttft_ratio_upper_95": paired_ratio_bound(
            samples["cold_ttft_glm"], samples["cold_ttft_dsv4"], side="upper"
        ),
    }
    checks = {
        "decode": metrics["decode_ratio_lower_95"] >= 0.80,
        "prefill_rate": metrics["prefill_ratio_lower_95"] >= 0.80,
        "prefill_time": metrics["prefill_time_ratio_upper_95"] <= 1.25,
        "warm_ttft": metrics["warm_ttft_ratio_upper_95"] <= 1.20,
        "cold_ttft": metrics["cold_ttft_ratio_upper_95"] <= 1.20,
    }
    return {
        "formula_version": 1,
        "metrics": metrics,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def context_verdict(observation: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed 1M context, retrieval and resource-safety formula."""
    required = {
        "context_cap",
        "processed_tokens",
        "retrieval_pass",
        "negative_control_pass",
        "completed_generation",
        "truncated",
        "oom",
        "xid",
        "available_memory_gib",
    }
    missing = sorted(required - observation.keys())
    if missing:
        raise ValueError(f"missing context fields: {', '.join(missing)}")
    for field in ("context_cap", "processed_tokens"):
        value = observation[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be an exact integer")
    memory_value = observation["available_memory_gib"]
    if isinstance(memory_value, bool) or not isinstance(
        memory_value, (int, float)
    ):
        raise ValueError("available memory must be an exact numeric value")
    try:
        memory = float(memory_value)
    except OverflowError as exc:
        raise ValueError("available memory is outside the numeric range") from exc
    if not math.isfinite(memory):
        raise ValueError("available memory is non-finite")
    for field in (
        "retrieval_pass",
        "negative_control_pass",
        "completed_generation",
        "truncated",
        "oom",
        "xid",
    ):
        if not isinstance(observation[field], bool):
            raise ValueError(f"{field} must be boolean")
    checks = {
        "context_cap": observation["context_cap"] == 1_048_576,
        "processed_tokens": observation["processed_tokens"] >= 1_000_000,
        "retrieval": observation["retrieval_pass"] is True,
        "negative_control": observation["negative_control_pass"] is True,
        "completed_generation": observation["completed_generation"] is True,
        "no_truncation": observation["truncated"] is False,
        "no_oom": observation["oom"] is False,
        "no_xid": observation["xid"] is False,
        "memory_floor": memory >= 10.0,
    }
    return {
        "formula_version": 1,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _weighted_upper_95(values: list[float], weights: list[int]) -> tuple[float, float]:
    total = float(sum(weights))
    mean = sum(value * weight for value, weight in zip(values, weights)) / total
    sum_w2 = float(sum(weight * weight for weight in weights))
    effective_n = total * total / sum_w2
    denominator = total - sum_w2 / total
    if denominator <= 0 or effective_n <= 1:
        raise ValueError("quality suite has no effective paired variance")
    variance = (
        sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights))
        / denominator
    )
    sem = math.sqrt(variance / effective_n)
    upper = mean + _t95(max(1, int(math.floor(effective_n - 1)))) * sem
    return mean, upper


def quality_verdict(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fixed 100-case paired lossy-fidelity scorer."""
    rows = list(cases)
    if len(rows) != 100:
        raise ValueError("quality suite requires exactly 100 paired cases")
    weights: list[int] = []
    nll_delta: list[float] = []
    top1_loss: list[float] = []
    for index, row in enumerate(rows):
        required = {
            "tokens",
            "baseline_nll_sum",
            "candidate_nll_sum",
            "baseline_top1_correct",
            "candidate_top1_correct",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"quality case {index} has wrong schema")
        tokens = row["tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError(f"quality case {index} has invalid token count")
        numeric = [float(row[name]) for name in ("baseline_nll_sum", "candidate_nll_sum")]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"quality case {index} has non-finite NLL")
        correct = [
            row["baseline_top1_correct"],
            row["candidate_top1_correct"],
        ]
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > tokens
            for value in correct
        ):
            raise ValueError(f"quality case {index} has invalid top-1 counts")
        weights.append(tokens)
        nll_delta.append((numeric[1] - numeric[0]) / tokens)
        top1_loss.append((correct[0] - correct[1]) / tokens)
    nll_mean, nll_upper = _weighted_upper_95(nll_delta, weights)
    top1_mean, top1_upper = _weighted_upper_95(top1_loss, weights)
    checks = {
        "delta_nll_mean": nll_mean <= 0.01,
        "delta_nll_upper_95": nll_upper <= 0.01,
        "top1_loss_pp": top1_mean * 100 <= 0.5,
        "top1_loss_upper_95_pp": top1_upper * 100 <= 0.5,
    }
    return {
        "formula_version": 1,
        "metrics": {
            "delta_nll": nll_mean,
            "delta_nll_upper_95": nll_upper,
            "top1_loss_pp": top1_mean * 100,
            "top1_loss_upper_95_pp": top1_upper * 100,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _score_w9_e2m1_fidelity_raw(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the bounded F32-physical E2M1 fidelity seam from raw arms."""
    if len(records) != 1:
        raise ValueError("W9 E2M1 scorer requires exactly one campaign")
    campaign = records[0]
    _require_exact_keys(
        campaign,
        {
            "record_type", "engine_candidate_hash", "seed_sha256",
            "binary_sha256", "baseline_environment_sha256",
            "candidate_environment_sha256", "fixture_sha256",
            "candidate_arm", "candidate_required_paths", "attempts",
        },
        "W9 E2M1 campaign",
    )
    if campaign["record_type"] != "w9_e2m1_fidelity_raw":
        raise ValueError("W9 E2M1 record type is invalid")
    engine = campaign["engine_candidate_hash"]
    if (
        not isinstance(engine, str) or len(engine) != 40
        or any(char not in "0123456789abcdef" for char in engine)
    ):
        raise ValueError("W9 E2M1 engine candidate is invalid")
    for field in (
        "seed_sha256", "binary_sha256", "baseline_environment_sha256",
        "candidate_environment_sha256", "fixture_sha256",
    ):
        if not _is_sha256(campaign[field]):
            raise ValueError(f"W9 E2M1 {field} is invalid")
    if campaign["baseline_environment_sha256"] == campaign["candidate_environment_sha256"]:
        raise ValueError("W9 E2M1 arms have identical environments")
    seed = campaign["seed_sha256"]
    candidate_arm = "A" if int(seed[:2], 16) % 2 == 0 else "B"
    if campaign["candidate_arm"] != candidate_arm:
        raise ValueError("W9 E2M1 candidate arm does not match the seed")
    schedule = "ABBA" if int(seed[2:4], 16) % 2 == 0 else "BAAB"
    required_paths = campaign["candidate_required_paths"]
    if (
        not isinstance(required_paths, list) or not required_paths
        or len(required_paths) != len(set(required_paths))
        or any(path not in {"normal", "fused"} for path in required_paths)
    ):
        raise ValueError("W9 E2M1 required paths are invalid")
    attempts = campaign["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 4:
        raise ValueError("W9 E2M1 campaign requires four attempts")
    attempt_keys = {
        "sequence", "arm", "process_identity", "binary_sha256",
        "environment_sha256", "fixture_sha256", "command_log",
        "safe_run_completed", "minimum_available_memory_gib", "swap_bytes",
        "oom", "xid", "cases",
    }
    case_keys = {"case_id", "tokens", "nll_sum", "top1_correct"}
    identities: set[str] = set()
    by_arm: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    minimum_memory = math.inf
    for sequence, attempt in enumerate(attempts):
        _require_exact_keys(attempt, attempt_keys, f"W9 E2M1 attempt {sequence}")
        arm = schedule[sequence]
        if attempt["sequence"] != sequence or attempt["arm"] != arm:
            raise ValueError("W9 E2M1 attempts are missing or reordered")
        identity = attempt["process_identity"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("W9 E2M1 process identity is invalid or reused")
        identities.add(identity)
        if attempt["binary_sha256"] != campaign["binary_sha256"]:
            raise ValueError("W9 E2M1 executed binary differs")
        expected_environment = campaign[
            "candidate_environment_sha256" if arm == candidate_arm
            else "baseline_environment_sha256"
        ]
        if attempt["environment_sha256"] != expected_environment:
            raise ValueError("W9 E2M1 executed environment differs")
        if attempt["fixture_sha256"] != campaign["fixture_sha256"]:
            raise ValueError("W9 E2M1 fixture differs")
        if attempt["safe_run_completed"] is not True:
            raise ValueError("W9 E2M1 safe run did not complete")
        memory = _finite_number(
            attempt["minimum_available_memory_gib"],
            "minimum_available_memory_gib",
        )
        if memory < 10.0:
            raise ValueError("W9 E2M1 memory floor was violated")
        minimum_memory = min(minimum_memory, memory)
        if (
            not isinstance(attempt["swap_bytes"], int)
            or isinstance(attempt["swap_bytes"], bool)
            or attempt["swap_bytes"] != 0
            or attempt["oom"] is not False or attempt["xid"] is not False
        ):
            raise ValueError("W9 E2M1 resource safety failed")
        command_log = attempt["command_log"]
        if not isinstance(command_log, str) or len(command_log.encode()) > 16 * 1024 * 1024:
            raise ValueError("W9 E2M1 command log is invalid")
        starts = re.findall(
            r"^ds4: GLM compact cache E2M1 fidelity seam=(on|off) physical=f32$",
            command_log, re.MULTILINE,
        )
        exits = re.findall(
            r"^ds4: GLM compact cache E2M1 fidelity attestation "
            r"mode=(on|off) synchronized=([01]) normal_rows=(\d+) fused_rows=(\d+)$",
            command_log, re.MULTILINE,
        )
        if len(starts) != 1 or len(exits) != 1:
            raise ValueError("W9 E2M1 attestation is missing or duplicated")
        exit_mode, synchronized, normal_text, fused_text = exits[0]
        if starts[0] != exit_mode or synchronized != "1":
            raise ValueError("W9 E2M1 mode changed or was not synchronized")
        counts = {"normal": int(normal_text), "fused": int(fused_text)}
        is_candidate = arm == candidate_arm
        if is_candidate:
            if exit_mode != "on" or any(counts[path] <= 0 for path in required_paths):
                raise ValueError("W9 E2M1 candidate device effect was not executed")
        elif exit_mode != "off" or any(counts.values()):
            raise ValueError("W9 E2M1 baseline is not default-off")
        cases = attempt["cases"]
        if not isinstance(cases, list) or len(cases) != 100:
            raise ValueError("W9 E2M1 attempt requires 100 cases")
        seen: set[str] = set()
        for case_index, case in enumerate(cases):
            _require_exact_keys(case, case_keys, f"W9 E2M1 case {case_index}")
            case_id, tokens, correct = (
                case["case_id"], case["tokens"], case["top1_correct"]
            )
            if (
                not isinstance(case_id, str) or not case_id or case_id in seen
                or not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0
                or not isinstance(correct, int) or isinstance(correct, bool)
                or not 0 <= correct <= tokens
            ):
                raise ValueError("W9 E2M1 case identity/count is invalid")
            _finite_number(case["nll_sum"], "nll_sum")
            seen.add(case_id)
        by_arm[arm].append({"counts": counts, "cases": cases})
    for arm, repeats in by_arm.items():
        if len(repeats) != 2 or repeats[0] != repeats[1]:
            raise ValueError(f"W9 E2M1 arm {arm} is not deterministic")
    baseline_arm = "B" if candidate_arm == "A" else "A"
    baseline_cases = by_arm[baseline_arm][0]["cases"]
    candidate_cases = by_arm[candidate_arm][0]["cases"]
    paired: list[dict[str, Any]] = []
    changed = False
    for baseline, candidate in zip(baseline_cases, candidate_cases):
        if baseline["case_id"] != candidate["case_id"] or baseline["tokens"] != candidate["tokens"]:
            raise ValueError("W9 E2M1 arms use unequal fixtures")
        changed |= baseline["nll_sum"] != candidate["nll_sum"] or baseline["top1_correct"] != candidate["top1_correct"]
        paired.append({
            "tokens": baseline["tokens"],
            "baseline_nll_sum": baseline["nll_sum"],
            "candidate_nll_sum": candidate["nll_sum"],
            "baseline_top1_correct": baseline["top1_correct"],
            "candidate_top1_correct": candidate["top1_correct"],
        })
    if not changed:
        raise ValueError("W9 E2M1 candidate arm is observationally inactive")
    quality = quality_verdict(paired)
    checks = {
        **quality["checks"],
        "candidate_device_effect_attested": True,
        "baseline_default_off": True,
        "fresh_counterbalanced_processes": len(identities) == 4,
        "repeat_determinism": True,
        "resource_safety": minimum_memory >= 10.0,
    }
    return {
        "scorer_id": "w9.e2m1-fidelity.v1",
        "formula_version": 1,
        "engine_candidate_hash": engine,
        "paired_case_count": len(paired),
        "attempt_count": len(attempts),
        "minimum_available_memory_gib": minimum_memory,
        "metrics": quality["metrics"],
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _score_w1_affine(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Authorize W1 affine fidelity only from a strict counterbalanced campaign."""
    if len(records) != 1:
        raise ValueError("W1 affine scorer requires one campaign record")
    campaign = records[0]
    _require_exact_keys(
        campaign,
        {
            "record_type",
            "engine_candidate_hash",
            "seed_sha256",
            "binary_sha256",
            "configuration_sha256",
            "fixture_sha256",
            "baseline_environment_sha256",
            "candidate_environment_sha256",
            "candidate_arm",
            "attempts",
        },
        "W1 affine campaign",
    )
    if campaign["record_type"] != "w1_affine_campaign":
        raise ValueError("W1 affine campaign record_type is invalid")
    engine_candidate = campaign["engine_candidate_hash"]
    if not (
        isinstance(engine_candidate, str)
        and len(engine_candidate) == 40
        and all(character in "0123456789abcdef" for character in engine_candidate)
    ):
        raise ValueError("W1 affine engine candidate hash is invalid")
    for field in (
        "seed_sha256",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256",
        "baseline_environment_sha256",
        "candidate_environment_sha256",
    ):
        if not _is_sha256(campaign[field]):
            raise ValueError(f"W1 affine {field} is invalid")
    if (
        campaign["baseline_environment_sha256"]
        == campaign["candidate_environment_sha256"]
    ):
        raise ValueError("W1 affine arms have identical environments")

    seed = campaign["seed_sha256"]
    expected_candidate_arm = "A" if int(seed[:2], 16) % 2 == 0 else "B"
    if campaign["candidate_arm"] != expected_candidate_arm:
        raise ValueError("W1 affine arm mapping does not match the public seed")
    first_schedule = "ABBA" if int(seed[2:4], 16) % 2 == 0 else "BAAB"
    other_schedule = "BAAB" if first_schedule == "ABBA" else "ABBA"
    expected_schedules = [
        first_schedule if block % 2 == 0 else other_schedule
        for block in range(5)
    ]

    attempts = campaign["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 20:
        raise ValueError("W1 affine campaign requires exactly 20 attempts")
    expected_attempt_keys = {
        "block",
        "sequence",
        "arm",
        "server_instance_id",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256_before",
        "fixture_sha256_after",
        "environment_sha256",
        "resolved_mode",
        "affine_store_count",
        "completed",
        "available_memory_gib",
        "swap_bytes",
        "oom",
        "xid",
        "failures",
        "cases",
    }
    expected_case_keys = {
        "case_id",
        "tokens",
        "nll_sum",
        "top1_correct",
    }
    servers: set[str] = set()
    by_block_arm: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for index, attempt in enumerate(attempts):
        _require_exact_keys(
            attempt, expected_attempt_keys, f"W1 affine attempt {index}"
        )
        expected_block, expected_sequence = divmod(index, 4)
        if (
            attempt["block"] != expected_block
            or attempt["sequence"] != expected_sequence
        ):
            raise ValueError("W1 affine attempts are missing, duplicated, or reordered")
        expected_arm = expected_schedules[expected_block][expected_sequence]
        if attempt["arm"] != expected_arm:
            raise ValueError("W1 affine schedule is not counterbalanced")
        server = attempt["server_instance_id"]
        if not isinstance(server, str) or not server or server in servers:
            raise ValueError("W1 affine attempts do not use fresh server instances")
        servers.add(server)
        for field in ("binary_sha256", "configuration_sha256"):
            if attempt[field] != campaign[field]:
                raise ValueError(f"W1 affine attempt {field} does not match campaign")
        if (
            attempt["fixture_sha256_before"] != campaign["fixture_sha256"]
            or attempt["fixture_sha256_after"] != campaign["fixture_sha256"]
        ):
            raise ValueError("W1 affine fixture bytes changed or are unbound")

        is_candidate = attempt["arm"] == campaign["candidate_arm"]
        expected_environment = campaign[
            "candidate_environment_sha256"
            if is_candidate
            else "baseline_environment_sha256"
        ]
        if attempt["environment_sha256"] != expected_environment:
            raise ValueError("W1 affine effective environment is wrong")
        mode = attempt["resolved_mode"]
        store_count = attempt["affine_store_count"]
        if (
            not isinstance(mode, int)
            or isinstance(mode, bool)
            or not isinstance(store_count, int)
            or isinstance(store_count, bool)
            or store_count < 0
        ):
            raise ValueError("W1 affine mode attestation is malformed")
        if is_candidate:
            if mode != 2 or store_count <= 0:
                raise ValueError("W1 affine candidate mode/store path was not executed")
        elif mode != 0 or store_count != 0:
            raise ValueError("W1 affine baseline is not default-off")
        if attempt["completed"] is not True:
            raise ValueError("W1 affine attempt did not complete")
        memory = _finite_number(
            attempt["available_memory_gib"], "available_memory_gib"
        )
        if memory < 10.0:
            raise ValueError("W1 affine attempt violates the memory floor")
        if (
            not isinstance(attempt["swap_bytes"], int)
            or isinstance(attempt["swap_bytes"], bool)
            or attempt["swap_bytes"] != 0
        ):
            raise ValueError("W1 affine attempt used swap")
        if attempt["oom"] is not False or attempt["xid"] is not False:
            raise ValueError("W1 affine attempt reports OOM or Xid")
        if attempt["failures"] != []:
            raise ValueError("W1 affine attempt contains failures")

        cases = attempt["cases"]
        if not isinstance(cases, list) or len(cases) != 20:
            raise ValueError("W1 affine attempt requires exactly 20 cases")
        case_ids: set[str] = set()
        for case_index, case in enumerate(cases):
            _require_exact_keys(
                case,
                expected_case_keys,
                f"W1 affine attempt {index} case {case_index}",
            )
            case_id = case["case_id"]
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise ValueError("W1 affine case IDs are invalid or duplicated")
            case_ids.add(case_id)
            tokens = case["tokens"]
            if (
                not isinstance(tokens, int)
                or isinstance(tokens, bool)
                or tokens <= 0
            ):
                raise ValueError("W1 affine case token count is invalid")
            _finite_number(case["nll_sum"], "nll_sum")
            correct = case["top1_correct"]
            if (
                not isinstance(correct, int)
                or isinstance(correct, bool)
                or not 0 <= correct <= tokens
            ):
                raise ValueError("W1 affine top-1 count is invalid")
        by_block_arm.setdefault((expected_block, expected_arm), []).append(attempt)

    quality_cases: list[dict[str, Any]] = []
    all_case_ids: set[str] = set()
    baseline_arm = "B" if campaign["candidate_arm"] == "A" else "A"
    for block in range(5):
        block_attempts: dict[str, dict[str, Any]] = {}
        for arm in ("A", "B"):
            repeats = by_block_arm.get((block, arm), [])
            if len(repeats) != 2:
                raise ValueError("W1 affine block does not contain two runs per arm")
            if (
                repeats[0]["cases"] != repeats[1]["cases"]
                or repeats[0]["affine_store_count"]
                != repeats[1]["affine_store_count"]
            ):
                raise ValueError("W1 affine repeated arm is not deterministic")
            block_attempts[arm] = repeats[0]
        baseline_cases = block_attempts[baseline_arm]["cases"]
        candidate_cases = block_attempts[campaign["candidate_arm"]]["cases"]
        for baseline_case, candidate_case in zip(
            baseline_cases, candidate_cases
        ):
            if (
                baseline_case["case_id"] != candidate_case["case_id"]
                or baseline_case["tokens"] != candidate_case["tokens"]
            ):
                raise ValueError("W1 affine arms use unequal fixtures")
            case_id = baseline_case["case_id"]
            if case_id in all_case_ids:
                raise ValueError("W1 affine cases repeat between blocks")
            all_case_ids.add(case_id)
            quality_cases.append(
                {
                    "tokens": baseline_case["tokens"],
                    "baseline_nll_sum": baseline_case["nll_sum"],
                    "candidate_nll_sum": candidate_case["nll_sum"],
                    "baseline_top1_correct": baseline_case["top1_correct"],
                    "candidate_top1_correct": candidate_case["top1_correct"],
                }
            )
    quality = quality_verdict(quality_cases)
    checks = {
        **quality["checks"],
        "counterbalanced_fresh_servers": len(servers) == 20,
        "content_complete_fixture_bound": True,
        "effective_modes_attested": True,
        "repeat_determinism": True,
        "resource_safety": True,
    }
    return {
        "scorer_id": "w1.affine-quality.v1",
        "formula_version": 1,
        "engine_candidate_hash": engine_candidate,
        "paired_case_count": len(quality_cases),
        "attempt_count": len(attempts),
        "metrics": quality["metrics"],
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _w1_single_match(pattern: str, text: str, label: str) -> tuple[str, ...]:
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"W1 {label} is missing or duplicated")
    match = matches[0]
    return (match,) if isinstance(match, str) else tuple(match)


def _w1_quality_cases(
    text: str, expected_case_ids: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.endswith("\n"):
        raise ValueError("W1 quality TSV is malformed")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required = {"id", "target_tokens", "nll", "target_top1_correct"}
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or not required.issubset(reader.fieldnames)
    ):
        raise ValueError("W1 quality TSV schema is invalid")
    rows = list(reader)
    if len(rows) != 20:
        raise ValueError("W1 quality TSV requires exactly 20 rows")
    if [row.get("id") for row in rows] != expected_case_ids:
        raise ValueError("W1 quality TSV does not match the selected fixture")
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            tokens = int(row["target_tokens"])
            nll_sum = float(row["nll"])
            top1_correct = int(row["target_top1_correct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("W1 quality TSV values are malformed") from exc
        if (
            tokens <= 0
            or not math.isfinite(nll_sum)
            or top1_correct < 0
            or top1_correct > tokens
        ):
            raise ValueError("W1 quality TSV values are invalid")
        result.append(
            {
                "case_id": row["id"],
                "tokens": tokens,
                "nll_sum": nll_sum,
                "top1_correct": top1_correct,
            }
        )
    return result


def _score_w1_affine_raw(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive W1 solely from raw process logs, telemetry and quality TSVs."""
    if len(records) != 1:
        raise ValueError("W1 raw scorer requires one campaign")
    campaign = records[0]
    real_packed = campaign.get("candidate_format") == "affine-int8-block16"
    campaign_keys = {
        "record_type",
        "harness_candidate_hash",
        "engine_candidate_hash",
        "composite_candidate_sha256",
        "seed_sha256",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256",
        "fixture_content_sha256",
        "model_content_sha256",
        "tokenizer_content_sha256",
        "engine_build_sha256",
        "engine_source_sha256",
        "build_log_sha256",
        "baseline_environment_sha256",
        "candidate_environment_sha256",
        "candidate_arm",
        "lineage",
        "fixture_blocks",
        "attempts",
    }
    if "candidate_format" in campaign:
        campaign_keys.add("candidate_format")
    _require_exact_keys(
        campaign,
        campaign_keys,
        "W1 raw campaign",
    )
    if "candidate_format" in campaign and not real_packed:
        raise ValueError("W1 raw candidate format is invalid")
    if campaign["record_type"] != "w1_affine_raw_campaign":
        raise ValueError("W1 raw campaign record type is invalid")
    for field in ("harness_candidate_hash", "engine_candidate_hash"):
        value = campaign[field]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"W1 raw {field} is invalid")
    for field in (
        "composite_candidate_sha256",
        "seed_sha256",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256",
        "fixture_content_sha256",
        "model_content_sha256",
        "tokenizer_content_sha256",
        "engine_build_sha256",
        "engine_source_sha256",
        "build_log_sha256",
        "baseline_environment_sha256",
        "candidate_environment_sha256",
    ):
        if not _is_sha256(campaign[field]):
            raise ValueError(f"W1 raw {field} is invalid")
    if (
        campaign["baseline_environment_sha256"]
        == campaign["candidate_environment_sha256"]
    ):
        raise ValueError("W1 raw arms have identical environments")
    validate_manifest_lineage(
        campaign["lineage"],
        "W1",
        campaign["harness_candidate_hash"],
    )
    lineage_seed = campaign["lineage"]["randomness"]["seed_sha256"]
    if campaign["seed_sha256"] != lineage_seed:
        raise ValueError("W1 raw seed does not match authenticated lineage")
    seed = lineage_seed
    if (
        campaign["lineage"]["freeze"].get("composite_candidate_sha256")
        != campaign["composite_candidate_sha256"]
    ):
        raise ValueError("W1 raw composite candidate differs from lineage")
    expected_candidate = "A" if int(seed[:2], 16) % 2 == 0 else "B"
    if campaign["candidate_arm"] != expected_candidate:
        raise ValueError("W1 raw candidate arm does not match the seed")
    first = "ABBA" if int(seed[2:4], 16) % 2 == 0 else "BAAB"
    other = "BAAB" if first == "ABBA" else "ABBA"
    expected_schedules = [
        first if block % 2 == 0 else other for block in range(5)
    ]

    blocks = campaign["fixture_blocks"]
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise ValueError("W1 raw fixture requires five blocks")
    block_cases: dict[int, list[str]] = {}
    all_fixture_cases: set[str] = set()
    for index, block in enumerate(blocks):
        _require_exact_keys(
            block,
            {"block", "manifest_sha256", "ordered_case_ids"},
            f"W1 raw fixture block {index}",
        )
        case_ids = block["ordered_case_ids"]
        if (
            block["block"] != index
            or not _is_sha256(block["manifest_sha256"])
            or not isinstance(case_ids, list)
            or len(case_ids) != 20
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(set(case_ids)) != 20
            or all_fixture_cases.intersection(case_ids)
        ):
            raise ValueError("W1 raw fixture block is invalid")
        all_fixture_cases.update(case_ids)
        block_cases[index] = case_ids
    if len(all_fixture_cases) != 100:
        raise ValueError("W1 raw fixture does not bind 100 unique cases")

    attempts = campaign["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 20:
        raise ValueError("W1 raw campaign requires 20 attempts")
    attempt_keys = {
        "block",
        "sequence",
        "arm",
        "fixture_content_sha256_before",
        "fixture_content_sha256_after",
        "model_identity_before",
        "model_identity_after",
        "evidence",
    }
    evidence_keys = {
        "launcher_log",
        "main_log",
        "cmd_log",
        "samples_log",
        "kernel_log",
        "quality_tsv",
        "journal_witness",
    }
    identities: set[tuple[str, str]] = set()
    model_identity: str | None = None
    derived: dict[tuple[int, str], list[dict[str, Any]]] = {}
    minimum_memory = math.inf
    for index, attempt in enumerate(attempts):
        _require_exact_keys(attempt, attempt_keys, f"W1 raw attempt {index}")
        block, sequence = divmod(index, 4)
        arm = expected_schedules[block][sequence]
        if (
            attempt["block"] != block
            or attempt["sequence"] != sequence
            or attempt["arm"] != arm
        ):
            raise ValueError("W1 raw attempts are missing or reordered")
        if (
            attempt["fixture_content_sha256_before"]
            != campaign["fixture_content_sha256"]
            or attempt["fixture_content_sha256_after"]
            != campaign["fixture_content_sha256"]
        ):
            raise ValueError("W1 raw fixture content changed")
        before_identity = attempt["model_identity_before"]
        after_identity = attempt["model_identity_after"]
        if (
            not isinstance(before_identity, str)
            or not before_identity
            or before_identity != after_identity
            or (model_identity is not None and before_identity != model_identity)
        ):
            raise ValueError("W1 raw model identity changed")
        model_identity = before_identity
        evidence = attempt["evidence"]
        _require_exact_keys(evidence, evidence_keys, f"W1 raw evidence {index}")
        if any(
            not isinstance(evidence[field], str)
            or len(evidence[field].encode()) > 16 * 1024 * 1024
            for field in evidence_keys
        ):
            raise ValueError("W1 raw evidence is invalid or oversized")

        launcher = evidence["launcher_log"]
        _w1_single_match(
            r"^SAFE_RUN_DONE rc=0 killed=no dir=(\S+)$",
            launcher,
            "launcher completion",
        )
        main = evidence["main_log"]
        if (
            "memory_swap_max=0" not in main
            or "memory_oom_group=1" not in main
        ):
            raise ValueError("W1 raw cgroup completion is invalid")
        executed_at_text = _w1_single_match(
            r"^(\S+) executed_candidate_verified pid=\d+ start_ticks=\d+",
            main,
            "executed-process timestamp",
        )[0]
        completed_at_text = _w1_single_match(
            r"^(\S+) SAFE_RUN end rc=0 killed=no\b.*$",
            main,
            "wrapper completion",
        )[0]
        executed_at = _utc_timestamp(
            executed_at_text, "W1 executed-process timestamp"
        )
        completed_at = _utc_timestamp(
            completed_at_text, "W1 wrapper completion timestamp"
        )
        if completed_at <= executed_at:
            raise ValueError("W1 raw lifecycle timing is invalid")
        binary = _w1_single_match(
            r"(?:^| )candidate_binary_sha256=([0-9a-f]{64})(?: |$)",
            main,
            "binary provenance",
        )[0]
        if binary != campaign["binary_sha256"]:
            raise ValueError("W1 raw executed binary differs")
        environment = _w1_single_match(
            r"(?:^| )executed_environment_sha256=([0-9a-f]{64})(?: |$)",
            main,
            "environment provenance",
        )[0]
        expected_environment = campaign[
            "candidate_environment_sha256"
            if arm == campaign["candidate_arm"]
            else "baseline_environment_sha256"
        ]
        if environment != expected_environment:
            raise ValueError("W1 raw executed environment differs")
        pid, start_ticks = _w1_single_match(
            r"executed_candidate_verified pid=(\d+) start_ticks=(\d+)",
            main,
            "process identity",
        )
        if (pid, start_ticks) in identities:
            raise ValueError("W1 raw attempts did not use fresh processes")
        identities.add((pid, start_ticks))
        try:
            witness = json.loads(
                evidence["journal_witness"],
                object_pairs_hook=lambda pairs: _unique_pairs(
                    pairs, "W1 journal witness"
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("W1 journal witness is malformed") from exc
        _require_exact_keys(
            witness,
            {
                "cursor",
                "realtime_timestamp",
                "boot_id",
                "invocation_id",
                "pid",
                "uid",
                "cgroup",
                "user_unit",
                "message",
            },
            "W1 journal witness",
        )
        if any(not isinstance(value, str) or not value for value in witness.values()):
            raise ValueError("W1 journal witness fields are absent")

        cmd = evidence["cmd_log"]
        mode = int(_w1_single_match(
            r"^ds4: GLM compact cache fidelity resolved_mode=(\d+)$",
            cmd,
            "resolved mode",
        )[0])
        exit_mode, store_rows, changed_values = map(
            int,
            _w1_single_match(
                r"^ds4: GLM compact cache fidelity attestation "
                r"resolved_mode=(\d+) affine_store_rows=(\d+) "
                r"affine_changed_values=(\d+)$",
                cmd,
                "device effect attestation",
            ),
        )
        if mode != exit_mode:
            raise ValueError("W1 raw mode changed during an attempt")
        is_candidate = arm == campaign["candidate_arm"]
        packed_store_rows = 0
        packed_read_values = 0
        storage_format = "f32"
        if real_packed:
            storage_format = _w1_single_match(
                r"^ds4: GLM compact cache storage format=([a-z0-9-]+)$",
                cmd,
                "storage format",
            )[0]
            exit_format, packed_rows_text, packed_reads_text = (
                _w1_single_match(
                    r"^ds4: GLM compact cache storage attestation "
                    r"format=([a-z0-9-]+) packed_store_rows=(\d+) "
                    r"packed_read_values=(\d+)$",
                    cmd,
                    "packed device effect attestation",
                )
            )
            packed_store_rows = int(packed_rows_text)
            packed_read_values = int(packed_reads_text)
            if storage_format != exit_format:
                raise ValueError("W1 raw storage format changed")
            if mode != 0 or store_rows != 0 or changed_values != 0:
                raise ValueError("W1 raw fake affine mode was not off")
            if is_candidate:
                if (
                    storage_format != "affine-int8-block16"
                    or packed_store_rows <= 0
                    or packed_read_values <= 0
                ):
                    raise ValueError(
                        "W1 raw packed CUDA effect was not executed"
                    )
            elif (
                storage_format != "f32"
                or packed_store_rows != 0
                or packed_read_values != 0
            ):
                raise ValueError("W1 raw baseline is not default-off")
        elif is_candidate:
            if mode != 2 or store_rows <= 0 or changed_values <= 0:
                raise ValueError("W1 raw affine CUDA effect was not executed")
        elif mode != 0 or store_rows != 0 or changed_values != 0:
            raise ValueError("W1 raw baseline is not default-off")

        sample_rows = []
        for line in evidence["samples_log"].splitlines():
            match = re.fullmatch(
                r"(\S+) mem_avail_kb=(\d+) eng_rss_kb=(\d+) "
                r"read_bytes=(\d+)",
                line,
            )
            if match is None:
                raise ValueError("W1 raw memory telemetry row is malformed")
            timestamp, available, rss, read_bytes = match.groups()
            sample_rows.append(
                (
                    _utc_timestamp(timestamp, "W1 memory timestamp"),
                    int(available),
                    int(rss),
                    int(read_bytes),
                )
            )
        if len(sample_rows) < 20:
            raise ValueError("W1 raw memory telemetry coverage is incomplete")
        sample_times = [row[0] for row in sample_rows]
        gaps = [
            (right - left).total_seconds()
            for left, right in zip(sample_times, sample_times[1:])
        ]
        first_delay = (sample_times[0] - executed_at).total_seconds()
        trailing_delay = (completed_at - sample_times[-1]).total_seconds()
        failed_coverage = []
        if any(gap <= 0 or gap > 0.75 for gap in gaps):
            failed_coverage.append("max_gap")
        if first_delay > 1.0:
            failed_coverage.append("late_first")
        if trailing_delay > 1.0:
            failed_coverage.append("early_final")
        if failed_coverage:
            raise ValueError(
                "W1 raw memory telemetry does not cover execution: "
                f"attempt={index} failed={','.join(failed_coverage)} "
                f"sample_count={len(sample_rows)} "
                f"min_gap_s={min(gaps)!r} max_gap_s={max(gaps)!r} "
                f"first_minus_executed_s={first_delay!r} "
                f"completed_minus_last_s={trailing_delay!r}"
            )
        memory_values = [row[1] for row in sample_rows]
        attempt_memory = min(memory_values) / 1048576
        if attempt_memory < 10.0:
            raise ValueError("W1 raw memory floor was violated")
        minimum_memory = min(minimum_memory, attempt_memory)
        expected_nonce = hashlib.sha256(
            f"{seed}:{index}:W1-witness".encode()
        ).hexdigest()
        witness_values = _w1_single_match(
            r"^W1_WITNESS nonce=([0-9a-f]{64}) "
            r"unit=(glm52-w1-[A-Za-z0-9_-]+) "
            r"binary=([0-9a-f]{64}) environment=([0-9a-f]{64}) "
            r"pid=(\d+) start_ticks=(\d+) rc=(\d+) killed=(\S+) "
            r"cmd_sha256=([0-9a-f]{64}) samples_sha256=([0-9a-f]{64}) "
            r"artifact_sha256=([0-9a-f]{64}) "
            r"artifact_identity=(\d+:\d+:\d+)$",
            witness["message"],
            "journal witness message",
        )
        (
            witnessed_nonce,
            witnessed_unit,
            witnessed_binary,
            witnessed_environment,
            witnessed_pid,
            witnessed_start,
            witnessed_rc,
            witnessed_killed,
            witnessed_cmd,
            witnessed_samples,
            witnessed_artifact,
            witnessed_artifact_identity,
        ) = witness_values
        expected_unit_prefix = (
            f"glm52-w1-{seed[:8]}-{index:02d}-{arm}-"
        )
        if (
            witnessed_nonce != expected_nonce
            or not witnessed_unit.startswith(expected_unit_prefix)
            or witnessed_binary != binary
            or witnessed_environment != environment
            or witnessed_pid != pid
            or witnessed_start != start_ticks
            or witnessed_rc != "0"
            or witnessed_killed != "no"
            or witnessed_cmd
            != hashlib.sha256(evidence["cmd_log"].encode()).hexdigest()
            or witnessed_samples
            != hashlib.sha256(evidence["samples_log"].encode()).hexdigest()
            or witnessed_artifact
            != hashlib.sha256(evidence["quality_tsv"].encode()).hexdigest()
            or not re.fullmatch(r"\d+:\d+:\d+", witnessed_artifact_identity)
            or witness["uid"] != "995"
            or witness["user_unit"] != witnessed_unit + ".service"
            or not witness["cgroup"].endswith(
                f"/{witnessed_unit}.service"
            )
        ):
            raise ValueError("W1 journal witness does not bind raw execution")
        if re.search(
            r"NVRM.*Xid|NV_ERR_NO_MEMORY|oom-kill|Out of memory|"
            r"Memory cgroup out of memory",
            "\n".join((evidence["kernel_log"], main, cmd)),
            re.I,
        ):
            raise ValueError("W1 raw kernel or memory fault was observed")
        cases = _w1_quality_cases(
            evidence["quality_tsv"], block_cases[block]
        )
        observation = {
            "cases": cases,
            "store_rows": store_rows,
            "changed_values": changed_values,
            "storage_format": storage_format,
            "packed_store_rows": packed_store_rows,
            "packed_read_values": packed_read_values,
        }
        derived.setdefault((block, arm), []).append(observation)

    quality_cases: list[dict[str, Any]] = []
    baseline_arm = "B" if campaign["candidate_arm"] == "A" else "A"
    for block in range(5):
        by_arm: dict[str, list[dict[str, Any]]] = {
            arm: derived.get((block, arm), []) for arm in ("A", "B")
        }
        for arm, repeats in by_arm.items():
            if len(repeats) != 2 or repeats[0] != repeats[1]:
                raise ValueError(
                    f"W1 raw block {block} arm {arm} is not deterministic"
                )
        baseline_cases = by_arm[baseline_arm][0]["cases"]
        candidate_cases = by_arm[campaign["candidate_arm"]][0]["cases"]
        for baseline_case, candidate_case in zip(
            baseline_cases, candidate_cases
        ):
            quality_cases.append(
                {
                    "tokens": baseline_case["tokens"],
                    "baseline_nll_sum": baseline_case["nll_sum"],
                    "candidate_nll_sum": candidate_case["nll_sum"],
                    "baseline_top1_correct": baseline_case["top1_correct"],
                    "candidate_top1_correct": candidate_case["top1_correct"],
                }
            )
    quality = quality_verdict(quality_cases)
    checks = {
        **quality["checks"],
        "authenticated_seed": True,
        "raw_process_evidence": True,
        "exact_fixture_mapping": True,
        "device_effect_attested": True,
        "fresh_counterbalanced_processes": len(identities) == 20,
        "model_identity_stable": model_identity is not None,
        "resource_safety": minimum_memory >= 10.0,
    }
    return {
        "scorer_id": "w1.affine-quality.v2",
        "formula_version": 2,
        "engine_candidate_hash": campaign["engine_candidate_hash"],
        "harness_candidate_hash": campaign["harness_candidate_hash"],
        "paired_case_count": len(quality_cases),
        "attempt_count": len(attempts),
        "minimum_available_memory_gib": minimum_memory,
        "metrics": quality["metrics"],
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_raw_record(record: dict[str, Any]) -> None:
    """Reject malformed, failed, short, or unidentifiable measurement arms."""
    if record.get("arm") not in {"A", "B"}:
        raise ValueError("arm must be A or B")
    for field in ("fixture_sha256", "binary_sha256"):
        if not _is_sha256(record.get(field)):
            raise ValueError(f"{field} is not a lowercase SHA-256")
    decode_tokens_per_second(record.get("token_timestamps", ()))
    evaluated = record.get("evaluated_tokens")
    if not isinstance(evaluated, int) or isinstance(evaluated, bool) or evaluated <= 0:
        raise ValueError("evaluated_tokens must be a positive integer")
    _finite_number(record.get("prefill_seconds"), "prefill_seconds")
    if record.get("failures") != []:
        raise ValueError("measurement record contains failures")


def validate_ab_blocks(
    records: Iterable[dict[str, Any]], *, flip: bool = False
) -> None:
    """Require five fresh-server ABBA/BAAB blocks with equal fixtures."""
    if not isinstance(flip, bool):
        raise ValueError("AB block orientation must be boolean")
    rows = list(records)
    if len(rows) != 20:
        raise ValueError("exactly five four-arm blocks are required")
    fixtures = {row.get("fixture_sha256") for row in rows}
    if len(fixtures) != 1 or not _is_sha256(next(iter(fixtures), None)):
        raise ValueError("all arms must use one valid fixture hash")
    boot_ids: list[str] = []
    for block in range(5):
        group = sorted(
            (row for row in rows if row.get("block") == block),
            key=lambda row: row.get("sequence", -1),
        )
        if len(group) != 4 or [row.get("sequence") for row in group] != list(range(4)):
            raise ValueError(f"block {block} is incomplete or mis-sequenced")
        expected = "ABBA" if (block + int(flip)) % 2 == 0 else "BAAB"
        if "".join(str(row.get("arm", "")) for row in group) != expected:
            raise ValueError(f"block {block} does not follow {expected}")
        group_boots = [row.get("server_boot_id") for row in group]
        if (
            any(not isinstance(value, str) or not value for value in group_boots)
            or len(set(group_boots)) != 4
        ):
            raise ValueError(f"block {block} does not use four fresh servers")
        boot_ids.extend(group_boots)
        identities = {
            arm: {
                (row.get("binary_sha256"), row.get("configuration_sha256"))
                for row in group
                if row.get("arm") == arm
            }
            for arm in ("A", "B")
        }
        for arm in ("A", "B"):
            if len(identities.get(arm, set())) != 1:
                raise ValueError(f"block {block} has inconsistent {arm} identity")
            binary, config = next(iter(identities[arm]))
            if not _is_sha256(binary) or not _is_sha256(config):
                raise ValueError(f"block {block} has invalid {arm} hashes")
        if identities["A"] == identities["B"]:
            raise ValueError(f"block {block} arms are identical")
    if len(set(boot_ids)) != 20:
        raise ValueError("every arm execution must use a fresh server")
    for arm in ("A", "B"):
        global_identities = {
            (row.get("binary_sha256"), row.get("configuration_sha256"))
            for row in rows
            if row.get("arm") == arm
        }
        if len(global_identities) != 1:
            raise ValueError(f"{arm} identity changes between blocks")


def _require_exact_keys(
    record: dict[str, Any], expected: set[str], label: str
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    missing = sorted(expected - record.keys())
    extra = sorted(record.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"{label} keys differ: missing={missing!r} extra={extra!r}"
        )


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside the numeric range") from exc
    if not math.isfinite(result) or result <= minimum:
        raise ValueError(f"{label} must be finite and greater than {minimum}")
    return result


def _score_w11(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("W11 requires exactly one context observation")
    expected = {
        "record_type",
        "binary_sha256",
        "configuration_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
        "stages",
        "retrieval_results",
        "negative_control_results",
        "memory_samples",
        "failure_events",
        "oom_events",
        "xid_events",
    }
    observation = records[0]
    _require_exact_keys(observation, expected, "W11 context observation")
    if observation["record_type"] != "context_observation":
        raise ValueError("W11 record_type is invalid")
    for field in (
        "binary_sha256",
        "configuration_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
    ):
        if not _is_sha256(observation[field]):
            raise ValueError(f"W11 {field} is invalid")

    stages = observation["stages"]
    expected_caps = [1_048_576]
    if not isinstance(stages, list) or len(stages) != len(expected_caps):
        raise ValueError("W11 requires exactly one direct 1M stage")
    processed: list[int] = []
    generations_complete = True
    no_truncation = True
    for index, (stage, expected_cap) in enumerate(zip(stages, expected_caps)):
        _require_exact_keys(
            stage,
            {
                "context_cap",
                "processed_tokens",
                "started_at_seconds",
                "finished_at_seconds",
                "completed_output_tokens",
                "token_timestamps",
                "output_sha256",
                "finish_reason",
                "truncated",
            },
            f"W11 stage {index}",
        )
        for field in ("context_cap", "processed_tokens", "completed_output_tokens"):
            if (
                not isinstance(stage[field], int)
                or isinstance(stage[field], bool)
                or stage[field] < 1
            ):
                raise ValueError(f"W11 stage {index} {field} must be positive integer")
        if stage["context_cap"] != expected_cap:
            raise ValueError("W11 context stages are not graduated exactly")
        if stage["processed_tokens"] > stage["context_cap"]:
            raise ValueError("W11 processed tokens exceed the context cap")
        started = _finite_number(
            stage["started_at_seconds"],
            "W11 stage start",
            minimum=-1.0,
        )
        finished = _finite_number(
            stage["finished_at_seconds"],
            "W11 stage finish",
            minimum=-1.0,
        )
        if finished <= started:
            raise ValueError("W11 stage timing is not positive")
        if index and started <= stages[index - 1]["finished_at_seconds"]:
            raise ValueError("W11 stage timing overlaps or is out of order")
        timestamps = stage["token_timestamps"]
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != stage["completed_output_tokens"]
        ):
            raise ValueError("W11 generation timestamp coverage is incomplete")
        numeric_timestamps = [
            _finite_number(value, "W11 token timestamp", minimum=-1.0)
            for value in timestamps
        ]
        if any(
            right <= left
            for left, right in zip(numeric_timestamps, numeric_timestamps[1:])
        ):
            raise ValueError("W11 token timestamps are not strictly increasing")
        if (
            numeric_timestamps[0] < started
            or numeric_timestamps[-1] > finished
        ):
            raise ValueError("W11 token timestamps fall outside the stage")
        if not _is_sha256(stage["output_sha256"]):
            raise ValueError("W11 output digest is invalid")
        if stage["finish_reason"] != "stop":
            raise ValueError("W11 requires a non-truncated stop finish")
        if not isinstance(stage["truncated"], bool):
            raise ValueError("W11 truncated must be boolean")
        processed.append(stage["processed_tokens"])
        generations_complete &= stage["completed_output_tokens"] > 0
        no_truncation &= stage["truncated"] is False

    retrieval = observation["retrieval_results"]
    if not isinstance(retrieval, list) or len(retrieval) < 3:
        raise ValueError("W11 requires at least three retrieval positions")
    retrieval_ids: set[str] = set()
    retrieval_positions: list[int] = []
    retrieval_pass = True
    for index, result in enumerate(retrieval):
        _require_exact_keys(
            result,
            {"case_id", "position", "expected_sha256", "observed_sha256"},
            f"W11 retrieval {index}",
        )
        case_id = result["case_id"]
        position = result["position"]
        if not isinstance(case_id, str) or not case_id or case_id in retrieval_ids:
            raise ValueError("W11 retrieval case IDs must be unique strings")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position >= processed[-1]
        ):
            raise ValueError("W11 retrieval position is invalid")
        if not _is_sha256(result["expected_sha256"]) or not _is_sha256(
            result["observed_sha256"]
        ):
            raise ValueError("W11 retrieval digest is invalid")
        retrieval_ids.add(case_id)
        retrieval_positions.append(position)
        retrieval_pass &= result["expected_sha256"] == result["observed_sha256"]
    position_coverage = (
        min(retrieval_positions) <= expected_caps[-1] // 4
        and any(
            expected_caps[-1] // 4 < position < 3 * expected_caps[-1] // 4
            for position in retrieval_positions
        )
        and max(retrieval_positions) >= 3 * expected_caps[-1] // 4
    )

    negative = observation["negative_control_results"]
    if not isinstance(negative, list) or not negative:
        raise ValueError("W11 requires a negative control")
    negative_ids: set[str] = set()
    negative_pass = True
    for index, result in enumerate(negative):
        _require_exact_keys(
            result,
            {"case_id", "expected_sha256", "observed_sha256"},
            f"W11 negative control {index}",
        )
        case_id = result["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in negative_ids:
            raise ValueError("W11 negative-control IDs must be unique strings")
        if not _is_sha256(result["expected_sha256"]) or not _is_sha256(
            result["observed_sha256"]
        ):
            raise ValueError("W11 negative-control digest is invalid")
        negative_ids.add(case_id)
        negative_pass &= result["expected_sha256"] == result["observed_sha256"]

    memory_samples = observation["memory_samples"]
    if not isinstance(memory_samples, list) or len(memory_samples) < 4:
        raise ValueError("W11 memory telemetry is incomplete")
    memory_times: list[float] = []
    memory: list[float] = []
    swap_current: list[int] = []
    for index, sample in enumerate(memory_samples):
        _require_exact_keys(
            sample,
            {"timestamp_seconds", "available_gib", "swap_current_bytes"},
            f"W11 memory sample {index}",
        )
        memory_times.append(
            _finite_number(
                sample["timestamp_seconds"],
                "W11 memory timestamp",
                minimum=-1.0,
            )
        )
        memory.append(
            _finite_number(sample["available_gib"], "W11 available memory")
        )
        swap_value = sample["swap_current_bytes"]
        if (
            not isinstance(swap_value, int)
            or isinstance(swap_value, bool)
            or swap_value < 0
        ):
            raise ValueError("W11 swap current bytes must be a nonnegative integer")
        swap_current.append(swap_value)
    if any(
        right <= left for left, right in zip(memory_times, memory_times[1:])
    ):
        raise ValueError("W11 memory timestamps are not strictly increasing")
    if (
        memory_times[0] > stages[0]["started_at_seconds"]
        or memory_times[-1] < stages[-1]["finished_at_seconds"]
        or any(
            right - left > 0.5
            for left, right in zip(memory_times, memory_times[1:])
        )
    ):
        raise ValueError("W11 memory telemetry does not cover execution at 4 Hz")
    event_fields = ("failure_events", "oom_events", "xid_events")
    for field in event_fields:
        if not isinstance(observation[field], list):
            raise ValueError(f"W11 {field} must be a list")
    checks = {
        "context_cap": stages[-1]["context_cap"] == 1_048_576,
        "processed_tokens": processed[-1] >= 1_000_000,
        "direct_one_million_stage": (
            len(stages) == 1 and stages[0]["context_cap"] == 1_048_576
        ),
        "retrieval": retrieval_pass,
        "retrieval_position_coverage": position_coverage,
        "negative_control": negative_pass,
        "completed_generation": generations_complete,
        "no_truncation": no_truncation,
        "no_failures": not observation["failure_events"],
        "no_oom": not observation["oom_events"],
        "no_xid": not observation["xid_events"],
        "memory_floor": min(memory) >= 10.0,
        "zero_swap": max(swap_current) == 0,
    }
    return {
        "scorer_id": "w11.context.v1",
        "formula_version": 2,
        "measurements": {
            "stage_context_caps": expected_caps,
            "stage_processed_tokens": processed,
            "retrieval_cases": len(retrieval),
            "negative_control_cases": len(negative),
            "memory_samples": len(memory),
            "minimum_available_memory_gib": min(memory),
            "maximum_swap_current_bytes": max(swap_current),
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def generate_w11_fixture(
    candidate_hash: str, seed_sha256: str
) -> dict[str, Any]:
    """Generate the complete registered W11 confirmation fixture."""
    if not (
        isinstance(candidate_hash, str)
        and len(candidate_hash) == 40
        and all(char in "0123456789abcdef" for char in candidate_hash)
    ):
        raise ValueError("W11 fixture candidate is invalid")
    if not _is_sha256(seed_sha256):
        raise ValueError("W11 fixture seed is invalid")

    def derived(label: str) -> str:
        return hashlib.sha256(
            (
                "w11-fixture.v2:"
                f"{candidate_hash}:{seed_sha256}:{label}"
            ).encode()
        ).hexdigest()

    position_ranges = (
        (16_384, 104_857),
        (419_430, 629_145),
        (943_718, 1_048_575),
    )
    retrieval_cases = []
    for index, (lower, upper) in enumerate(position_ranges):
        position = lower + int(derived(f"position:{index}")[:16], 16) % (
            upper - lower + 1
        )
        marker = derived(f"retrieval-marker:{index}")
        retrieval_cases.append(
            {
                "case_id": f"needle-{index}",
                "position": position,
                "expected_sha256": hashlib.sha256(marker.encode()).hexdigest(),
            }
        )
    absent_marker = "ABSENT-" + derived("negative-marker:0")
    return {
        "schema_version": 1,
        "candidate_hash": candidate_hash,
        "seed_sha256": seed_sha256,
        "generator_version": "w11-fixture.v2",
        "context_cap": 1_048_576,
        "stage_context_caps": [1_048_576],
        "retrieval_cases": retrieval_cases,
        "negative_control_cases": [
            {
                "case_id": "absent-0",
                "expected_sha256": hashlib.sha256(
                    absent_marker.encode()
                ).hexdigest(),
            }
        ],
    }


def _verify_w1_journal_authority(record: dict[str, Any]) -> None:
    """Require each embedded witness to exist with journal-owned metadata."""
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 20:
        raise ValueError("W1 journal authority requires twenty attempts")
    for index, attempt in enumerate(attempts):
        try:
            witness = json.loads(
                attempt["evidence"]["journal_witness"],
                object_pairs_hook=lambda pairs: _unique_pairs(
                    pairs, f"W1 journal witness {index}"
                ),
            )
            cursor = witness["cursor"]
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("W1 journal witness is malformed") from exc
        completed = subprocess.run(
            [
                "/usr/bin/journalctl",
                "--no-pager",
                "-o",
                "json",
                "--cursor",
                cursor,
                "-n",
                "1",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env={
                "HOME": "/nonexistent",
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
            },
        )
        rows = []
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        if len(rows) != 1:
            raise ValueError("W1 journal witness is not externally persisted")
        row = rows[0]
        expected = {
            "__CURSOR": witness.get("cursor"),
            "__REALTIME_TIMESTAMP": witness.get("realtime_timestamp"),
            "_BOOT_ID": witness.get("boot_id"),
            "_SYSTEMD_INVOCATION_ID": witness.get("invocation_id"),
            "_PID": witness.get("pid"),
            "_UID": witness.get("uid"),
            "_SYSTEMD_CGROUP": witness.get("cgroup"),
            "_SYSTEMD_UNIT": witness.get("user_unit"),
            "MESSAGE": witness.get("message"),
        }
        if (
            any(str(row.get(field, "")) != value for field, value in expected.items())
            or witness.get("uid") != "995"
            or not str(witness.get("cgroup", "")).endswith(
                "/" + str(witness.get("user_unit", ""))
            )
        ):
            raise ValueError("W1 journal witness trusted metadata differs")


def _w1_attempt_tree_manifest(attempt: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(attempt.rglob("*")):
        relative = path.relative_to(attempt).as_posix()
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("W1 controller attempt contains a symlink")
        if path.is_dir():
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError("W1 controller attempt contains an unsafe file")
        total += details.st_size
        if len(rows) >= 1024 or total > 2 * 1024 * 1024 * 1024:
            raise ValueError("W1 controller attempt exceeds authority limits")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": details.st_size,
            }
        )
    return rows


def validate_w1_root_receipt(attempt: Path, record: dict[str, Any]) -> None:
    """Bind W1 acceptance to immutable evidence produced by the root authority."""
    composite = record.get("composite_candidate_sha256")
    if not _is_sha256(composite):
        raise ValueError("W1 root receipt composite is invalid")
    try:
        authority_attempt_root = W1_AUTHORITY_ATTEMPT_ROOT.resolve(strict=True)
        attempt_parent = attempt.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("W1 root attempt authority is absent") from exc
    if attempt_parent != authority_attempt_root:
        raise ValueError("W1 attempt is outside the root authority")
    authority_root = W1_AUTHORITY_RECEIPT_ROOT.parent
    receipt_path = W1_AUTHORITY_RECEIPT_ROOT / composite
    for path in (
        authority_root,
        W1_AUTHORITY_ATTEMPT_ROOT,
        attempt,
        W1_AUTHORITY_RECEIPT_ROOT,
        receipt_path,
    ):
        try:
            details = path.lstat()
        except OSError as exc:
            raise ValueError("W1 root receipt is absent") from exc
        if (
            details.st_uid != 0
            or details.st_mode & 0o022
            or stat.S_ISLNK(details.st_mode)
        ):
            raise ValueError("W1 root receipt ownership or mode is unsafe")
    for path in attempt.rglob("*"):
        details = path.lstat()
        if details.st_uid != 0 or details.st_mode & 0o022:
            raise ValueError("W1 attempt evidence ownership or mode is unsafe")
    receipt = _read_strict_json(receipt_path)
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "terminal_state",
            "service_returncode",
            "campaign_returncode",
            "request_id",
            "harness_commit",
            "engine_commit",
            "model_sha256",
            "composite_candidate_sha256",
            "controller_attempt_manifest_sha256",
            "controller_attempt_files",
        },
        "W1 root receipt",
    )
    rows = _w1_attempt_tree_manifest(attempt)
    rows_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        receipt["schema_version"] != 2
        or receipt["terminal_state"] != "PASS"
        or receipt["service_returncode"] != 0
        or receipt["harness_commit"] != record.get("harness_candidate_hash")
        or receipt["engine_commit"] != record.get("engine_candidate_hash")
        or receipt["model_sha256"] != record.get("model_content_sha256")
        or receipt["composite_candidate_sha256"] != composite
        or receipt["controller_attempt_files"] != rows
        or receipt["controller_attempt_manifest_sha256"] != rows_sha256
    ):
        raise ValueError("W1 root receipt does not bind the controller attempt")


def validate_record_artifact_bindings(
    gate: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    artifact_paths: dict[str, Path] | None = None,
) -> None:
    """Require raw candidate identities to equal the hashed manifest artifacts."""
    workstream_gates = {f"W{index}" for index in range(1, 11)} | {"switch"}
    if gate in workstream_gates:
        if len(records) != 1:
            raise ValueError(f"{gate} requires one raw workstream record")
        record = records[0]
        for field in (
            "binary_sha256",
            "configuration_sha256",
            "fixture_sha256",
        ):
            if record.get(field) != manifest.get(field):
                label = field.removesuffix("_sha256").replace("_", " ")
                raise ValueError(
                    f"{gate} raw {label} identity does not match manifest"
                )
        if gate == "W1":
            if artifact_paths is None:
                raise ValueError("W1 artifact bindings are unavailable")
            if record.get("harness_candidate_hash") != manifest.get(
                "candidate_hash"
            ):
                raise ValueError("W1 raw harness candidate does not match manifest")
            if record.get("engine_build_sha256") != manifest.get("diff_sha256"):
                raise ValueError("W1 raw engine build does not match manifest")
            if record.get("engine_source_sha256") != manifest.get(
                "engine_source_sha256"
            ):
                raise ValueError("W1 raw engine source does not match manifest")
            if record.get("build_log_sha256") != manifest.get(
                "build_log_sha256"
            ):
                raise ValueError("W1 raw build log does not match manifest")

            evidence_path = artifact_paths.get("evidence")
            if evidence_path is None or _read_strict_json(evidence_path) != record:
                raise ValueError("W1 evidence artifact does not equal raw evidence")
            _verify_w1_journal_authority(record)

            model = _read_strict_json(artifact_paths["model"])
            _require_exact_keys(
                model,
                {"schema_version", "content_sha256", "identity"},
                "W1 model descriptor",
            )
            if (
                model["schema_version"] != 1
                or model["content_sha256"] != record.get("model_content_sha256")
                or not isinstance(model["identity"], str)
                or not model["identity"]
                or any(
                    attempt.get("model_identity_before") != model["identity"]
                    or attempt.get("model_identity_after") != model["identity"]
                    for attempt in record.get("attempts", [])
                    if isinstance(attempt, dict)
                )
            ):
                raise ValueError("W1 model descriptor does not bind raw evidence")

            tokenizer = _read_strict_json(artifact_paths["tokenizer"])
            _require_exact_keys(
                tokenizer,
                {"schema_version", "lineage", "content_sha256"},
                "W1 tokenizer descriptor",
            )
            if (
                tokenizer["schema_version"] != 1
                or tokenizer["lineage"] != "embedded-in-model-container"
                or tokenizer["content_sha256"]
                != record.get("tokenizer_content_sha256")
            ):
                raise ValueError("W1 tokenizer descriptor does not bind raw evidence")

            fixture = _read_strict_json(artifact_paths["fixture"])
            _require_exact_keys(
                fixture,
                {"schema_version", "content_sha256", "blocks"},
                "W1 fixture descriptor",
            )
            projected_blocks = [
                {
                    "block": block.get("block"),
                    "manifest_sha256": block.get("manifest_sha256"),
                    "ordered_case_ids": block.get("ordered_case_ids"),
                }
                for block in fixture.get("blocks", [])
                if isinstance(block, dict)
            ]
            if (
                fixture["schema_version"] != 1
                or fixture["content_sha256"]
                != record.get("fixture_content_sha256")
                or projected_blocks != record.get("fixture_blocks")
            ):
                raise ValueError("W1 fixture descriptor does not bind raw evidence")

            engine_build = _read_strict_json(artifact_paths["diff"])
            if (
                not isinstance(engine_build, dict)
                or engine_build.get("schema_version") != 1
                or engine_build.get("commit")
                != record.get("engine_candidate_hash")
                or engine_build.get("quality_binary_sha256")
                != record.get("binary_sha256")
                or engine_build.get("status_porcelain") != ""
                or engine_build.get("cuda_test_passed") is not True
                or engine_build.get("clean_build_transcript_sha256")
                != record.get("build_log_sha256")
                or "gguf-tools/quality-testing/score_official.o"
                not in engine_build.get("object_sha256", {})
            ):
                raise ValueError("W1 engine build does not bind raw evidence")
            engine_bundle = artifact_paths.get("engine_source")
            if engine_bundle is None:
                raise ValueError("W1 engine source bundle is unavailable")
            bundle_heads = subprocess.run(
                ["/usr/bin/git", "bundle", "list-heads", str(engine_bundle)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                env=_git_env(),
            )
            if (
                bundle_heads.returncode != 0
                or not any(
                    line.split()[0] == record.get("engine_candidate_hash")
                    for line in bundle_heads.stdout.splitlines()
                    if line.split()
                )
            ):
                raise ValueError("W1 engine bundle lacks the frozen commit")

            configuration = _read_strict_json(artifact_paths["configuration"])
            expected_configuration = {
                "harness_candidate_hash": record.get("harness_candidate_hash"),
                "engine_candidate_hash": record.get("engine_candidate_hash"),
                "composite_candidate_sha256": record.get(
                    "composite_candidate_sha256"
                ),
                "binary_sha256": record.get("binary_sha256"),
                "model_content_sha256": record.get("model_content_sha256"),
                "tokenizer_content_sha256": record.get(
                    "tokenizer_content_sha256"
                ),
                "engine_build_sha256": record.get("engine_build_sha256"),
                "engine_source_sha256": record.get("engine_source_sha256"),
                "build_log_sha256": record.get("build_log_sha256"),
                "fixture_sha256": record.get("fixture_sha256"),
                "fixture_content_sha256": record.get("fixture_content_sha256"),
                "lineage": record.get("lineage"),
            }
            if not isinstance(configuration, dict) or any(
                configuration.get(field) != value
                for field, value in expected_configuration.items()
            ):
                raise ValueError("W1 configuration does not bind raw evidence")
        return
    if gate == "foundation":
        candidates = [
            record.get("glm_baseline")
            for record in records
            if isinstance(record, dict)
        ]
        references = [
            record.get("dsv4_baseline")
            for record in records
            if isinstance(record, dict)
        ]
    elif gate == "parity":
        candidates = [
            record
            for record in records
            if isinstance(record, dict) and record.get("profile") == "glm52"
        ]
        references = [
            record
            for record in records
            if isinstance(record, dict) and record.get("profile") == "dsv4"
        ]
    else:
        candidates = []
        references = []
    if gate in {"foundation", "parity"}:
        if not candidates or any(not isinstance(record, dict) for record in candidates):
            raise ValueError(f"{gate} candidate raw records are missing")
        for index, record in enumerate(candidates):
            for field in (
                "binary_sha256",
                "configuration_sha256",
                "fixture_sha256",
            ):
                if record.get(field) != manifest.get(field):
                    label = field.removesuffix("_sha256").replace("_", " ")
                    raise ValueError(
                        f"{gate} raw {label} identity does not match manifest "
                        f"at candidate record {index}"
                    )
        if not references or any(
            not isinstance(record, dict) for record in references
        ):
            raise ValueError(f"{gate} DeepSeek raw records are missing")
        approved_reference = _load_approved_dsv4_profile(
            manifest.get("candidate_hash")
        )
        for index, record in enumerate(references):
            for field in ("binary_sha256", "configuration_sha256"):
                if record.get(field) != approved_reference[field]:
                    label = field.removesuffix("_sha256").replace("_", " ")
                    raise ValueError(
                        f"{gate} raw {label} identity does not match "
                        f"approved DeepSeek profile at reference record {index}"
                    )
            if record.get("fixture_sha256") != manifest.get("fixture_sha256"):
                raise ValueError(
                    f"{gate} raw fixture identity does not match manifest "
                    f"at reference record {index}"
                )
        return
    if gate != "W11":
        return
    fields = (
        "binary_sha256",
        "configuration_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
    )
    for index, record in enumerate(records):
        for field in fields:
            if record.get(field) != manifest.get(field):
                label = field.removesuffix("_sha256").replace("_", " ")
                raise ValueError(
                    f"W11 raw {label} identity does not match manifest "
                    f"at record {index}"
                )
    if artifact_paths is None:
        return
    fixture_path = artifact_paths.get("fixture")
    if fixture_path is None:
        raise ValueError("W11 fixture artifact is unavailable")
    try:
        fixture = _read_strict_json(fixture_path)
    except ValueError as exc:
        raise ValueError(f"W11 fixture is invalid: {exc}") from exc
    _require_exact_keys(
        fixture,
        {
            "schema_version",
            "candidate_hash",
            "seed_sha256",
            "generator_version",
            "context_cap",
            "stage_context_caps",
            "retrieval_cases",
            "negative_control_cases",
        },
        "W11 fixture",
    )
    if fixture["schema_version"] != 1 or fixture["context_cap"] != 1_048_576:
        raise ValueError("W11 fixture schema or context cap is invalid")
    lineage = manifest.get("lineage")
    expected_seed = (
        lineage.get("randomness", {}).get("seed_sha256")
        if isinstance(lineage, dict)
        and isinstance(lineage.get("randomness"), dict)
        else None
    )
    if fixture["candidate_hash"] != manifest.get("candidate_hash"):
        raise ValueError("W11 fixture candidate does not match manifest")
    if not _is_sha256(expected_seed) or fixture["seed_sha256"] != expected_seed:
        raise ValueError("W11 fixture seed does not match manifest lineage")
    if fixture["generator_version"] != "w11-fixture.v2":
        raise ValueError("W11 fixture generator is not registered")
    expected_caps = [1_048_576]
    if fixture["stage_context_caps"] != expected_caps:
        raise ValueError("W11 fixture direct context plan is invalid")
    generated_fixture = generate_w11_fixture(
        manifest.get("candidate_hash"), expected_seed
    )
    if fixture != generated_fixture:
        raise ValueError(
            "W11 fixture is not the deterministic registered seed output"
        )
    if len(records) != 1:
        raise ValueError("W11 fixture binding requires one raw record")
    record = records[0]
    raw_caps = [
        stage.get("context_cap")
        for stage in record.get("stages", [])
        if isinstance(stage, dict)
    ]
    if raw_caps != fixture["stage_context_caps"]:
        raise ValueError("W11 raw stages do not match fixture")

    fixture_retrieval: dict[str, tuple[int, str]] = {}
    retrieval_cases = fixture["retrieval_cases"]
    if not isinstance(retrieval_cases, list) or len(retrieval_cases) < 3:
        raise ValueError("W11 fixture retrieval coverage is incomplete")
    for index, case in enumerate(retrieval_cases):
        _require_exact_keys(
            case,
            {"case_id", "position", "expected_sha256"},
            f"W11 fixture retrieval {index}",
        )
        case_id = case["case_id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in fixture_retrieval
            or not isinstance(case["position"], int)
            or isinstance(case["position"], bool)
            or not _is_sha256(case["expected_sha256"])
        ):
            raise ValueError("W11 fixture retrieval case is invalid")
        fixture_retrieval[case_id] = (
            case["position"],
            case["expected_sha256"],
        )
    raw_retrieval = {
        result.get("case_id"): (
            result.get("position"),
            result.get("expected_sha256"),
        )
        for result in record.get("retrieval_results", [])
        if isinstance(result, dict)
    }
    if raw_retrieval != fixture_retrieval:
        raise ValueError("W11 fixture retrieval expectations do not match raw")

    fixture_negative: dict[str, str] = {}
    negative_cases = fixture["negative_control_cases"]
    if not isinstance(negative_cases, list) or not negative_cases:
        raise ValueError("W11 fixture negative controls are missing")
    for index, case in enumerate(negative_cases):
        _require_exact_keys(
            case,
            {"case_id", "expected_sha256"},
            f"W11 fixture negative control {index}",
        )
        case_id = case["case_id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in fixture_negative
            or not _is_sha256(case["expected_sha256"])
        ):
            raise ValueError("W11 fixture negative control is invalid")
        fixture_negative[case_id] = case["expected_sha256"]
    raw_negative = {
        result.get("case_id"): result.get("expected_sha256")
        for result in record.get("negative_control_results", [])
        if isinstance(result, dict)
    }
    if raw_negative != fixture_negative:
        raise ValueError(
            "W11 fixture negative-control expectations do not match raw"
        )


def _score_parity(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "record_type",
        "block",
        "sequence",
        "arm",
        "profile",
        "server_boot_id",
        "fixture_sha256",
        "binary_sha256",
        "configuration_sha256",
        "token_timestamps",
        "evaluated_tokens",
        "prefill_seconds",
        "warm_ttft_seconds",
        "cold_ttft_seconds",
        "available_memory_gib",
        "truncated",
        "oom",
        "xid",
        "failures",
    }
    if len(records) != 20:
        raise ValueError("parity requires exactly 20 matched arm records")
    for index, record in enumerate(records):
        _require_exact_keys(record, expected, f"parity record {index}")
        if record["record_type"] != "matched_arm":
            raise ValueError(f"parity record {index} has invalid record_type")
        if (
            not isinstance(record["block"], int)
            or isinstance(record["block"], bool)
            or not isinstance(record["sequence"], int)
            or isinstance(record["sequence"], bool)
        ):
            raise ValueError("parity block and sequence must be exact integers")
        if record["profile"] not in {"glm52", "dsv4"}:
            raise ValueError("parity profile is invalid")
        validate_raw_record(record)
        for field in (
            "warm_ttft_seconds",
            "cold_ttft_seconds",
            "available_memory_gib",
        ):
            _finite_number(record[field], field)
        if record["available_memory_gib"] < 10.0:
            raise ValueError("parity record violates the memory floor")
        for field in ("truncated", "oom", "xid"):
            if record[field] is not False:
                raise ValueError(f"parity record has {field}=true")
    validate_ab_blocks(records)

    arm_profiles = {
        arm: {record["profile"] for record in records if record["arm"] == arm}
        for arm in ("A", "B")
    }
    if any(len(values) != 1 for values in arm_profiles.values()):
        raise ValueError("A/B profile mapping changes between runs")
    if arm_profiles["A"] == arm_profiles["B"]:
        raise ValueError("A/B profiles are identical")

    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "decode_glm",
            "decode_dsv4",
            "prefill_glm",
            "prefill_dsv4",
            "prefill_time_glm",
            "prefill_time_dsv4",
            "warm_ttft_glm",
            "warm_ttft_dsv4",
            "cold_ttft_glm",
            "cold_ttft_dsv4",
        )
    }
    for block in range(5):
        for profile, suffix in (("glm52", "glm"), ("dsv4", "dsv4")):
            group = [
                record
                for record in records
                if record["block"] == block and record["profile"] == profile
            ]
            if len(group) != 2:
                raise ValueError(
                    f"block {block} does not contain two {profile} executions"
                )
            samples[f"decode_{suffix}"].append(
                statistics.fmean(
                    decode_tokens_per_second(record["token_timestamps"])
                    for record in group
                )
            )
            samples[f"prefill_{suffix}"].append(
                statistics.fmean(
                    record["evaluated_tokens"] / record["prefill_seconds"]
                    for record in group
                )
            )
            samples[f"prefill_time_{suffix}"].append(
                statistics.fmean(record["prefill_seconds"] for record in group)
            )
            samples[f"warm_ttft_{suffix}"].append(
                statistics.fmean(
                    record["warm_ttft_seconds"] for record in group
                )
            )
            samples[f"cold_ttft_{suffix}"].append(
                statistics.fmean(
                    record["cold_ttft_seconds"] for record in group
                )
            )
    result = performance_verdict(samples)
    return {"scorer_id": "parity.performance.v1", "samples": samples, **result}


def reviewed_measurements_digest(records: Iterable[dict[str, Any]]) -> str:
    """Bind a review to the exact ordered matched-arm observations."""
    rows = list(records)
    try:
        encoded = json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reviewed measurements are not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _score_reviewed_no_go(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Authorize candidate-level NO_GO from decisive matched data and reviews."""
    arms = [
        record for record in records if record.get("record_type") == "matched_arm"
    ]
    reviews = [
        record for record in records if record.get("record_type") == "no_go_review"
    ]
    if len(arms) + len(reviews) != len(records):
        raise ValueError("reviewed NO_GO contains an unknown record type")
    parity = _score_parity(arms)
    if len(reviews) != 2:
        raise ValueError("reviewed NO_GO requires exactly two persistent reviewers")

    samples = parity["samples"]
    decisive_failures = {
        "decode": paired_ratio_bound(
            samples["decode_glm"], samples["decode_dsv4"], side="upper"
        )
        < 0.80,
        "prefill_rate": paired_ratio_bound(
            samples["prefill_glm"], samples["prefill_dsv4"], side="upper"
        )
        < 0.80,
        "prefill_time": paired_ratio_bound(
            samples["prefill_time_glm"],
            samples["prefill_time_dsv4"],
            side="lower",
        )
        > 1.25,
        "warm_ttft": paired_ratio_bound(
            samples["warm_ttft_glm"],
            samples["warm_ttft_dsv4"],
            side="lower",
        )
        > 1.20,
        "cold_ttft": paired_ratio_bound(
            samples["cold_ttft_glm"],
            samples["cold_ttft_dsv4"],
            side="lower",
        )
        > 1.20,
    }
    measurement_digest = reviewed_measurements_digest(arms)
    expected = {
        "record_type",
        "reviewer",
        "candidate_hash",
        "review_round",
        "reviewed_measurements_sha256",
        "claimed_score",
        "critical",
        "high",
        "medium",
        "low",
        "prior_issue_status",
        "verdict",
    }
    canonical = {"gap_reviewer", "adversarial_reviewer"}
    reviewers: set[str] = set()
    candidates: set[str] = set()
    rounds: set[int] = set()
    scores: dict[str, int] = {}
    narratives: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for index, review in enumerate(reviews):
        _require_exact_keys(review, expected, f"NO_GO review {index}")
        reviewer = review["reviewer"]
        if reviewer not in canonical or reviewer in reviewers:
            raise ValueError("reviewed NO_GO persistent reviewer identity is invalid")
        reviewers.add(reviewer)
        candidate = review["candidate_hash"]
        if not (
            isinstance(candidate, str)
            and len(candidate) == 40
            and all(character in "0123456789abcdef" for character in candidate)
        ):
            raise ValueError("reviewed NO_GO candidate hash is invalid")
        candidates.add(candidate)
        review_round = review["review_round"]
        if (
            not isinstance(review_round, int)
            or isinstance(review_round, bool)
            or review_round < 1
        ):
            raise ValueError("reviewed NO_GO round is invalid")
        rounds.add(review_round)
        if review["reviewed_measurements_sha256"] != measurement_digest:
            raise ValueError("reviewed measurements do not match the matched arms")
        issues = {
            severity: _review_issues(review[severity], severity)
            for severity in ("critical", "high", "medium", "low")
        }
        issue_ids = [
            issue["id"] for values in issues.values() for issue in values
        ]
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("one NO_GO review issue appears at multiple severities")
        score = review["claimed_score"]
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 100
        ):
            raise ValueError("NO_GO reviewer score must be an integer from 0 to 100")
        if review["verdict"] not in {"ACCEPT", "REJECT"}:
            raise ValueError("NO_GO reviewer narrative verdict is invalid")
        prior = review["prior_issue_status"]
        if not isinstance(prior, list):
            raise ValueError("NO_GO prior_issue_status must be a list")
        prior_ids: list[str] = []
        for entry in prior:
            _require_exact_keys(entry, {"id", "status"}, "NO_GO prior issue")
            issue_id = _review_issue_ids([entry["id"]], "NO_GO prior issue")[0]
            if entry["status"] not in {
                "OPEN",
                "VERIFIED",
                "FALSIFIED",
                "FIXED",
                "DEFERRED",
            }:
                raise ValueError("NO_GO prior issue status is invalid")
            prior_ids.append(issue_id)
        if len(set(prior_ids)) != len(prior_ids):
            raise ValueError("NO_GO prior_issue_status contains duplicate IDs")
        scores[reviewer] = score
        narratives[reviewer] = review["verdict"]
        counts[reviewer] = {
            severity: len(values) for severity, values in issues.items()
        }
    if reviewers != canonical:
        raise ValueError("reviewed NO_GO persistent reviewer pair is incomplete")
    if len(candidates) != 1:
        raise ValueError("NO_GO reviewers inspected different candidates")
    if len(rounds) != 1:
        raise ValueError("NO_GO reviewers reported different rounds")

    checks = {
        "matched_parity_failed": parity["verdict"] == "FAIL",
        "decisive_matched_failure": any(decisive_failures.values()),
        "no_critical": all(value["critical"] == 0 for value in counts.values()),
        "no_high": all(value["high"] == 0 for value in counts.values()),
    }
    return {
        "scorer_id": "parity.reviewed-no-go.v1",
        "formula_version": 1,
        "candidate_hash": next(iter(candidates)),
        "review_round": next(iter(rounds)),
        "reviewed_measurements_sha256": measurement_digest,
        "parity": parity,
        "decisive_failures": decisive_failures,
        "reviewer_scores": scores,
        "reviewer_verdicts": narratives,
        "reviewer_issue_counts": counts,
        "checks": checks,
        "decision": "NO_GO",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _score_foundation_baseline(
    baseline: dict[str, Any], expected_profile: str
) -> dict[str, float]:
    expected = {
        "profile",
        "server_instance_id",
        "fixture_sha256",
        "binary_sha256",
        "configuration_sha256",
        "token_timestamps",
        "evaluated_tokens",
        "prefill_seconds",
        "warm_ttft_seconds",
        "cold_ttft_seconds",
        "available_memory_gib",
        "truncated",
        "oom",
        "xid",
        "failures",
    }
    _require_exact_keys(baseline, expected, f"{expected_profile} baseline")
    if baseline["profile"] != expected_profile:
        raise ValueError(f"{expected_profile} baseline profile is wrong")
    if (
        not isinstance(baseline["server_instance_id"], str)
        or not baseline["server_instance_id"]
    ):
        raise ValueError("foundation server_instance_id is invalid")
    for field in (
        "fixture_sha256",
        "binary_sha256",
        "configuration_sha256",
    ):
        if not _is_sha256(baseline[field]):
            raise ValueError(f"foundation baseline {field} is invalid")
    decode = decode_tokens_per_second(baseline["token_timestamps"])
    evaluated = baseline["evaluated_tokens"]
    if (
        not isinstance(evaluated, int)
        or isinstance(evaluated, bool)
        or evaluated <= 0
    ):
        raise ValueError("foundation evaluated_tokens must be a positive integer")
    prefill = _finite_number(baseline["prefill_seconds"], "prefill_seconds")
    warm = _finite_number(
        baseline["warm_ttft_seconds"], "warm_ttft_seconds"
    )
    cold = _finite_number(
        baseline["cold_ttft_seconds"], "cold_ttft_seconds"
    )
    memory = _finite_number(
        baseline["available_memory_gib"], "available_memory_gib"
    )
    if memory < 10.0:
        raise ValueError("foundation baseline violates the memory floor")
    for field in ("truncated", "oom", "xid"):
        if baseline[field] is not False:
            raise ValueError(f"foundation baseline has {field}=true")
    if baseline["failures"] != []:
        raise ValueError("foundation baseline contains failures")
    return {
        "decode_tok_s": decode,
        "prefill_tok_s": evaluated / prefill,
        "prefill_seconds": prefill,
        "warm_ttft_seconds": warm,
        "cold_ttft_seconds": cold,
        "available_memory_gib": memory,
    }


def _score_foundation(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("foundation requires exactly one observation")
    expected = {
        "record_type",
        "upstream_commit",
        "source_clean",
        "clean_build",
        "model_artifacts_verified",
        "tokenizer_artifacts_verified",
        "bandwidth_gb_s",
        "glm_baseline",
        "dsv4_baseline",
    }
    record = records[0]
    _require_exact_keys(record, expected, "foundation observation")
    if record["record_type"] != "foundation_observation":
        raise ValueError("foundation record_type is invalid")
    commit = record["upstream_commit"]
    if not (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("foundation upstream_commit is invalid")
    for field in (
        "source_clean",
        "clean_build",
        "model_artifacts_verified",
        "tokenizer_artifacts_verified",
    ):
        if record[field] is not True and record[field] is not False:
            raise ValueError(f"foundation {field} must be boolean")
    bandwidth = record["bandwidth_gb_s"]
    if not isinstance(bandwidth, list) or len(bandwidth) != 5:
        raise ValueError("foundation requires exactly five bandwidth samples")
    bandwidth_values = [
        _finite_number(value, "bandwidth_gb_s") for value in bandwidth
    ]
    glm = record["glm_baseline"]
    dsv4 = record["dsv4_baseline"]
    if not isinstance(glm, dict) or not isinstance(dsv4, dict):
        raise ValueError("foundation baselines must be objects")
    glm_metrics = _score_foundation_baseline(glm, "glm52")
    dsv4_metrics = _score_foundation_baseline(dsv4, "dsv4")
    if glm["fixture_sha256"] != dsv4["fixture_sha256"]:
        raise ValueError("foundation baselines use unequal fixtures")
    if glm["server_instance_id"] == dsv4["server_instance_id"]:
        raise ValueError("foundation baselines reuse one server instance")
    glm_identity = (glm["binary_sha256"], glm["configuration_sha256"])
    dsv4_identity = (dsv4["binary_sha256"], dsv4["configuration_sha256"])
    if glm_identity == dsv4_identity:
        raise ValueError("foundation baseline identities are identical")
    checks = {
        field: record[field] is True
        for field in (
            "source_clean",
            "clean_build",
            "model_artifacts_verified",
            "tokenizer_artifacts_verified",
        )
    }
    return {
        "scorer_id": "foundation.v1",
        "formula_version": 1,
        "measurements": {
            "bandwidth_gb_s": bandwidth_values,
            "bandwidth_mean_gb_s": statistics.fmean(bandwidth_values),
            "bandwidth_min_gb_s": min(bandwidth_values),
            "glm_baseline": glm_metrics,
            "dsv4_baseline": dsv4_metrics,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _review_issue_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if any(
        not isinstance(issue, str)
        or not issue
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in issue
        )
        for issue in value
    ):
        raise ValueError(f"{label} contains an invalid issue ID")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate issue IDs")
    return value


def _review_issues(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    expected = {
        "id",
        "evidence",
        "affected_gate",
        "reproduction_instructions",
        "proposed_acceptance_test",
    }
    issues: list[dict[str, str]] = []
    ids: list[str] = []
    for index, issue in enumerate(value):
        _require_exact_keys(issue, expected, f"{label} issue {index}")
        issue_id = _review_issue_ids([issue["id"]], label)[0]
        for field in expected - {"id"}:
            if not isinstance(issue[field], str) or not issue[field].strip():
                raise ValueError(f"{label} issue {field} must be non-empty")
        ids.append(issue_id)
        issues.append(issue)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate issue IDs")
    return issues


def _score_review(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 2:
        raise ValueError("review requires exactly two persistent reviewers")
    expected = {
        "record_type",
        "reviewer",
        "candidate_hash",
        "review_round",
        "claimed_score",
        "critical",
        "high",
        "medium",
        "low",
        "prior_issue_status",
        "verdict",
    }
    canonical = {"gap_reviewer", "adversarial_reviewer"}
    reviewers: set[str] = set()
    candidate_hashes: set[str] = set()
    rounds: set[int] = set()
    scores: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        _require_exact_keys(record, expected, f"review record {index}")
        if record["record_type"] != "review":
            raise ValueError("review record_type is invalid")
        reviewer = record["reviewer"]
        if reviewer not in canonical:
            raise ValueError("reviewer is not one of the persistent canonical pair")
        if reviewer in reviewers:
            raise ValueError("persistent reviewer is duplicated")
        reviewers.add(reviewer)
        candidate = record["candidate_hash"]
        if not (
            isinstance(candidate, str)
            and len(candidate) == 40
            and all(character in "0123456789abcdef" for character in candidate)
        ):
            raise ValueError("review candidate_hash is invalid")
        candidate_hashes.add(candidate)
        review_round = record["review_round"]
        if (
            not isinstance(review_round, int)
            or isinstance(review_round, bool)
            or review_round < 1
        ):
            raise ValueError("review_round must be a positive integer")
        rounds.add(review_round)
        issues = {
            severity: _review_issues(record[severity], severity)
            for severity in ("critical", "high", "medium", "low")
        }
        flattened = [
            issue["id"] for values in issues.values() for issue in values
        ]
        if len(set(flattened)) != len(flattened):
            raise ValueError("one review issue appears at multiple severities")
        claimed = record["claimed_score"]
        if (
            not isinstance(claimed, int)
            or isinstance(claimed, bool)
            or not 0 <= claimed <= 100
        ):
            raise ValueError("reviewer-assigned score must be an integer from 0 to 100")
        if record["verdict"] not in {"ACCEPT", "REJECT"}:
            raise ValueError("reviewer verdict is invalid")
        prior = record["prior_issue_status"]
        if not isinstance(prior, list):
            raise ValueError("prior_issue_status must be a list")
        prior_ids: list[str] = []
        for entry in prior:
            _require_exact_keys(entry, {"id", "status"}, "prior issue status")
            issue_id = _review_issue_ids([entry["id"]], "prior issue")[0]
            if entry["status"] not in {
                "OPEN",
                "VERIFIED",
                "FALSIFIED",
                "FIXED",
                "DEFERRED",
            }:
                raise ValueError("prior issue status is invalid")
            prior_ids.append(issue_id)
        if len(set(prior_ids)) != len(prior_ids):
            raise ValueError("prior_issue_status contains duplicate IDs")
        scores[reviewer] = claimed
        counts[reviewer] = {
            severity: len(values) for severity, values in issues.items()
        }
    if reviewers != canonical:
        raise ValueError("the persistent reviewer pair is incomplete")
    if len(candidate_hashes) != 1:
        raise ValueError("reviewers inspected different candidate hashes")
    if len(rounds) != 1:
        raise ValueError("reviewers reported different review rounds")
    checks = {
        "no_critical": all(value["critical"] == 0 for value in counts.values()),
        "no_high": all(value["high"] == 0 for value in counts.values()),
    }
    return {
        "scorer_id": "review.final.v1",
        "formula_version": 3,
        "candidate_hash": next(iter(candidate_hashes)),
        "review_round": next(iter(rounds)),
        "scores": scores,
        "issue_counts": counts,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _score_workstream(
    records: list[dict[str, Any]], gate: str
) -> dict[str, Any]:
    """Score one W1-W10 or switch observation from strict raw fields."""
    supported = {f"W{index}" for index in range(1, 11)} | {"switch"}
    if gate not in supported:
        raise ValueError(f"workstream scorer does not support {gate}")
    if len(records) != 1:
        raise ValueError(f"{gate} requires exactly one workstream observation")
    record = records[0]
    _require_exact_keys(
        record,
        {
            "record_type",
            "gate",
            "binary_sha256",
            "configuration_sha256",
            "fixture_sha256",
            "workflow",
            "metrics",
            "failures",
        },
        f"{gate} workstream observation",
    )
    if record["record_type"] != "workstream_observation" or record["gate"] != gate:
        raise ValueError(f"{gate} workstream identity is invalid")
    for field in ("binary_sha256", "configuration_sha256", "fixture_sha256"):
        if not _is_sha256(record[field]):
            raise ValueError(f"{gate} {field} is invalid")
    workflow_keys = {
        "test_committed",
        "red_confirmed",
        "implementation_default_off",
        "candidate_frozen",
        "post_freeze_randomness",
        "clean_build",
        "blinded_ab",
        "diff_scan_clean",
        "mutation_rejected",
    }
    workflow = record["workflow"]
    _require_exact_keys(workflow, workflow_keys, f"{gate} workflow")
    if any(not isinstance(workflow[key], bool) for key in workflow_keys):
        raise ValueError(f"{gate} workflow fields must be boolean")
    if not isinstance(record["failures"], list):
        raise ValueError(f"{gate} failures must be a list")
    metrics = record["metrics"]

    def exact(expected: set[str]) -> None:
        _require_exact_keys(metrics, expected, f"{gate} metrics")

    def booleans(fields: Iterable[str]) -> dict[str, bool]:
        result = {}
        for field in fields:
            if not isinstance(metrics[field], bool):
                raise ValueError(f"{gate} {field} must be boolean")
            result[field] = metrics[field]
        return result

    derived: dict[str, float] = {}
    if gate == "W1":
        fields = {
            "f16_tested",
            "block_e4m3_tested",
            "f32_rope",
            "fidelity_pass",
            "retrieval_pass",
            "available_memory_gib",
        }
        exact(fields)
        checks = booleans(fields - {"available_memory_gib"})
        memory = _finite_number(
            metrics["available_memory_gib"], "W1 available memory"
        )
        checks["memory_floor"] = memory >= 10.0
    elif gate == "W2":
        exact({"byte_identical", "baseline_hit_rate", "candidate_hit_rate"})
        checks = booleans({"byte_identical"})
        baseline = _finite_number(
            metrics["baseline_hit_rate"], "W2 baseline hit rate", minimum=-1.0
        )
        candidate = _finite_number(
            metrics["candidate_hit_rate"], "W2 candidate hit rate", minimum=-1.0
        )
        if baseline > 1.0 or candidate > 1.0:
            raise ValueError("W2 hit rates must be at most one")
        derived["hit_rate_gain_pp"] = (candidate - baseline) * 100.0
        checks["hit_rate_gain"] = derived["hit_rate_gain_pp"] >= 3.0
    elif gate == "W3":
        exact(
            {
                "byte_identical",
                "event_safe",
                "baseline_seconds",
                "candidate_seconds",
            }
        )
        checks = booleans({"byte_identical", "event_safe"})
        if (
            len(metrics["baseline_seconds"]) != 5
            or len(metrics["candidate_seconds"]) != 5
        ):
            raise ValueError("W3 requires five paired timing samples")
        derived["completed_time_ratio_upper_95"] = paired_ratio_bound(
            metrics["candidate_seconds"],
            metrics["baseline_seconds"],
            side="upper",
        )
        checks["completed_time_improvement"] = (
            derived["completed_time_ratio_upper_95"] <= 0.95
        )
    elif gate == "W4":
        exact(
            {
                "ids_identical",
                "logits_identical",
                "baseline_topk_seconds",
                "candidate_topk_seconds",
                "baseline_prefill_seconds",
                "candidate_prefill_seconds",
            }
        )
        checks = booleans({"ids_identical", "logits_identical"})
        sample_fields = (
            "baseline_topk_seconds",
            "candidate_topk_seconds",
            "baseline_prefill_seconds",
            "candidate_prefill_seconds",
        )
        if any(len(metrics[field]) != 5 for field in sample_fields):
            raise ValueError("W4 requires five paired samples per timing metric")
        derived["topk_speedup_lower_95"] = paired_ratio_bound(
            metrics["baseline_topk_seconds"],
            metrics["candidate_topk_seconds"],
            side="lower",
        )
        derived["prefill_speedup_lower_95"] = paired_ratio_bound(
            metrics["baseline_prefill_seconds"],
            metrics["candidate_prefill_seconds"],
            side="lower",
        )
        checks["topk_speedup"] = derived["topk_speedup_lower_95"] >= 2.0
        checks["prefill_speedup"] = derived["prefill_speedup_lower_95"] >= 1.05
    elif gate == "W5":
        exact(
            {
                "scores_identical",
                "ids_identical",
                "logits_identical",
                "baseline_allocation_bytes",
                "candidate_allocation_bytes",
            }
        )
        checks = booleans(
            {"scores_identical", "ids_identical", "logits_identical"}
        )
        baseline = _finite_number(
            metrics["baseline_allocation_bytes"], "W5 baseline allocation"
        )
        candidate = _finite_number(
            metrics["candidate_allocation_bytes"], "W5 candidate allocation"
        )
        derived["allocation_ratio"] = candidate / baseline
        checks["half_allocation"] = derived["allocation_ratio"] == 0.5
    elif gate == "W6":
        exact(
            {
                "outputs_identical",
                "width2_measured",
                "width4_measured",
                "selected_width",
                "width2_seconds",
                "width4_seconds",
                "baseline_load_bytes",
                "selected_load_bytes",
            }
        )
        checks = booleans(
            {"outputs_identical", "width2_measured", "width4_measured"}
        )
        width = metrics["selected_width"]
        if isinstance(width, bool) or width not in {2, 4}:
            raise ValueError("W6 selected width must be 2 or 4")
        width2 = _finite_number(metrics["width2_seconds"], "W6 width-2 time")
        width4 = _finite_number(metrics["width4_seconds"], "W6 width-4 time")
        baseline_load = _finite_number(
            metrics["baseline_load_bytes"], "W6 baseline loads"
        )
        selected_load = _finite_number(
            metrics["selected_load_bytes"], "W6 selected loads"
        )
        checks["fastest_selected"] = width == (2 if width2 <= width4 else 4)
        checks["load_reduced"] = selected_load < baseline_load
    elif gate == "W7":
        exact(
            {
                "complete_dumps_equal",
                "max_abs_logit_delta",
                "argmax_identical",
                "checkpoint_correct",
                "global_guard_preserved",
            }
        )
        checks = booleans(
            {
                "complete_dumps_equal",
                "argmax_identical",
                "checkpoint_correct",
                "global_guard_preserved",
            }
        )
        delta = _finite_number(
            metrics["max_abs_logit_delta"], "W7 max logit delta", minimum=-1.0
        )
        checks["logit_delta"] = delta < 1e-2
    elif gate == "W8":
        fields = {
            "checksums_verified",
            "corruption_failed_closed",
            "selected_rows_exact",
            "selected_block_cache",
            "context_1m_pass",
            "retrieval_pass",
            "available_memory_gib",
        }
        exact(fields)
        checks = booleans(fields - {"available_memory_gib"})
        memory = _finite_number(
            metrics["available_memory_gib"], "W8 available memory"
        )
        checks["memory_floor"] = memory >= 10.0
    elif gate == "W9":
        exact(
            {
                "real_capture",
                "capture_width",
                "query_weighted_error",
                "maximum_allowed_error",
            }
        )
        checks = booleans({"real_capture"})
        width = metrics["capture_width"]
        if not isinstance(width, int) or isinstance(width, bool):
            raise ValueError("W9 capture width must be an integer")
        error = _finite_number(
            metrics["query_weighted_error"], "W9 query-weighted error", minimum=-1.0
        )
        limit = _finite_number(
            metrics["maximum_allowed_error"], "W9 error limit", minimum=-1.0
        )
        checks["capture_width"] = width == 512
        checks["offline_falsifier"] = error <= limit
    elif gate == "W10":
        exact(
            {
                "data_frozen",
                "splits_frozen",
                "seeds_frozen",
                "storage_ratio",
                "maximum_storage_ratio",
                "runtime_ratio",
                "maximum_runtime_ratio",
                "fidelity_pass",
            }
        )
        checks = booleans(
            {"data_frozen", "splits_frozen", "seeds_frozen", "fidelity_pass"}
        )
        storage = _finite_number(metrics["storage_ratio"], "W10 storage ratio")
        storage_limit = _finite_number(
            metrics["maximum_storage_ratio"], "W10 storage limit"
        )
        runtime = _finite_number(metrics["runtime_ratio"], "W10 runtime ratio")
        runtime_limit = _finite_number(
            metrics["maximum_runtime_ratio"], "W10 runtime limit"
        )
        checks["storage"] = storage <= storage_limit
        checks["runtime"] = runtime <= runtime_limit
    else:
        fields = {
            "serialized",
            "idempotent",
            "hashes_verified",
            "environment_allowlisted",
            "identity_safe_stop",
            "authenticated_completion",
            "unauthenticated_rejected",
            "memwatch_pass",
            "semantic_output_pass",
            "rollback_pass",
            "reboot_restore_pass",
            "fault_matrix_pass",
            "transition_cycle_pass",
        }
        exact(fields)
        checks = booleans(fields)

    checks["workflow"] = all(workflow.values())
    checks["no_failures"] = not record["failures"]
    # This compact schema is useful for preserving terminal diagnostics, but
    # its workflow and fidelity fields are self-attestations rather than raw
    # observations. It must never authorize PASS. A gate-specific raw scorer
    # is required before any workstream can qualify a release.
    checks["raw_evidence_authority"] = False
    return {
        "scorer_id": "workstream.terminal.v1",
        "formula_version": 1,
        "gate": gate,
        "derived_metrics": derived,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _w7_f32_vector(encoded: Any, label: str, expected_count: int) -> tuple[float, ...]:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{label} must be non-empty base64")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        expected_bytes = expected_count * 4
        if len(compressed) > expected_bytes + 1024:
            raise ValueError(f"{label} compressed input is oversized")
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, expected_bytes + 1)
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise ValueError(f"{label} is not canonical compressed F32") from exc
    if (
        not inflater.eof or inflater.unused_data or inflater.unconsumed_tail or
        len(raw) != expected_bytes
    ):
        raise ValueError(f"{label} has the wrong F32 length")
    values = struct.unpack(f"<{expected_count}f", raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains non-finite values")
    return values


def _w7_i32_vector(encoded: Any, label: str, expected_count: int) -> tuple[int, ...]:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{label} must be non-empty base64")
    expected_bytes = expected_count * 4
    try:
        compressed = base64.b64decode(encoded, validate=True)
        if len(compressed) > expected_bytes + 1024:
            raise ValueError(f"{label} compressed input is oversized")
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, expected_bytes + 1)
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise ValueError(f"{label} is not canonical compressed I32") from exc
    if (
        not inflater.eof or inflater.unused_data or inflater.unconsumed_tail or
        len(raw) != expected_bytes
    ):
        raise ValueError(f"{label} has the wrong I32 length")
    return struct.unpack(f"<{expected_count}i", raw)


def _w7_utf8_bytes(encoded: Any, label: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{label} must be non-empty base64")
    try:
        value = base64.b64decode(encoded, validate=True)
        if not value or len(value) > 16 * 1024 * 1024:
            raise ValueError(f"{label} has invalid length")
        value.decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 base64") from exc
    return value


_W7_SAFE_WRAPPER_SHA256 = "6e4d382bc5e5818787af8c17aae7a0750ca3ab7b36471f21355789d194b2e801"
_W7_STEM_PATH = ROOT / "results/glm52-gates/harness/fixture-glm-long8.json"
_W7_STEM_FILE_SHA256 = "4b46547667dd4d84b8da83c0ccca358e725e33e8de8bbfe25701ab0d878ae469"
_W7_STEM_TEXT_SHA256 = "a2ff948826dd9a1b4c74fe599ba4b668ae4285e44b275006afc9d3c7541655cf"
_W7_POOL_PATH = ROOT / "results/glm52-gates/harness/w7-production-fixture-pool-v1.json"
_W7_POOL_SHA256 = "c71f1c9c90164baae00492befed68765fd9bee40fef3de8c3b291cc06794ecb9"
_W7_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
_W7_TOKENIZER_INIT_SHA256 = "eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4"
_W7_TOKENIZER_NATIVE_SHA256 = "fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910"
_W7_SERVER_SOURCE_SHA256 = "d48d748edb56220727875d705f8487406c0f4f5b64b4d28ec0b829eb5ce87f07"
_W7_RENDER_ORACLE_SOURCE_SHA256 = "9590b8eaa238e311ca0468e6983280b798cbb94c3d727920f5e839ac8ee20539"
_W7_RENDER_ORACLE_BINARY_SHA256 = "6bd6896581db71bdb76a9afdb59a9254b151ade22017e17f111fd3345fb5ad66"
_W7_RENDER_PREFIX = b"[gMASK]<sop><|system|>Reasoning Effort: High<|system|>You are a helpful assistant<|user|>"
_W7_RENDER_SUFFIX = b"<|assistant|><think>"
_W7_PRIMARY_SUFFIX = "\n\n[W7 primary fixed] Explain why a restored prefix must be rewound before this appended request."
_W7_CONFIRMATION_SUFFIXES = tuple(
    f"\n\n[W7 confirmation {index:02d}] " + instruction
    for index, instruction in enumerate((
        "Name the cache frontier invariant and give one counterexample.",
        "Describe the UTF-8/BPE boundary risk in exactly three sentences.",
        "Return a concise checklist for validating a resumed append.",
        "Explain why exact replay alone cannot qualify suffix divergence.",
        "State which rows must be invalidated before evaluating the suffix.",
        "Contrast a cold restart with a live rewind without using a table.",
        "Give a deterministic test for a checkpoint selected at a byte boundary.",
        "Identify the first observable symptom of stale frontier contamination.",
        "Explain why a longer stored record with wrong lineage must be ignored.",
        "State the required relationship among selected, common, live, and prompt.",
        "Describe a mutation that proves a short payload read is rejected.",
        "Explain why generated-token equality does not replace logit comparison.",
        "List the state members that make a GLM checkpoint complete.",
        "Describe how to prove evaluator ranges contain no hidden prefix work.",
        "Explain why the default-off flag must not affect non-GLM models.",
        "Summarize the fail-closed behavior for a malformed checkpoint.",
    ))
)
_W7_CASE_CONTRACT = {
    "cold-primary": ("glm", "unset", "primary-cold", "cold"),
    "strict-primary": ("glm", "unset", "primary-extension", "cold"),
    "candidate-primary": ("glm", "1", "primary-extension", "resume"),
    "exact-off": ("glm", "unset", "exact-replay", "replay"),
    "exact-on": ("glm", "1", "exact-replay", "replay"),
    "divergence-off": ("glm", "unset", "divergence", "reject"),
    "divergence-on": ("glm", "1", "divergence", "reject"),
    "shorten-off": ("glm", "unset", "shorten", "reject"),
    "shorten-on": ("glm", "1", "shorten", "reject"),
    "malformed-off": ("glm", "unset", "malformed-checkpoint", "reject"),
    "malformed-on": ("glm", "1", "malformed-checkpoint", "reject"),
    "wrong-lineage-off": ("glm", "unset", "wrong-lineage", "reject"),
    "wrong-lineage-on": ("glm", "1", "wrong-lineage", "reject"),
    "non-glm-off": ("non-glm", "unset", "primary-extension", "reject"),
    "non-glm-on": ("non-glm", "1", "primary-extension", "reject"),
    "flag-zero": ("glm", "0", "primary-extension", "cold"),
    "flag-empty": ("glm", "", "primary-extension", "cold"),
    "flag-garbage": ("glm", "garbage", "primary-extension", "cold"),
}


def _w7_caller_wire(variant: str | int) -> bytes:
    if _sha256(_W7_STEM_PATH) != _W7_STEM_FILE_SHA256:
        raise ValueError("W7 frozen stem file hash is invalid")
    document = _read_strict_json(_W7_STEM_PATH)
    if not isinstance(document, dict) or not isinstance(document.get("prompt"), str):
        raise ValueError("W7 frozen stem document is invalid")
    stem = document["prompt"].encode("utf-8")
    if hashlib.sha256(stem).hexdigest() != _W7_STEM_TEXT_SHA256:
        raise ValueError("W7 frozen stem text hash is invalid")
    if variant == "primary-fixed":
        suffix = _W7_PRIMARY_SUFFIX
    elif isinstance(variant, int) and not isinstance(variant, bool) and 0 <= variant < 16:
        suffix = _W7_CONFIRMATION_SUFFIXES[variant]
    else:
        raise ValueError("W7 frozen fixture variant is invalid")
    return stem + suffix.encode("utf-8")


def _w7_fixture_pool() -> tuple[dict[str | int, dict[str, Any]], tuple[int, ...], dict[str, int]]:
    if _sha256(_W7_POOL_PATH) != _W7_POOL_SHA256:
        raise ValueError("W7 production fixture pool hash is invalid")
    document = _read_strict_json(_W7_POOL_PATH)
    _require_exact_keys(
        document,
        {"schema", "tokenizer", "render_contract", "oracle", "live", "inventory_recipe", "variants"},
        "W7 production fixture pool",
    )
    if document["schema"] != "glm52-w7-production-fixture-pool-v2":
        raise ValueError("W7 production fixture pool schema is invalid")
    expected_tokenizer = {
        "tokenizer_sha256": _W7_TOKENIZER_SHA256,
        "runtime_init_sha256": _W7_TOKENIZER_INIT_SHA256,
        "runtime_native_sha256": _W7_TOKENIZER_NATIVE_SHA256,
        "add_special_tokens": False,
    }
    if document["tokenizer"] != expected_tokenizer:
        raise ValueError("W7 production tokenizer identity is invalid")
    if document["render_contract"] != {
        "api": "/v1/completions", "context_tokens": 8192, "model": "default",
        "reasoning_effort": "high", "thinking": True,
        "system": "You are a helpful assistant", "oracle": "frozen-ds4-server-c-parser",
    }:
        raise ValueError("W7 production render contract is invalid")
    if document["oracle"] != {
        "ds4_server_source_sha256": _W7_SERVER_SOURCE_SHA256,
        "oracle_source_sha256": _W7_RENDER_ORACLE_SOURCE_SHA256,
        "oracle_binary_sha256": _W7_RENDER_ORACLE_BINARY_SHA256,
    }:
        raise ValueError("W7 C render oracle identity is invalid")
    recipe = document["inventory_recipe"]
    if recipe != {
        "alignment_tokens": 4, "older_delta_tokens": -4,
        "wrong_lineage_delta_tokens": 1, "malformed_delta_tokens": 2,
    }:
        raise ValueError("W7 checkpoint inventory recipe is invalid")
    live = document["live"]
    _require_exact_keys(
        live,
        {
            "suffix_utf8", "caller_wire_sha256", "rendered_wire_utf8_b64",
            "rendered_wire_sha256", "token_ids_zlib_b64", "token_count",
        },
        "W7 live fixture",
    )
    if live["suffix_utf8"] != "\n\nOne two three four five six seven.":
        raise ValueError("W7 live fixture suffix is invalid")
    stem = _w7_caller_wire("primary-fixed")[:-len(_W7_PRIMARY_SUFFIX.encode("utf-8"))]
    live_caller = stem + live["suffix_utf8"].encode("utf-8")
    if live["caller_wire_sha256"] != hashlib.sha256(live_caller).hexdigest():
        raise ValueError("W7 live caller wire is invalid")
    live_wire = _w7_utf8_bytes(live["rendered_wire_utf8_b64"], "W7 live rendered wire")
    if (
        live["rendered_wire_sha256"] != hashlib.sha256(live_wire).hexdigest() or
        live_wire != _W7_RENDER_PREFIX + live_caller + _W7_RENDER_SUFFIX
    ):
        raise ValueError("W7 live C-rendered wire is invalid")
    live_tokens = _w7_i32_vector(
        live["token_ids_zlib_b64"], "W7 live fixture tokens", live["token_count"],
    )
    variants = document["variants"]
    if not isinstance(variants, list) or len(variants) != 17:
        raise ValueError("W7 production fixture variants are incomplete")
    by_variant: dict[str | int, dict[str, Any]] = {}
    for index, item in enumerate(variants):
        _require_exact_keys(
            item,
            {
                "variant", "caller_wire_sha256", "rendered_wire_utf8_b64",
                "rendered_wire_sha256", "canonical_token_ids_zlib_b64",
                "wire_token_end_offsets_zlib_b64", "prompt_tokens", "common_tokens",
                "live_tokens", "selected_tokens",
            },
            f"W7 production fixture variant {index}",
        )
        variant = item["variant"]
        expected_variant: str | int = "primary-fixed" if index == 0 else index - 1
        if variant != expected_variant or variant in by_variant:
            raise ValueError("W7 production fixture variant identity is invalid")
        caller_wire = _w7_caller_wire(variant)
        if item["caller_wire_sha256"] != hashlib.sha256(caller_wire).hexdigest():
            raise ValueError("W7 production caller wire hash is invalid")
        wire = _w7_utf8_bytes(
            item["rendered_wire_utf8_b64"], f"W7 production fixture {variant} wire",
        )
        if (
            item["rendered_wire_sha256"] != hashlib.sha256(wire).hexdigest() or
            wire != _W7_RENDER_PREFIX + caller_wire + _W7_RENDER_SUFFIX
        ):
            raise ValueError("W7 production C-rendered wire is invalid")
        canonical = _w7_i32_vector(
            item["canonical_token_ids_zlib_b64"],
            f"W7 production fixture {variant} tokens", item["prompt_tokens"],
        )
        offsets = _w7_i32_vector(
            item["wire_token_end_offsets_zlib_b64"],
            f"W7 production fixture {variant} offsets", item["prompt_tokens"],
        )
        common = 0
        while (
            common < len(canonical) and common < len(live_tokens) and
            canonical[common] == live_tokens[common]
        ):
            common += 1
        if (
            offsets[-1] != len(wire) or
            any(now < before for before, now in zip((0,) + offsets, offsets)) or
            item["common_tokens"] != common or item["live_tokens"] != len(live_tokens) or
            item["selected_tokens"] != common - (common % recipe["alignment_tokens"])
        ):
            raise ValueError("W7 production fixture geometry is invalid")
        by_variant[variant] = {
            **item, "wire": wire, "canonical": canonical, "offsets": offsets,
        }
    return by_variant, live_tokens, recipe


def _w7_expected_payload_bytes(token_count: int) -> int:
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
        raise ValueError("W7 payload token count is invalid")
    return (
        13 * 4 + token_count * 4 + 154880 * 4 + 78 * 4 * 2 +
        token_count * (78 * (512 + 64) * 4 + 21 * 128 * 4)
    )


def _w7_token_sha256(values: tuple[int, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest()


def _w7_state_manifest(value: Any, label: str, lineage: str, prompt: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(
        value,
        {
            "format", "config", "total_bytes", "content_sha256",
            "token_lineage_sha256", "logical_frontiers", "sections",
        },
        label,
    )
    if value["format"] != "ds4-glm-session-canonical-v2":
        raise ValueError(f"{label} format is invalid")
    indexer_layers = [0, 1, 2] + list(range(6, 78, 4))
    expected_config = {
        "normal_layers": 78, "vocab": 154880, "kv_lora": 512,
        "rope": 64, "indexer_dim": 128, "full_live": 0,
        "compact_live": prompt, "indexer_layers": indexer_layers,
    }
    if value["config"] != expected_config:
        raise ValueError(f"{label} production GLM state configuration is invalid")
    if value["token_lineage_sha256"] != lineage:
        raise ValueError(f"{label} token lineage is invalid")
    frontiers = value["logical_frontiers"]
    if (
        not isinstance(frontiers, list) or len(frontiers) != 78 or
        any(item != prompt for item in frontiers)
    ):
        raise ValueError(f"{label} logical frontiers are incomplete")
    sections = value["sections"]
    expected_members: list[tuple[str, str, list[int]]] = [
        ("header", "u32", [13]),
        ("checkpoint_tokens", "u32", [prompt]),
        ("logits", "f32", [154880]),
        ("compact_live_rows", "u32", [78]),
        ("index_live_rows", "u32", [78]),
    ]
    for layer in range(78):
        expected_members.extend([
            (f"layer.{layer:02d}.kv_lora", "f32", [prompt, 512]),
            (f"layer.{layer:02d}.k_rope", "f32", [prompt, 64]),
        ])
        if layer in indexer_layers:
            expected_members.append(
                (f"layer.{layer:02d}.indexer_key", "f32", [prompt, 128])
            )
    if not isinstance(sections, list) or len(sections) != len(expected_members):
        raise ValueError(f"{label} state section coverage is incomplete")
    offset = 0
    content_hashes: list[str] = []
    for index, (section, expected_member) in enumerate(zip(sections, expected_members)):
        if not isinstance(section, dict):
            raise ValueError(f"{label} section {index} must be an object")
        _require_exact_keys(
            section,
            {"name", "dtype", "shape", "offset", "byte_count", "content_sha256"},
            f"{label} section {index}",
        )
        expected_name, expected_dtype, expected_shape = expected_member
        expected_bytes = math.prod(expected_shape) * 4
        if (
            section["name"] != expected_name or section["dtype"] != expected_dtype or
            section["shape"] != expected_shape or section["offset"] != offset or
            section["byte_count"] != expected_bytes
        ):
            raise ValueError(f"{label} section coverage is not canonical")
        byte_count = section["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError(f"{label} section byte count is invalid")
        if not _is_sha256(section["content_sha256"]):
            raise ValueError(f"{label} section hash is invalid")
        offset += byte_count
        content_hashes.append(section["content_sha256"])
    if value["total_bytes"] != offset or offset <= 0:
        raise ValueError(f"{label} total byte coverage is invalid")
    if offset != _w7_expected_payload_bytes(prompt):
        raise ValueError(f"{label} total bytes do not match production payload formula")
    manifest_root = hashlib.sha256("".join(content_hashes).encode("ascii")).hexdigest()
    if value["content_sha256"] != manifest_root:
        raise ValueError(f"{label} content root is invalid")
    return value


def _w7_fixture_material(regime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
            {
                key: row[key] for key in (
                    "case_id", "model_family", "flag_value", "fixture_mutation",
                    "wire_text_utf8_b64", "wire_token_end_offsets_zlib_b64",
                    "canonical_token_ids_zlib_b64", "live_token_count",
                    "live_token_ids_zlib_b64", "prompt_tokens",
                    "checkpoint_inventory",
                )
            }
            for row in regime["cases"]
        ]


def _w7_fixture_digest(regime: dict[str, Any]) -> str:
    material = {
        "fixture_variant": regime["fixture_variant"],
        "cases": _w7_fixture_material(regime),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _score_w7_resume(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive W7 exclusively from inventories, ordered primitive events and dumps."""
    if len(records) != 1:
        raise ValueError("W7 resume requires exactly one raw observation")
    record = records[0]
    _require_exact_keys(
        record,
        {
            "record_type", "gate", "attempt_id", "execution_nonce_sha256",
            "binary_sha256", "configuration_sha256",
            "fixture_sha256", "guard_off_present", "confirmation_seed_sha256",
            "regimes", "failures",
        },
        "W7 resume observation",
    )
    if record["record_type"] != "w7_resume_observation" or record["gate"] != "W7":
        raise ValueError("W7 resume observation identity is invalid")
    for field in (
        "attempt_id", "execution_nonce_sha256", "binary_sha256",
        "configuration_sha256", "fixture_sha256",
        "confirmation_seed_sha256",
    ):
        if not _is_sha256(record[field]):
            raise ValueError(f"W7 {field} is invalid")
    if record["guard_off_present"] is not False:
        raise ValueError("W7 legacy guard bypass must be absent")
    if not isinstance(record["failures"], list):
        raise ValueError("W7 failures must be a list")
    regimes = record["regimes"]
    if not isinstance(regimes, list) or len(regimes) != 3:
        raise ValueError("W7 requires one primary and two confirmation regimes")
    regime_ids = [item.get("regime_id") if isinstance(item, dict) else None for item in regimes]
    if regime_ids != ["primary", "confirmation-1", "confirmation-2"]:
        raise ValueError("W7 regime identities are invalid")
    fixture_pool, frozen_live_tokens, inventory_recipe = _w7_fixture_pool()
    expected_attempt_id = hashlib.sha256(
        bytes.fromhex(record["binary_sha256"]) +
        bytes.fromhex(record["configuration_sha256"]) +
        bytes.fromhex(record["confirmation_seed_sha256"]) +
        bytes.fromhex(_W7_POOL_SHA256) +
        bytes.fromhex(record["execution_nonce_sha256"])
    ).hexdigest()
    if record["attempt_id"] != expected_attempt_id:
        raise ValueError("W7 attempt identity is not derived from immutable inputs")

    case_keys = {
        "case_id", "invocation_id", "model_family", "flag_value",
        "fixture_mutation", "wire_text_utf8_b64", "wire_token_end_offsets_zlib_b64",
        "canonical_token_ids_zlib_b64",
        "live_token_count", "live_token_ids_zlib_b64",
        "restored_checkpoint_token_ids_zlib_b64", "sync_tokens_sha256",
        "final_lineage_sha256", "prompt_tokens",
        "checkpoint_inventory", "events", "output_token_ids",
        "logits_f32_zlib_b64", "state_manifest", "safety",
    }
    inventory_keys = {
        "record_id", "path_sha256", "device", "inode", "size", "mtime_ns",
        "payload_bytes", "token_length", "rendered_prefix_bytes",
        "rendered_prefix_sha256", "token_sha256",
        "lineage_sha256", "payload_sha256", "structurally_valid",
        "matches_wire_prefix", "matches_lineage",
    }
    event_keys = {
        "seq", "invocation_id", "kind", "record_id", "checkpoint_tokens",
        "payload_sha256", "start", "end", "byte_count", "status",
    }
    safety_keys = {
        "wrapper_path", "wrapper_sha256", "lock_path", "lock_owner_uid",
        "lock_mode_octal", "lock_acquired", "cgroup_path",
        "cgroup_identity_sha256", "start_available_gib", "minimum_available_gib",
        "bare_engine_detected", "terminal_rc", "oom_events_before",
        "oom_events_after", "xid_count", "swap_used_before_bytes",
        "swap_used_after_bytes", "survivor_pids",
    }
    all_fixture_hashes: set[str] = set()
    all_selection_seeds: set[str] = set()
    all_fixture_content_hashes: set[str] = set()
    all_invocation_ids: set[str] = set()
    maximum_logit_delta = 0.0
    selected_checkpoints: list[int] = []
    all_checks: list[bool] = []

    for regime_index, regime in enumerate(regimes):
        _require_exact_keys(
            regime,
            {
                "regime_id", "fixture_variant", "fixture_sha256",
                "selection_seed_sha256", "cases",
            },
            f"W7 regime {regime_index}",
        )
        for field in ("fixture_sha256", "selection_seed_sha256"):
            if not _is_sha256(regime[field]):
                raise ValueError(f"W7 {regime['regime_id']} {field} is invalid")
        expected_selection_seed = hashlib.sha256(
            bytes.fromhex(record["confirmation_seed_sha256"]) +
            regime["regime_id"].encode("ascii")
        ).hexdigest()
        if regime["selection_seed_sha256"] != expected_selection_seed:
            raise ValueError(f"W7 {regime['regime_id']} selection seed is not bound")
        if regime["regime_id"] == "primary":
            expected_variant: str | int = "primary-fixed"
        else:
            base_variant = int(expected_selection_seed[:4], 16) % 8
            expected_variant = (
                base_variant if regime["regime_id"] == "confirmation-1"
                else 8 + base_variant
            )
        if regime["fixture_variant"] != expected_variant:
            raise ValueError(f"W7 {regime['regime_id']} fixture variant is not seed-selected")
        expected_fixture = fixture_pool[expected_variant]
        expected_wire = expected_fixture["wire"]
        all_fixture_hashes.add(regime["fixture_sha256"])
        all_selection_seeds.add(regime["selection_seed_sha256"])
        cases = regime["cases"]
        if not isinstance(cases, list) or len(cases) != len(_W7_CASE_CONTRACT):
            raise ValueError(f"W7 {regime['regime_id']} case matrix is incomplete")
        by_id: dict[str, dict[str, Any]] = {}
        derived: dict[str, dict[str, Any]] = {}
        for case_index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValueError("W7 case must be an object")
            _require_exact_keys(case, case_keys, f"W7 case {regime_index}:{case_index}")
            case_id = case["case_id"]
            if case_id not in _W7_CASE_CONTRACT or case_id in by_id:
                raise ValueError("W7 case IDs are missing, duplicated, or unknown")
            family, flag, mutation, expected_mode = _W7_CASE_CONTRACT[case_id]
            if (
                case["model_family"] != family or case["flag_value"] != flag or
                case["fixture_mutation"] != mutation
            ):
                raise ValueError(f"W7 {case_id} does not match its frozen contract")
            expected_invocation = (
                f"invoke:{record['attempt_id']}:{regime['regime_id']}:{case_id}"
            )
            if case["invocation_id"] != expected_invocation or expected_invocation in all_invocation_ids:
                raise ValueError(f"W7 {case_id} invocation identity is invalid")
            all_invocation_ids.add(expected_invocation)
            wire_bytes = _w7_utf8_bytes(case["wire_text_utf8_b64"], f"W7 {case_id} wire text")
            if wire_bytes != expected_wire:
                raise ValueError(f"W7 {case_id} fixture is not frozen and seed-selected")
            for field in ("live_token_count", "prompt_tokens"):
                value = case[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"W7 {case_id} {field} is invalid")
            if case["prompt_tokens"] <= 0:
                raise ValueError(f"W7 {case_id} token frontiers are invalid")
            canonical = _w7_i32_vector(
                case["canonical_token_ids_zlib_b64"],
                f"W7 {case_id} canonical tokens", case["prompt_tokens"],
            )
            if (
                canonical != expected_fixture["canonical"] or
                case["prompt_tokens"] != expected_fixture["prompt_tokens"]
            ):
                raise ValueError(f"W7 {case_id} does not use frozen production tokens")
            if any(token < 0 or token >= 154880 for token in canonical):
                raise ValueError(f"W7 {case_id} canonical token is outside vocabulary")
            wire_offsets = _w7_i32_vector(
                case["wire_token_end_offsets_zlib_b64"],
                f"W7 {case_id} wire token offsets", case["prompt_tokens"],
            )
            if (
                not wire_offsets or wire_offsets[-1] != len(wire_bytes) or
                any(current < previous for previous, current in zip((0,) + wire_offsets, wire_offsets))
            ):
                raise ValueError(f"W7 {case_id} wire/token boundary map is invalid")
            if wire_offsets != expected_fixture["offsets"]:
                raise ValueError(f"W7 {case_id} does not use frozen production offsets")
            if case["live_token_count"] == 0:
                if case["live_token_ids_zlib_b64"] is not None:
                    raise ValueError(f"W7 {case_id} zero live lineage has encoded tokens")
                live_tokens: tuple[int, ...] = ()
            else:
                live_tokens = _w7_i32_vector(
                    case["live_token_ids_zlib_b64"], f"W7 {case_id} live tokens",
                    case["live_token_count"],
                )
                if any(token < 0 or token >= 154880 for token in live_tokens):
                    raise ValueError(f"W7 {case_id} live token is outside vocabulary")
            if case_id == "cold-primary":
                expected_live: tuple[int, ...] = ()
            elif case_id in {"exact-off", "exact-on"}:
                expected_live = expected_fixture["canonical"]
            else:
                expected_live = frozen_live_tokens
            if live_tokens != expected_live:
                raise ValueError(f"W7 {case_id} does not use frozen production live lineage")
            common_tokens = 0
            while (
                common_tokens < len(live_tokens) and
                common_tokens < len(canonical) and
                live_tokens[common_tokens] == canonical[common_tokens]
            ):
                common_tokens += 1
            expected_common = (
                0 if case_id == "cold-primary" else
                case["prompt_tokens"] if case_id in {"exact-off", "exact-on"} else
                expected_fixture["common_tokens"]
            )
            if common_tokens != expected_common:
                raise ValueError(f"W7 {case_id} common frontier is not frozen")
            lineage = _w7_token_sha256(canonical)
            if case["sync_tokens_sha256"] != lineage or case["final_lineage_sha256"] != lineage:
                raise ValueError(f"W7 {case_id} synchronized lineage is invalid")

            inventory = case["checkpoint_inventory"]
            if not isinstance(inventory, list):
                raise ValueError(f"W7 {case_id} inventory is invalid")
            inventory_by_id: dict[str, dict[str, Any]] = {}
            for item_index, item in enumerate(inventory):
                if not isinstance(item, dict):
                    raise ValueError(f"W7 {case_id} inventory item is invalid")
                _require_exact_keys(item, inventory_keys, f"W7 {case_id} inventory {item_index}")
                record_id = item["record_id"]
                if not isinstance(record_id, str) or not record_id or record_id in inventory_by_id:
                    raise ValueError(f"W7 {case_id} inventory identity is invalid")
                for field in (
                    "path_sha256", "rendered_prefix_sha256", "token_sha256",
                    "lineage_sha256", "payload_sha256",
                ):
                    if not _is_sha256(item[field]):
                        raise ValueError(f"W7 {case_id} inventory hash is invalid")
                for field in (
                    "device", "inode", "size", "mtime_ns", "payload_bytes", "token_length",
                    "rendered_prefix_bytes",
                ):
                    value = item[field]
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ValueError(f"W7 {case_id} inventory number is invalid")
                for field in ("structurally_valid", "matches_wire_prefix", "matches_lineage"):
                    if not isinstance(item[field], bool):
                        raise ValueError(f"W7 {case_id} inventory flag is invalid")
                token_length = item["token_length"]
                computed_prefix = (
                    _w7_token_sha256(canonical[:token_length])
                    if token_length <= len(canonical) else None
                )
                if item["size"] < item["payload_bytes"]:
                    raise ValueError(f"W7 {case_id} payload extent exceeds record size")
                if (
                    item["structurally_valid"] and
                    item["payload_bytes"] != _w7_expected_payload_bytes(token_length)
                ):
                    raise ValueError(f"W7 {case_id} payload extent contradicts GLM format")
                if item["matches_lineage"] != (item["lineage_sha256"] == computed_prefix):
                    raise ValueError(f"W7 {case_id} lineage receipt contradicts inventory")
                prefix_bytes = item["rendered_prefix_bytes"]
                expected_prefix_bytes = (
                    wire_offsets[token_length - 1]
                    if 0 < token_length <= len(wire_offsets) else None
                )
                computed_rendered_prefix = (
                    hashlib.sha256(wire_bytes[:prefix_bytes]).hexdigest()
                    if prefix_bytes <= len(wire_bytes) else None
                )
                if prefix_bytes != expected_prefix_bytes:
                    raise ValueError(f"W7 {case_id} rendered prefix length contradicts token boundary map")
                if item["matches_wire_prefix"] != (item["rendered_prefix_sha256"] == computed_rendered_prefix):
                    raise ValueError(f"W7 {case_id} wire receipt contradicts inventory")
                if item["structurally_valid"] and item["token_sha256"] != computed_prefix:
                    raise ValueError(f"W7 {case_id} checkpoint tokens contradict canonical prefix")
                inventory_by_id[record_id] = item
            selected_fixture = expected_fixture["selected_tokens"]
            full_recipe = [
                ("older", selected_fixture + inventory_recipe["older_delta_tokens"], True, True, True),
                ("selected", selected_fixture, True, True, True),
                ("wrong-lineage", selected_fixture + inventory_recipe["wrong_lineage_delta_tokens"], True, True, False),
                ("malformed", selected_fixture + inventory_recipe["malformed_delta_tokens"], False, True, True),
            ]
            if case_id in {
                "strict-primary", "candidate-primary", "flag-zero",
                "flag-empty", "flag-garbage",
            }:
                expected_inventory_recipe = full_recipe
            elif case_id.startswith("malformed-"):
                expected_inventory_recipe = [full_recipe[3]]
            elif case_id.startswith("wrong-lineage-"):
                expected_inventory_recipe = [full_recipe[2]]
            elif case_id.startswith("divergence-"):
                expected_inventory_recipe = [
                    ("selected", selected_fixture, True, False, True),
                ]
            elif case_id.startswith("shorten-"):
                expected_inventory_recipe = [
                    ("selected", expected_fixture["common_tokens"] + 1, True, True, True),
                ]
            else:
                expected_inventory_recipe = []
            observed_recipe = [
                (
                    item["record_id"], item["token_length"], item["structurally_valid"],
                    item["matches_wire_prefix"], item["matches_lineage"],
                )
                for item in inventory
            ]
            if observed_recipe != expected_inventory_recipe:
                raise ValueError(f"W7 {case_id} checkpoint inventory recipe is not frozen")
            eligible = [
                item for item in inventory
                if item["structurally_valid"] and item["matches_wire_prefix"] and
                item["matches_lineage"] and item["token_length"] <= common_tokens
            ]
            expected_record = None
            if eligible:
                longest = max(item["token_length"] for item in eligible)
                winners = [item for item in eligible if item["token_length"] == longest]
                if len(winners) != 1:
                    raise ValueError(f"W7 {case_id} longest checkpoint is ambiguous")
                expected_record = winners[0]
            selection_cases = {
                "strict-primary", "candidate-primary", "flag-zero",
                "flag-empty", "flag-garbage",
            }
            if case_id in selection_cases:
                if [item["record_id"] for item in inventory] != [
                    "older", "selected", "wrong-lineage", "malformed",
                ]:
                    raise ValueError(f"W7 {case_id} required decoy inventory is incomplete")
                older, selected, wrong, malformed = inventory
                if not (
                    older["structurally_valid"] and older["matches_wire_prefix"] and older["matches_lineage"] and
                    selected["structurally_valid"] and selected["matches_wire_prefix"] and selected["matches_lineage"] and
                    selected["token_length"] > older["token_length"] and
                    wrong["structurally_valid"] and wrong["matches_wire_prefix"] and not wrong["matches_lineage"] and
                    not malformed["structurally_valid"] and malformed["matches_wire_prefix"] and malformed["matches_lineage"] and
                    expected_record is selected
                ):
                    raise ValueError(f"W7 {case_id} decoy semantics are invalid")

            events = case["events"]
            if not isinstance(events, list):
                raise ValueError(f"W7 {case_id} events are invalid")
            kinds: dict[str, list[dict[str, Any]]] = {}
            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(f"W7 {case_id} event must be an object")
                _require_exact_keys(event, event_keys, f"W7 {case_id} event {event_index}")
                if event["seq"] != event_index or event["invocation_id"] != case["invocation_id"]:
                    raise ValueError(f"W7 {case_id} event ordering/identity is invalid")
                if event["kind"] not in {
                    "matcher_candidate", "matcher_selected", "payload_read",
                    "reset", "invalidate", "evaluate",
                } or event["status"] not in {"observed", "ok"}:
                    raise ValueError(f"W7 {case_id} event type/status is invalid")
                for field in ("checkpoint_tokens", "start", "end", "byte_count"):
                    value = event[field]
                    if value is not None and (
                        not isinstance(value, int) or isinstance(value, bool) or value < 0
                    ):
                        raise ValueError(f"W7 {case_id} event number is invalid")
                if event["record_id"] is not None and event["record_id"] not in inventory_by_id:
                    raise ValueError(f"W7 {case_id} event references an unknown record")
                if event["payload_sha256"] is not None and not _is_sha256(event["payload_sha256"]):
                    raise ValueError(f"W7 {case_id} event payload hash is invalid")
                kinds.setdefault(event["kind"], []).append(event)
            candidate_events = kinds.get("matcher_candidate", [])
            if [event["record_id"] for event in candidate_events] != [item["record_id"] for item in inventory]:
                raise ValueError(f"W7 {case_id} matcher candidates do not bind inventory")
            for event, item in zip(candidate_events, inventory):
                if (
                    event["checkpoint_tokens"] != item["token_length"] or
                    event["payload_sha256"] != item["payload_sha256"] or
                    event["status"] != "observed" or event["byte_count"] != 0 or
                    event["start"] is not None or event["end"] is not None
                ):
                    raise ValueError(f"W7 {case_id} matcher candidate receipt is invalid")
            selected_events = kinds.get("matcher_selected", [])
            read_events = kinds.get("payload_read", [])
            resets = kinds.get("reset", [])
            invalidates = kinds.get("invalidate", [])
            evaluations = kinds.get("evaluate", [])
            selected_event = selected_events[0] if len(selected_events) == 1 else None
            read_event = read_events[0] if len(read_events) == 1 else None
            if len(selected_events) > 1 or len(read_events) > 1:
                raise ValueError(f"W7 {case_id} has duplicate selection/read events")
            if selected_event is not None:
                if expected_record is None or any(
                    (
                        selected_event["record_id"] != expected_record["record_id"],
                        selected_event["checkpoint_tokens"] != expected_record["token_length"],
                        selected_event["payload_sha256"] != expected_record["payload_sha256"],
                        selected_event["status"] != "ok",
                        selected_event["start"] is not None,
                        selected_event["end"] is not None,
                        selected_event["byte_count"] != 0,
                    )
                ):
                    raise ValueError(f"W7 {case_id} selected the wrong checkpoint")
            if read_event is not None:
                if selected_event is None or any(
                    (
                        read_event["record_id"] != selected_event["record_id"],
                        read_event["checkpoint_tokens"] != selected_event["checkpoint_tokens"],
                        read_event["payload_sha256"] != selected_event["payload_sha256"],
                        read_event["status"] != "ok",
                        read_event["byte_count"] != expected_record["payload_bytes"],
                        read_event["start"] is not None, read_event["end"] is not None,
                    )
                ):
                    raise ValueError(f"W7 {case_id} payload read is not selected-record-bound")
            eval_ranges = []
            for event in evaluations:
                if (
                    event["status"] != "ok" or event["start"] is None or event["end"] is None or
                    event["end"] <= event["start"] or event["byte_count"] != 0
                ):
                    raise ValueError(f"W7 {case_id} evaluation receipt is invalid")
                eval_ranges.append([event["start"], event["end"]])
            for event in resets + invalidates:
                if any(
                    event[field] is not None
                    for field in ("record_id", "checkpoint_tokens", "payload_sha256", "start", "end")
                ) or event["byte_count"] != 0 or event["status"] != "ok":
                    raise ValueError(f"W7 {case_id} reset/invalidate receipt is malformed")
            def evaluation_covers(start: int, end: int) -> bool:
                return bool(eval_ranges) and (
                    eval_ranges[0][0] == start and eval_ranges[-1][1] == end and
                    all(left[1] == right[0] for left, right in zip(eval_ranges, eval_ranges[1:]))
                )
            if resets:
                mode = "cold"
            elif read_event is not None:
                mode = "resume"
            elif not events and expected_mode == "replay":
                mode = "replay"
            elif not events or (
                set(kinds) == {"matcher_candidate"} and
                selected_event is None and read_event is None
            ):
                mode = "reject"
            else:
                mode = "invalid"
            if mode != expected_mode:
                raise ValueError(f"W7 {case_id} operation-derived mode is {mode}, expected {expected_mode}")

            restored_encoded = case["restored_checkpoint_token_ids_zlib_b64"]
            if read_event is not None:
                selected_tokens = int(read_event["checkpoint_tokens"])
                restored = _w7_i32_vector(
                    restored_encoded, f"W7 {case_id} restored tokens", selected_tokens,
                )
                if (
                    restored != canonical[:selected_tokens] or
                    restored != live_tokens[:selected_tokens]
                ):
                    raise ValueError(f"W7 {case_id} restored tokens differ from canonical prefix")
            elif restored_encoded is not None:
                raise ValueError(f"W7 {case_id} has restored tokens without a payload read")

            if regime["regime_id"] != "primary" and case_id in {
                "strict-primary", "candidate-primary",
            }:
                if not (
                    selected_tokens <= common_tokens < len(live_tokens) < case["prompt_tokens"] and
                    len(live_tokens) - common_tokens >= 8 and
                    case["prompt_tokens"] - len(live_tokens) >= 4
                ):
                    raise ValueError(
                        f"W7 {case_id} confirmation lacks the preregistered divergent append"
                    )

            if case_id == "cold-primary":
                behavior = (
                    not inventory and len(resets) == 1 and not invalidates and
                    read_event is None and evaluation_covers(0, case["prompt_tokens"]) and
                    len(live_tokens) == 0 and common_tokens == 0 and
                    [event["kind"] for event in events[:1]] == ["reset"] and
                    all(event["kind"] == "evaluate" for event in events[1:])
                )
            elif case_id in {"strict-primary", "flag-zero", "flag-empty", "flag-garbage"}:
                behavior = (
                    len(inventory) == 4 and read_event is not None and len(resets) == 1 and
                    not invalidates and evaluation_covers(0, case["prompt_tokens"]) and
                    [event["kind"] for event in events[:7]] ==
                    ["matcher_candidate"] * 4 + ["matcher_selected", "payload_read", "reset"] and
                    all(event["kind"] == "evaluate" for event in events[7:])
                )
            elif case_id == "candidate-primary":
                behavior = (
                    len(inventory) == 4 and expected_record is not None and
                    read_event is not None and not resets and not invalidates and
                    evaluation_covers(expected_record["token_length"], case["prompt_tokens"]) and
                    [event["kind"] for event in events[:6]] ==
                    ["matcher_candidate"] * 4 + ["matcher_selected", "payload_read"] and
                    all(event["kind"] == "evaluate" for event in events[6:])
                )
                if behavior:
                    selected_checkpoints.append(expected_record["token_length"])
            elif expected_mode == "replay":
                behavior = (
                    not inventory and not events and live_tokens == canonical and
                    common_tokens == case["prompt_tokens"]
                )
            else:
                behavior = (
                    selected_event is None and read_event is None and not resets and
                    not invalidates and not evaluations
                )
                if case_id.startswith("malformed-"):
                    behavior = behavior and len(inventory) == 1 and not inventory[0]["structurally_valid"] and [event["kind"] for event in events] == ["matcher_candidate"]
                elif case_id.startswith("wrong-lineage-"):
                    behavior = behavior and len(inventory) == 1 and not inventory[0]["matches_lineage"] and [event["kind"] for event in events] == ["matcher_candidate"]
                elif case_id.startswith("divergence-"):
                    behavior = behavior and len(inventory) == 1 and not inventory[0]["matches_wire_prefix"] and [event["kind"] for event in events] == ["matcher_candidate"]
                elif case_id.startswith("shorten-"):
                    behavior = behavior and len(inventory) == 1 and inventory[0]["token_length"] > common_tokens and [event["kind"] for event in events] == ["matcher_candidate"]
                else:
                    behavior = behavior and not inventory and not events

            safety = case["safety"]
            if not isinstance(safety, dict):
                raise ValueError(f"W7 {case_id} safety receipt is invalid")
            _require_exact_keys(safety, safety_keys, f"W7 {case_id} safety")
            if (
                not isinstance(safety["lock_owner_uid"], int) or
                isinstance(safety["lock_owner_uid"], bool)
            ):
                raise ValueError(f"W7 {case_id} lock owner is invalid")
            if (
                safety["wrapper_path"] != "results/glm52-gates/harness/glm_safe_run.sh" or
                safety["wrapper_sha256"] != _W7_SAFE_WRAPPER_SHA256 or
                safety["lock_path"] != "/run/dsv4/inference.lock" or
                safety["lock_owner_uid"] != 0 or safety["lock_mode_octal"] != "0660" or
                safety["lock_acquired"] is not True or
                not isinstance(safety["cgroup_path"], str) or
                not safety["cgroup_path"].startswith("/sys/fs/cgroup/system.slice/") or
                not _is_sha256(safety["cgroup_identity_sha256"]) or
                safety["bare_engine_detected"] is not False or safety["terminal_rc"] != 0 or
                safety["oom_events_after"] != safety["oom_events_before"] or
                safety["xid_count"] != 0 or
                safety["swap_used_after_bytes"] > safety["swap_used_before_bytes"] or
                safety["survivor_pids"] != []
            ):
                raise ValueError(f"W7 {case_id} containment receipt is unsafe")
            for field in ("start_available_gib", "minimum_available_gib"):
                _finite_number(safety[field], f"W7 {case_id} {field}", minimum=0.0)
            if safety["start_available_gib"] < 110.0 or safety["minimum_available_gib"] < 10.0:
                behavior = False
            for field in (
                "terminal_rc", "oom_events_before", "oom_events_after", "xid_count",
                "swap_used_before_bytes", "swap_used_after_bytes",
            ):
                if not isinstance(safety[field], int) or isinstance(safety[field], bool) or safety[field] < 0:
                    raise ValueError(f"W7 {case_id} safety counter is invalid")
            by_id[case_id] = case
            derived[case_id] = {
                "canonical": canonical, "lineage": lineage, "wire": wire_bytes,
                "live": live_tokens, "common": common_tokens, "behavior": behavior,
            }
        if set(by_id) != set(_W7_CASE_CONTRACT):
            raise ValueError(f"W7 {regime['regime_id']} case matrix is incomplete")
        if regime["fixture_sha256"] != _w7_fixture_digest(regime):
            raise ValueError(f"W7 {regime['regime_id']} fixture hash is not content-derived")
        fixture_content_hash = hashlib.sha256(
            json.dumps(
                _w7_fixture_material(regime), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        all_fixture_content_hashes.add(fixture_content_hash)

        cold = by_id["cold-primary"]
        strict = by_id["strict-primary"]
        candidate = by_id["candidate-primary"]
        identity_ids = (
            "cold-primary", "strict-primary", "candidate-primary",
            "exact-off", "exact-on",
        )
        primary_identity = (
            len({derived[case_id]["wire"] for case_id in identity_ids}) == 1 and
            len({derived[case_id]["canonical"] for case_id in identity_ids}) == 1 and
            len({by_id[case_id]["prompt_tokens"] for case_id in identity_ids}) == 1
        )
        if regime["regime_id"] == "primary":
            primary_identity = primary_identity and (
                len(derived["strict-primary"]["live"]) ==
                len(derived["candidate-primary"]["live"]) == expected_fixture["live_tokens"] and
                derived["strict-primary"]["common"] ==
                derived["candidate-primary"]["common"] == expected_fixture["common_tokens"] and
                candidate["prompt_tokens"] == expected_fixture["prompt_tokens"] and
                selected_checkpoints[-1:] == [expected_fixture["selected_tokens"]]
            )
        comparison_cases = (
            cold, strict, candidate, by_id["exact-off"], by_id["exact-on"],
        )
        outputs = [case["output_token_ids"] for case in comparison_cases]
        if any(
            not isinstance(output, list) or len(output) != 64 or
            any(not isinstance(token, int) or isinstance(token, bool) or token < 0 or token >= 154880 for token in output)
            for output in outputs
        ):
            raise ValueError(f"W7 {regime['regime_id']} completed output is invalid")
        output_equal = all(output == outputs[0] for output in outputs[1:])
        logits = [
            _w7_f32_vector(case["logits_f32_zlib_b64"], f"W7 {case['case_id']} logits", 154880)
            for case in comparison_cases
        ]
        deltas = [
            max(abs(left - right) for left, right in zip(logits[0], comparison))
            for comparison in logits[1:]
        ]
        if any(not math.isfinite(delta) for delta in deltas):
            raise ValueError("W7 derived logit delta is non-finite")
        maximum_logit_delta = max(maximum_logit_delta, *deltas)
        argmaxes = [max(range(len(vector)), key=vector.__getitem__) for vector in logits]
        state_manifests = [
            _w7_state_manifest(
                case["state_manifest"], f"W7 {case['case_id']} state",
                derived[case["case_id"]]["lineage"], case["prompt_tokens"],
            )
            for case in comparison_cases
        ]
        state_equal = all(manifest == state_manifests[0] for manifest in state_manifests[1:])
        all_checks.extend([
            primary_identity,
            all(item["behavior"] for item in derived.values()),
            output_equal,
            max(deltas) < 1e-2 and len(set(argmaxes)) == 1,
            state_equal,
        ])

    if (
        len(all_fixture_hashes) != 3 or len(all_selection_seeds) != 3 or
        len(all_fixture_content_hashes) != 3
    ):
        raise ValueError("W7 confirmation fixtures/seeds are not independent")
    expected_top_fixture = hashlib.sha256(
        "".join(regime["fixture_sha256"] for regime in regimes).encode("ascii")
    ).hexdigest()
    if record["fixture_sha256"] != expected_top_fixture:
        raise ValueError("W7 top-level fixture hash is not regime-content-bound")
    checks = {
        "three_independent_regimes": len(all_fixture_hashes) == len(all_selection_seeds) == 3,
        "receipt_derived_case_matrix": all(all_checks),
        "full_logits": maximum_logit_delta < 1e-2,
        "complete_state_manifests": all(all_checks),
        "no_failures": not record["failures"],
    }
    return {
        "scorer_id": "w7.resume.v1",
        "formula_version": 1,
        "gate": "W7",
        "derived_metrics": {
            "max_abs_logit_delta": maximum_logit_delta,
            "qualified_regimes": 3,
            "selected_checkpoint_tokens": selected_checkpoints,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


_FROZEN_SCORER_IDENTITIES = frozenset(
    {
        (
            "W1",
            "3879eb01a2a427be76373b847d832738f1f86552",
            "w1.affine-quality.v2",
            "b322d78612d51eb714039c38fe79d512"
            "1428681610815d7453dbb6e69ad5a1e6",
        ),
    }
)


def scorer_descriptor_matches(
    gate: str,
    candidate_hash: str,
    scorer_id: str,
    implementation_sha256: str,
) -> bool:
    """Accept the current scorer or one exact pre-registered frozen identity."""
    if implementation_sha256 == registered_scorer_digest(scorer_id):
        return True
    return (
        gate,
        candidate_hash,
        scorer_id,
        implementation_sha256,
    ) in _FROZEN_SCORER_IDENTITIES


def registered_scorer_digest(scorer_id: str) -> str:
    """Hash only the fixed formula and dependencies for one scorer version."""
    dependencies: dict[str, tuple[Any, ...]] = {
        "foundation.v1": (
            _score_foundation,
            _score_foundation_baseline,
            _require_exact_keys,
            _finite_number,
            decode_tokens_per_second,
            _is_sha256,
        ),
        "w11.context.v1": (
            _score_w11,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
        "parity.performance.v1": (
            _score_parity,
            performance_verdict,
            paired_ratio_bound,
            _finite_positive,
            _t95,
            validate_raw_record,
            validate_ab_blocks,
            decode_tokens_per_second,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
        "parity.reviewed-no-go.v1": (
            _score_reviewed_no_go,
            _score_parity,
            reviewed_measurements_digest,
            performance_verdict,
            paired_ratio_bound,
            _finite_positive,
            _t95,
            validate_raw_record,
            validate_ab_blocks,
            decode_tokens_per_second,
            _review_issues,
            _review_issue_ids,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
        "review.final.v1": (
            _score_review,
            _review_issues,
            _review_issue_ids,
            _require_exact_keys,
        ),
        "workstream.terminal.v1": (
            _score_workstream,
            paired_ratio_bound,
            _finite_positive,
            _t95,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
        "w7.resume.v1": (
            _score_w7_resume,
            _w7_f32_vector,
            _w7_i32_vector,
            _w7_utf8_bytes,
            _w7_caller_wire,
            _w7_fixture_pool,
            _w7_token_sha256,
            _w7_expected_payload_bytes,
            _w7_state_manifest,
            _w7_fixture_material,
            _w7_fixture_digest,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
        "w1.affine-quality.v2": (
            _score_w1_affine_raw,
            _w1_single_match,
            _w1_quality_cases,
            quality_verdict,
            _weighted_upper_95,
            _t95,
            _require_exact_keys,
            _finite_number,
            _is_sha256,
        ),
    }
    functions = dependencies.get(scorer_id)
    if functions is None:
        raise ValueError(f"unknown registered scorer: {scorer_id}")
    trust_boundary = (
        registered_scorer_digest,
        score_registered_gate,
        validate_attempt,
        validate_manifest_lineage,
        _fetch_public_drand,
        _git_commit_time,
        _utc_timestamp,
        _finite_number,
        _is_sha256,
        validate_source_provenance,
        validate_profile_artifact_bindings,
        validate_record_artifact_bindings,
        validate_w1_root_receipt,
        _w1_attempt_tree_manifest,
        _verify_w1_journal_authority,
        generate_w11_fixture,
        _load_approved_dsv4_profile,
        _read_strict_json,
        _unique_pairs,
        _sha256,
        scorer_descriptor_matches,
    )
    digest = hashlib.sha256()
    digest.update(f"scorer_id={scorer_id}\n".encode())
    for function in functions + trust_boundary:
        digest.update(f"function={function.__name__}\n".encode())
        digest.update(inspect.getsource(function).encode())
    digest.update(
        json.dumps(
            sorted(_FROZEN_SCORER_IDENTITIES),
            separators=(",", ":"),
        ).encode()
    )
    if scorer_id == "parity.performance.v1":
        digest.update(
            json.dumps(_T95, sort_keys=True, separators=(",", ":")).encode()
        )
    if scorer_id == "w7.resume.v1":
        digest.update(
            json.dumps(
                {
                    "case_contract": _W7_CASE_CONTRACT,
                    "safe_wrapper_sha256": _W7_SAFE_WRAPPER_SHA256,
                    "stem_file_sha256": _W7_STEM_FILE_SHA256,
                    "stem_text_sha256": _W7_STEM_TEXT_SHA256,
                    "primary_suffix": _W7_PRIMARY_SUFFIX,
                    "confirmation_suffixes": _W7_CONFIRMATION_SUFFIXES,
                    "fixture_pool_sha256": _W7_POOL_SHA256,
                    "tokenizer_sha256": _W7_TOKENIZER_SHA256,
                    "tokenizer_init_sha256": _W7_TOKENIZER_INIT_SHA256,
                    "tokenizer_native_sha256": _W7_TOKENIZER_NATIVE_SHA256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def score_registered_gate(
    gate: str, scorer_id: str, records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute an authoritative terminal verdict from strict raw records."""
    rows = list(records)
    registered = {
        ("foundation", "foundation.v1"): _score_foundation,
        ("W1", "w1.affine-quality.v2"): _score_w1_affine_raw,
        ("W11", "w11.context.v1"): _score_w11,
        ("parity", "parity.performance.v1"): _score_parity,
        ("parity", "parity.reviewed-no-go.v1"): _score_reviewed_no_go,
        ("review", "review.final.v1"): _score_review,
        ("W7", "w7.resume.v1"): _score_w7_resume,
    }
    if scorer_id == "workstream.terminal.v1" and gate in {
        *(f"W{index}" for index in range(1, 11)),
        "switch",
    }:
        return _score_workstream(rows, gate)
    scorer = registered.get((gate, scorer_id))
    if scorer is None:
        raise ValueError(
            f"no fixed terminal scorer {scorer_id!r} is registered for {gate}"
        )
    return scorer(rows)


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path.name} contains non-finite value {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path.name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {path.name}: {exc}") from exc


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must carry an explicit UTC offset")
    return parsed


def validate_manifest_lineage(
    lineage: Any,
    gate: str,
    candidate_hash: str,
    relay_fetcher: Any = None,
    commit_time_fetcher: Any = None,
) -> None:
    """Require public randomness obtained strictly after the candidate freeze."""
    if not isinstance(lineage, dict):
        raise ValueError("manifest lineage must be an object")
    _require_exact_keys(lineage, {"freeze", "randomness"}, "manifest lineage")
    freeze = lineage["freeze"]
    randomness = lineage["randomness"]
    if not isinstance(freeze, dict) or not isinstance(randomness, dict):
        raise ValueError("freeze and randomness lineage must be objects")
    freeze_keys = {"candidate_hash", "frozen_at"}
    if gate == "W1":
        freeze_keys.add("composite_candidate_sha256")
    _require_exact_keys(freeze, freeze_keys, "freeze lineage")
    _require_exact_keys(
        randomness,
        {
            "source",
            "round",
            "randomness",
            "signature",
            "obtained_at",
            "seed_sha256",
        },
        "randomness lineage",
    )
    if freeze["candidate_hash"] != candidate_hash:
        raise ValueError("freeze lineage candidate does not match manifest")
    frozen_at = _utc_timestamp(freeze["frozen_at"], "frozen_at")
    if commit_time_fetcher is not None:
        try:
            committed_at = _utc_timestamp(
                commit_time_fetcher(candidate_hash), "commit timestamp"
            )
        except Exception as exc:
            raise ValueError(f"cannot derive commit timestamp: {exc}") from exc
        if gate == "W1":
            if frozen_at < committed_at:
                raise ValueError("frozen_at predates the candidate commit")
        elif frozen_at != committed_at:
            raise ValueError("frozen_at does not equal the commit timestamp")
    obtained_at = _utc_timestamp(randomness["obtained_at"], "obtained_at")
    if obtained_at <= frozen_at:
        raise ValueError("public randomness was not obtained after the freeze")
    if randomness["source"] != "drand-default":
        raise ValueError("randomness source is not the registered drand chain")
    round_number = randomness["round"]
    if (
        not isinstance(round_number, int)
        or isinstance(round_number, bool)
        or round_number < 1
    ):
        raise ValueError("drand round must be a positive integer")
    beacon_unix = 1_595_431_050 + (round_number - 1) * 30
    beacon_time = datetime.fromtimestamp(beacon_unix, timezone.utc)
    if beacon_time <= frozen_at:
        raise ValueError("drand round was published before the candidate freeze")
    if obtained_at < beacon_time:
        raise ValueError("obtained_at predates the drand round publication")
    signature = randomness["signature"]
    if not (
        isinstance(signature, str)
        and len(signature) == 192
        and all(character in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("drand signature is invalid")
    expected_randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    if randomness["randomness"] != expected_randomness:
        raise ValueError("drand randomness is not SHA-256(signature)")
    seed_candidate = candidate_hash
    if gate == "W1":
        seed_candidate = freeze["composite_candidate_sha256"]
        if not _is_sha256(seed_candidate):
            raise ValueError("composite candidate digest is invalid")
    expected_seed = hashlib.sha256(
        f"{seed_candidate}:{expected_randomness}:{gate}".encode()
    ).hexdigest()
    if randomness["seed_sha256"] != expected_seed:
        raise ValueError("confirmation seed derivation is invalid")
    if relay_fetcher is not None:
        expected_beacon = {
            "round": round_number,
            "randomness": expected_randomness,
            "signature": signature,
        }
        for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
            try:
                published = relay_fetcher(host, round_number)
            except Exception as exc:
                raise ValueError(
                    f"public drand relays are unavailable: {host}: {exc}"
                ) from exc
            if not isinstance(published, dict) or any(
                published.get(field) != value
                for field, value in expected_beacon.items()
            ):
                raise ValueError(
                    f"public drand relays do not authenticate the beacon: {host}"
                )


def _fetch_public_drand(host: str, round_number: int) -> dict[str, Any]:
    if host not in {"api.drand.sh", "api2.drand.sh", "api3.drand.sh"}:
        raise ValueError("unregistered drand relay")
    response = subprocess.run(
        [
            "/usr/bin/curl",
            "--disable",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "10",
            "--proto",
            "=https",
            f"https://{host}/public/{round_number}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
    )
    if response.returncode != 0:
        raise ValueError(
            response.stderr.decode("utf-8", errors="replace").strip()
            or f"curl exited {response.returncode}"
        )
    try:
        value = json.loads(
            response.stdout.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(
                pairs, f"drand relay {host}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid drand response from {host}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid drand response from {host}")
    return value


def _git_commit_time(
    candidate_hash: str, repository: Path = ROOT
) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "show", "-s", "--format=%cI", candidate_hash],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_git_env(),
    )
    if result.returncode != 0:
        raise ValueError("git cannot resolve candidate timestamp")
    try:
        value = datetime.fromisoformat(result.stdout.strip())
    except ValueError as exc:
        raise ValueError("git returned an invalid candidate timestamp") from exc
    if value.tzinfo is None:
        raise ValueError("git candidate timestamp lacks an offset")
    return value.astimezone(timezone.utc).isoformat()


def _load_approved_dsv4_profile(candidate_hash: Any) -> dict[str, Any]:
    """Load the exact DeepSeek reference identity frozen in the candidate."""
    if not (
        isinstance(candidate_hash, str)
        and len(candidate_hash) == 40
        and all(char in "0123456789abcdef" for char in candidate_hash)
    ):
        raise ValueError("approved DeepSeek profile candidate is invalid")
    result = subprocess.run(
        ["/usr/bin/git", "show", f"{candidate_hash}:configs/dsv4-profile.json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_git_env(),
    )
    if result.returncode != 0:
        raise ValueError("approved DeepSeek profile is absent from candidate")
    try:
        profile = json.loads(
            result.stdout.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(
                pairs, "approved DeepSeek profile"
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite profile value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"approved DeepSeek profile is invalid: {exc}") from exc
    _require_exact_keys(
        profile,
        {
            "schema_version",
            "profile",
            "binary_sha256",
            "configuration_sha256",
            "build_manifest_sha256",
            "weights_manifest_sha256",
            "shared_libraries",
            "model_files",
        },
        "approved DeepSeek profile",
    )
    if profile["schema_version"] != 2 or profile["profile"] != "dsv4":
        raise ValueError("approved DeepSeek profile identity is invalid")
    for field in (
        "binary_sha256",
        "configuration_sha256",
        "build_manifest_sha256",
        "weights_manifest_sha256",
    ):
        if not _is_sha256(profile[field]):
            raise ValueError(f"approved DeepSeek profile {field} is invalid")
    for field in ("shared_libraries", "model_files"):
        values = profile[field]
        if (
            not isinstance(values, dict)
            or not values
            or any(
                not isinstance(name, str)
                or not name
                or not _is_sha256(digest)
                for name, digest in values.items()
            )
        ):
            raise ValueError(f"approved DeepSeek profile {field} is invalid")
    return profile


def validate_source_provenance(
    source_path: Path,
    candidate_hash: str,
    *,
    repository: Path = ROOT,
) -> None:
    """Bind the source descriptor to the frozen repository commit and tree."""
    try:
        descriptor = _read_strict_json(source_path)
    except ValueError as exc:
        raise ValueError(f"source provenance is invalid: {exc}") from exc
    if not isinstance(descriptor, dict):
        raise ValueError("source provenance must be an object")
    _require_exact_keys(
        descriptor,
        {"schema_version", "candidate_hash", "git_tree"},
        "source provenance",
    )
    if descriptor["schema_version"] != 1:
        raise ValueError("source provenance schema is invalid")
    if descriptor["candidate_hash"] != candidate_hash:
        raise ValueError("source provenance candidate does not match manifest")
    tree = subprocess.run(
        ["/usr/bin/git", "rev-parse", f"{candidate_hash}^{{tree}}"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_git_env(),
    )
    expected_tree = tree.stdout.strip()
    if (
        tree.returncode != 0
        or len(expected_tree) != 40
        or descriptor["git_tree"] != expected_tree
    ):
        raise ValueError("source provenance git tree does not match candidate")


def validate_profile_artifact_bindings(
    manifest: dict[str, Any], artifact_paths: dict[str, Path]
) -> None:
    """Bind terminal GLM artifacts to the profile committed in the candidate."""
    if manifest.get("gate") not in {"foundation", "W11", "parity"}:
        return
    candidate_hash = manifest.get("candidate_hash")
    profile_result = subprocess.run(
        ["/usr/bin/git", "show", f"{candidate_hash}:configs/glm52-profile.json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_git_env(),
    )
    if profile_result.returncode != 0:
        raise ValueError("approved GLM profile is absent from candidate")
    try:
        profile = json.loads(
            profile_result.stdout.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(
                pairs, "approved GLM profile"
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite profile value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"approved GLM profile is invalid: {exc}") from exc
    _require_exact_keys(
        profile,
        {
            "schema_version",
            "profile",
            "binary_sha256",
            "model_sha256",
            "tokenizer_sha256",
            "context_cap",
            "build_manifest_sha256",
            "promotion",
            "runtime",
            "artifact_sha256",
        },
        "approved GLM profile",
    )
    if (
        profile["schema_version"] != 3
        or profile["profile"] != "glm52"
        or profile["context_cap"] != 1_048_576
    ):
        raise ValueError("approved GLM profile identity or context cap is invalid")
    for field in (
        "binary_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "build_manifest_sha256",
    ):
        if not _is_sha256(profile[field]):
            raise ValueError(f"approved GLM profile {field} is invalid")
    promotion = profile["promotion"]
    _require_exact_keys(
        promotion,
        {
            "gate",
            "engine_commit",
            "binary_freeze_sha256",
            "owner_decision_sha256",
            "review_sha256",
        },
        "approved GLM promotion",
    )
    if (
        promotion["gate"]
        != "W7.1a-stable-model-cache-generation-owner-adoption"
        or not isinstance(promotion["engine_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", promotion["engine_commit"]) is None
        or any(
            not _is_sha256(promotion[field])
            for field in (
                "binary_freeze_sha256",
                "owner_decision_sha256",
                "review_sha256",
            )
        )
    ):
        raise ValueError("approved GLM promotion binding is invalid")
    for field in ("binary_sha256", "model_sha256"):
        if not _is_sha256(profile[field]) or manifest.get(field) != profile[field]:
            raise ValueError(
                f"manifest {field} does not match approved GLM profile"
            )
    runtime = profile["runtime"]
    _require_exact_keys(
        runtime,
        {"engine_environment", "launch_arguments", "benchmark", "safety"},
        "approved GLM runtime",
    )
    expected_environment = {
        "DS4_CUDA_EXPERT_CACHE_GB": "0",
        "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
        "DS4_CUDA_FETCH_THREADS": "6",
        "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_CUDA_STABLE_MODEL_REMAP": "1",
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    expected_arguments = [
        "--cuda",
        "-m",
        "{model}",
        "-c",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        "{port}",
        "--ssd-streaming",
        "--ssd-streaming-cache-experts",
        "40GB",
    ]
    expected_benchmark = {
        "fixture_context_tokens": 0,
        "max_completion_tokens": 160,
        "minimum_completion_tokens": 128,
        "raw_token_timing_required": True,
    }
    expected_safety = {
        "kill_floor_gib": 40,
        "minimum_start_gib": 110,
        "sample_hz": 4,
        "swap_max_bytes": 0,
        "timeout_seconds": 2400,
        "virtual_memory_limit_kib": 419_430_400,
    }
    if (
        runtime["engine_environment"] != expected_environment
        or runtime["launch_arguments"] != expected_arguments
        or runtime["benchmark"] != expected_benchmark
        or runtime["safety"] != expected_safety
    ):
        raise ValueError("approved GLM runtime configuration is invalid")
    expected_artifacts = {
        "scripts/11_build_glm52_repro.sh",
        "results/glm52-goal/harness/decisive_matched.sh",
        "results/glm52-goal/harness/glm_decisive_arm.sh",
        "results/glm52-gates/harness/glm_safe_run.sh",
        "results/glm52-gates/harness/glm_cgroup_run.sh",
        "results/glm52-gates/harness/glm_evidence_export.py",
        "scripts/30_bench_speed.py",
    }
    artifact_hashes = profile["artifact_sha256"]
    if (
        not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != expected_artifacts
        or any(not _is_sha256(value) for value in artifact_hashes.values())
    ):
        raise ValueError("approved GLM artifact hash map is invalid")
    candidate_artifacts = {
        **artifact_hashes,
        "configs/build-manifests/glm52-ds4-repro.json": profile[
            "build_manifest_sha256"
        ],
        "results/glm52-gates/W7-cache-generation-freeze-v9.json": promotion[
            "binary_freeze_sha256"
        ],
        "results/glm52-gates/W7-cache-generation-W7.1a-owner-adoption.json": promotion[
            "owner_decision_sha256"
        ],
        "results/glm52-gates/W7-cache-generation-W7.1a-review-r295.json": promotion[
            "review_sha256"
        ],
    }
    for relative_path, expected_digest in candidate_artifacts.items():
        artifact_result = subprocess.run(
            ["/usr/bin/git", "show", f"{candidate_hash}:{relative_path}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env=_git_env(),
        )
        if (
            artifact_result.returncode != 0
            or hashlib.sha256(artifact_result.stdout).hexdigest()
            != expected_digest
        ):
            raise ValueError(
                f"approved GLM artifact does not match candidate: {relative_path}"
            )
    configuration = artifact_paths.get("configuration")
    if (
        configuration is None
        or configuration.read_bytes() != profile_result.stdout
        or manifest.get("configuration_sha256")
        != hashlib.sha256(profile_result.stdout).hexdigest()
    ):
        raise ValueError(
            "configuration artifact does not match approved GLM profile"
        )


def validate_attempt(
    attempt: Path,
    *,
    root_authority_pending: bool = False,
    source_repository: Path | None = None,
) -> None:
    """Validate the mandatory evidence triplet without trusting narration."""
    if attempt.is_symlink():
        raise ValueError("attempt directory must not be a symlink")
    if not attempt.is_dir():
        raise ValueError("attempt path is not a directory")
    required_hashes = {
        "source_sha256",
        "diff_sha256",
        "binary_sha256",
        "scorer_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
        "configuration_sha256",
    }
    manifest = _read_strict_json(attempt / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("gate") not in GATE_ORDER:
        raise ValueError("manifest gate is missing or unknown")
    candidate_hash = manifest.get("candidate_hash")
    if not (
        isinstance(candidate_hash, str)
        and len(candidate_hash) == 40
        and all(char in "0123456789abcdef" for char in candidate_hash)
    ):
        raise ValueError("manifest candidate_hash is invalid")
    repository = (
        source_repository.resolve()
        if source_repository is not None
        else ROOT
    )
    if source_repository is not None:
        try:
            repository_details = source_repository.lstat()
        except FileNotFoundError as exc:
            raise ValueError("source repository is absent") from exc
        if (
            not root_authority_pending
            or manifest["gate"] != "W1"
            or source_repository.is_symlink()
            or not stat.S_ISDIR(repository_details.st_mode)
        ):
            raise ValueError("source repository override is not authorized")
    candidate_check = subprocess.run(
        ["/usr/bin/git", "cat-file", "-e", f"{candidate_hash}^{{commit}}"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_git_env(),
    )
    if candidate_check.returncode != 0:
        raise ValueError("manifest candidate_hash is not a repository commit")
    validate_manifest_lineage(
        manifest.get("lineage"),
        manifest["gate"],
        candidate_hash,
        relay_fetcher=_fetch_public_drand,
        commit_time_fetcher=lambda candidate: _git_commit_time(
            candidate, repository
        ),
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts map is missing")
    root = attempt.resolve()
    artifact_paths: dict[str, Path] = {}
    if manifest["gate"] == "W1":
        for artifact_name, field in (
            ("evidence", "evidence_sha256"),
            ("engine_source", "engine_source_sha256"),
            ("build_log", "build_log_sha256"),
        ):
            if not _is_sha256(manifest.get(field)):
                raise ValueError(f"manifest {field} is invalid")
            relative = artifacts.get(artifact_name)
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"manifest artifact {artifact_name} is missing")
            artifact = (attempt / relative).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError(
                    f"manifest artifact {artifact_name} escapes or is absent"
                )
            if _sha256(artifact) != manifest[field]:
                raise ValueError(
                    f"manifest artifact {artifact_name} hash mismatch"
                )
            artifact_paths[artifact_name] = artifact
    for field in required_hashes:
        if not _is_sha256(manifest.get(field)):
            raise ValueError(f"manifest {field} is invalid")
        artifact_name = field.removesuffix("_sha256")
        relative = artifacts.get(artifact_name)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"manifest artifact {artifact_name} is missing")
        artifact = (attempt / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"manifest artifact {artifact_name} escapes or is absent")
        if _sha256(artifact) != manifest[field]:
            raise ValueError(f"manifest artifact {artifact_name} hash mismatch")
        artifact_paths[artifact_name] = artifact
    if manifest["gate"] == "W1":
        with artifact_paths["binary"].open("rb") as executable:
            if executable.read(4) != b"\x7fELF":
                raise ValueError("W1 quality binary is not an ELF executable")
    validate_source_provenance(
        artifact_paths["source"],
        candidate_hash,
        repository=repository,
    )
    validate_profile_artifact_bindings(manifest, artifact_paths)
    raw_path = attempt / "raw.jsonl"
    try:
        lines = raw_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"invalid raw.jsonl: {exc}") from exc
    if not lines:
        raise ValueError("raw.jsonl is empty")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value {value}")
                ),
                object_pairs_hook=lambda pairs: _unique_pairs(
                    pairs, f"raw.jsonl line {number}"
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"raw.jsonl line {number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"raw.jsonl line {number} is not an object")
        records.append(record)
    if manifest["gate"] == "W1" and not root_authority_pending:
        if len(records) != 1:
            raise ValueError("W1 root receipt requires one raw record")
        validate_w1_root_receipt(attempt, records[0])
    validate_record_artifact_bindings(
        manifest["gate"], manifest, records, artifact_paths
    )
    summary = _read_strict_json(attempt / "summary.json")
    formula_version = (
        summary.get("formula_version") if isinstance(summary, dict) else None
    )
    if (
        not isinstance(formula_version, int)
        or isinstance(formula_version, bool)
        or formula_version < 1
    ):
        raise ValueError("summary has no fixed formula version")
    if summary.get("verdict") not in {"PASS", "FAIL", "NO_RESULT"}:
        raise ValueError("summary verdict is invalid")
    scorer_id = summary.get("scorer_id")
    if not isinstance(scorer_id, str) or not scorer_id:
        raise ValueError(
            f"no fixed terminal scorer is registered for {manifest['gate']}"
        )
    descriptor = _read_strict_json(artifact_paths["scorer"])
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "schema_version",
        "scorer_id",
        "implementation_sha256",
    }:
        raise ValueError("scorer artifact is not a strict descriptor")
    if descriptor.get("schema_version") != 1:
        raise ValueError("scorer descriptor schema is invalid")
    if descriptor.get("scorer_id") != scorer_id:
        raise ValueError("scorer descriptor ID does not match summary")
    if not scorer_descriptor_matches(
        manifest["gate"],
        candidate_hash,
        scorer_id,
        descriptor.get("implementation_sha256"),
    ):
        raise ValueError("scorer descriptor does not match fixed implementation")
    recomputed = score_registered_gate(manifest["gate"], scorer_id, records)
    if (
        manifest["gate"] == "review"
        or scorer_id == "parity.reviewed-no-go.v1"
    ) and recomputed.get("candidate_hash") != candidate_hash:
        raise ValueError("reviewed candidate does not match manifest candidate")
    if summary != recomputed:
        raise ValueError("summary does not exactly match fixed scorer output")


def _unique_pairs(pairs: list[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        result[key] = value
    return result


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "gates": {
            gate: {"status": "PENDING", "attempts": [], "reason": None}
            for gate in GATE_ORDER
        },
    }


def _validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != 1:
        raise GoalError("unsupported state schema")
    gates = state.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATE_ORDER):
        raise GoalError("state gate set is incomplete or unknown")
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") not in STATUSES:
            raise GoalError(f"{name}: invalid status")
        if not isinstance(gate.get("attempts"), list):
            raise GoalError(f"{name}: attempts is not a list")
        if gate.get("status") in TERMINAL_STATUSES and not gate["attempts"]:
            raise GoalError(f"{name}: terminal status has no evidence attempt")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _load_state(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "state.json"
    if not path.exists():
        state = _initial_state()
        _ingest_attempts(state_dir, state)
        _atomic_json(path, state)
        return state
    try:
        state = _read_strict_json(path)
    except (OSError, ValueError) as exc:
        raise GoalError(f"cannot read state: {exc}") from exc
    _validate_state(state)
    if _ingest_attempts(state_dir, state):
        _validate_state(state)
        state["updated_at"] = _utcnow()
        _atomic_json(path, state)
    return state


def _gate_status_from_summary(
    gate: str,
    summary: dict[str, Any],
    *,
    candidate_format: str | None = None,
) -> tuple[str, str | None]:
    """Map a valid scorer verdict to controller progress without overclaiming."""
    verdict = summary.get("verdict")
    reason = summary.get("reason")
    if (
        gate == "W1"
        and summary.get("scorer_id") == "w1.affine-quality.v2"
        and verdict == "PASS"
    ):
        if candidate_format == "affine-int8-block16":
            return (
                "RED_CONFIRMED",
                W1_PACKED_RETRIEVAL_REASON,
            )
        return (
            "RED_CONFIRMED",
            "affine fidelity diagnostic passed; real packed storage, memory, "
            "checkpoint, and retrieval qualification remain unfinished",
        )
    return verdict, reason


def _ingest_attempts(state_dir: Path, state: dict[str, Any]) -> bool:
    """Discover immutable attempt directories and ingest fixed verdicts."""
    def attempt_order(path: Path) -> tuple[int, int | str]:
        suffix = path.name.removeprefix("attempt-")
        if path.name.startswith("attempt-") and suffix.isdigit():
            return (0, int(suffix))
        return (1, path.name)

    changed = False
    for name in GATE_ORDER:
        gate_dir = (
            W1_AUTHORITY_ATTEMPT_ROOT
            if name == "W1"
            else state_dir / name
        )
        attempts = (
            sorted(
                (path for path in gate_dir.iterdir() if path.is_dir()),
                key=attempt_order,
            )
            if gate_dir.is_dir()
            else []
        )
        relative = [
            (
                str(path)
                if name == "W1"
                else str(path.relative_to(state_dir))
            )
            for path in attempts
        ]
        gate = state["gates"][name]
        if gate["attempts"] != relative:
            gate["attempts"] = relative
            changed = True
        if not attempts:
            if name == "W1" and gate["status"] in TERMINAL_STATUSES:
                gate["status"] = "PENDING"
                gate["reason"] = "awaiting root-authoritative W1 attempt"
                changed = True
                continue
            if gate["status"] in TERMINAL_STATUSES:
                raise GoalError(f"{name}: terminal evidence attempt disappeared")
            continue
        latest = attempts[-1]
        try:
            if not latest.name.startswith("attempt-") or not latest.name[8:].isdigit():
                raise ValueError("attempt directory name is invalid")
            manifest = _read_strict_json(latest / "manifest.json")
            if manifest.get("gate") != name:
                raise ValueError(
                    f"manifest gate {manifest.get('gate')!r} does not match {name!r}"
                )
            validate_attempt(latest)
            summary = _read_strict_json(latest / "summary.json")
            candidate_format = None
            if name == "W1":
                configuration = _read_strict_json(latest / "configuration.json")
                candidate_format = configuration.get("candidate_format")
            status, reason = _gate_status_from_summary(
                name,
                summary,
                candidate_format=candidate_format,
            )
        except ValueError as exc:
            # The anti-cheating contract makes malformed or unauthoritative
            # evidence a preserved terminal failure. A later, higher-numbered
            # immutable attempt may supersede it.
            status = "FAIL"
            reason = f"invalid evidence in {latest.name}: {exc}"
        if gate["status"] != status or gate.get("reason") != reason:
            gate["status"] = status
            gate["reason"] = reason
            changed = True
    return changed


def _parity_release_decision(
    state_dir: Path, state: dict[str, Any]
) -> str:
    gate = state["gates"]["parity"]
    if gate["status"] != "PASS" or not gate["attempts"]:
        return "UNPROVEN"
    attempt = (state_dir / gate["attempts"][-1]).resolve()
    if not attempt.is_relative_to(state_dir.resolve()):
        return "UNPROVEN"
    try:
        validate_attempt(attempt)
        summary = _read_strict_json(attempt / "summary.json")
    except (OSError, ValueError):
        return "UNPROVEN"
    if (
        summary.get("scorer_id") == "parity.performance.v1"
        and summary.get("verdict") == "PASS"
        and "decision" not in summary
    ):
        return "PASS"
    if (
        summary.get("scorer_id") == "parity.reviewed-no-go.v1"
        and summary.get("verdict") == "PASS"
        and summary.get("decision") == "NO_GO"
    ):
        return "NO_GO"
    return "UNPROVEN"


def _release_verdict(state_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    gates = state["gates"]
    if gates["foundation"]["status"] != "PASS":
        failed.append("foundation")
    for name in (f"W{index}" for index in range(1, 12)):
        if gates[name]["status"] not in TERMINAL_STATUSES:
            failed.append(name)
    if gates["W11"]["status"] != "PASS":
        failed.append("W11")
    for name in ("switch", "review"):
        if gates[name]["status"] != "PASS":
            failed.append(name)
    parity_decision = _parity_release_decision(state_dir, state)
    if parity_decision not in {"PASS", "NO_GO"}:
        failed.append("parity")
    unique_failed = list(dict.fromkeys(failed))
    return {
        "schema_version": 1,
        "release_qualified": not unique_failed,
        "failed_requirements": unique_failed,
        "parity_decision": parity_decision,
    }


def _selected_gate(state: dict[str, Any]) -> str | None:
    if state["gates"]["foundation"]["status"] != "PASS":
        return "foundation"

    # The packed W1 campaign established storage, memory and fidelity, but its
    # 1M retrieval gate needs exact NVMe cKV staging. Route directly through
    # that dependency and the 1M qualification instead of repeating the same
    # nonterminal 20-arm campaign or spending time on lower-value gates.
    w1 = state["gates"]["W1"]
    if (
        w1["status"] == "RED_CONFIRMED"
        and w1.get("reason") == W1_PACKED_RETRIEVAL_REASON
    ):
        if state["gates"]["W8"]["status"] not in TERMINAL_STATUSES:
            return "W8"
        if state["gates"]["W11"]["status"] != "PASS":
            return "W11"
        return "W1"

    required_pass = {"foundation", "W11", "switch", "parity", "review"}
    for name in GATE_ORDER:
        status = state["gates"][name]["status"]
        if name in required_pass:
            if status != "PASS":
                return name
        elif status not in TERMINAL_STATUSES:
            return name
    return None


def _registered_default_runner(
    gate: str,
    *,
    root: Path = ROOT,
    registry: dict[str, str] | None = None,
) -> int | None:
    approved = DEFAULT_RUNNER_SHA256 if registry is None else registry
    expected = approved.get(gate)
    if expected is None:
        return None
    path = root / "scripts" / "glm52-runners" / gate
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        return None
    if not os.access(path, os.X_OK) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return None

    source = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    sealed = os.memfd_create(
        f"glm52-runner-{gate}", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    )
    try:
        opened = os.fstat(source)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o111 == 0
            or (opened.st_dev, opened.st_ino)
            != (details.st_dev, details.st_ino)
        ):
            return None
        digest = hashlib.sha256()
        copied = 0
        while block := os.read(source, 1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(sealed, view)
                if written <= 0:
                    raise OSError("registered runner memfd write made no progress")
                copied += written
                view = view[written:]
        if digest.hexdigest() != expected:
            return None
        if os.fstat(sealed).st_size != copied:
            return None
        sealed_digest = hashlib.sha256()
        offset = 0
        while block := os.pread(sealed, 1024 * 1024, offset):
            sealed_digest.update(block)
            offset += len(block)
        if offset != copied or sealed_digest.hexdigest() != expected:
            return None
        os.fchmod(sealed, 0o500)
        fcntl.fcntl(
            sealed,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE,
        )
        os.lseek(sealed, 0, os.SEEK_SET)
        result = sealed
        sealed = -1
        return result
    finally:
        os.close(source)
        if sealed >= 0:
            os.close(sealed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dispatch(state_dir: Path, command: str) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "controller.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        while True:
            state = _load_state(state_dir)
            selected = _selected_gate(state)
            if selected is None:
                event = {
                    "command": command,
                    "selected_gate": None,
                    "time": _utcnow(),
                    "action": "terminal_not_release_qualified",
                }
                break
            runner: Path | None = None
            if state_dir.resolve() != DEFAULT_STATE_DIR.resolve():
                candidate = state_dir / "runners" / selected
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    runner = candidate
            runner_descriptor = (
                None if runner is not None else _registered_default_runner(selected)
            )
            if runner is None and runner_descriptor is None:
                event = {
                    "command": command,
                    "selected_gate": selected,
                    "time": _utcnow(),
                    "action": "awaiting_registered_runner",
                }
                break
            executable = (
                str(runner)
                if runner is not None
                else f"/proc/self/fd/{runner_descriptor}"
            )
            try:
                completed = subprocess.run(
                    [executable, str(state_dir), selected],
                    cwd=ROOT,
                    check=False,
                    timeout=14_400,
                    pass_fds=(runner_descriptor,) if runner_descriptor is not None else (),
                    env={
                        "HOME": os.environ.get("HOME", ""),
                        "PATH": os.environ.get("PATH", ""),
                        "LANG": os.environ.get("LANG", "C.UTF-8"),
                    },
                )
            finally:
                if runner_descriptor is not None:
                    os.close(runner_descriptor)
            if completed.returncode != 0:
                event = {
                    "command": command,
                    "selected_gate": selected,
                    "time": _utcnow(),
                    "action": "runner_failed",
                    "runner_returncode": completed.returncode,
                }
                break
            refreshed = _load_state(state_dir)
            if _selected_gate(refreshed) == selected:
                event = {
                    "command": command,
                    "selected_gate": selected,
                    "time": _utcnow(),
                    "action": "runner_produced_no_terminal_evidence",
                }
                break
    events = state_dir / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("resume")
    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    release = subparsers.add_parser("release-check")
    release.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = _load_state(args.state_dir)
        if args.command == "status":
            if args.json:
                print(json.dumps(state, sort_keys=True, allow_nan=False))
            else:
                selected = _selected_gate(state)
                print(f"next={selected or 'none'}")
                for name in GATE_ORDER:
                    print(f"{name}: {state['gates'][name]['status']}")
            return 0
        if args.command == "release-check":
            verdict = _release_verdict(args.state_dir, state)
            if args.json:
                print(json.dumps(verdict, sort_keys=True, allow_nan=False))
            else:
                print(
                    "qualified"
                    if verdict["release_qualified"]
                    else "not qualified: "
                    + ",".join(verdict["failed_requirements"])
                )
            return 0 if verdict["release_qualified"] else 1
        print(
            json.dumps(
                _dispatch(args.state_dir, args.command),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (GoalError, ValueError, OSError) as exc:
        print(f"glm52_goal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
