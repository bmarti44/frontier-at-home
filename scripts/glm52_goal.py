#!/usr/bin/env python3
"""Fail-closed controller and fixed acceptance formulas for the GLM-5.2 goal.

The controller deliberately does not infer PASS from prose or engine logs.  A
gate can advance only through a registered runner which writes a manifest,
raw.jsonl and summary.json under the immutable attempt directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / "results" / "glm52-goal"
STATUSES = frozenset({"PENDING", "RED_CONFIRMED", "PASS", "FAIL", "NO_RESULT"})
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "NO_RESULT"})
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
    timestamps = [float(value) for value in token_timestamps]
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
    memory = float(observation["available_memory_gib"])
    if not math.isfinite(memory):
        raise ValueError("available memory is non-finite")
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
    seconds = float(record.get("prefill_seconds", math.nan))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("prefill_seconds must be finite and positive")
    if record.get("failures") != []:
        raise ValueError("measurement record contains failures")


def validate_ab_blocks(records: Iterable[dict[str, Any]]) -> None:
    """Require five fresh-server ABBA/BAAB blocks with equal fixtures."""
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
        expected = "ABBA" if block % 2 == 0 else "BAAB"
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
    result = float(value)
    if not math.isfinite(result) or result <= minimum:
        raise ValueError(f"{label} must be finite and greater than {minimum}")
    return result


def _score_w11(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError("W11 requires exactly one context observation")
    expected = {
        "record_type",
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
    observation = records[0]
    _require_exact_keys(observation, expected, "W11 context observation")
    if observation["record_type"] != "context_observation":
        raise ValueError("W11 record_type is invalid")
    result = context_verdict(observation)
    return {"scorer_id": "w11.context.v1", **result}


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
            severity: _review_issue_ids(record[severity], severity)
            for severity in ("critical", "high", "medium", "low")
        }
        flattened = [issue for values in issues.values() for issue in values]
        if len(set(flattened)) != len(flattened):
            raise ValueError("one review issue appears at multiple severities")
        expected_score = max(
            0,
            100
            - 25 * len(issues["critical"])
            - 10 * len(issues["high"])
            - 3 * len(issues["medium"])
            - len(issues["low"]),
        )
        claimed = record["claimed_score"]
        if (
            not isinstance(claimed, int)
            or isinstance(claimed, bool)
            or claimed != expected_score
        ):
            raise ValueError("claimed reviewer score does not match the rubric")
        expected_verdict = (
            "ACCEPT"
            if expected_score >= 90
            and not issues["critical"]
            and not issues["high"]
            else "REJECT"
        )
        if record["verdict"] != expected_verdict:
            raise ValueError("reviewer verdict does not match score and issues")
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
        scores[reviewer] = expected_score
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
        "both_scores_at_least_90": all(score >= 90 for score in scores.values()),
        "no_critical": all(value["critical"] == 0 for value in counts.values()),
        "no_high": all(value["high"] == 0 for value in counts.values()),
    }
    return {
        "scorer_id": "review.final.v1",
        "formula_version": 1,
        "candidate_hash": next(iter(candidate_hashes)),
        "review_round": next(iter(rounds)),
        "scores": scores,
        "issue_counts": counts,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def score_registered_gate(
    gate: str, scorer_id: str, records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute an authoritative terminal verdict from strict raw records."""
    rows = list(records)
    registered = {
        ("foundation", "foundation.v1"): _score_foundation,
        ("W11", "w11.context.v1"): _score_w11,
        ("parity", "parity.performance.v1"): _score_parity,
        ("review", "review.final.v1"): _score_review,
    }
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


def validate_attempt(attempt: Path) -> None:
    """Validate the mandatory evidence triplet without trusting narration."""
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
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts map is missing")
    root = attempt.resolve()
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
    summary = _read_strict_json(attempt / "summary.json")
    if not isinstance(summary, dict) or summary.get("formula_version") != 1:
        raise ValueError("summary has no fixed formula version")
    if summary.get("verdict") not in {"PASS", "FAIL", "NO_RESULT"}:
        raise ValueError("summary verdict is invalid")
    scorer_id = summary.get("scorer_id")
    if not isinstance(scorer_id, str) or not scorer_id:
        raise ValueError(
            f"no fixed terminal scorer is registered for {manifest['gate']}"
        )
    if manifest["scorer_sha256"] != _sha256(Path(__file__)):
        raise ValueError("manifest scorer is not the executing fixed scorer")
    recomputed = score_registered_gate(manifest["gate"], scorer_id, records)
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


def _ingest_attempts(state_dir: Path, state: dict[str, Any]) -> bool:
    """Discover immutable attempt directories and ingest fixed verdicts."""
    changed = False
    for name in GATE_ORDER:
        gate_dir = state_dir / name
        attempts = (
            sorted(path for path in gate_dir.iterdir() if path.is_dir())
            if gate_dir.is_dir()
            else []
        )
        relative = [str(path.relative_to(state_dir)) for path in attempts]
        gate = state["gates"][name]
        if gate["attempts"] != relative:
            gate["attempts"] = relative
            changed = True
        if not attempts:
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
            status = summary["verdict"]
            reason = summary.get("reason")
        except ValueError as exc:
            # Invalid or unauthoritative evidence must not terminalize a gate.
            # Keeping it PENDING lets a registered runner supersede it with a
            # later immutable attempt.
            status = "PENDING"
            reason = f"invalid evidence in {latest.name}: {exc}"
        if gate["status"] != status or gate.get("reason") != reason:
            gate["status"] = status
            gate["reason"] = reason
            changed = True
    return changed


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
    parity_ok = gates["parity"]["status"] == "PASS"
    # A summary boolean is not reviewer authority. Keep NO_GO disabled until a
    # registered scorer recomputes the physical bound and validates immutable
    # attestations from both persistent reviewers.
    parity_no_go = False
    if not (parity_ok or parity_no_go):
        failed.append("parity")
    unique_failed = list(dict.fromkeys(failed))
    return {
        "schema_version": 1,
        "release_qualified": not unique_failed,
        "failed_requirements": unique_failed,
        "parity_decision": (
            "PASS" if parity_ok else "NO_GO" if parity_no_go else "UNPROVEN"
        ),
    }


def _selected_gate(state: dict[str, Any]) -> str | None:
    for name in GATE_ORDER:
        if state["gates"][name]["status"] not in TERMINAL_STATUSES:
            return name
    return None


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
            candidates = [ROOT / "scripts" / "glm52-runners" / selected]
            if state_dir.resolve() != DEFAULT_STATE_DIR.resolve():
                candidates.insert(0, state_dir / "runners" / selected)
            runner = next(
                (
                    path
                    for path in candidates
                    if path.is_file() and os.access(path, os.X_OK)
                ),
                None,
            )
            if runner is None:
                event = {
                    "command": command,
                    "selected_gate": selected,
                    "time": _utcnow(),
                    "action": "awaiting_registered_runner",
                }
                break
            completed = subprocess.run(
                [str(runner), str(state_dir), selected],
                cwd=ROOT,
                check=False,
                timeout=14_400,
                env={
                    "HOME": os.environ.get("HOME", ""),
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                },
            )
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
