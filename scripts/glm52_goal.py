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
import inspect
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
    expected_caps = [131_072, 262_144, 524_288, 1_048_576]
    minimum_processed = [130_000, 260_000, 520_000, 1_000_000]
    if not isinstance(stages, list) or len(stages) != len(expected_caps):
        raise ValueError("W11 requires exactly four graduated stages")
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
        if stage["finish_reason"] not in {"stop", "length"}:
            raise ValueError("W11 finish reason is invalid")
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
    for index, sample in enumerate(memory_samples):
        _require_exact_keys(
            sample,
            {"timestamp_seconds", "available_gib"},
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
        "graduated_stages": all(
            actual >= minimum
            for actual, minimum in zip(processed, minimum_processed)
        )
        and all(right > left for left, right in zip(processed, processed[1:])),
        "retrieval": retrieval_pass,
        "retrieval_position_coverage": position_coverage,
        "negative_control": negative_pass,
        "completed_generation": generations_complete,
        "no_truncation": no_truncation,
        "no_failures": not observation["failure_events"],
        "no_oom": not observation["oom_events"],
        "no_xid": not observation["xid_events"],
        "memory_floor": min(memory) >= 10.0,
    }
    return {
        "scorer_id": "w11.context.v1",
        "formula_version": 1,
        "measurements": {
            "stage_context_caps": expected_caps,
            "stage_processed_tokens": processed,
            "retrieval_cases": len(retrieval),
            "negative_control_cases": len(negative),
            "memory_samples": len(memory),
            "minimum_available_memory_gib": min(memory),
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
                "w11-fixture.v1:"
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
        "generator_version": "w11-fixture.v1",
        "context_cap": 1_048_576,
        "stage_context_caps": [131_072, 262_144, 524_288, 1_048_576],
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
        for field in (
            "binary_sha256",
            "configuration_sha256",
            "fixture_sha256",
        ):
            if records[0].get(field) != manifest.get(field):
                label = field.removesuffix("_sha256").replace("_", " ")
                raise ValueError(
                    f"{gate} raw {label} identity does not match manifest"
                )
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
    if fixture["generator_version"] != "w11-fixture.v1":
        raise ValueError("W11 fixture generator is not registered")
    expected_caps = [131_072, 262_144, 524_288, 1_048_576]
    if fixture["stage_context_caps"] != expected_caps:
        raise ValueError("W11 fixture stage graduation is invalid")
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
        generate_w11_fixture,
        _load_approved_dsv4_profile,
        _read_strict_json,
        _unique_pairs,
        _sha256,
    )
    digest = hashlib.sha256()
    digest.update(f"scorer_id={scorer_id}\n".encode())
    for function in functions + trust_boundary:
        digest.update(f"function={function.__name__}\n".encode())
        digest.update(inspect.getsource(function).encode())
    if scorer_id == "parity.performance.v1":
        digest.update(
            json.dumps(_T95, sort_keys=True, separators=(",", ":")).encode()
        )
    return digest.hexdigest()


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
    _require_exact_keys(
        freeze, {"candidate_hash", "frozen_at"}, "freeze lineage"
    )
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
        if frozen_at != committed_at:
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
    expected_seed = hashlib.sha256(
        f"{candidate_hash}:{expected_randomness}:{gate}".encode()
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


def _git_commit_time(candidate_hash: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cI", candidate_hash],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
        },
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


def _load_approved_dsv4_profile(candidate_hash: Any) -> dict[str, str]:
    """Load the exact DeepSeek reference identity frozen in the candidate."""
    if not (
        isinstance(candidate_hash, str)
        and len(candidate_hash) == 40
        and all(char in "0123456789abcdef" for char in candidate_hash)
    ):
        raise ValueError("approved DeepSeek profile candidate is invalid")
    result = subprocess.run(
        ["git", "show", f"{candidate_hash}:configs/dsv4-profile.json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
        },
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
        },
        "approved DeepSeek profile",
    )
    if profile["schema_version"] != 1 or profile["profile"] != "dsv4":
        raise ValueError("approved DeepSeek profile identity is invalid")
    for field in ("binary_sha256", "configuration_sha256"):
        if not _is_sha256(profile[field]):
            raise ValueError(f"approved DeepSeek profile {field} is invalid")
    return profile


def validate_source_provenance(source_path: Path, candidate_hash: str) -> None:
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
        ["git", "rev-parse", f"{candidate_hash}^{{tree}}"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
        },
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
        ["git", "show", f"{candidate_hash}:configs/glm52-profile.json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
        },
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
        {"profile", "binary_sha256", "model_sha256", "context_cap"},
        "approved GLM profile",
    )
    if profile["profile"] != "glm52" or profile["context_cap"] != 1_048_576:
        raise ValueError("approved GLM profile identity or context cap is invalid")
    for field in ("binary_sha256", "model_sha256"):
        if not _is_sha256(profile[field]) or manifest.get(field) != profile[field]:
            raise ValueError(
                f"manifest {field} does not match approved GLM profile"
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


def validate_attempt(attempt: Path) -> None:
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
    candidate_check = subprocess.run(
        ["git", "cat-file", "-e", f"{candidate_hash}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
        },
    )
    if candidate_check.returncode != 0:
        raise ValueError("manifest candidate_hash is not a repository commit")
    validate_manifest_lineage(
        manifest.get("lineage"),
        manifest["gate"],
        candidate_hash,
        relay_fetcher=_fetch_public_drand,
        commit_time_fetcher=_git_commit_time,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts map is missing")
    root = attempt.resolve()
    artifact_paths: dict[str, Path] = {}
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
    validate_source_provenance(artifact_paths["source"], candidate_hash)
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
    validate_record_artifact_bindings(
        manifest["gate"], manifest, records, artifact_paths
    )
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
    if descriptor.get("implementation_sha256") != registered_scorer_digest(
        scorer_id
    ):
        raise ValueError("scorer descriptor does not match fixed implementation")
    recomputed = score_registered_gate(manifest["gate"], scorer_id, records)
    if (
        manifest["gate"] == "review"
        and recomputed.get("candidate_hash") != candidate_hash
    ):
        raise ValueError("review candidate does not match manifest candidate")
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
    def attempt_order(path: Path) -> tuple[int, int | str]:
        suffix = path.name.removeprefix("attempt-")
        if path.name.startswith("attempt-") and suffix.isdigit():
            return (0, int(suffix))
        return (1, path.name)

    changed = False
    for name in GATE_ORDER:
        gate_dir = state_dir / name
        attempts = (
            sorted(
                (path for path in gate_dir.iterdir() if path.is_dir()),
                key=attempt_order,
            )
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
