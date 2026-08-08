#!/usr/bin/env python3
"""Fixed, fail-closed scorer for the preregistered W7.1 matched campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from typing import Any


T_ONE_SIDED_95_DF4 = 2.131846786326649
MIN_MEMORY_KB = 10 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FORMULA = (
    "five fresh-server ABBA/BAAB blocks; byte-identical outputs/logits; "
    "decode=(N-1)/(tN-t1) over >=128 output-token timestamps; block ratios use "
    "the mean of two ON arms divided by the mean of two OFF arms; one-sided "
    "95% log-ratio t bounds with df=4; TTFT upper bound <=0.95 and decode "
    "lower bound >=1.00"
)


class InvalidCampaign(ValueError):
    """Raised internally when an input violates the frozen evidence contract."""


def _exact_int(value: object) -> bool:
    return type(value) is int


def _sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCampaign(message)


def _ratio_bound(ratios: list[float], *, upper: bool) -> float:
    _require(len(ratios) == 5, "confidence bounds require exactly five blocks")
    _require(all(math.isfinite(value) and value > 0.0 for value in ratios), "invalid ratio")
    logs = [math.log(value) for value in ratios]
    center = statistics.fmean(logs)
    margin = T_ONE_SIDED_95_DF4 * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(center + margin if upper else center - margin)


def _fail(message: str) -> dict[str, object]:
    return {
        "formula": FORMULA,
        "checks": {"input_and_acceptance_checks": False},
        "observed": {},
        "failure": message,
        "verdict": "FAIL",
    }


def _validated_row(row: object) -> dict[str, Any]:
    _require(isinstance(row, dict), "row is not an object")
    required = {
        "block", "position", "arm", "run_id", "binary_sha256", "model_sha256",
        "common_config_sha256", "request_sha256", "stable_remap", "request_start_ns",
        "token_timestamps_ns", "output_token_ids", "output_sha256",
        "generated_text_sha256", "generated_text_bytes",
        "final_logits_sha256", "logit_sequence_sha256", "server_fresh", "safety",
    }
    _require(set(row) == required, "row keys do not match the frozen schema")
    _require(_exact_int(row["block"]) and 0 <= row["block"] < 5, "invalid block")
    _require(_exact_int(row["position"]) and 0 <= row["position"] < 4, "invalid position")
    _require(row["arm"] in {"off", "on"}, "invalid arm")
    _require(isinstance(row["run_id"], str) and bool(row["run_id"]), "invalid run id")
    for name in (
        "binary_sha256", "model_sha256", "common_config_sha256", "request_sha256",
        "output_sha256", "final_logits_sha256", "logit_sequence_sha256",
        "generated_text_sha256",
    ):
        _require(_sha256(row[name]), f"invalid {name}")
    _require(_exact_int(row["stable_remap"]) and row["stable_remap"] in {0, 1}, "invalid flag")
    _require(row["stable_remap"] == (1 if row["arm"] == "on" else 0), "flag/arm mismatch")
    _require(_exact_int(row["request_start_ns"]) and row["request_start_ns"] >= 0, "invalid start")

    timestamps = row["token_timestamps_ns"]
    token_ids = row["output_token_ids"]
    _require(isinstance(timestamps, list) and len(timestamps) >= 128, "short timestamp series")
    _require(isinstance(token_ids, list) and len(token_ids) == len(timestamps), "token/timing mismatch")
    _require(all(_exact_int(value) and value >= 0 for value in timestamps), "invalid timestamp")
    _require(all(_exact_int(value) and value >= 0 for value in token_ids), "invalid token id")
    token_bytes = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    _require(
        hashlib.sha256(token_bytes).hexdigest() == row["output_sha256"],
        "output digest does not bind token IDs",
    )
    _require(timestamps[0] > row["request_start_ns"], "nonpositive TTFT")
    _require(all(right > left for left, right in zip(timestamps, timestamps[1:])), "timestamps not strict")
    _require(row["server_fresh"] is True, "server was not fresh")
    _require(
        _exact_int(row["generated_text_bytes"]) and row["generated_text_bytes"] >= 0,
        "invalid generated text size",
    )

    safety = row["safety"]
    safety_keys = {
        "containment_rc", "minimum_mem_available_kb", "swap_growth_bytes",
        "cgroup_max_delta", "cgroup_oom_delta", "cgroup_oom_kill_delta", "xid_count",
        "surviving_descendants", "false_generation_flushes",
    }
    _require(isinstance(safety, dict) and set(safety) == safety_keys, "invalid safety schema")
    _require(all(_exact_int(safety[name]) for name in safety_keys), "non-integer safety value")
    for name in (
        "containment_rc", "swap_growth_bytes", "cgroup_max_delta", "cgroup_oom_delta",
        "cgroup_oom_kill_delta", "xid_count", "surviving_descendants",
    ):
        _require(safety[name] == 0, f"unsafe {name}")
    _require(safety["minimum_mem_available_kb"] >= MIN_MEMORY_KB, "memory floor violated")
    _require(safety["false_generation_flushes"] >= 0, "invalid flush count")
    if row["arm"] == "on":
        _require(safety["false_generation_flushes"] == 0, "ON arm false flush")
    else:
        _require(safety["false_generation_flushes"] > 0, "OFF control did not reproduce defect")
    return row


def score_campaign_rows(rows: object, schedules: object) -> dict[str, object]:
    """Score raw campaign rows; malformed or incomplete evidence always fails."""
    try:
        _require(isinstance(schedules, list) and len(schedules) == 5, "need five schedules")
        _require(all(schedule in {"ABBA", "BAAB"} for schedule in schedules), "invalid schedule")
        _require(isinstance(rows, list) and len(rows) == 20, "need exactly twenty rows")
        validated = [_validated_row(row) for row in rows]

        by_slot: dict[tuple[int, int], dict[str, Any]] = {}
        for row in validated:
            slot = (row["block"], row["position"])
            _require(slot not in by_slot, "duplicate block position")
            by_slot[slot] = row
        _require(len(by_slot) == 20, "incomplete block positions")
        for block, schedule in enumerate(schedules):
            for position, letter in enumerate(schedule):
                expected_arm = "off" if letter == "A" else "on"
                _require(by_slot[(block, position)]["arm"] == expected_arm, "arm schedule mismatch")

        _require(len({row["run_id"] for row in validated}) == 20, "duplicate run id")
        for field in ("binary_sha256", "model_sha256", "common_config_sha256", "request_sha256"):
            _require(len({row[field] for row in validated}) == 1, f"unequal {field}")
        for field in (
            "output_token_ids", "output_sha256", "final_logits_sha256",
            "logit_sequence_sha256", "generated_text_sha256", "generated_text_bytes",
        ):
            first = validated[0][field]
            _require(all(row[field] == first for row in validated[1:]), f"unequal {field}")

        row_metrics: dict[str, dict[str, float]] = {}
        for row in validated:
            timestamps = row["token_timestamps_ns"]
            ttft_s = (timestamps[0] - row["request_start_ns"]) / 1_000_000_000.0
            decode_tps = (len(timestamps) - 1) * 1_000_000_000.0 / (timestamps[-1] - timestamps[0])
            _require(math.isfinite(ttft_s) and ttft_s > 0.0, "invalid TTFT")
            _require(math.isfinite(decode_tps) and decode_tps > 0.0, "invalid decode")
            row_metrics[row["run_id"]] = {"ttft_s": ttft_s, "decode_tps": decode_tps}

        ttft_ratios: list[float] = []
        decode_ratios: list[float] = []
        for block in range(5):
            block_rows = [by_slot[(block, position)] for position in range(4)]
            off = [row_metrics[row["run_id"]] for row in block_rows if row["arm"] == "off"]
            on = [row_metrics[row["run_id"]] for row in block_rows if row["arm"] == "on"]
            _require(len(off) == len(on) == 2, "unbalanced block")
            ttft_ratios.append(
                statistics.fmean(item["ttft_s"] for item in on)
                / statistics.fmean(item["ttft_s"] for item in off)
            )
            decode_ratios.append(
                statistics.fmean(item["decode_tps"] for item in on)
                / statistics.fmean(item["decode_tps"] for item in off)
            )

        ttft_upper = _ratio_bound(ttft_ratios, upper=True)
        decode_lower = _ratio_bound(decode_ratios, upper=False)
        checks = {
            "schema_and_schedule_valid": True,
            "fresh_servers_and_unique_runs": True,
            "equal_fixtures_and_frozen_identity": True,
            "byte_identical_output_tokens": True,
            "byte_identical_output_and_final_logits": True,
            "byte_identical_full_logit_sequence": True,
            "at_least_128_output_timestamps": True,
            "safety_and_memory_floor": True,
            "off_control_reproduced_false_flushes": True,
            "on_false_generation_flushes_zero": True,
            "ttft_ratio_upper_95_le_0_95": ttft_upper <= 0.95,
            "decode_ratio_lower_95_ge_1_00": decode_lower >= 1.0,
        }
        observed = {
            "ttft_block_ratios": ttft_ratios,
            "decode_block_ratios": decode_ratios,
            "ttft_ratio_geometric_mean": math.exp(statistics.fmean(map(math.log, ttft_ratios))),
            "decode_ratio_geometric_mean": math.exp(statistics.fmean(map(math.log, decode_ratios))),
            "ttft_ratio_upper_95": ttft_upper,
            "decode_ratio_lower_95": decode_lower,
            "minimum_mem_available_kb": min(row["safety"]["minimum_mem_available_kb"] for row in validated),
            "output_tokens_per_run": len(validated[0]["output_token_ids"]),
            "logit_max_abs_delta": 0.0,
            "logit_argmax_equal": True,
            "blocks": 5,
            "runs": 20,
        }
        return {
            "formula": FORMULA,
            "checks": checks,
            "observed": observed,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    except (InvalidCampaign, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        return _fail(f"{type(error).__name__}: {error}")
