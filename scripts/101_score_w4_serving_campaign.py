#!/usr/bin/env python3
"""Fail-closed scorer for W4's matched serving-prefill confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


T_ONE_SIDED_95_DF4 = 2.131846786326649
MIN_PROMPT_TOKENS = 16_000
MIN_MEMORY_KIB = 10 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FORMULA = (
    "five fresh-server ABBA/BAAB blocks; same novel >=16000-token request; "
    "external prefill time=response_complete_ns-request_start_ns with max_tokens=0 "
    "and cache_write_tokens=prompt_tokens; block speedup=mean(OFF seconds)/mean(ON "
    "seconds); one-sided 95% lower log-ratio t bound with df=4 >=1.05; exact "
    "semantic response and final logits; independently replayed W4 CUDA top-k "
    "lower-95 speedup >=2.0 with exact selected IDs"
)


class InvalidCampaign(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCampaign(message)


def _integer(value: object) -> bool:
    return type(value) is int


def _sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _ratio_lower95(ratios: list[float]) -> float:
    _require(len(ratios) == 5, "confidence bound requires five blocks")
    _require(all(math.isfinite(value) and value > 0 for value in ratios),
             "invalid ratio")
    logs = [math.log(value) for value in ratios]
    return math.exp(
        statistics.fmean(logs)
        - T_ONE_SIDED_95_DF4 * statistics.stdev(logs) / math.sqrt(5)
    )


def _validated_row(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "row is not an object")
    row = value
    required = {
        "block", "position", "arm", "run_id", "binary_sha256",
        "model_sha256", "common_config_sha256", "request_sha256",
        "topk_cub", "request_start_ns", "response_complete_ns",
        "prompt_tokens", "cached_tokens", "cache_write_tokens",
        "response_semantic_sha256", "final_logits_sha256",
        "logit_sequence_sha256", "topk_marker_count", "server_fresh", "safety",
    }
    _require(set(row) == required, "row keys differ")
    _require(_integer(row["block"]) and 0 <= row["block"] < 5, "invalid block")
    _require(_integer(row["position"]) and 0 <= row["position"] < 4,
             "invalid position")
    _require(row["arm"] in {"off", "on"}, "invalid arm")
    _require(isinstance(row["run_id"], str) and bool(row["run_id"]),
             "invalid run id")
    for name in (
        "binary_sha256", "model_sha256", "common_config_sha256",
        "request_sha256", "response_semantic_sha256", "final_logits_sha256",
        "logit_sequence_sha256",
    ):
        _require(_sha256(row[name]), f"invalid {name}")
    expected_flag = 1 if row["arm"] == "on" else 0
    _require(_integer(row["topk_cub"]) and row["topk_cub"] == expected_flag,
             "flag/arm mismatch")
    _require(_integer(row["topk_marker_count"]) and
             row["topk_marker_count"] == expected_flag,
             "effective marker mismatch")
    for name in (
        "request_start_ns", "response_complete_ns", "prompt_tokens",
        "cached_tokens", "cache_write_tokens",
    ):
        _require(_integer(row[name]) and row[name] >= 0, f"invalid {name}")
    _require(row["response_complete_ns"] > row["request_start_ns"],
             "nonpositive external prefill time")
    _require(row["prompt_tokens"] >= MIN_PROMPT_TOKENS, "prompt too short")
    _require(row["cached_tokens"] == 0, "request was not novel")
    _require(row["cache_write_tokens"] == row["prompt_tokens"],
             "evaluated-token accounting differs")
    _require(row["server_fresh"] is True, "server was not fresh")

    safety = row["safety"]
    safety_keys = {
        "containment_rc", "minimum_mem_available_kib", "swap_growth_bytes",
        "cgroup_max_delta", "cgroup_oom_delta", "cgroup_oom_kill_delta",
        "xid_count", "surviving_descendants",
    }
    _require(isinstance(safety, dict) and set(safety) == safety_keys,
             "invalid safety schema")
    _require(all(_integer(safety[name]) for name in safety_keys),
             "non-integer safety value")
    for name in safety_keys - {"minimum_mem_available_kib"}:
        _require(safety[name] == 0, f"unsafe {name}")
    _require(safety["minimum_mem_available_kib"] >= MIN_MEMORY_KIB,
             "memory floor violated")
    return row


def _validate_microgate(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "microgate is not an object")
    required = {
        "block_a_ms", "block_b_ms", "selected_ids_sha256",
        "speedup_lower_95", "required_speedup_lower_95", "verdict",
    }
    _require(required.issubset(value), "microgate fields missing")
    a = value["block_a_ms"]
    b = value["block_b_ms"]
    _require(isinstance(a, list) and isinstance(b, list) and
             len(a) == len(b) == 5, "microgate samples differ")
    _require(all(type(item) in {int, float} and math.isfinite(float(item)) and
                 float(item) > 0 for item in a + b), "invalid microgate timing")
    recomputed = _ratio_lower95([float(left) / float(right)
                                 for left, right in zip(a, b)])
    _require(type(value["speedup_lower_95"]) in {int, float} and
             math.isclose(float(value["speedup_lower_95"]), recomputed,
                          rel_tol=1e-12, abs_tol=1e-12),
             "microgate speedup does not replay")
    _require(_sha256(value["selected_ids_sha256"]), "invalid selected IDs digest")
    _require(value["required_speedup_lower_95"] == 2.0 and
             value["verdict"] == "PASS", "microgate did not pass")
    return {"topk_speedup_lower_95": recomputed,
            "selected_ids_sha256": value["selected_ids_sha256"]}


def _fail(message: str) -> dict[str, object]:
    return {"schema": "glm52-w4-serving-summary-v1", "formula": FORMULA,
            "checks": {"input_and_acceptance_checks": False},
            "observed": {}, "failure": message, "verdict": "FAIL"}


def score_campaign_rows(rows: object, schedules: object,
                        microgate: object) -> dict[str, object]:
    try:
        _require(isinstance(schedules, list) and len(schedules) == 5,
                 "need five schedules")
        _require(all(item in {"ABBA", "BAAB"} for item in schedules),
                 "invalid schedule")
        _require(isinstance(rows, list) and len(rows) == 20,
                 "need exactly twenty rows")
        validated = [_validated_row(row) for row in rows]
        micro = _validate_microgate(microgate)

        slots: dict[tuple[int, int], dict[str, Any]] = {}
        for row in validated:
            slot = (row["block"], row["position"])
            _require(slot not in slots, "duplicate block position")
            slots[slot] = row
        for block, schedule in enumerate(schedules):
            for position, letter in enumerate(schedule):
                expected = "off" if letter == "A" else "on"
                _require(slots.get((block, position), {}).get("arm") == expected,
                         "arm schedule mismatch")
        _require(len({row["run_id"] for row in validated}) == 20,
                 "duplicate run id")
        for name in (
            "binary_sha256", "model_sha256", "common_config_sha256",
            "request_sha256", "prompt_tokens", "response_semantic_sha256",
            "final_logits_sha256", "logit_sequence_sha256",
        ):
            _require(len({row[name] for row in validated}) == 1,
                     f"unequal {name}")

        seconds = {
            row["run_id"]:
            (row["response_complete_ns"] - row["request_start_ns"]) / 1e9
            for row in validated
        }
        block_ratios: list[float] = []
        off_seconds: list[float] = []
        on_seconds: list[float] = []
        for block in range(5):
            block_rows = [slots[(block, position)] for position in range(4)]
            off = [seconds[row["run_id"]] for row in block_rows
                   if row["arm"] == "off"]
            on = [seconds[row["run_id"]] for row in block_rows
                  if row["arm"] == "on"]
            _require(len(off) == len(on) == 2, "unbalanced block")
            off_mean = statistics.fmean(off)
            on_mean = statistics.fmean(on)
            off_seconds.append(off_mean)
            on_seconds.append(on_mean)
            block_ratios.append(off_mean / on_mean)
        prefill_lower95 = _ratio_lower95(block_ratios)
        prompt_tokens = validated[0]["prompt_tokens"]
        checks = {
            "ids_identical": True,
            "logits_identical": True,
            "semantic_response_identical": True,
            "topk_speedup": micro["topk_speedup_lower_95"] >= 2.0,
            "prefill_speedup": prefill_lower95 >= 1.05,
            "all_safety_checks": True,
        }
        return {
            "schema": "glm52-w4-serving-summary-v1",
            "formula": FORMULA,
            "checks": checks,
            "observed": {
                "prompt_tokens": prompt_tokens,
                "off_block_seconds": off_seconds,
                "on_block_seconds": on_seconds,
                "prefill_block_speedup_ratios": block_ratios,
                "prefill_speedup_lower_95": prefill_lower95,
                "off_prefill_tokens_per_second": [prompt_tokens / value
                                                     for value in off_seconds],
                "on_prefill_tokens_per_second": [prompt_tokens / value
                                                    for value in on_seconds],
                **micro,
            },
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    except (InvalidCampaign, KeyError, TypeError, ValueError, OverflowError) as error:
        return _fail(str(error))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        schedules = json.loads((args.run_dir / "schedules.json").read_text())
        microgate = json.loads((args.run_dir / "microgate-summary.json").read_text())
        rows = [json.loads(line) for line in
                (args.run_dir / "raw.jsonl").read_text().splitlines() if line]
        result = score_campaign_rows(rows, schedules, microgate)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result = _fail(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
