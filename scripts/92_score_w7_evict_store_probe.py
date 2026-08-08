#!/usr/bin/env python3
"""Fixed scorer shell for the preregistered W7.2 bounded probe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


FORMULA = (
    "two fresh-server equal-fixture OFF/ON arms; exact output-token, UTF-8, and "
    "three-logit equality; warm append TTFT_OFF-TTFT_ON >=0.5s; "
    "decode_ON/decode_OFF >=0.99 over >=128 output timestamps"
)


SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MIN_MEMORY_KB = 10 * 1024 * 1024


class InvalidProbe(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidProbe(message)


def _validated_row(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "row is not an object")
    row = value
    required = {
        "arm", "position", "run_id", "binary_sha256", "model_sha256",
        "common_config_sha256", "request_sha256", "diagnostic_skip",
        "request_start_ns", "token_timestamps_ns", "output_token_ids",
        "output_sha256", "generated_text_sha256", "generated_text_bytes",
        "logit_sha256s", "selected_checkpoint_tokens", "evict_store_count",
        "skip_marker_count", "activation_marker_count", "server_fresh", "safety",
    }
    _require(set(row) == required, "row keys do not match frozen schema")
    _require(row["arm"] in {"off", "on"}, "invalid arm")
    _require(type(row["position"]) is int and row["position"] in {0, 1}, "invalid position")
    _require(isinstance(row["run_id"], str) and bool(row["run_id"]), "invalid run id")
    for name in (
        "binary_sha256", "model_sha256", "common_config_sha256", "request_sha256",
        "output_sha256", "generated_text_sha256",
    ):
        _require(isinstance(row[name], str) and SHA_RE.fullmatch(row[name]) is not None, f"invalid {name}")
    _require(
        isinstance(row["logit_sha256s"], list) and len(row["logit_sha256s"]) == 3
        and all(isinstance(item, str) and SHA_RE.fullmatch(item) is not None for item in row["logit_sha256s"]),
        "invalid synchronized-logit digests",
    )
    expected_on = 1 if row["arm"] == "on" else 0
    _require(type(row["diagnostic_skip"]) is int and row["diagnostic_skip"] == expected_on, "flag/arm mismatch")
    _require(type(row["request_start_ns"]) is int and row["request_start_ns"] >= 0, "invalid request start")
    timestamps = row["token_timestamps_ns"]
    tokens = row["output_token_ids"]
    _require(isinstance(timestamps, list) and len(timestamps) >= 128, "short timestamp series")
    _require(isinstance(tokens, list) and len(tokens) == len(timestamps), "token/timing mismatch")
    _require(all(type(item) is int and item >= 0 for item in timestamps), "invalid timestamp")
    _require(all(type(item) is int and item >= 0 for item in tokens), "invalid token id")
    _require(timestamps[0] > row["request_start_ns"], "nonpositive TTFT")
    _require(all(right > left for left, right in zip(timestamps, timestamps[1:])), "timestamps not strict")
    token_digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest()
    _require(token_digest == row["output_sha256"], "output digest does not bind token IDs")
    _require(type(row["generated_text_bytes"]) is int and row["generated_text_bytes"] >= 0, "invalid text size")
    _require(row["selected_checkpoint_tokens"] == 5044, "wrong checkpoint selected")
    _require(type(row["evict_store_count"]) is int, "invalid evict-store count")
    _require(type(row["skip_marker_count"]) is int, "invalid skip-marker count")
    _require(type(row["activation_marker_count"]) is int, "invalid activation-marker count")
    if row["arm"] == "off":
        _require(row["evict_store_count"] == 1, "OFF did not reproduce one evict store")
        _require(row["skip_marker_count"] == 0 and row["activation_marker_count"] == 0, "OFF emitted diagnostic markers")
    else:
        _require(row["evict_store_count"] == 0, "ON performed evict store")
        _require(row["skip_marker_count"] == 1 and row["activation_marker_count"] == 1, "ON marker cardinality mismatch")
    _require(row["server_fresh"] is True, "server was not fresh")
    safety = row["safety"]
    safety_keys = {
        "containment_rc", "minimum_mem_available_kb", "swap_growth_bytes",
        "cgroup_max_delta", "cgroup_oom_delta", "cgroup_oom_kill_delta",
        "xid_count", "surviving_descendants",
    }
    _require(isinstance(safety, dict) and set(safety) == safety_keys, "invalid safety schema")
    _require(all(type(safety[name]) is int for name in safety_keys), "non-integer safety value")
    _require(safety["minimum_mem_available_kb"] >= MIN_MEMORY_KB, "memory floor violated")
    for name in safety_keys - {"minimum_mem_available_kb"}:
        _require(safety[name] == 0, f"unsafe {name}")
    return row


def _fail(message: str) -> dict[str, object]:
    return {
        "formula": FORMULA,
        "checks": {"input_and_acceptance_checks": False},
        "observed": {},
        "failure": message,
        "verdict": "FAIL",
    }


def score_probe_rows(rows: object, order: object) -> dict[str, object]:
    """Score the bounded two-arm falsifier; every malformed input fails closed."""
    try:
        _require(isinstance(order, list) and len(order) == 2 and set(order) == {"off", "on"}, "invalid arm order")
        _require(isinstance(rows, list) and len(rows) == 2, "need exactly two rows")
        validated = [_validated_row(row) for row in rows]
        _require([row["arm"] for row in validated] == order, "arm order mismatch")
        _require([row["position"] for row in validated] == [0, 1], "position mismatch")
        _require(len({row["arm"] for row in validated}) == 2, "missing or duplicate arm")
        _require(len({row["run_id"] for row in validated}) == 2, "duplicate run id")
        for field in ("binary_sha256", "model_sha256", "common_config_sha256", "request_sha256"):
            _require(len({row[field] for row in validated}) == 1, f"unequal {field}")
        off = next(row for row in validated if row["arm"] == "off")
        on = next(row for row in validated if row["arm"] == "on")
        for field in (
            "output_token_ids", "output_sha256", "generated_text_sha256",
            "generated_text_bytes", "logit_sha256s",
        ):
            _require(off[field] == on[field], f"unequal {field}")
        def metrics(row: dict[str, Any]) -> tuple[float, float]:
            times = row["token_timestamps_ns"]
            ttft = (times[0] - row["request_start_ns"]) / 1_000_000_000.0
            decode = (len(times) - 1) * 1_000_000_000.0 / (times[-1] - times[0])
            _require(math.isfinite(ttft) and ttft > 0, "invalid TTFT")
            _require(math.isfinite(decode) and decode > 0, "invalid decode")
            return ttft, decode
        off_ttft, off_decode = metrics(off)
        on_ttft, on_decode = metrics(on)
        saved = off_ttft - on_ttft
        decode_ratio = on_decode / off_decode
        checks = {
            "schema_order_and_equal_fixture": True,
            "fresh_safe_servers": True,
            "off_one_store_on_one_skip": True,
            "same_authenticated_checkpoint_5044": True,
            "byte_identical_output_tokens_and_utf8": True,
            "byte_identical_three_synchronized_logits": True,
            "warm_append_seconds_saved_ge_0_5": saved >= 0.5,
            "decode_ratio_ge_0_99": decode_ratio >= 0.99,
        }
        return {
            "formula": FORMULA,
            "checks": checks,
            "observed": {
                "off_ttft_s": off_ttft,
                "on_ttft_s": on_ttft,
                "warm_append_seconds_saved": saved,
                "off_decode_tps": off_decode,
                "on_decode_tps": on_decode,
                "decode_ratio": decode_ratio,
                "output_tokens_per_arm": len(off["output_token_ids"]),
                "minimum_mem_available_kb": min(row["safety"]["minimum_mem_available_kb"] for row in validated),
            },
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    except (InvalidProbe, KeyError, StopIteration, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        return _fail(f"{type(error).__name__}: {error}")
