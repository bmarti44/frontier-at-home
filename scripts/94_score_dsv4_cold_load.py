#!/usr/bin/env python3
"""Fixed, fail-closed scorer for matched DSV4 cold-load campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from typing import Any


T_ONE_SIDED_95_DF4 = 2.131846786326649
MAX_READY_RATIO_UPPER_95 = 0.5
MAX_ON_READY_SECONDS_UPPER_95 = 30.0
MAX_ON_TENSOR_SECONDS_UPPER_95 = 20.393206603359758
MAX_CACHE_RESIDENT_BYTES = 1024**3
MIN_MEMORY_KB = 10 * 1024 * 1024
MIN_PHYSICAL_READ_FRACTION = 0.90
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FORMULA = (
    "five fresh-server blocks with randomness-derived ABBA/BAAB order; "
    "block ready ratio=mean(ON launch-to-health)/mean(OFF launch-to-health); "
    "one-sided 95% Student-t upper bounds with df=4 over five block values; "
    "ready ratio upper<=0.5, ON ready upper<=30s, ON synchronized tensor-load "
    "upper<=20.393206603359758s"
)


class InvalidCampaign(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCampaign(message)


def _exact_int(value: object) -> bool:
    return type(value) is int


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _schedule(randomness_hex: str) -> list[str]:
    _require(_sha(randomness_hex), "invalid randomness")
    seed = bytes.fromhex(randomness_hex)
    domain = b"frontier-at-home/dsv4-cold-load/v1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0 else "BAAB"
        for block in range(5)
    ]


def _upper(values: list[float], *, logarithmic: bool = False) -> float:
    _require(len(values) == 5, "confidence bound requires five blocks")
    _require(all(math.isfinite(value) and value > 0.0 for value in values), "invalid metric")
    samples = [math.log(value) for value in values] if logarithmic else values
    center = statistics.fmean(samples)
    margin = T_ONE_SIDED_95_DF4 * statistics.stdev(samples) / math.sqrt(5)
    result = center + margin
    return math.exp(result) if logarithmic else result


def _fail(message: str) -> dict[str, object]:
    return {
        "formula": FORMULA,
        "checks": {"input_and_acceptance_checks": False},
        "observed": {},
        "failure": message,
        "verdict": "FAIL",
    }


def _validate_manifest(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "manifest is not an object")
    required = {
        "schema_version", "candidate_hash", "runner_sha256", "scorer_sha256",
        "model_sha256", "configuration_sha256", "binary_sha256",
        "drand_verifier_sha256", "drand_node_sha256", "model_bytes",
        "randomness", "schedules",
    }
    _require(set(value) == required, "manifest keys do not match frozen schema")
    _require(value["schema_version"] == 1, "invalid manifest schema")
    for name in (
        "candidate_hash", "runner_sha256", "scorer_sha256", "model_sha256",
        "configuration_sha256", "binary_sha256", "drand_verifier_sha256",
        "drand_node_sha256",
    ):
        _require(_sha(value[name]), f"invalid manifest {name}")
    _require(_exact_int(value["model_bytes"]) and value["model_bytes"] > 0, "invalid model bytes")
    randomness = value["randomness"]
    _require(isinstance(randomness, dict) and set(randomness) == {"value", "receipt_sha256"}, "invalid randomness receipt")
    _require(_sha(randomness["value"]) and _sha(randomness["receipt_sha256"]), "invalid randomness binding")
    schedules = _schedule(randomness["value"])
    _require(value["schedules"] == schedules, "schedule is not randomness-derived")
    return value


def _validate_row(value: object, manifest: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, dict), "row is not an object")
    required = {
        "schema_version", "block", "position", "arm", "run_id", "candidate_hash",
        "model_sha256", "configuration_sha256", "binary_sha256",
        "runtime_closure_sha256", "runtime_closure_count",
        "process_launch_monotonic_ns", "health_ready_monotonic_ns",
        "tensor_load_start_monotonic_ns", "tensor_load_end_monotonic_ns",
        "server_pid", "server_start_ticks", "server_fresh", "physical_read_bytes",
        "cache_resident_bytes_before", "direct_shard_count", "direct_required",
        "semantic_sha256", "first_token_logit_sha256", "authenticated_health",
        "authenticated_completion", "unauthenticated_rejected",
        "minimum_mem_available_kb", "swap_growth_bytes", "cgroup_oom_delta",
        "cgroup_oom_kill_delta", "cgroup_max_delta", "xid_count", "surviving_descendants",
        "systemd_result", "systemd_exec_main_code", "systemd_exec_main_status",
        "systemd_memory_peak_bytes", "systemd_memory_swap_peak_bytes",
    }
    _require(set(value) == required, "row keys do not match frozen schema")
    _require(value["schema_version"] == 1, "invalid row schema")
    _require(_exact_int(value["block"]) and 0 <= value["block"] < 5, "invalid block")
    _require(_exact_int(value["position"]) and 0 <= value["position"] < 4, "invalid position")
    _require(value["arm"] in {"off", "on"}, "invalid arm")
    _require(isinstance(value["run_id"], str) and bool(value["run_id"]), "invalid run id")
    for row_name, manifest_name in (
        ("candidate_hash", "candidate_hash"),
        ("model_sha256", "model_sha256"),
        ("configuration_sha256", "configuration_sha256"),
        ("binary_sha256", "binary_sha256"),
    ):
        _require(value[row_name] == manifest[manifest_name], f"stale {row_name}")
    for name in ("semantic_sha256", "first_token_logit_sha256", "runtime_closure_sha256"):
        _require(_sha(value[name]), f"invalid {name}")
    integer_fields = (
        "process_launch_monotonic_ns", "health_ready_monotonic_ns",
        "tensor_load_start_monotonic_ns", "tensor_load_end_monotonic_ns",
        "server_pid", "server_start_ticks", "physical_read_bytes",
        "cache_resident_bytes_before", "direct_shard_count", "runtime_closure_count",
        "minimum_mem_available_kb",
        "swap_growth_bytes", "cgroup_oom_delta", "cgroup_oom_kill_delta", "xid_count",
        "cgroup_max_delta", "surviving_descendants", "systemd_exec_main_code",
        "systemd_exec_main_status", "systemd_memory_peak_bytes",
        "systemd_memory_swap_peak_bytes",
    )
    _require(all(_exact_int(value[name]) for name in integer_fields), "non-integer row value")
    start = value["process_launch_monotonic_ns"]
    tensor_start = value["tensor_load_start_monotonic_ns"]
    tensor_end = value["tensor_load_end_monotonic_ns"]
    ready = value["health_ready_monotonic_ns"]
    _require(0 <= start < tensor_start < tensor_end <= ready, "invalid timing order")
    _require(value["server_pid"] > 0 and value["server_start_ticks"] > 0, "invalid process identity")
    _require(value["runtime_closure_count"] > 0, "empty runtime closure")
    _require(value["server_fresh"] is True, "server is not fresh")
    _require(
        value["physical_read_bytes"] >= math.ceil(manifest["model_bytes"] * MIN_PHYSICAL_READ_FRACTION),
        "insufficient physical reads",
    )
    _require(value["cache_resident_bytes_before"] <= MAX_CACHE_RESIDENT_BYTES, "warm cache")
    expected_direct = value["arm"] == "on"
    _require(value["direct_required"] is expected_direct, "direct arm mismatch")
    _require(value["direct_shard_count"] == (3 if expected_direct else 0), "direct shard observation mismatch")
    for name in ("authenticated_health", "authenticated_completion", "unauthenticated_rejected"):
        _require(value[name] is True, f"failed {name}")
    _require(value["minimum_mem_available_kb"] >= MIN_MEMORY_KB, "memory floor violated")
    for name in (
        "swap_growth_bytes", "cgroup_oom_delta", "cgroup_oom_kill_delta", "cgroup_max_delta", "xid_count",
        "surviving_descendants", "systemd_exec_main_code", "systemd_exec_main_status",
        "systemd_memory_swap_peak_bytes",
    ):
        _require(value[name] == 0, f"unsafe {name}")
    _require(value["systemd_result"] == "success", "unsafe systemd result")
    _require(0 < value["systemd_memory_peak_bytes"] <= 104 * 1024**3, "invalid systemd memory peak")
    return value


def score_campaign(manifest: object, rows: object) -> dict[str, object]:
    """Return PASS only for a complete, matched and safe campaign."""
    try:
        bound_manifest = _validate_manifest(manifest)
        _require(isinstance(rows, list) and len(rows) == 20, "need exactly twenty rows")
        valid = [_validate_row(row, bound_manifest) for row in rows]
        by_slot: dict[tuple[int, int], dict[str, Any]] = {}
        for row in valid:
            slot = (row["block"], row["position"])
            _require(slot not in by_slot, "duplicate block position")
            by_slot[slot] = row
        _require(len(by_slot) == 20, "incomplete block positions")
        for block, schedule in enumerate(bound_manifest["schedules"]):
            for position, letter in enumerate(schedule):
                expected = "off" if letter == "A" else "on"
                _require(by_slot[(block, position)]["arm"] == expected, "arm schedule mismatch")
        _require(len({row["run_id"] for row in valid}) == 20, "duplicate run id")
        _require(len({(row["server_pid"], row["server_start_ticks"]) for row in valid}) == 20, "reused server")
        for name in ("semantic_sha256", "first_token_logit_sha256"):
            _require(len({row[name] for row in valid}) == 1, f"unequal {name}")
        _require(
            len({(row["runtime_closure_sha256"], row["runtime_closure_count"]) for row in valid}) == 1,
            "unequal observed runtime closure",
        )

        ready_ratios: list[float] = []
        on_ready_means: list[float] = []
        on_tensor_means: list[float] = []
        for block in range(5):
            block_rows = [by_slot[(block, position)] for position in range(4)]
            off = [row for row in block_rows if row["arm"] == "off"]
            on = [row for row in block_rows if row["arm"] == "on"]
            _require(len(off) == len(on) == 2, "unbalanced block")

            def ready_seconds(row: dict[str, Any]) -> float:
                return (row["health_ready_monotonic_ns"] - row["process_launch_monotonic_ns"]) / 1e9

            def tensor_seconds(row: dict[str, Any]) -> float:
                return (row["tensor_load_end_monotonic_ns"] - row["tensor_load_start_monotonic_ns"]) / 1e9

            off_ready = statistics.fmean(ready_seconds(row) for row in off)
            on_ready = statistics.fmean(ready_seconds(row) for row in on)
            ready_ratios.append(on_ready / off_ready)
            on_ready_means.append(on_ready)
            on_tensor_means.append(statistics.fmean(tensor_seconds(row) for row in on))

        ready_upper = _upper(ready_ratios, logarithmic=True)
        on_ready_upper = _upper(on_ready_means)
        on_tensor_upper = _upper(on_tensor_means)
        checks = {
            "schema_schedule_and_hash_bindings": True,
            "twenty_unique_fresh_servers": True,
            "matched_semantics_and_first_token_logits": True,
            "cold_physical_read_coverage": True,
            "direct_descriptors_match_arms": True,
            "auth_completion_and_safety": True,
            "ready_ratio_upper_95_le_0_5": ready_upper <= MAX_READY_RATIO_UPPER_95,
            "on_ready_seconds_upper_95_le_30": on_ready_upper <= MAX_ON_READY_SECONDS_UPPER_95,
            "on_tensor_seconds_upper_95_le_fio_half_rate": on_tensor_upper <= MAX_ON_TENSOR_SECONDS_UPPER_95,
        }
        return {
            "formula": FORMULA,
            "checks": checks,
            "observed": {
                "ready_block_ratios": ready_ratios,
                "ready_ratio_upper_95": ready_upper,
                "on_ready_seconds_upper_95": on_ready_upper,
                "on_tensor_seconds_upper_95": on_tensor_upper,
                "minimum_mem_available_kb": min(row["minimum_mem_available_kb"] for row in valid),
                "minimum_physical_read_bytes": min(row["physical_read_bytes"] for row in valid),
                "maximum_cache_resident_bytes_before": max(row["cache_resident_bytes_before"] for row in valid),
                "blocks": 5,
                "runs": 20,
            },
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    except (InvalidCampaign, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        return _fail(f"{type(error).__name__}: {error}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("raw_jsonl")
    parser.add_argument("summary")
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(args.raw_jsonl, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary = score_campaign(manifest, rows)
    with open(args.summary, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
