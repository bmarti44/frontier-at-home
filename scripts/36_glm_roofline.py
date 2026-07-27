#!/usr/bin/env python3
"""Compute a GLM decode engineering forecast from frozen raw measurements.

The retained non-loader/non-MoE time is measured implementation time, not a
physical lower bound. Consequently this report cannot authorize a global
NO_GO by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


LOAD_RE = re.compile(
    r"^LOADPROF L(\d+) .* total_ms=([0-9.]+)$"
)
MOE_RE = re.compile(
    r"^.*CUDA MoE profile .* total=([0-9.]+) ms$"
)
TOKEN_RE = re.compile(
    r"^DS4_TOKEN_TIMING request=(\S+) index=(\d+) "
    r"monotonic_ns=(\d+) token=(-?\d+)$"
)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_rep(path: Path) -> dict[str, Any]:
    document = load_json(path)
    if document.get("suite_valid") is not True:
        raise ValueError(f"{path}: suite is not valid")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 1 or cells[0].get("valid") is not True:
        raise ValueError(f"{path}: expected one valid cell")
    reps = cells[0].get("reps")
    if not isinstance(reps, list) or len(reps) != 1 or reps[0].get("valid") is not True:
        raise ValueError(f"{path}: expected one valid repetition")
    return reps[0]


def raw_intervals_ms(rep: dict[str, Any]) -> list[float]:
    timestamps = rep.get("token_timestamps_ns")
    if (
        rep.get("timing_source") != "server_raw_token_log"
        or not isinstance(timestamps, list)
        or len(timestamps) < 128
        or any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps)
        or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("clean result lacks complete raw token timestamps")
    return [(right - left) / 1_000_000 for left, right in zip(timestamps, timestamps[1:])]


def profile_groups(path: Path, pattern: re.Pattern[str]) -> list[list[float]]:
    groups: list[list[float]] = []
    current: list[float] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            match = pattern.fullmatch(line)
            if match:
                current.append(float(match.group(match.lastindex or 1)))
            elif TOKEN_RE.fullmatch(line):
                groups.append(current)
                current = []
    if len(groups) < 128:
        raise ValueError(f"{path}: fewer than 128 token profile groups")
    usable = groups[1:]
    if any(len(group) != 75 for group in usable):
        raise ValueError(f"{path}: every decode token must cover exactly 75 routed layers")
    return usable


def minimum_memory_gib(path: Path) -> float:
    values: list[float] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            for field in line.split():
                if field.startswith("mem_avail_kb="):
                    values.append(int(field.split("=", 1)[1]) / 1_048_576)
    if not values:
        raise ValueError(f"{path}: no memory samples")
    return min(values)


def compute_roofline(
    *,
    dsv4_decode_tok_s: float,
    clean_intervals_ms: list[float],
    loader_groups_ms: list[list[float]],
    moe_groups_ms: list[list[float]],
    bandwidth_gb_s: float,
    expert_mib_per_layer_lower_bound: float = 74.0,
    routed_layers: int = 75,
) -> dict[str, Any]:
    numeric = (dsv4_decode_tok_s, bandwidth_gb_s, expert_mib_per_layer_lower_bound)
    if any(not math.isfinite(value) or value <= 0 for value in numeric):
        raise ValueError("roofline inputs must be finite and positive")
    if (
        len(clean_intervals_ms) != len(loader_groups_ms)
        or len(clean_intervals_ms) != len(moe_groups_ms)
    ):
        raise ValueError("profile and timing interval counts differ")
    wall_ms = statistics.fmean(clean_intervals_ms)
    loader_ms = statistics.fmean(sum(group) for group in loader_groups_ms)
    moe_ms = statistics.fmean(sum(group) for group in moe_groups_ms)
    non_loader_non_moe_ms = wall_ms - loader_ms - moe_ms
    if non_loader_non_moe_ms <= 0:
        raise ValueError("profile decomposition has a non-positive residual")

    expert_bytes = expert_mib_per_layer_lower_bound * 2**20 * routed_layers
    physical_expert_read_floor_ms = expert_bytes / (bandwidth_gb_s * 1e9) * 1000
    optimistic_ms = non_loader_non_moe_ms + physical_expert_read_floor_ms
    optimistic_tok_s = 1000 / optimistic_ms
    required_tok_s = 0.80 * dsv4_decode_tok_s

    indexer_1m_bytes = 21 * 1_000_000 * 128 * 4
    indexer_1m_floor_ms = indexer_1m_bytes / (bandwidth_gb_s * 1e9) * 1000
    optimistic_1m_tok_s = 1000 / (optimistic_ms + indexer_1m_floor_ms)
    target_total_ms = 1000 / required_tok_s
    allowed_residual_ms = target_total_ms - physical_expert_read_floor_ms
    residual_reduction_needed_ms = max(
        0.0, non_loader_non_moe_ms - allowed_residual_ms
    )
    residual_reduction_needed_fraction = (
        residual_reduction_needed_ms / non_loader_non_moe_ms
    )
    return {
        "formula_version": 1,
        "assumptions": {
            "loader_time_removed_completely": True,
            "measured_moe_time_replaced_by_physical_read_floor": True,
            "expert_mib_per_layer_lower_bound": expert_mib_per_layer_lower_bound,
            "routed_layers": routed_layers,
            "memory_bandwidth_gb_s_upper_bound": bandwidth_gb_s,
        },
        "measurements": {
            "dsv4_decode_tok_s": dsv4_decode_tok_s,
            "clean_glm_wall_ms_per_token": wall_ms,
            "loader_ms_per_token_removed": loader_ms,
            "measured_moe_ms_per_token_replaced": moe_ms,
            "non_loader_non_moe_ms_per_token": non_loader_non_moe_ms,
            "physical_expert_read_floor_ms": physical_expert_read_floor_ms,
            "indexer_1m_read_floor_ms": indexer_1m_floor_ms,
        },
        "roofline": {
            "required_80_percent_tok_s": required_tok_s,
            "forecast_short_context_tok_s": optimistic_tok_s,
            "forecast_short_context_ratio": optimistic_tok_s / dsv4_decode_tok_s,
            "forecast_1m_tok_s": optimistic_1m_tok_s,
            "forecast_1m_ratio": optimistic_1m_tok_s / dsv4_decode_tok_s,
            "target_total_ms": target_total_ms,
            "allowed_residual_ms": allowed_residual_ms,
            "residual_reduction_needed_ms": residual_reduction_needed_ms,
            "residual_reduction_needed_fraction": residual_reduction_needed_fraction,
        },
        "physical_no_go_established": False,
        "decision": "NO_RESULT",
        "reason": (
            "forecast retains measured residual time without a physical lower "
            "bound; it cannot establish a global NO_GO"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsv4-result", type=Path, required=True)
    parser.add_argument("--clean-glm-result", type=Path, required=True)
    parser.add_argument("--loader-log", type=Path, required=True)
    parser.add_argument("--moe-log", type=Path, required=True)
    parser.add_argument("--clean-samples", type=Path, required=True)
    parser.add_argument("--loader-samples", type=Path, required=True)
    parser.add_argument("--moe-samples", type=Path, required=True)
    parser.add_argument("--bandwidth-gb-s", type=float, default=273.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dsv4 = valid_rep(args.dsv4_result)
    clean = valid_rep(args.clean_glm_result)
    report = compute_roofline(
        dsv4_decode_tok_s=float(dsv4["decode_tok_s"]),
        clean_intervals_ms=raw_intervals_ms(clean),
        loader_groups_ms=profile_groups(args.loader_log, LOAD_RE),
        moe_groups_ms=profile_groups(args.moe_log, MOE_RE),
        bandwidth_gb_s=args.bandwidth_gb_s,
    )
    memory = {
        "clean_min_gib": minimum_memory_gib(args.clean_samples),
        "loader_profile_min_gib": minimum_memory_gib(args.loader_samples),
        "moe_profile_min_gib": minimum_memory_gib(args.moe_samples),
    }
    report["memory"] = memory
    report["checks"] = {
        "all_memory_at_least_10_gib": min(memory.values()) >= 10.0,
        "at_least_128_intervals": len(raw_intervals_ms(clean)) >= 127,
        "physical_no_go_established": report["physical_no_go_established"],
    }
    report["artifacts"] = {
        str(path): sha256(path)
        for path in (
            args.dsv4_result,
            args.clean_glm_result,
            args.loader_log,
            args.moe_log,
            args.clean_samples,
            args.loader_samples,
            args.moe_samples,
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Preserve the forecast but fail closed until every retained component is
    # a justified physical floor.
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
