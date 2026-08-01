#!/usr/bin/env python3
"""Thin Rung 0.1 lifecycle wrapper around the existing speed scorer."""

from __future__ import annotations

import math
import statistics
from typing import Any

from scripts.glm52_goal import paired_ratio_bound, validate_ab_blocks


SLAB_PATH = "/home/bmarti44/.cache/glm52-rung0-artifacts/glm52-experts-v2.slab"
SLAB_SHA256 = (
    "62961905a685e16e3e8f5f98e189511e"
    "b2e65ee6eda7e1a860c1ec58959e5518"
)
MODEL_SHA256 = (
    "a49de64c5020432bdae23de36a423a96"
    "60a5621bc0db8d12b66bd8814b07fea0"
)


def arm_schedule() -> tuple[tuple[int, int, str], ...]:
    """Return the preregistered five-block execution order."""
    rows: list[tuple[int, int, str]] = []
    for block in range(5):
        order = "ABBA" if block % 2 == 0 else "BAAB"
        rows.extend((block, sequence, arm) for sequence, arm in enumerate(order))
    return tuple(rows)


def canonical_engine_environment(mode: str) -> dict[str, str]:
    """Return the exact timed engine environment for one arm."""
    if mode not in {"off", "on"}:
        raise ValueError("mode must be off or on")
    result = {
        "DS4_CUDA_EXPERT_CACHE_GB": "68",
        "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
        "DS4_CUDA_FETCH_THREADS": "8",
        "DS4_CUDA_LOAD_PROFILE": "1",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_GLM_TP_DEBUG": "1",
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    if mode == "on":
        result.update(
            {
                "DS4_CUDA_EXPERT_SLAB_PATH": SLAB_PATH,
                "DS4_CUDA_EXPERT_SLAB_SHA256": SLAB_SHA256,
                "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256": MODEL_SHA256,
            }
        )
    return result


def score_campaign(records: list[dict[str, Any]], nll: dict[str, Any]) -> dict[str, Any]:
    """Validate raw arms and apply the fixed Rung 0.1 formulas."""
    expected_keys = {
        "schema_version",
        "block",
        "sequence",
        "arm",
        "mode",
        "server_instance_id",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256",
        "suite_valid",
        "reps",
        "engine",
        "external_io",
        "safety",
    }
    if len(records) != 20:
        raise ValueError("campaign requires exactly 20 arms")

    def sha256(value: Any, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} is not a lowercase SHA-256")
        return value

    def positive(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} is not numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{label} is not positive and finite")
        return result

    validation_rows = []
    binaries: set[str] = set()
    configurations: dict[str, set[str]] = {"off": set(), "on": set()}
    output_signatures: dict[int, set[tuple[Any, ...]]] = {0: set(), 1: set()}
    access_streams: set[str] = set()
    per_row: dict[tuple[int, int], tuple[float, float, float]] = {}
    io_throughput: dict[str, list[float]] = {"off": [], "on": []}

    for index, record in enumerate(records):
        if set(record) != expected_keys or record["schema_version"] != 1:
            raise ValueError(f"arm {index} has an invalid schema")
        mode = record["mode"]
        arm = record["arm"]
        if (arm, mode) not in {("A", "off"), ("B", "on")}:
            raise ValueError("arm-to-mode mapping is invalid")
        binary = sha256(record["binary_sha256"], "binary_sha256")
        configuration = sha256(
            record["configuration_sha256"], "configuration_sha256"
        )
        fixture = sha256(record["fixture_sha256"], "fixture_sha256")
        binaries.add(binary)
        configurations[mode].add(configuration)
        validation_rows.append(
            {
                "block": record["block"],
                "sequence": record["sequence"],
                "arm": arm,
                "server_boot_id": record["server_instance_id"],
                "fixture_sha256": fixture,
                "binary_sha256": binary,
                "configuration_sha256": configuration,
            }
        )
        if record["suite_valid"] is not True:
            raise ValueError("speed suite is invalid")
        reps = record["reps"]
        if not isinstance(reps, list) or len(reps) != 2:
            raise ValueError("each arm requires exactly two measured reps")
        decode_rates: list[float] = []
        ttfts: list[float] = []
        prompt_rates: list[float] = []
        for rep_index, rep in enumerate(reps):
            if not isinstance(rep, dict) or rep.get("valid") is not True:
                raise ValueError("measured rep is invalid")
            timestamps = rep.get("sse_token_timestamps_ns")
            token_count = rep.get("client_completion_tokens")
            token_ids = rep.get("token_ids")
            if (
                not isinstance(timestamps, list)
                or len(timestamps) < 128
                or isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count != len(timestamps)
                or not isinstance(token_ids, list)
                or len(token_ids) != token_count
                or any(isinstance(value, bool) or not isinstance(value, int) for value in timestamps)
                or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
            ):
                raise ValueError("client-observed token timing is incomplete")
            elapsed = (timestamps[-1] - timestamps[0]) / 1_000_000_000
            decode_rates.append((token_count - 1) / positive(elapsed, "decode elapsed"))
            ttft = positive(rep.get("ttft_s"), "TTFT")
            prompt_tokens = rep.get("client_prompt_tokens")
            if (
                isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0
            ):
                raise ValueError("client prompt token count is invalid")
            ttfts.append(ttft)
            prompt_rates.append(prompt_tokens / ttft)
            signature = (
                sha256(rep.get("request_sha256"), "request_sha256"),
                sha256(
                    rep.get("generated_reasoning_sha256"),
                    "generated_reasoning_sha256",
                ),
                sha256(
                    rep.get("generated_content_sha256"),
                    "generated_content_sha256",
                ),
                token_count,
                tuple(token_ids),
            )
            output_signatures[rep_index].add(signature)
        per_row[(record["block"], record["sequence"])] = (
            statistics.fmean(decode_rates),
            statistics.fmean(ttfts),
            statistics.fmean(prompt_rates),
        )

        engine = record["engine"]
        if not isinstance(engine, dict) or engine.get("slab_mode") != mode:
            raise ValueError("resolved slab mode is invalid")
        reads = engine.get("slab_reads")
        peak_qd = engine.get("slab_peak_qd")
        if (
            isinstance(reads, bool)
            or not isinstance(reads, int)
            or isinstance(peak_qd, bool)
            or not isinstance(peak_qd, int)
            or reads < 0
            or peak_qd < 0
        ):
            raise ValueError("slab counters are invalid")
        if mode == "off" and (reads != 0 or peak_qd != 0):
            raise ValueError("default-off arm performed slab I/O")
        if mode == "on" and (reads <= 0 or peak_qd < 2):
            raise ValueError("slab arm lacks positive concurrent reads")
        if engine.get("arena_pin_ok") is not True or engine.get("trace_lines") != 0:
            raise ValueError("timed engine instrumentation or arena pin is invalid")
        access_streams.add(
            sha256(engine.get("access_stream_sha256"), "access stream")
        )

        external = record["external_io"]
        if not isinstance(external, dict):
            raise ValueError("external I/O record is absent")
        read_bytes = positive(external.get("read_bytes_delta"), "read bytes")
        io_elapsed = positive(external.get("elapsed_seconds"), "I/O elapsed")
        samples = external.get("sample_count")
        external_qd = external.get("peak_read_qd")
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < 2
            or isinstance(external_qd, bool)
            or not isinstance(external_qd, int)
            or external_qd < 0
            or (mode == "on" and external_qd < 2)
        ):
            raise ValueError("external completed-I/O coverage is invalid")
        io_throughput[mode].append(read_bytes / io_elapsed)

        safety = record["safety"]
        if not isinstance(safety, dict):
            raise ValueError("safety evidence is absent")
        if positive(safety.get("minimum_available_gib"), "available memory") < 10:
            raise ValueError("whole-system memory floor was violated")
        for field in (
            "cgroup_high_events",
            "cgroup_max_events",
            "cgroup_oom_events",
            "cgroup_swap_bytes",
        ):
            if safety.get(field) != 0:
                raise ValueError(f"safety evidence has nonzero {field}")
        if safety.get("xid") is not False or safety.get("survivors") != []:
            raise ValueError("Xid or survivor invalidates the arm")
        if safety.get("failures") != []:
            raise ValueError("arm contains a safety failure")

    validate_ab_blocks(validation_rows)
    if len(binaries) != 1:
        raise ValueError("campaign used more than one binary")
    if any(len(values) != 1 for values in configurations.values()):
        raise ValueError("arm configuration changed between blocks")
    if configurations["off"] == configurations["on"]:
        raise ValueError("campaign arms are identical")
    if any(len(signatures) != 1 for signatures in output_signatures.values()):
        raise ValueError("paired output bytes or token IDs differ")
    if len(access_streams) != 1:
        raise ValueError("expert access streams differ between arms")

    if set(nll) != {
        "case_count",
        "token_weighted_delta_nll",
        "top1_loss_pp",
        "deterministic",
    }:
        raise ValueError("NLL summary schema is invalid")
    if (
        nll["case_count"] != 100
        or nll["token_weighted_delta_nll"] != 0.0
        or nll["top1_loss_pp"] != 0.0
        or nll["deterministic"] is not True
    ):
        raise ValueError("lossless transport requires exact-zero paired NLL")

    decode_off: list[float] = []
    decode_on: list[float] = []
    ttft_off: list[float] = []
    ttft_on: list[float] = []
    prompt_rate_off: list[float] = []
    prompt_rate_on: list[float] = []
    for block in range(5):
        for arm, decode_target, ttft_target, prompt_target in (
            ("A", decode_off, ttft_off, prompt_rate_off),
            ("B", decode_on, ttft_on, prompt_rate_on),
        ):
            values = [
                per_row[(block, record["sequence"])]
                for record in records
                if record["block"] == block and record["arm"] == arm
            ]
            if len(values) != 2:
                raise ValueError("block does not contain two instances per arm")
            decode_target.append(statistics.fmean(value[0] for value in values))
            ttft_target.append(statistics.fmean(value[1] for value in values))
            prompt_target.append(statistics.fmean(value[2] for value in values))

    decode_lower = paired_ratio_bound(decode_on, decode_off, side="lower")
    ttft_upper = paired_ratio_bound(ttft_on, ttft_off, side="upper")
    verdict = "PASS" if decode_lower > 1.0 and ttft_upper <= 1.05 else "FAIL"
    return {
        "scorer_id": "glm.rung0.slab.v1",
        "verdict": verdict,
        "decode_ratio_lower_95": decode_lower,
        "warm_ttft_ratio_upper_95": ttft_upper,
        "decode_tps": {"off": decode_off, "on": decode_on},
        "warm_ttft_seconds": {"off": ttft_off, "on": ttft_on},
        "diagnostic_prompt_rate": {
            "label": "client-token-count divided by TTFT; not synchronized prefill",
            "off": prompt_rate_off,
            "on": prompt_rate_on,
        },
        "external_read_bytes_per_second": {
            mode: statistics.fmean(values) for mode, values in io_throughput.items()
        },
        "nll": dict(nll),
    }
