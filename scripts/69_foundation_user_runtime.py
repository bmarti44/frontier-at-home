#!/usr/bin/env python3
"""Fixed runtime primitives for sudo-free matched foundation measurements.

This module deliberately contains no authority to start a model by itself.
The registered foundation runner invokes it inside an already-created user
systemd cgroup after freezing candidate and artifact identities.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from pathlib import Path
from typing import Any


GIB = 1 << 30
CGROUP_LIMITS = {
    "dsv4": (105 * GIB, 110 * GIB, 0),
    "glm52": (68 * GIB, 72 * GIB, 0),
}


def _bounded_port(port: int) -> str:
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer from 1024 through 65535")
    return str(port)


def server_invocation(
    profile: str, binary: Path, model: Path, port: int
) -> tuple[list[str], dict[str, str]]:
    """Return the only approved server command/environment for each profile."""
    port_text = _bounded_port(port)
    binary_text = str(binary)
    model_text = str(model)
    if profile == "dsv4":
        return (
            [
                binary_text,
                "--model",
                model_text,
                "--alias",
                "deepseek-v4-flash",
                "--host",
                "127.0.0.1",
                "--port",
                port_text,
                "-c",
                "1048576",
                "-np",
                "1",
                "-ngl",
                "999",
                "-b",
                "512",
                "-ub",
                "256",
                "--no-warmup",
                "--cache-ram",
                "0",
                "--no-mmap",
            ],
            {},
        )
    if profile == "glm52":
        return (
            [
                binary_text,
                "--cuda",
                "-m",
                model_text,
                "-c",
                "8192",
                "--host",
                "127.0.0.1",
                "--port",
                port_text,
                "--ssd-streaming",
                "--ssd-streaming-cache-experts",
                "40GB",
            ],
            {
                "DS4_CUDA_EXPERT_CACHE_GB": "40",
                "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                "DS4_CUDA_FETCH_THREADS": "6",
                "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                "DS4_GLM_COMPACT_CACHE_AFFINE_INT8": "1",
                "DS4_TOKEN_TIMING_LOG": "1",
            },
        )
    raise ValueError("profile must be dsv4 or glm52")


def validate_cgroup(
    profile: str, memory_high: int, memory_max: int, swap_max: int
) -> None:
    expected = CGROUP_LIMITS.get(profile)
    if expected is None:
        raise ValueError("unknown cgroup profile")
    values = (memory_high, memory_max, swap_max)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("cgroup values must be integers")
    if values != expected:
        raise ValueError(
            f"unsafe {profile} cgroup limits: observed={values} expected={expected}"
        )


def verify_artifact(path: Path, expected_sha256: str) -> None:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("artifact digest is invalid")
    try:
        int(expected_sha256, 16)
        details = path.lstat()
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError("artifact identity is invalid") from exc
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise ValueError("artifact must be a plain file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"artifact hash mismatch: {path}")


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not finite and positive")
    return result


def baseline_from_result(
    result: dict[str, Any],
    *,
    profile: str,
    server_instance_id: str,
    fixture_sha256: str,
    binary_sha256: str,
    configuration_sha256: str,
    available_memory_gib: float,
) -> dict[str, Any]:
    """Convert one fixed two-repetition speed result into foundation raw data."""
    if profile not in {"dsv4", "glm52"} or result.get("suite_valid") is not True:
        raise ValueError("benchmark identity or suite validity is invalid")
    metadata = result.get("metadata")
    cells = result.get("cells")
    if (
        not isinstance(metadata, dict)
        or metadata.get("reps") != 2
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], dict)
        or cells[0].get("ctx_tokens") != 0
        or cells[0].get("valid") is not True
    ):
        raise ValueError("benchmark shape is invalid")
    reps = cells[0].get("reps")
    if (
        not isinstance(reps, list)
        or len(reps) != 2
        or any(not isinstance(rep, dict) or rep.get("valid") is not True for rep in reps)
    ):
        raise ValueError("cold/warm repetitions are invalid")
    cold, warm = reps
    timestamps = warm.get("token_timestamps_ns")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) < 128
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in timestamps)
        or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("raw token timestamps are invalid")
    prompt_tokens = warm.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError("evaluated prompt token count is invalid")
    prefill_rate = _positive_number(warm.get("prefill_tok_s"), "prefill rate")
    memory = _positive_number(available_memory_gib, "available memory")
    if memory < 10.0:
        raise ValueError("foundation memory floor was violated")
    identities = (server_instance_id, fixture_sha256, binary_sha256, configuration_sha256)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in identities
    ):
        raise ValueError("foundation identity hash is invalid")
    return {
        "profile": profile,
        "server_instance_id": server_instance_id,
        "fixture_sha256": fixture_sha256,
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "token_timestamps": [value / 1_000_000_000 for value in timestamps],
        "evaluated_tokens": prompt_tokens,
        "prefill_seconds": prompt_tokens / prefill_rate,
        "warm_ttft_seconds": _positive_number(warm.get("ttft_s"), "warm TTFT"),
        "cold_ttft_seconds": _positive_number(cold.get("ttft_s"), "cold TTFT"),
        "available_memory_gib": memory,
        "truncated": False,
        "oom": False,
        "xid": False,
        "failures": [],
    }


if __name__ == "__main__":
    raise SystemExit("69_foundation_user_runtime.py is a library; use its registered runner")
