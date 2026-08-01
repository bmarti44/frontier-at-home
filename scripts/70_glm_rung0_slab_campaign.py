#!/usr/bin/env python3
"""Thin Rung 0.1 lifecycle wrapper around the existing speed scorer."""

from __future__ import annotations

from typing import Any


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
    raise NotImplementedError((records, nll))
