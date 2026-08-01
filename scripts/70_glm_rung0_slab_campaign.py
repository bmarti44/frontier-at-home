#!/usr/bin/env python3
"""Thin Rung 0.1 lifecycle wrapper around the existing speed scorer."""

from __future__ import annotations

from typing import Any


def arm_schedule() -> tuple[tuple[int, int, str], ...]:
    """Return the preregistered five-block execution order."""
    return ()


def canonical_engine_environment(mode: str) -> dict[str, str]:
    """Return the exact timed engine environment for one arm."""
    raise NotImplementedError(mode)


def score_campaign(records: list[dict[str, Any]], nll: dict[str, Any]) -> dict[str, Any]:
    """Validate raw arms and apply the fixed Rung 0.1 formulas."""
    raise NotImplementedError((records, nll))

