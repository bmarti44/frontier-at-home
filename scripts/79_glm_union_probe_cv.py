#!/usr/bin/env python3
"""Run frozen train-only grouped CV for the GLM direct expert-union probe."""

from __future__ import annotations

import numpy as np


def accumulate_request_metrics(
    requests: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    budgets: tuple[int, ...] = (16, 32, 64),
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return exact per-request sums/counts for one K and one routed layer."""
    raise NotImplementedError


def aggregate_request_metrics(
    layers: list[dict[str, dict[str, dict[str, float | int]]]],
) -> dict[str, dict[str, float]]:
    """Aggregate event sums within request, then macro-average across requests."""
    raise NotImplementedError
