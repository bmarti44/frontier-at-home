#!/usr/bin/env python3
"""Train and score frozen GLM multi-token expert-union probes."""

from __future__ import annotations

import numpy as np


K_VALUES = (2, 4, 8)
BUDGETS = (16, 32, 64)
N_EXPERT = 256


def future_union_targets(
    request_index: np.ndarray,
    layer: np.ndarray,
    token_position: np.ndarray,
    selected_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid source-row indices and exact future-K expert-union labels."""
    raise NotImplementedError


def score_rankings(
    row_indices: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    request_index: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, object]:
    """Score macro-request and event-weighted recall/precision without rounding."""
    raise NotImplementedError

