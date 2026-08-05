#!/usr/bin/env python3
"""Train and score frozen GLM multi-token expert-union probes."""

from __future__ import annotations

import numpy as np


K_VALUES = (2, 4, 8)
BUDGETS = (16, 32, 64)
N_EXPERT = 256
SPLIT_COUNTS = {
    "train-fit": 55,
    "train-precision-diagnostic": 5,
    "calibration": 20,
    "test": 20,
}


def partition_request_rows(
    request_index: np.ndarray,
    request_metadata: list[dict[str, object]],
    expected_counts: dict[str, int] = SPLIT_COUNTS,
) -> dict[str, np.ndarray]:
    """Return exact row indices for the preregistered request-grouped splits."""
    if (
        not isinstance(request_index, np.ndarray) or request_index.ndim != 1 or
        request_index.size == 0 or not np.issubdtype(request_index.dtype, np.integer) or
        np.any(request_index <= 0) or np.any(np.diff(request_index.astype(np.int64)) < 0) or
        not isinstance(request_metadata, list) or not request_metadata or
        not isinstance(expected_counts, dict) or set(expected_counts) != set(SPLIT_COUNTS) or
        any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in expected_counts.values())
    ):
        raise ValueError("split input schema is invalid")
    metadata_by_request: dict[int, dict[str, object]] = {}
    case_ids: set[str] = set()
    split_counts = {name: 0 for name in expected_counts}
    for row in request_metadata:
        if not isinstance(row, dict):
            raise ValueError("request metadata row is malformed")
        request = row.get("request_index")
        case = row.get("case_id")
        split = row.get("split")
        if (
            not isinstance(request, int) or isinstance(request, bool) or request <= 0 or
            request in metadata_by_request or not isinstance(case, str) or not case or
            case in case_ids or split not in expected_counts
        ):
            raise ValueError("request metadata identity or split is invalid")
        metadata_by_request[request] = row
        case_ids.add(case)
        split_counts[str(split)] += 1
    if split_counts != expected_counts:
        raise ValueError("request split counts differ from the frozen plan")
    observed_requests = set(int(value) for value in np.unique(request_index))
    if observed_requests != set(metadata_by_request):
        raise ValueError("request rows and metadata are not a bijection")
    split_by_request = {
        request: str(row["split"]) for request, row in metadata_by_request.items()
    }
    result = {
        split: np.flatnonzero(np.asarray([
            split_by_request[int(request)] == split for request in request_index
        ], dtype=np.bool_)).astype(np.int64)
        for split in expected_counts
    }
    combined = np.concatenate(list(result.values()))
    if (
        combined.size != request_index.size or
        not np.array_equal(np.sort(combined), np.arange(request_index.size))
    ):
        raise ValueError("split rows are incomplete or overlap")
    return result


def future_union_targets(
    request_index: np.ndarray,
    layer: np.ndarray,
    token_position: np.ndarray,
    selected_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid source-row indices and exact future-K expert-union labels."""
    arrays = (request_index, layer, token_position, selected_ids)
    if any(not isinstance(value, np.ndarray) for value in arrays):
        raise ValueError("future-union inputs must be numpy arrays")
    if (
        request_index.ndim != 1 or layer.ndim != 1 or token_position.ndim != 1 or
        selected_ids.ndim != 2 or selected_ids.shape[0] != request_index.size or
        layer.size != request_index.size or token_position.size != request_index.size or
        request_index.size == 0 or k not in K_VALUES or
        not all(np.issubdtype(value.dtype, np.integer) for value in arrays) or
        np.any(request_index <= 0) or np.any(layer < 0) or np.any(token_position < 0) or
        np.any(selected_ids < 0) or np.any(selected_ids >= N_EXPERT) or
        any(np.unique(row).size != row.size for row in selected_ids)
    ):
        raise ValueError("future-union input schema is invalid")

    groups: list[tuple[int, int]] = []
    starts: list[int] = []
    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for index, key in enumerate(zip(request_index.tolist(), layer.tolist())):
        current = (int(key[0]), int(key[1]))
        if current != previous:
            if current in seen:
                raise ValueError("request/layer group is repeated or reordered")
            if previous is not None and current <= previous:
                raise ValueError("request/layer groups are not in canonical order")
            seen.add(current)
            groups.append(current)
            starts.append(index)
            previous = current
    starts.append(request_index.size)

    output_rows: list[int] = []
    output_targets: list[np.ndarray] = []
    for group_index in range(len(groups)):
        start, stop = starts[group_index], starts[group_index + 1]
        positions = token_position[start:stop].astype(np.int64, copy=False)
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
            raise ValueError("token positions are gapped, duplicated, or reordered")
        for source in range(start, stop - k):
            target = np.zeros(N_EXPERT, dtype=np.bool_)
            target[selected_ids[source + 1:source + k + 1].reshape(-1)] = True
            if not target.any():
                raise ValueError("future expert union is empty")
            output_rows.append(source)
            output_targets.append(target)
    if not output_rows:
        raise ValueError("no complete future-K window exists")
    return np.asarray(output_rows, dtype=np.int64), np.stack(output_targets)


def score_rankings(
    row_indices: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    request_index: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, object]:
    """Score macro-request and event-weighted recall/precision without rounding."""
    if (
        not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1 or
        not isinstance(targets, np.ndarray) or targets.shape != (row_indices.size, N_EXPERT) or
        not isinstance(rankings, np.ndarray) or rankings.shape != targets.shape or
        not isinstance(request_index, np.ndarray) or request_index.ndim != 1 or
        row_indices.size == 0 or not np.issubdtype(row_indices.dtype, np.integer) or
        not np.issubdtype(rankings.dtype, np.integer) or
        np.any(row_indices < 0) or np.any(row_indices >= request_index.size) or
        np.any(rankings < 0) or np.any(rankings >= N_EXPERT) or
        any(np.unique(row).size != N_EXPERT for row in rankings) or
        any(not isinstance(value, int) or isinstance(value, bool) or
            not 1 <= value <= N_EXPERT for value in budgets) or
        len(set(budgets)) != len(budgets)
    ):
        raise ValueError("ranking scorer input schema is invalid")
    if not np.issubdtype(targets.dtype, np.bool_):
        if not np.issubdtype(targets.dtype, np.integer) or np.any((targets != 0) & (targets != 1)):
            raise ValueError("targets must be Boolean")
        targets = targets.astype(np.bool_)
    target_sizes = targets.sum(axis=1)
    if np.any(target_sizes <= 0):
        raise ValueError("target union is empty")
    requests = request_index[row_indices]
    unique_requests = np.unique(requests)
    if np.any(unique_requests <= 0):
        raise ValueError("request identity is invalid")

    by_budget: dict[str, dict[str, float]] = {}
    event_rows = np.arange(row_indices.size)[:, None]
    for budget in budgets:
        predicted = rankings[:, :budget]
        hits = targets[event_rows, predicted].sum(axis=1).astype(np.float64)
        recall = hits / target_sizes
        precision = hits / budget
        wasted = budget - hits
        coverage = (hits == target_sizes).astype(np.float64)

        def macro(values: np.ndarray) -> float:
            return float(np.mean([
                np.mean(values[requests == request]) for request in unique_requests
            ]))

        by_budget[str(budget)] = {
            "macro_request_recall": macro(recall),
            "macro_request_precision": macro(precision),
            "macro_request_wasted_experts": macro(wasted),
            "macro_request_full_set_coverage": macro(coverage),
            "event_weighted_recall": float(np.mean(recall)),
            "event_weighted_precision": float(np.mean(precision)),
            "event_weighted_wasted_experts": float(np.mean(wasted)),
            "event_weighted_full_set_coverage": float(np.mean(coverage)),
        }
    return {
        "requests": int(unique_requests.size),
        "events": int(row_indices.size),
        "budgets": list(budgets),
        "by_budget": by_budget,
    }
