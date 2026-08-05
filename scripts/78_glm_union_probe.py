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


def split_compact_arrays(
    arrays: dict[str, np.ndarray], split_rows: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Project a combined compact corpus into isolated split archives."""
    if (
        not isinstance(arrays, dict) or not arrays or
        not isinstance(split_rows, dict) or set(split_rows) != set(SPLIT_COUNTS) or
        any(not isinstance(value, np.ndarray) for value in arrays.values()) or
        "request_index" not in arrays or "layer" not in arrays or
        "hidden_fp16_holdout_row" not in arrays or "hidden_fp16_holdout" not in arrays
    ):
        raise ValueError("compact split input schema is invalid")
    row_count = arrays["request_index"].shape[0]
    holdout_names = {"hidden_fp16_holdout_row", "hidden_fp16_holdout"}
    main_names = set(arrays) - holdout_names
    if (
        row_count <= 0 or any(value.ndim == 0 or value.shape[0] != row_count
                              for name, value in arrays.items() if name in main_names) or
        any(np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
            for value in arrays.values())
    ):
        raise ValueError("compact split row coverage or finiteness differs")
    holdout_row = arrays["hidden_fp16_holdout_row"]
    holdout_hidden = arrays["hidden_fp16_holdout"]
    if (
        holdout_row.ndim != 1 or not np.issubdtype(holdout_row.dtype, np.integer) or
        holdout_hidden.ndim != 2 or holdout_hidden.shape[0] != holdout_row.size or
        np.any(holdout_row < 0) or np.any(holdout_row >= row_count) or
        (holdout_row.size > 1 and np.any(np.diff(holdout_row.astype(np.int64)) <= 0))
    ):
        raise ValueError("compact split holdout coverage is invalid")
    normalized_rows: dict[str, np.ndarray] = {}
    for split, rows in split_rows.items():
        if (
            not isinstance(rows, np.ndarray) or rows.ndim != 1 or
            not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or
            np.any(rows >= row_count) or
            (rows.size > 1 and np.any(np.diff(rows.astype(np.int64)) <= 0))
        ):
            raise ValueError(f"split row indices are invalid: {split}")
        normalized_rows[split] = rows.astype(np.int64, copy=False)
    combined = np.concatenate(list(normalized_rows.values()))
    if combined.size != row_count or not np.array_equal(np.sort(combined), np.arange(row_count)):
        raise ValueError("compact split rows are incomplete or overlap")

    result: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in normalized_rows.items():
        projected = {name: arrays[name][rows] for name in main_names}
        old_to_new = np.full(row_count, -1, dtype=np.int64)
        old_to_new[rows] = np.arange(rows.size)
        selected_holdout = old_to_new[holdout_row.astype(np.int64)] >= 0
        projected["hidden_fp16_holdout_row"] = old_to_new[
            holdout_row[selected_holdout].astype(np.int64)
        ].astype(np.uint32)
        projected["hidden_fp16_holdout"] = holdout_hidden[selected_holdout]
        if (
            projected["hidden_fp16_holdout"].shape[0] !=
            projected["hidden_fp16_holdout_row"].size or
            any(value.shape[0] != rows.size for name, value in projected.items()
                if name not in holdout_names)
        ):
            raise ValueError("projected compact split is inconsistent")
        result[split] = projected
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
