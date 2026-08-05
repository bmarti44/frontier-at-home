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
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(targets, np.ndarray) or targets.shape != (requests.size, 256) or
        not isinstance(rankings, np.ndarray) or rankings.shape != targets.shape or
        not np.issubdtype(rankings.dtype, np.integer) or
        np.any(rankings < 0) or np.any(rankings >= 256) or
        any(np.unique(row).size != 256 for row in rankings) or
        any(not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 256
            for budget in budgets) or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("CV metric input schema is invalid")
    if targets.dtype != np.bool_:
        if not np.issubdtype(targets.dtype, np.integer) or np.any((targets != 0) & (targets != 1)):
            raise ValueError("CV targets must be Boolean")
        targets = targets.astype(np.bool_)
    sizes = targets.sum(axis=1)
    if np.any(sizes <= 0):
        raise ValueError("CV target union is empty")
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    event_rows = np.arange(requests.size)[:, None]
    for budget in budgets:
        hits = targets[event_rows, rankings[:, :budget]].sum(axis=1).astype(np.float64)
        values = {
            "recall_sum": hits / sizes,
            "precision_sum": hits / budget,
            "wasted_sum": budget - hits,
            "coverage_sum": (hits == sizes).astype(np.float64),
        }
        per_request = {}
        for request in np.unique(requests):
            mask = requests == request
            per_request[str(int(request))] = {
                name: float(value[mask].sum()) for name, value in values.items()
            } | {"events": int(mask.sum())}
        output[str(budget)] = per_request
    return output


def aggregate_request_metrics(
    layers: list[dict[str, dict[str, dict[str, float | int]]]],
) -> dict[str, dict[str, float]]:
    """Aggregate event sums within request, then macro-average across requests."""
    if not isinstance(layers, list) or not layers or any(not isinstance(layer, dict) for layer in layers):
        raise ValueError("CV layer metrics are missing")
    budgets = set(layers[0])
    if not budgets or any(set(layer) != budgets for layer in layers):
        raise ValueError("CV layer budget sets differ")
    output: dict[str, dict[str, float]] = {}
    required = {"recall_sum", "precision_sum", "wasted_sum", "coverage_sum", "events"}
    for budget in sorted(budgets, key=int):
        requests = set(layers[0][budget])
        if not requests or any(set(layer[budget]) != requests for layer in layers):
            raise ValueError("CV request coverage differs across layers")
        totals = {
            request: {name: 0.0 for name in required - {"events"}} | {"events": 0}
            for request in requests
        }
        for layer in layers:
            for request, record in layer[budget].items():
                if set(record) != required or not isinstance(record["events"], int) or record["events"] <= 0:
                    raise ValueError("CV request metric record is malformed")
                for name in required - {"events"}:
                    value = record[name]
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("CV request metric is non-finite")
                    totals[request][name] += float(value)
                totals[request]["events"] += record["events"]
        request_means = {
            name: [totals[request][name] / totals[request]["events"] for request in requests]
            for name in required - {"events"}
        }
        event_count = sum(totals[request]["events"] for request in requests)
        output[budget] = {
            "requests": len(requests),
            "events": event_count,
            "macro_request_recall": float(np.mean(request_means["recall_sum"])),
            "macro_request_precision": float(np.mean(request_means["precision_sum"])),
            "macro_request_wasted_experts": float(np.mean(request_means["wasted_sum"])),
            "macro_request_full_set_coverage": float(np.mean(request_means["coverage_sum"])),
            "event_weighted_recall": sum(totals[r]["recall_sum"] for r in requests) / event_count,
            "event_weighted_precision": sum(totals[r]["precision_sum"] for r in requests) / event_count,
            "event_weighted_wasted_experts": sum(totals[r]["wasted_sum"] for r in requests) / event_count,
            "event_weighted_full_set_coverage": sum(totals[r]["coverage_sum"] for r in requests) / event_count,
        }
    return output
