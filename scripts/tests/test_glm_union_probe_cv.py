#!/usr/bin/env python3
"""Acceptance tests for train-only GLM probe CV metric aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/79_glm_union_probe_cv.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe_cv", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CVMetricTests(unittest.TestCase):
    def test_production_main_runs_only_after_metric_definitions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertGreater(source.rfind('if __name__ == "__main__"'), source.rfind("def aggregate_request_metrics"))

    def test_aggregation_is_request_macro_not_event_or_layer_macro(self) -> None:
        requests = np.asarray([1, 1, 2], dtype=np.uint16)
        targets = np.zeros((3, 256), dtype=np.bool_)
        targets[0, [0, 1]] = True
        targets[1, [2, 3]] = True
        targets[2, [9, 10]] = True
        rankings = np.tile(np.arange(256, dtype=np.uint16), (3, 1))
        rankings[2] = np.concatenate(([9, 10], np.arange(0, 9), np.arange(11, 256)))
        first = MODULE.accumulate_request_metrics(requests, targets, rankings, budgets=(1, 2))
        second = MODULE.accumulate_request_metrics(requests, targets, rankings, budgets=(1, 2))
        result = MODULE.aggregate_request_metrics([first, second])
        self.assertEqual(result["2"]["requests"], 2)
        self.assertEqual(result["2"]["events"], 6)
        self.assertEqual(result["2"]["macro_request_recall"], 0.75)
        self.assertEqual(result["2"]["event_weighted_recall"], 2 / 3)

    def test_aggregation_rejects_duplicate_rankings_and_request_drift(self) -> None:
        requests = np.asarray([1], dtype=np.uint16)
        targets = np.zeros((1, 256), dtype=np.bool_)
        targets[0, 1] = True
        rankings = np.tile(np.arange(256, dtype=np.uint16), (1, 1))
        rankings[0, 2] = rankings[0, 1]
        with self.assertRaises(ValueError):
            MODULE.accumulate_request_metrics(requests, targets, rankings)
        valid = np.tile(np.arange(256, dtype=np.uint16), (1, 1))
        one = MODULE.accumulate_request_metrics(requests, targets, valid)
        changed = {budget: {"2": values["1"]} for budget, values in one.items()}
        with self.assertRaises(ValueError):
            MODULE.aggregate_request_metrics([one, changed])

    def test_fold_weights_exclude_validation_lengths_and_equalize_each_k(self) -> None:
        request = np.asarray([1] * 10 + [2] * 4 + [3] * 6, dtype=np.uint16)
        rows = np.arange(20, dtype=np.int64)
        valid = np.ones((20, 3), dtype=np.bool_)
        valid[8:, 2] = False
        folds = np.asarray([0] * 10 + [1] * 10, dtype=np.uint8)
        weights, fitting = MODULE.fold_training_weights(request, rows, valid, folds, 0)
        np.testing.assert_array_equal(fitting, np.arange(10, 20))
        self.assertTrue(np.all(weights[:10] == 0))
        for k_index in range(3):
            active = (weights[:, k_index] > 0)
            if not active.any():
                continue
            masses = [
                float(weights[(request == identity) & active, k_index].sum())
                for identity in np.unique(request[active])
            ]
            self.assertAlmostEqual(min(masses), max(masses))
        mutated = valid.copy()
        mutated[:10, 1:] = False
        changed, _ = MODULE.fold_training_weights(request, rows, mutated, folds, 0)
        np.testing.assert_array_equal(weights[10:], changed[10:])


if __name__ == "__main__":
    unittest.main()
