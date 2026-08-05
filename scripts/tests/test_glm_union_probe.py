#!/usr/bin/env python3
"""Acceptance tests for exact future-union construction and scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/78_glm_union_probe.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnionTargetTests(unittest.TestCase):
    def fixture(self):
        request = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint16)
        layer = np.asarray([3] * 9, dtype=np.uint16)
        position = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.uint32)
        selected = np.asarray([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [10, 11], [11, 12], [12, 13], [13, 14],
        ], dtype=np.uint8)
        return request, layer, position, selected

    def test_future_union_is_exact_and_never_crosses_request_boundary(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        np.testing.assert_array_equal(rows, [0, 1, 2, 5, 6])
        expected = [
            {1, 2, 3}, {2, 3, 4}, {3, 4, 5},
            {11, 12, 13}, {12, 13, 14},
        ]
        self.assertEqual(target.shape, (5, 256))
        for row, wanted in zip(target, expected):
            self.assertEqual(set(np.flatnonzero(row)), wanted)

    def test_future_union_rejects_noncontiguous_or_reordered_rows(self) -> None:
        request, layer, position, selected = self.fixture()
        for mutation in ("gap", "reorder", "bad_k", "duplicate_expert"):
            with self.subTest(mutation=mutation):
                changed_position = position.copy()
                changed_selected = selected.copy()
                k = 2
                if mutation == "gap":
                    changed_position[2] = 7
                elif mutation == "reorder":
                    changed_position[[1, 2]] = changed_position[[2, 1]]
                elif mutation == "bad_k":
                    k = 3
                else:
                    changed_selected[0] = [1, 1]
                with self.assertRaises(ValueError):
                    MODULE.future_union_targets(
                        request, layer, changed_position, changed_selected, k=k,
                    )

    def test_scoring_uses_unweighted_request_macro_and_reports_waste(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        rankings = np.tile(np.arange(256, dtype=np.uint16), (len(rows), 1))
        result = MODULE.score_rankings(
            rows, target, rankings, request, budgets=(2, 4),
        )
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["events"], 5)
        self.assertEqual(result["budgets"], [2, 4])
        self.assertEqual(result["by_budget"]["2"]["macro_request_recall"], 1 / 9)
        self.assertEqual(result["by_budget"]["4"]["macro_request_recall"], 4 / 9)
        self.assertEqual(result["by_budget"]["2"]["event_weighted_wasted_experts"], 5 / 3)

    def test_scoring_rejects_duplicate_rankings_or_cross_row_mismatch(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        rankings = np.tile(np.arange(256, dtype=np.uint16), (len(rows), 1))
        rankings[0, 1] = rankings[0, 0]
        with self.assertRaises(ValueError):
            MODULE.score_rankings(rows, target, rankings, request)
        with self.assertRaises(ValueError):
            MODULE.score_rankings(rows[:-1], target, rankings, request)


if __name__ == "__main__":
    unittest.main()

