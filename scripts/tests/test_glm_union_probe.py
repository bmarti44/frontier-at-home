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
        self.assertEqual(result["by_budget"]["2"]["macro_request_recall"], 1 / 18)
        self.assertEqual(result["by_budget"]["4"]["macro_request_recall"], 1 / 3)
        self.assertEqual(result["by_budget"]["2"]["event_weighted_wasted_experts"], 9 / 5)

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

    def test_partition_is_request_grouped_complete_and_disjoint(self) -> None:
        request = np.asarray([1, 1, 2, 3, 3, 4], dtype=np.uint16)
        metadata = [
            {"request_index": 1, "case_id": "a", "split": "train-fit"},
            {"request_index": 2, "case_id": "b", "split": "train-precision-diagnostic"},
            {"request_index": 3, "case_id": "c", "split": "calibration"},
            {"request_index": 4, "case_id": "d", "split": "test"},
        ]
        expected = {name: 1 for name in MODULE.SPLIT_COUNTS}
        observed = MODULE.partition_request_rows(request, metadata, expected)
        np.testing.assert_array_equal(observed["train-fit"], [0, 1])
        np.testing.assert_array_equal(observed["train-precision-diagnostic"], [2])
        np.testing.assert_array_equal(observed["calibration"], [3, 4])
        np.testing.assert_array_equal(observed["test"], [5])
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(observed.values()))), np.arange(6),
        )

    def test_partition_rejects_missing_duplicate_or_wrong_split_metadata(self) -> None:
        request = np.asarray([1, 2, 3, 4], dtype=np.uint16)
        valid = [
            {"request_index": 1, "case_id": "a", "split": "train-fit"},
            {"request_index": 2, "case_id": "b", "split": "train-precision-diagnostic"},
            {"request_index": 3, "case_id": "c", "split": "calibration"},
            {"request_index": 4, "case_id": "d", "split": "test"},
        ]
        expected = {name: 1 for name in MODULE.SPLIT_COUNTS}
        mutations = [
            valid[:-1],
            [valid[0], valid[0], *valid[2:]],
            [{**valid[0], "split": "test"}, *valid[1:]],
            [{**valid[0], "case_id": "b"}, *valid[1:]],
        ]
        for metadata in mutations:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                MODULE.partition_request_rows(request, metadata, expected)

    def test_split_arrays_remaps_fp16_holdout_without_leaking_rows(self) -> None:
        rows = {
            "train-fit": np.asarray([0, 1]),
            "train-precision-diagnostic": np.asarray([2]),
            "calibration": np.asarray([3, 4]),
            "test": np.asarray([5]),
        }
        arrays = {
            "request_index": np.asarray([1, 1, 2, 3, 3, 4], dtype=np.uint16),
            "layer": np.asarray([3] * 6, dtype=np.uint16),
            "token_position": np.asarray([0, 1, 0, 0, 1, 0], dtype=np.uint32),
            "selected_ids": np.arange(12, dtype=np.uint8).reshape(6, 2),
            "hidden_fp16_holdout_row": np.asarray([0, 2, 5], dtype=np.uint32),
            "hidden_fp16_holdout": np.asarray([[10], [20], [30]], dtype=np.float16),
        }
        split = MODULE.split_compact_arrays(arrays, rows)
        np.testing.assert_array_equal(split["train-fit"]["request_index"], [1, 1])
        np.testing.assert_array_equal(split["calibration"]["request_index"], [3, 3])
        np.testing.assert_array_equal(
            split["train-fit"]["hidden_fp16_holdout_row"], [0],
        )
        np.testing.assert_array_equal(
            split["train-precision-diagnostic"]["hidden_fp16_holdout"], [[20]],
        )
        np.testing.assert_array_equal(split["test"]["hidden_fp16_holdout_row"], [0])
        self.assertEqual(sum(value["layer"].size for value in split.values()), 6)

    def test_split_arrays_rejects_bad_holdout_or_row_coverage(self) -> None:
        base = {
            "request_index": np.asarray([1, 2], dtype=np.uint16),
            "layer": np.asarray([3, 3], dtype=np.uint16),
            "hidden_fp16_holdout_row": np.asarray([0], dtype=np.uint32),
            "hidden_fp16_holdout": np.asarray([[1]], dtype=np.float16),
        }
        rows = {
            "train-fit": np.asarray([0]),
            "train-precision-diagnostic": np.asarray([], dtype=np.int64),
            "calibration": np.asarray([], dtype=np.int64),
            "test": np.asarray([1]),
        }
        mutations = []
        duplicate = {key: value.copy() for key, value in base.items()}
        duplicate["hidden_fp16_holdout_row"] = np.asarray([0, 0], dtype=np.uint32)
        duplicate["hidden_fp16_holdout"] = np.asarray([[1], [1]], dtype=np.float16)
        mutations.append(duplicate)
        out_of_range = {key: value.copy() for key, value in base.items()}
        out_of_range["hidden_fp16_holdout_row"] = np.asarray([2], dtype=np.uint32)
        mutations.append(out_of_range)
        short = {key: value.copy() for key, value in base.items()}
        short["layer"] = short["layer"][:1]
        mutations.append(short)
        for arrays in mutations:
            with self.subTest(arrays=arrays), self.assertRaises(ValueError):
                MODULE.split_compact_arrays(arrays, rows)


if __name__ == "__main__":
    unittest.main()
