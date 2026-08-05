#!/usr/bin/env python3
"""Tests for the frozen GLM probe feature-precision comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/80_glm_union_probe_precision.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe_precision", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrecisionDiagnosticTests(unittest.TestCase):
    def test_paired_metrics_preserve_rows_and_measure_top32_overlap(self) -> None:
        requests = np.asarray([1, 1, 2], dtype=np.uint16)
        targets = np.zeros((3, 256), dtype=np.bool_)
        targets[0, [0, 1]] = True
        targets[1, [2, 3]] = True
        targets[2, [4, 5]] = True
        q4 = np.tile(-np.arange(256, dtype=np.float32), (3, 1))
        fp16 = q4.copy()
        q4[1, 2] = 1000
        fp16[2] = np.roll(fp16[2], 64)
        result = MODULE.diagnostic_pair_metrics(requests, targets, q4, fp16)
        self.assertEqual(result["q4"]["32"]["1"]["events"], 2)
        self.assertEqual(result["fp16"]["32"]["2"]["events"], 1)
        self.assertEqual(result["top32_overlap"]["1"]["events"], 2)
        self.assertEqual(result["top32_overlap"]["2"]["overlap_sum"], 0)
        np.testing.assert_array_equal(result["evidence"]["request"], requests)

    def test_paired_metrics_reject_unpaired_nonfinite_or_empty_inputs(self) -> None:
        requests = np.asarray([1], dtype=np.uint16)
        targets = np.zeros((1, 256), dtype=np.bool_)
        targets[0, 1] = True
        logits = np.zeros((1, 256), dtype=np.float32)
        mutations = [
            (requests[:0], targets[:0], logits[:0], logits[:0]),
            (requests, targets, logits[:, :-1], logits),
            (requests, targets, logits, np.full_like(logits, np.nan)),
        ]
        for values in mutations:
            with self.subTest(shapes=[value.shape for value in values]), self.assertRaises(ValueError):
                MODULE.diagnostic_pair_metrics(*values)

    def test_overlap_aggregation_is_request_macro_and_event_weighted(self) -> None:
        result = MODULE.aggregate_overlap([
            {"1": {"overlap_sum": 32, "events": 1}},
            {"1": {"overlap_sum": 0, "events": 1}, "2": {"overlap_sum": 96, "events": 3}},
        ])
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["events"], 5)
        self.assertEqual(result["event_weighted_overlap"], 0.8)
        self.assertEqual(result["macro_request_overlap"], 0.75)


if __name__ == "__main__":
    unittest.main()
