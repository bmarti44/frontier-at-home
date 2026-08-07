#!/usr/bin/env python3
"""Production-path contract for the real-capture W9 FP4 falsifier."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/93_score_w9_fp4_falsifier.py"
SPEC = importlib.util.spec_from_file_location("w9_fp4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W9Fp4FalsifierTests(unittest.TestCase):
    def test_e2m1_quantizer_uses_exact_blocks_and_finite_values(self) -> None:
        rows = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]],
                        dtype=np.float32)
        quantized = MODULE.e2m1_quantize(rows, block_width=8)
        np.testing.assert_array_equal(quantized, rows)
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.e2m1_quantize(np.array([[np.nan]], dtype=np.float32), 1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MODULE.e2m1_quantize(rows, block_width=3)

    def test_hadamard_rotation_preserves_dot_products(self) -> None:
        rng = np.random.default_rng(8)
        left = rng.normal(size=(7, 8)).astype(np.float32)
        right = rng.normal(size=(5, 8)).astype(np.float32)
        signs = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)
        rotated_left = MODULE.hadamard_rotate(left, signs)
        rotated_right = MODULE.hadamard_rotate(right, signs)
        np.testing.assert_allclose(rotated_left @ rotated_right.T,
                                   left @ right.T, rtol=2e-6, atol=2e-6)

    def test_split_is_seeded_disjoint_complete_and_stable(self) -> None:
        first = MODULE.split_indices(32, bytes.fromhex("11" * 32), b"keys")
        second = MODULE.split_indices(32, bytes.fromhex("11" * 32), b"keys")
        self.assertEqual(first, second)
        calibration, heldout = first
        self.assertEqual(len(calibration), 16)
        self.assertEqual(set(calibration) & set(heldout), set())
        self.assertEqual(set(calibration) | set(heldout), set(range(32)))

    def test_channel_correction_fits_calibration_rows_only(self) -> None:
        quantized = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        reference = quantized * np.array([2.0, 0.5], dtype=np.float32)
        alpha = MODULE.fit_channel_correction(reference, quantized)
        np.testing.assert_allclose(alpha, [2.0, 0.5], rtol=0, atol=1e-7)

    def test_query_weighted_error_uses_only_heldout_selected_pairs(self) -> None:
        keys = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [8.0, 8.0]],
                        dtype=np.float32)
        candidate = keys.copy()
        candidate[1] = [0.0, 1.0]
        candidate[3] = [100.0, 100.0]  # not selected and must be irrelevant
        queries = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        selected = np.array([[0, 4, 4], [1, 2, 4]], dtype=np.uint32)
        metric = MODULE.query_weighted_error(
            queries, keys, candidate, selected, selected_sentinel=4,
            heldout_queries=np.array([False, True]),
            heldout_keys=np.array([True, True, True, False]),
        )
        # Held-out logits are [2, 1], errors [-1, 0].
        self.assertEqual(metric["pairs"], 2)
        self.assertAlmostEqual(metric["relative_rmse"], math.sqrt(1.0 / 5.0), places=7)

    def test_selected_rows_reject_duplicates_and_out_of_range_ids(self) -> None:
        queries = np.zeros((1, 1, 2), dtype=np.float32)
        keys = np.zeros((2, 2), dtype=np.float32)
        mask_q = np.array([True])
        mask_k = np.array([True, True])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.query_weighted_error(
                queries, keys, keys, np.array([[0, 0]], dtype=np.uint32), 2,
                mask_q, mask_k)
        with self.assertRaisesRegex(ValueError, "range"):
            MODULE.query_weighted_error(
                queries, keys, keys, np.array([[3, 2]], dtype=np.uint32), 2,
                mask_q, mask_k)

    def test_fixed_gate_threshold_and_candidates_are_source_bound(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "MAXIMUM_RELATIVE_RMSE = 0.05",
            '"plain_e2m1_f32_scale"',
            '"hadamard_e2m1_f32_scale"',
            '"hadamard_e2m1_f32_scale_channel_correction"',
            "manifest.json", "raw.jsonl", "summary.json",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
