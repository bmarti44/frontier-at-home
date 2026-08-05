#!/usr/bin/env python3
"""Acceptance tests for lossless-label, measured-loss feature compaction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/77_compact_glm_union_trace.py"
SPEC = importlib.util.spec_from_file_location("glm_union_trace_compact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompactArrayTests(unittest.TestCase):
    def fixture(self):
        hidden = np.array([
            [-7.0, -3.5, -1.0, 0.0, 1.0, 3.5, 6.0, 7.0],
            [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0],
        ], dtype=np.float32)
        logits = np.array([
            [0.1, 0.7, -0.2, 1.0, 0.3, 0.9, -0.4, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        selected = np.array([[7, 6], [6, 7]], dtype=np.int32)
        return hidden, logits, bias, selected

    def test_preserves_labels_and_stable_effective_score_topk(self) -> None:
        hidden, logits, bias, selected = self.fixture()
        compact, metrics = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=4,
        )
        np.testing.assert_array_equal(compact["selected_ids"], selected.astype(np.uint8))
        np.testing.assert_array_equal(compact["top_ids"][0], [7, 6, 3, 5])
        np.testing.assert_array_equal(compact["top_ids"][1], [6, 7, 0, 1])
        gathered = np.take_along_axis(logits, compact["top_ids"], axis=1)
        np.testing.assert_array_equal(compact["top_logits"], gathered.astype(np.float16))
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["hidden_values"], 16)

    def test_int4_round_trip_obeys_per_row_half_step_bound(self) -> None:
        hidden, logits, bias, selected = self.fixture()
        compact, metrics = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=4,
        )
        restored = MODULE.unpack_hidden_int4(
            compact["hidden_q4"], compact["hidden_scale"], hidden.shape[1],
        )
        error = np.abs(restored - hidden)
        bound = compact["hidden_scale"].astype(np.float32)[:, None] / 2 + 1e-6
        self.assertTrue(np.all(error <= bound))
        self.assertAlmostEqual(metrics["hidden_max_abs_error"], float(error.max()), places=6)
        self.assertGreaterEqual(metrics["hidden_nrmse"], 0.0)

    def test_captured_probabilities_preserve_fp32_tie_order(self) -> None:
        hidden = np.array([[0.0, 1.0]], dtype=np.float32)
        logits = np.zeros((1, 4), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        probabilities = np.full((1, 4), 0.5, dtype=np.float32)
        probabilities[0, 1] = np.nextafter(np.float32(0.5), np.float32(1.0))
        selected = np.array([[1, 0]], dtype=np.int32)
        compact, _ = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=2,
            router_probs=probabilities,
        )
        np.testing.assert_array_equal(compact["top_ids"], [[1, 0]])

    def test_rejects_nonfinite_and_malformed_inputs(self) -> None:
        hidden, logits, bias, selected = self.fixture()
        for mutation in ("nan", "rows", "selected", "top_k"):
            with self.subTest(mutation=mutation):
                h, g, b, s = hidden.copy(), logits.copy(), bias.copy(), selected.copy()
                top_k = 4
                if mutation == "nan":
                    h[0, 0] = np.nan
                elif mutation == "rows":
                    g = g[:1]
                elif mutation == "selected":
                    s[0, 0] = 9
                else:
                    top_k = 9
                with self.assertRaises(ValueError):
                    MODULE.compact_arrays(h, g, b, s, top_k=top_k)


if __name__ == "__main__":
    unittest.main()
