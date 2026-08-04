#!/usr/bin/env python3
"""Acceptance and source contract for R0-UPGRADE a."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/72_glm_shared_router_score.py"
PATCH = ROOT / "results/glm52-gates/harness/ds4-shared-router-correction.patch"
SPEC = importlib.util.spec_from_file_location("shared_router_score", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(layer: int, actual: range, baseline: range, shared: range) -> str:
    values = lambda items: " ".join(str(item) for item in items)
    return (
        f"PREDPAIR L{layer} actual: {values(actual)}"
        f" base: {values(baseline)} shared: {values(shared)}"
    )


class SharedRouterScorerTests(unittest.TestCase):
    def score_rows(self, rows: list[str]) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return MODULE.score(path)

    def test_accepts_matched_trace_with_two_point_recall_gain(self) -> None:
        rows = [
            row(4 + index % 74, range(8), range(8), range(8))
            for index in range(1000)
        ]
        # Lower the baseline by exactly two of eight experts in 80 samples:
        # 160 / 8000 = 0.02 absolute recall.
        for index in range(80):
            rows[index] = row(4 + index % 74, range(8), range(2, 10), range(8))
        result = self.score_rows(rows)
        self.assertEqual(result["verdict"], "PASS")
        self.assertAlmostEqual(result["absolute_recall_gain"], 0.02)

    def test_rejects_too_few_rows(self) -> None:
        result = self.score_rows([row(4, range(8), range(2, 10), range(8))] * 999)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["minimum_samples"])

    def test_rejects_no_gain(self) -> None:
        result = self.score_rows([row(4, range(8), range(8), range(8))] * 1000)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["shared_recall_gain"])

    def test_rejects_malformed_or_duplicate_ids(self) -> None:
        rows = [row(4, range(8), range(2, 10), range(8))] * 1000
        rows[0] = "PREDPAIR L4 actual: 0 0 1 2 3 4 5 6 base: 2 3 4 5 6 7 8 9 shared: 0 1 2 3 4 5 6 7"
        result = self.score_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["malformed_rows"], 1)


class SharedRouterSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PATCH.read_text(encoding="utf-8") if PATCH.exists() else ""

    def test_correction_is_explicit_and_default_off(self) -> None:
        self.assertIn('getenv("DS4_GLM_PREFETCH_SHARED_CORRECTION")', self.source)
        self.assertIn("shared-expert router correction enabled", self.source)

    def test_correction_uses_shared_residual_and_next_layer_norm(self) -> None:
        for marker in (
            "ds4_gpu_add_tensor(pf_corrected_state, after_attn, ffn_sum",
            "lnext->ffn_norm->abs_offset",
            "ds4_gpu_rms_norm_weight_tensor(pf_corrected_norm",
            "g->batch_router_logits",
        ):
            self.assertIn(marker, self.source)

    def test_probe_logs_matched_actual_baseline_and_shared_sets(self) -> None:
        self.assertIn("PREDPAIR L%u actual:", self.source)
        self.assertIn('getenv("DS4_GLM_PREDACC_SHARED")', self.source)

    def test_prefetch_hint_stays_after_current_selected_load(self) -> None:
        self.assertIn("shared correction waits for current selected load", self.source)


if __name__ == "__main__":
    unittest.main()
