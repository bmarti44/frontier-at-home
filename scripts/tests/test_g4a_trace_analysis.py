#!/usr/bin/env python3
"""Offline tests for future-token expert-union trace analysis."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "results/glm52-gates/harness/g4a_trace_analysis.py"


class G4ATraceAnalysisTests(unittest.TestCase):
    def run_trace(
        self, rows: list[str], cache_slots: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.log"
            trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
            command = ["python3", str(ANALYZER), str(trace)]
            if cache_slots is not None:
                command.append(str(cache_slots))
            return subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_reports_exact_future_union_size(self):
        rows = [
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
            "XTRACE L3 N8: 8 9 10 11 12 13 14 15",
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
        ]
        completed = self.run_trace(rows)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "future-union K=2 samples=2 mean=16.000 p95=16",
            completed.stdout,
        )

    def test_frequency_prior_is_fit_only_on_training_prefix(self):
        rows = ["XTRACE L3 N8: 0 1 2 3 4 5 6 7"] * 10
        completed = self.run_trace(rows)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "frequency-prior K=2 budget=8 samples=1 recall=1.000000 "
            "set_coverage=1.000000",
            completed.stdout,
        )

    def test_reports_causal_least_stale_against_lru_and_oracle(self):
        rows = [f"XTRACE L3 N8: {expert}" for expert in (1, 2, 3, 1, 2, 3)]
        completed = self.run_trace(rows, cache_slots=2)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "cache-policy slots=2 accesses=6 lru_hit=0.000000 "
            "belady_hit=0.333333 causal_interval_hit=0.333333 "
            "oracle_gain_pp=33.333333 causal_gain_pp=33.333333 "
            "decision=PROCEED_DIAGNOSTIC",
            completed.stdout,
        )

    def test_cache_policy_rejects_nonpositive_capacity(self):
        completed = self.run_trace(["XTRACE L3 N8: 1"], cache_slots=0)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("cache slots must be positive", completed.stderr)

    def test_markov_history_baseline_uses_training_prefix_only(self):
        rows = [
            f"XTRACE L3 N8: {expert}"
            for expert in (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
        ]
        completed = self.run_trace(rows)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "markov-history K=2 budget=2 samples=1 recall=1.000000 "
            "set_coverage=1.000000 frequency_recall=0.500000 "
            "recall_gain_pp=50.000000",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
