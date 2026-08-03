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
    def test_reports_exact_future_union_size(self):
        rows = [
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
            "XTRACE L3 N8: 8 9 10 11 12 13 14 15",
            "XTRACE L3 N8: 0 1 2 3 4 5 6 7",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.log"
            trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
            completed = subprocess.run(
                ["python3", str(ANALYZER), str(trace)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "future-union K=2 samples=2 mean=16.000 p95=16",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
