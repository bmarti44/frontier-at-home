#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/87_score_w7_resume_production.py"
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_production_v1.sh"
SPEC = importlib.util.spec_from_file_location("w7_equivalence_scorer", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7ProductionContractTest(unittest.TestCase):
    def test_candidate_frontier_requires_automatic_restore_without_diagnostic(self) -> None:
        automatic_log = (
            "kv cache hit text tokens=5044 file=/tmp/checkpoint.kv\n"
            "ds4: GLM sync start=5044 prompt=5066 suffix=22\n"
            "ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1\n"
        )
        self.assertTrue(MODULE._candidate_frontier_observed(automatic_log))
        self.assertFalse(
            MODULE._candidate_frontier_observed(
                automatic_log
                + "ds4: GLM restored-frontier diagnostic: authoritative "
                "checkpoint=5044 compact_rows=5044 prior_frontier=5044\n"
            )
        )

    def test_harness_never_injects_diagnostic_opt_in(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertNotIn("DS4_GLM_RESTORED_FRONTIER_DIAGNOSTIC", source)
        self.assertIn(
            "readonly BIN=/home/bmarti44/.cache/glm52-w7-3ba062e-runtime/ds4-server",
            source,
        )


if __name__ == "__main__":
    unittest.main()
