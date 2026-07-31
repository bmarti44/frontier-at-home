#!/usr/bin/env python3
"""Contracts for post-freeze, raw W1 telemetry confirmation probes."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/67_run_w1_telemetry_probe.py"
SCORER = ROOT / "scripts/68_score_w1_telemetry_probe.py"
LOAD_SOURCE = ROOT / "scripts/fixtures/w1_direct_io_load.c"


def load_scorer():
    spec = importlib.util.spec_from_file_location("w1_telemetry_scorer", SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load W1 telemetry scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W1TelemetryProbeSourceTests(unittest.TestCase):
    def test_fixed_runner_and_scorer_preserve_raw_evidence(self):
        for path in (RUNNER, SCORER, LOAD_SOURCE):
            self.assertTrue(path.is_file(), f"missing fixed probe artifact: {path}")
        runner = RUNNER.read_text(encoding="utf-8")
        scorer = SCORER.read_text(encoding="utf-8")
        load = LOAD_SOURCE.read_text(encoding="utf-8")
        for required in (
            "git status --porcelain",
            "git show -s --format=%cI",
            "GLM_SAFE_RUN_AS_CURRENT_USER",
            "manifest.json",
            "raw.jsonl",
            "summary.json",
            "O_DIRECT",
        ):
            self.assertIn(required, runner + scorer + load)
        self.assertIn("posix_memalign", load)
        self.assertIn("F_GETFL", load)
        self.assertIn("max_gap_s", scorer)
        self.assertIn("first_minus_executed_s", scorer)
        self.assertIn("completed_minus_last_s", scorer)
        self.assertIn("0.75", scorer)

    def test_scorer_rejects_raw_or_summary_mutation(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            scorer.write_test_package(package)
            accepted = scorer.verify_package(package)
            self.assertEqual(accepted["verdict"], "PASS")

            raw = package / "raw.jsonl"
            original = raw.read_text(encoding="utf-8")
            raw.write_text(original.replace("mem_avail_kb=90000000",
                                            "mem_avail_kb=80000000", 1),
                           encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw evidence hash"):
                scorer.verify_package(package)
            raw.write_text(original, encoding="utf-8")

            summary = package / "summary.json"
            value = json.loads(summary.read_text(encoding="utf-8"))
            value["probes"][0]["max_gap_s"] = 0.1
            summary.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary differs"):
                scorer.verify_package(package)


if __name__ == "__main__":
    unittest.main()
