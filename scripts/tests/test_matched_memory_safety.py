#!/usr/bin/env python3
"""Memory-safety contract for matched engine measurements."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "03_memory_guard.py"
HARNESS = ROOT / "results" / "glm52-goal" / "harness" / "decisive_matched.sh"
GLM_SAFE = ROOT / "results" / "glm52-gates" / "harness" / "glm_safe_run.sh"
DSV4_LAUNCHER = ROOT / "scripts" / "21_serve_llamacpp.sh"


class MemoryGuardTests(unittest.TestCase):
    def test_requires_stable_consecutive_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemAvailable: 120000000 kB\n", encoding="ascii")
            result = subprocess.run(
                [
                    "python3",
                    str(GUARD),
                    "--required-gib",
                    "110",
                    "--stable-samples",
                    "3",
                    "--interval-seconds",
                    "0",
                    "--timeout-seconds",
                    "0",
                    "--meminfo",
                    str(meminfo),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"stable_samples_observed":3', result.stdout)

    def test_rejects_insufficient_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemAvailable: 80000000 kB\n", encoding="ascii")
            result = subprocess.run(
                [
                    "python3",
                    str(GUARD),
                    "--required-gib",
                    "110",
                    "--stable-samples",
                    "1",
                    "--interval-seconds",
                    "0",
                    "--timeout-seconds",
                    "0",
                    "--meminfo",
                    str(meminfo),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 1)


class MatchedHarnessContractTests(unittest.TestCase):
    def test_harness_waits_for_full_release_and_serializes_engines(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("03_memory_guard.py", source)
        self.assertIn("--required-gib 110", source)
        self.assertIn("--stable-samples 3", source)
        self.assertIn("/run/dsv4/inference.lock", source)
        self.assertIn("MATCHED_BLOCKS:-5", source)

    def test_harness_does_not_lower_emergency_floor(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertNotIn("GLM_SAFE_KILL_FLOOR_GIB=10", source)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=18", source)
        safe_source = GLM_SAFE.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_MIN_START_GIB=110", source)
        self.assertIn("setsid timeout", safe_source)

    def test_production_watchdog_reserves_18_gib(self):
        source = DSV4_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_WATCHDOG_FLOOR_GIB:-18", source)
        self.assertIn('--threshold-gib "$watchdog_floor_gib"', source)


if __name__ == "__main__":
    unittest.main()
