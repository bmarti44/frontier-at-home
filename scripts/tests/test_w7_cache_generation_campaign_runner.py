#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
SPEC = importlib.util.spec_from_file_location("w7_cache_campaign_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7CacheGenerationCampaignRunnerTest(unittest.TestCase):
    def test_self_test_validates_dependencies_without_starting_engine(self) -> None:
        before = subprocess.run(
            ["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True,
            check=False,
        ).stdout
        completed = subprocess.run(
            ["/usr/bin/python3", str(RUNNER), "--self-test"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        after = subprocess.run(
            ["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True,
            check=False,
        ).stdout
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "W7_CACHE_GENERATION_CAMPAIGN_SELFTEST_OK\n")
        self.assertEqual(after, before)

    def test_schedule_is_deterministic_and_uses_only_abba_baab(self) -> None:
        seed = "a" * 64
        first = MODULE.derive_schedules(seed)
        self.assertEqual(first, MODULE.derive_schedules(seed))
        self.assertEqual(len(first), 5)
        self.assertTrue(all(value in {"ABBA", "BAAB"} for value in first))
        self.assertNotEqual(first, MODULE.derive_schedules("b" * 64))
        for invalid in ("", "a" * 63, "A" * 64, "z" * 64):
            with self.assertRaises(ValueError):
                MODULE.derive_schedules(invalid)

    def test_runner_declares_fixed_containment_and_measurement_surface(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required = (
            "glm_cgroup_run.sh", "glm_safe_run.sh", "DS4_TOKEN_TIMING_LOG",
            "DS4_GLM_LOGIT_DUMP_ALL", "DS4_CUDA_STABLE_MODEL_REMAP",
            "GLM_SAFE_MEMORY_HIGH_GIB", "GLM_SAFE_KILL_FLOOR_GIB",
            "GLM_SAFE_MIN_START_GIB", "GLM_SAFE_TIMEOUT_S",
            "MemorySwapMax", "false_generation_flushes", "server_fresh",
            "manifest.json", "raw.jsonl", "summary.json",
            "score_campaign_rows", "pgrep", "ds4-server",
        )
        for value in required:
            self.assertIn(value, source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)


if __name__ == "__main__":
    unittest.main()
