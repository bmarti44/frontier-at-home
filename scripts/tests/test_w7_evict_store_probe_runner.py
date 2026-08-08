#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/93_run_w7_evict_store_probe.py"
SPEC = importlib.util.spec_from_file_location("w7_evict_store_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7EvictStoreProbeRunnerTests(unittest.TestCase):
    def test_self_test_checks_dependencies_without_engine(self) -> None:
        before = subprocess.run(["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True).stdout
        completed = subprocess.run(
            ["/usr/bin/python3", str(RUNNER), "--self-test"],
            capture_output=True, text=True, timeout=30,
        )
        after = subprocess.run(["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True).stdout
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "W7_EVICT_STORE_PROBE_SELFTEST_OK\n")
        self.assertEqual(after, before)

    def test_order_is_deterministic_and_balanced(self) -> None:
        for seed in ("a" * 64, "b" * 64, "0" * 64, "f" * 64):
            order = MODULE.derive_order(seed)
            self.assertEqual(order, MODULE.derive_order(seed))
            self.assertEqual(set(order), {"off", "on"})
        for invalid in ("", "a" * 63, "A" * 64, "z" * 64):
            with self.assertRaises(ValueError):
                MODULE.derive_order(invalid)

    def test_runner_uses_frozen_lifecycle_and_bounded_flag(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for required in (
            "91_run_w7_cache_generation_campaign.py", "glm_safe_run.sh",
            "glm_cgroup_run_w7_evict_store_v1.sh", MODULE.FLAG,
            "MemorySwapMax", "minimum_start_GiB", "model content identity mismatch",
            "evict_store_count", "selected_checkpoint_tokens", "logit_sha256s",
            "manifest.json", "raw.jsonl", "summary.json", "public_randomness",
        ):
            self.assertIn(required, source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)


if __name__ == "__main__":
    unittest.main()
