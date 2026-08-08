#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
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
            "install_campaign_signal_handlers", "finalize_failure_triplet",
        ):
            self.assertIn(required, source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)

    def test_post_attempt_failure_is_preserved_as_triplet(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-evict-finalize-") as temporary:
            attempt = Path(temporary)
            MODULE._ACTIVE_ATTEMPT = attempt
            MODULE._ACTIVE_CANDIDATE = "a" * 40
            try:
                MODULE.finalize_failure_triplet(RuntimeError("injected"))
            finally:
                MODULE._ACTIVE_ATTEMPT = None
                MODULE._ACTIVE_CANDIDATE = None
            self.assertEqual((attempt / "raw.jsonl").read_bytes(), b"")
            summary = json.loads((attempt / "summary.json").read_text())
            manifest = json.loads((attempt / "manifest.json").read_text())
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertIn("RuntimeError: injected", summary["failure"])
            self.assertEqual(manifest["schema"], "glm52-w7-evict-store-probe-failure-v1")
            self.assertEqual(manifest["candidate_hash"], "a" * 40)
            self.assertEqual(manifest["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
