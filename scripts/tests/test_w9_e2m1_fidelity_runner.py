#!/usr/bin/env python3
"""Production-path tests for the bounded W9 E2M1 fidelity runner."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/104_run_w9_e2m1_fidelity.py"
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run_w9_e2m1_v1.sh"
BASE_LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"


def load_runner():
    spec = importlib.util.spec_from_file_location("w9_e2m1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W9E2M1FidelityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_launcher_is_frozen_base_plus_only_the_e2m1_flag(self):
        base = BASE_LAUNCHER.read_text().splitlines()
        candidate = LAUNCHER.read_text().splitlines()
        self.assertEqual(
            [line for line in candidate if "E2M1_FAKE" not in line], base
        )
        self.assertEqual(
            sum("DS4_GLM_COMPACT_CACHE_E2M1_FAKE" in line for line in candidate),
            1,
        )

    def test_schedule_is_seed_derived_and_counterbalanced(self):
        self.assertEqual(self.runner.schedule("00" * 32), "ABBA")
        self.assertEqual(self.runner.schedule("0001" + "00" * 30), "BAAB")

    def test_artifacts_are_exclusive_and_content_bound(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            descriptor = self.runner.write_artifact(root, "main.log", b"first\n")
            self.assertEqual(descriptor["path"], "main.log")
            self.assertEqual(descriptor["bytes"], 6)
            self.assertEqual(
                descriptor["sha256"], hashlib.sha256(b"first\n").hexdigest()
            )
            with self.assertRaises(FileExistsError):
                self.runner.write_artifact(root, "main.log", b"replacement\n")

    def test_runner_uses_containment_and_writes_only_artifact_descriptors(self):
        source = RUNNER.read_text()
        for token in (
            "GLM_SAFE_EXPECTED_BINARY_SHA256",
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST",
            "GLM_SAFE_EXPECTED_ENV_SHA256",
            "DS4_GLM_COMPACT_CACHE_E2M1_FAKE",
            "SAFE_RUN_DONE",
            "subprocess.run",
            "write_artifact",
            "artifact_descriptors",
        ):
            self.assertIn(token, source)
        self.assertNotIn('"main_log": main_log', source)
        self.assertNotIn('"command_log": command_log', source)

    def test_preflight_rejects_an_active_engine_or_low_memory(self):
        with self.assertRaises(RuntimeError):
            self.runner.preflight(active_pids=[123], available_kib=120 * 1024 * 1024)
        with self.assertRaises(RuntimeError):
            self.runner.preflight(active_pids=[], available_kib=100 * 1024 * 1024)
        self.runner.preflight(active_pids=[], available_kib=111 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
