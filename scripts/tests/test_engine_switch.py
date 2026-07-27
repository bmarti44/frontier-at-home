#!/usr/bin/env python3
"""Safety contract for the turnkey engine switch."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "52_engine_switch.sh"


class EngineSwitchTests(unittest.TestCase):
    def run_switch(self, root: Path, *args: str):
        env = {
            "PATH": os.environ["PATH"],
            "ENGINE_SWITCH_TESTING": "1",
            "ENGINE_SWITCH_TEST_ROOT": str(root),
        }
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )

    def test_status_json_is_machine_readable_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_switch(root, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual(status["active_profile"], None)
            self.assertEqual(status["state"], "inactive")
            self.assertFalse((root / "actions.log").exists())

    def test_implementation_has_lock_and_no_broad_pkill(self):
        source = SCRIPT.read_text()
        self.assertIn("flock", source)
        self.assertNotIn("pkill", source)
        self.assertIn("start_ticks", source)
        self.assertIn("rollback", source)
        self.assertIn("release-check", source)
        self.assertNotIn("DS4_CUDA_EXPERT_CACHE_GB=72", source)
        self.assertIn("memwatch_start_ticks", source)
        self.assertIn("DISARM %s %s %s", source)
        self.assertIn("GLM switching remains disabled", source)
        self.assertIn("wait_model_ready", source)
        self.assertIn("Waiting for %s load", source)
        self.assertIn("deadline=$((SECONDS + 1800))", source)

    def test_unqualified_glm_is_rejected_before_active_profile_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            result = self.run_switch(root, "glm52")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not qualified", result.stderr.lower())
            self.assertFalse((root / "actions.log").exists())
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )

    def test_concurrent_status_calls_return_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processes = [
                subprocess.Popen(
                    ["bash", str(SCRIPT), "status", "--json"],
                    cwd=ROOT,
                    env={
                        "PATH": os.environ["PATH"],
                        "ENGINE_SWITCH_TESTING": "1",
                        "ENGINE_SWITCH_TEST_ROOT": str(root),
                    },
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(8)
            ]
            for process in processes:
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                json.loads(stdout)


if __name__ == "__main__":
    unittest.main()
