#!/usr/bin/env python3
"""Safety contract for the turnkey engine switch."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "52_engine_switch.sh"
GLM_PROFILE = ROOT / "configs" / "glm52-profile.json"
DSV4_PROFILE = ROOT / "configs" / "dsv4-profile.json"
DSV4_SERVICE = ROOT / "configs/systemd/deepseek-v4-flash-llamacpp.service"
DSV4_BUILD = ROOT / "configs/build-manifests/llamacpp-fusion.json"
GLM_CONFIRMATION = (
    ROOT / "results/glm52-goal/evidence/"
    "w1-confirmation-iq2-549d12c-r6324494-manifest.json"
)


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
        self.assertIn("trap 'rollback \"$command\"' EXIT", source)
        self.assertIn("expected=deepseek-v4-flash", source)
        self.assertIn('expected == item["id"].lower()', source)
        self.assertIn("DSV4_ALLOW_RETRY_AFTER_FAILED_START || true", source)

    def test_frozen_profiles_pin_the_verified_production_candidates(self):
        glm = json.loads(GLM_PROFILE.read_text())
        dsv4 = json.loads(DSV4_PROFILE.read_text())
        confirmation = json.loads(GLM_CONFIRMATION.read_text())
        dsv4_build = json.loads(DSV4_BUILD.read_text())
        self.assertEqual(
            glm,
            {
                "binary_sha256": confirmation["source"]["binary_sha256"],
                "context_cap": 1_048_576,
                "model_sha256": confirmation["artifacts"]["model_sha256"],
                "profile": "glm52",
            },
        )
        self.assertEqual(dsv4["schema_version"], 1)
        self.assertEqual(dsv4["profile"], "dsv4")
        self.assertEqual(
            dsv4["binary_sha256"],
            dsv4_build["binaries"]["llama-server"]["sha256"],
        )
        self.assertEqual(
            dsv4["configuration_sha256"],
            hashlib.sha256(DSV4_SERVICE.read_bytes()).hexdigest(),
        )

    def test_authenticated_probe_keeps_bearer_secret_out_of_argv(self):
        source = SCRIPT.read_text()
        self.assertNotIn('-H "Authorization: Bearer $key"', source)
        self.assertIn("clean_curl --config -", source)
        self.assertIn("printf 'header = \"Authorization: Bearer %s\"", source)

    def test_switch_subprocesses_use_a_frozen_environment_allowlist(self):
        source = SCRIPT.read_text()
        self.assertIn("clean_python()", source)
        self.assertIn("clean_curl()", source)
        self.assertIn("dsv4_launcher()", source)
        self.assertIn("env -i", source)
        self.assertIn('/usr/bin/curl --disable "$@"', source)
        self.assertNotIn("PYTHONOPTIMIZE", source)

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
