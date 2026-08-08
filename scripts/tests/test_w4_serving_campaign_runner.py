#!/usr/bin/env python3

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "w4_serving_runner", ROOT / "scripts/102_run_w4_serving_campaign.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W4ServingContainmentTest(unittest.TestCase):
    def test_safe_run_candidate_directory_contains_named_binary(self) -> None:
        self.assertEqual(RUNNER.BIN.name, "ds4-server")
        self.assertEqual(RUNNER.CANDIDATE_SRC, RUNNER.BIN.parent)
        self.assertTrue((RUNNER.CANDIDATE_SRC / "ds4-server").is_file())

    def test_containment_forwards_exact_topk_flag(self) -> None:
        source = (ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh").read_text()
        self.assertIn("  DS4_CUDA_TOPK2048_CUB \\\n", source)

    def test_arm_environments_differ_only_by_topk_flag_and_logit_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            off, _ = RUNNER.environment_for_arm("off", parent / "off", "/proc/self/fd/9", "1:2")
            on, _ = RUNNER.environment_for_arm("on", parent / "on", "/proc/self/fd/9", "1:2")
        ignored = {
            "DS4_CUDA_TOPK2048_CUB", "DS4_GLM_LOGIT_DUMP",
            "GLM_SAFE_EXPECTED_ENV_SHA256", "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST",
            "GLM_SAFE_FINAL_ARTIFACTS",
        }
        self.assertEqual({k: v for k, v in off.items() if k not in ignored},
                         {k: v for k, v in on.items() if k not in ignored})
        self.assertNotIn("DS4_CUDA_TOPK2048_CUB", off)
        self.assertEqual(on["DS4_CUDA_TOPK2048_CUB"], "1")
        self.assertEqual(off["DS4_CUDA_STABLE_MODEL_REMAP"], "1")
        self.assertEqual(on["DS4_CUDA_STABLE_MODEL_REMAP"], "1")

    def test_request_is_deterministic_and_non_generating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first_sha = RUNNER.make_request(first)
            second_sha = RUNNER.make_request(second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            doc = json.loads(first_bytes)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(doc["max_tokens"], 0)
        self.assertFalse(doc["stream"])
        self.assertGreater(len(doc["prompt"]), 90_000)

    def test_schedule_is_deterministic_and_domain_separated(self) -> None:
        seed = hashlib.sha256(b"post-freeze randomness").hexdigest()
        self.assertEqual(RUNNER.derive_schedules(seed), RUNNER.derive_schedules(seed))
        self.assertEqual(len(RUNNER.derive_schedules(seed)), 5)

    def test_campaign_rejects_dead_user_manager_before_large_hashes(self) -> None:
        with mock.patch.object(RUNNER, "user_systemd_available", return_value=False), \
             mock.patch.object(RUNNER, "verify_dependencies",
                               side_effect=AssertionError("must not hash dependencies")):
            with self.assertRaisesRegex(RUNNER.CampaignError,
                                        "user-systemd containment is unavailable"):
                RUNNER.campaign("0" * 40, Path("unused-receipt.json"))

    def test_failure_finalizer_emits_w4_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "1" * 40
            try:
                RUNNER.finalize_failure(RuntimeError("synthetic failure"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["schema"], "glm52-w4-serving-campaign-failure-v1")
        self.assertEqual(manifest["scorer_sha256"], RUNNER.SCORER_SHA256)
        self.assertEqual(manifest["binary_sha256"], RUNNER.BINARY_SHA256)


if __name__ == "__main__":
    unittest.main()
