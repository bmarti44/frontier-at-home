#!/usr/bin/env python3
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/glm52-profile.json"
CAMPAIGN_PROFILE = ROOT / "configs/glm52-lossless-plateau-profile.json"
CAMPAIGN = ROOT / "results/glm52-goal/harness/decisive_matched.sh"
ARM = ROOT / "results/glm52-goal/harness/glm_decisive_arm.sh"


class GlmLosslessPlateauTests(unittest.TestCase):
    def test_matched_glm_arm_is_the_adopted_w7_1a_profile(self):
        serving_profile = json.loads(PROFILE.read_text())
        profile = json.loads(CAMPAIGN_PROFILE.read_text())
        expected = profile["binary_sha256"]
        self.assertEqual(expected, serving_profile["binary_sha256"])
        self.assertEqual(
            profile["runtime"]["engine_environment"][
                "DS4_CUDA_STABLE_MODEL_REMAP"
            ],
            "1",
        )

        campaign = CAMPAIGN.read_text()
        arm = ARM.read_text()
        self.assertIn(f"GLM_SAFE_EXPECTED_BINARY_SHA256={expected}", campaign)
        self.assertIn("GLM_CANDIDATE_SRC=/home/bmarti44/.cache/", campaign)
        self.assertIn("CTX=32768", campaign)
        self.assertIn("--context-levels 0,28672", campaign)
        self.assertNotIn("restore_dsv4", campaign)
        self.assertNotIn("sudo -n", campaign)
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER=1", campaign)
        for process_name in ("ds4-server", "llama-server", "fio"):
            self.assertIn(process_name, campaign)
        self.assertIn("ss -H -ltn", campaign)
        self.assertIn("verify_campaign_artifacts", campaign)
        self.assertIn('[[ $actual_binary_sha256 == "$EXPECTED_BINARY_SHA256" ]]', arm)
        self.assertNotIn('sha256sum -- "$MODEL"', arm)
        self.assertIn("model.device-inode-size", arm)
        self.assertIn("process.environment", arm)
        self.assertIn("process.command", arm)
        self.assertIn('"$SRC/ds4-server" --cuda -m "$MODEL" -c 32768', arm)
        self.assertIn("DS4_CUDA_STABLE_MODEL_REMAP=1", arm)
        self.assertIn("stable_model_remap=1", arm)
        self.assertNotIn("DS4_KV_SKIP_PRELOAD_EVICT_STORE_DIAGNOSTIC", arm)

        identity_check = arm.index(
            '[[ $actual_binary_sha256 == "$EXPECTED_BINARY_SHA256" ]]'
        )
        launch = arm.index('"$SRC/ds4-server" --cuda')
        self.assertLess(identity_check, launch)

        self.assertEqual(profile["model_supported_context_cap"], 1_048_576)
        self.assertEqual(profile["measured_server_context_cap"], 32_768)
        self.assertNotIn("context_cap", profile)

    def test_profile_hashes_the_exact_matched_harnesses(self):
        profile = json.loads(CAMPAIGN_PROFILE.read_text())
        bindings = profile["artifact_sha256"]
        for path in (
            "results/glm52-goal/harness/decisive_matched.sh",
            "results/glm52-goal/harness/glm_decisive_arm.sh",
            "scripts/30_bench_speed.py",
            "scripts/56_collect_matched_evidence.py",
        ):
            self.assertEqual(
                bindings[path],
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                path,
            )


if __name__ == "__main__":
    unittest.main()
