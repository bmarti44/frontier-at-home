#!/usr/bin/env python3
import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/glm52-profile.json"
CAMPAIGN_PROFILE = ROOT / "configs/glm52-lossless-plateau-profile.json"
CAMPAIGN = ROOT / "results/glm52-goal/harness/decisive_matched.sh"
ARM = ROOT / "results/glm52-goal/harness/glm_decisive_arm.sh"
DSV4_ARM = ROOT / "results/glm52-goal/harness/dsv4_decisive_arm.sh"
COLLECTOR = ROOT / "scripts/56_collect_matched_evidence.py"
DSV4_PROFILE = ROOT / "configs/dsv4-matched-32k-profile.json"
FREEZE_RECEIPT = ROOT / "results/glm52-gates/lossless-plateau-candidate7-preaudit.json"
GLM_CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
DSV4_CGROUP = ROOT / "results/glm52-gates/harness/dsv4_matched_cgroup_run.sh"


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
        dsv4_arm = DSV4_ARM.read_text()
        self.assertIn(f"GLM_BINARY_SHA256={expected}", campaign)
        self.assertIn("GLM_CANDIDATE_SRC=/home/bmarti44/.cache/", campaign)
        self.assertIn('"$SRC/ds4-server" --cuda -m "$MODEL" -c 32768', arm)
        self.assertIn("--context-levels 0,28672", arm)
        self.assertIn("--context-levels 0,28672", dsv4_arm)
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
            "results/glm52-goal/harness/dsv4_decisive_arm.sh",
            "results/glm52-gates/harness/glm_cgroup_run.sh",
            "results/glm52-gates/harness/glm_safe_run.sh",
            "scripts/30_bench_speed.py",
            "scripts/56_collect_matched_evidence.py",
        ):
            self.assertEqual(
                bindings[path],
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                path,
            )

    def test_campaign_retains_execution_bytes_and_holds_one_global_lock(self):
        campaign = CAMPAIGN.read_text()
        self.assertIn("MATCHED_RETAINED_RUNTIME", campaign)
        self.assertIn(
            "retained/results/glm52-goal/harness/decisive_matched.sh", campaign
        )
        self.assertIn('git -C "$REPO" show', campaign)
        self.assertIn("INFERENCE_LOCK_FD", campaign)
        self.assertIn("GLM_SAFE_PARENT_LOCK_FD", campaign)
        self.assertLess(
            campaign.index("exec {INFERENCE_LOCK_FD}"),
            campaign.index("\nverify_campaign_artifacts\n"),
        )

    def test_parent_lock_validation_binds_the_shared_open_description_not_flocker_pid(self):
        # flock(1) initiates the lock in a short-lived child.  Linux retains that
        # child's PID in fdinfo after the campaign shell inherits the locked open
        # file description, so requiring the fdinfo PID to equal the shell PID
        # rejects the real production launch.  Parent PID/start-ticks still bind
        # the direct caller; the fdinfo check must bind the unique kernel key.
        for path in (GLM_CGROUP, DSV4_CGROUP):
            source = path.read_text()
            self.assertNotIn('$6 == pid && $7 == key', source, str(path))
            self.assertIn('$7 == key', source, str(path))

    def test_dsv_containment_allows_bounded_helper_drain_before_survivor_verdict(self):
        source = DSV4_CGROUP.read_text()
        wait_index = source.index('wait "$wrapper_pid"')
        drain_index = source.index("while read -r cgroup_pid", wait_index)
        verdict_index = source.index("contained descendants survived command exit")
        self.assertLess(wait_index, drain_index)
        self.assertLess(drain_index, verdict_index)
        self.assertIn("seq 1 20", source[wait_index:verdict_index])
        self.assertIn("sleep 0.05", source[drain_index:verdict_index])
        self.assertNotIn("< <(", source[drain_index:verdict_index])

    def test_dsv4_arm_disables_prompt_cache_and_records_all_shard_checkpoints(self):
        arm = DSV4_ARM.read_text()
        self.assertIn("--no-cache-prompt", arm)
        self.assertIn("model.shards.jsonl", arm)
        for checkpoint in ("prelaunch", "ready", "post_requests"):
            self.assertIn(checkpoint, arm)
        for self_authored in (
            '"healthy": True',
            '"memwatch_alive": True',
            '"server_alive": True',
            '"watchdog_armed": True',
        ):
            self.assertNotIn(self_authored, arm)

    def test_collector_uses_raw_prompt_events_and_canonical_safety_records(self):
        collector = COLLECTOR.read_text()
        self.assertIn("_parse_production_prompt_counts", collector)
        self.assertIn("_parse_canonical_safety", collector)
        self.assertIn("safety.wrapper.out", collector)
        self.assertNotIn('if "SAFE_RUN_DONE rc=0" not in safety', collector)

    def test_dsv4_profile_declares_owner_accepted_containment_envelope(self):
        profile = json.loads(DSV4_PROFILE.read_text())
        self.assertEqual(
            profile["safety"],
            {
                "kill_floor_gib": 8,
                "minimum_start_gib": 110,
                "memory_high_gib": 105,
                "memory_max_gib": 107,
                "sample_hz": 4,
                "swap_max_bytes": 0,
                "timeout_seconds": 5400,
            },
        )

    def test_campaign_binds_isolated_python_and_reviewed_runtime_commit(self):
        campaign = CAMPAIGN.read_text()
        bench = (ROOT / "scripts/30_bench_speed.py").read_text()
        for profile_path in (CAMPAIGN_PROFILE, DSV4_PROFILE):
            runtime = json.loads(profile_path.read_text()).get("python_runtime")
            self.assertIsInstance(runtime, dict)
            self.assertEqual(
                set(runtime),
                {
                    "executable_path", "executable_sha256", "stdlib_path",
                    "stdlib_tree_sha256", "libpython_path", "libpython_sha256",
                    "tokenizer_native_path", "tokenizer_native_sha256",
                },
            )
        self.assertIn("lossless-plateau-candidate6-preaudit.json", campaign)
        self.assertIn("reviewed runtime commit", campaign)
        self.assertIn("/usr/bin/python3.12", campaign)
        self.assertIn("-I", campaign)
        self.assertIn("-B", campaign)
        self.assertIn("-S", campaign)
        self.assertIn("env -i", campaign)
        self.assertIn("MATCHED_TOKENIZER_NATIVE_PATH", bench)
        self.assertIn("ExtensionFileLoader", bench)

    def test_freeze_receipt_names_the_exact_existing_runtime_commit(self):
        receipt = json.loads(FREEZE_RECEIPT.read_text())
        commit = receipt["candidate_commit"]
        observed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"{commit}^{{commit}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(observed.returncode, 0, observed.stderr)
        self.assertEqual(observed.stdout.strip(), commit)
        tree = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"{commit}^{{tree}}"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        self.assertEqual(tree, receipt["candidate_tree"])
        paths = set()
        for profile_path in (CAMPAIGN_PROFILE, DSV4_PROFILE):
            profile = json.loads(profile_path.read_text())
            paths.add(profile_path.relative_to(ROOT).as_posix())
            paths.update(profile["artifact_sha256"])
        for relative in sorted(paths):
            frozen = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                stdout=subprocess.PIPE,
                check=True,
            ).stdout
            self.assertEqual(frozen, (ROOT / relative).read_bytes(), relative)


if __name__ == "__main__":
    unittest.main()
