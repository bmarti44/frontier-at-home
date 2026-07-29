#!/usr/bin/env python3
"""Security contract for the one-time, root-owned GLM W1 authority."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = ROOT / "scripts" / "65_glm52_w1_submit.py"
INSTALLER = ROOT / "scripts" / "66_install_glm52_w1_attestor.sh"
RUNNER = ROOT / "scripts" / "glm52-runners" / "W1"
CONTROLLER = ROOT / "scripts" / "glm52_goal.py"


def load_submitter():
    spec = importlib.util.spec_from_file_location("glm52_w1_submit", SUBMITTER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load GLM W1 submitter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootAttestorContractTests(unittest.TestCase):
    def test_submitter_accepts_hashes_only(self):
        submitter = load_submitter()
        sha1 = "1" * 40
        sha256 = "2" * 64
        self.assertEqual(
            submitter.parse_request(["run", sha1, sha1, sha256]),
            ("run", sha1, sha1, sha256),
        )
        self.assertEqual(
            submitter.parse_request(["status", sha256]),
            ("status", sha256),
        )
        for malformed in (
            [],
            ["run", sha1, sha1],
            ["run", "../repo", sha1, sha256],
            ["run", sha1, sha1, sha256, "--command=id"],
            ["status", sha256, "extra"],
            ["shell", sha256],
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    submitter.parse_request(malformed)

    def test_submitter_has_fixed_trust_roots_and_no_shell_escape(self):
        submitter = load_submitter()
        self.assertEqual(
            submitter.REPOSITORY,
            Path("/home/bmarti44/spark-deepseek-v4-flash"),
        )
        self.assertEqual(
            submitter.ENGINE_REPOSITORY,
            Path("/home/bmarti44/.cache/glm52-w1-real-capture-a37"),
        )
        self.assertEqual(submitter.STATE_ROOT, Path("/var/lib/glm52-w1"))
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("eval(", source)

    def test_installer_closes_docker_root_equivalence_and_is_hash_pinned(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertIn(
            "/usr/sbin/gpasswd -d bmarti44 docker",
            source,
        )
        self.assertRegex(
            source,
            r"systemctl disable --now docker\.socket docker\.service",
        )
        self.assertIn("/usr/sbin/visudo -cf", source)
        self.assertIn("NOPASSWD: /usr/local/sbin/glm52-w1-submit *", source)
        self.assertNotRegex(source, r"NOPASSWD:\\s*ALL")
        expected = hashlib.sha256(SUBMITTER.read_bytes()).hexdigest()
        match = re.search(
            r"^readonly SUBMITTER_SHA256='([0-9a-f]{32})''([0-9a-f]{32})'$",
            source,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual("".join(match.groups()), expected)

    def test_installer_requires_clean_exact_head(self):
        source = INSTALLER.read_text(encoding="utf-8")
        for required in (
            "must run as root",
            "candidate is not HEAD",
            "repository is not clean",
            "reviewed submitter digest differs",
            "install -o root -g root -m 0755",
            "install -d -o root -g root -m 0755",
            "/usr/local/libexec/glm52-w1/harness",
            "checkout --detach",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_untrusted_engine_builds_as_dsv4_in_a_root_cgroup(self):
        campaign = (
            ROOT / "scripts/glm52_w1_affine_campaign.py"
        ).read_text(encoding="utf-8")
        self.assertIn("untrusted=ROOT_AUTHORITY", campaign)
        self.assertIn('"--uid=dsv4"', campaign)
        self.assertIn('"MemoryMax=40G"', campaign)
        self.assertIn('"MemorySwapMax=0"', campaign)
        self.assertIn('"ProtectHome=read-only"', campaign)
        self.assertIn("_seal_candidate_tree(engine_source)", campaign)
        self.assertIn(
            r"/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}",
            campaign,
        )
        safe = (
            ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            r"/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}/crashlog",
            safe,
        )

    def test_controller_runner_cannot_bypass_root_authority(self):
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("/usr/local/sbin/glm52-w1-submit", runner)
        self.assertNotIn('python3 "$CAMPAIGN" run', runner)
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("validate_w1_root_receipt", controller)
        self.assertIn("/var/lib/glm52-w1/by-composite", controller)

    def test_submitter_does_not_ingest_user_campaign_trees(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        launcher = (
            ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("shutil.copytree", source)
        self.assertNotIn("/home/bmarti44/.local/state", source)
        self.assertNotIn("--uid=bmarti44", source + launcher)
        self.assertIn("--uid=dsv4", launcher)
        self.assertIn("MemorySwapMax=0", launcher)
        self.assertIn("OOMPolicy=kill", launcher)

    def test_failed_receipt_replay_stays_failed(self):
        submitter = load_submitter()
        self.assertTrue(hasattr(submitter, "receipt_exit_code"))
        self.assertEqual(
            submitter.receipt_exit_code(
                {"terminal_state": "PASS", "service_returncode": 0}
            ),
            0,
        )
        for receipt in (
            {"terminal_state": "FAIL", "service_returncode": 0},
            {"terminal_state": "PASS", "service_returncode": 137},
            {"terminal_state": "INCOMPLETE", "service_returncode": 0},
            {},
        ):
            with self.subTest(receipt=receipt):
                self.assertNotEqual(submitter.receipt_exit_code(receipt), 0)

    def test_failed_campaign_cannot_publish_authoritative_receipt(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("authority_pass = run_result.returncode == 0", source)
        self.assertIn("if authority_pass:", source)

    def test_evidence_manifest_rejects_symlinks(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "regular").write_text("evidence", encoding="utf-8")
            (root / "escape").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "symlink"):
                submitter._tree_manifest(root)

    def test_w1_scorer_digest_binds_journal_authority(self):
        spec = importlib.util.spec_from_file_location(
            "glm52_goal_for_digest", CONTROLLER
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        goal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(goal)
        before = goal.registered_scorer_digest("w1.affine-quality.v2")

        def rejected_mutation(record):
            return None

        with mock.patch.object(
            goal, "_verify_w1_journal_authority", rejected_mutation
        ):
            after = goal.registered_scorer_digest("w1.affine-quality.v2")
        self.assertNotEqual(before, after)

    def test_controller_ignores_user_owned_w1_attempt_directory(self):
        spec = importlib.util.spec_from_file_location(
            "glm52_goal_for_discovery", CONTROLLER
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        goal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(goal)
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            local = state_dir / "W1" / "attempt-999"
            local.mkdir(parents=True)
            (local / "manifest.json").write_text("{}", encoding="utf-8")
            state = goal._initial_state()
            with mock.patch.object(
                goal,
                "W1_AUTHORITY_ATTEMPT_ROOT",
                state_dir / "root-authority",
            ):
                goal._ingest_attempts(state_dir, state)
            self.assertEqual(state["gates"]["W1"]["attempts"], [])
            self.assertEqual(state["gates"]["W1"]["status"], "PENDING")

    def test_controller_rejects_writable_authority_attempt_root(self):
        spec = importlib.util.spec_from_file_location(
            "glm52_goal_for_authority_mode", CONTROLLER
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        goal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(goal)
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            attempts = state / "controller-attempts"
            attempt = attempts / "attempt-001"
            receipts = state / "by-composite"
            attempt.mkdir(parents=True)
            receipts.mkdir()
            attempts.chmod(0o777)
            composite = "1" * 64
            with (
                mock.patch.object(goal, "W1_AUTHORITY_ATTEMPT_ROOT", attempts),
                mock.patch.object(goal, "W1_AUTHORITY_RECEIPT_ROOT", receipts),
                self.assertRaisesRegex(ValueError, "ownership or mode"),
            ):
                goal.validate_w1_root_receipt(
                    attempt, {"composite_candidate_sha256": composite}
                )


if __name__ == "__main__":
    unittest.main()
