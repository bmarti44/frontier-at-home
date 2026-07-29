#!/usr/bin/env python3
"""Security contract for the one-time, root-owned GLM W1 authority."""

from __future__ import annotations

import hashlib
import importlib.util
import fcntl
import re
import subprocess
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
            r"systemctl disable --now docker\.socket docker\.service containerd\.service",
        )
        self.assertIn("/usr/sbin/groupdel docker", source)
        self.assertIn("/usr/bin/pgrep -x dockerd", source)
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
        self.assertIn('f"RuntimeMaxSec={timeout}s"', campaign)
        self.assertIn('"TasksMax=4096"', campaign)
        self.assertIn('"/usr/bin/systemctl", "stop"', campaign)
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
        self.assertIn(
            '$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- ',
            safe,
        )
        self.assertTrue(
            (
                ROOT
                / "results/glm52-gates/harness/glm_safe_run.sh"
            ).resolve().parents[3].joinpath("scripts/03_memory_guard.py").is_file()
        )

    def test_timed_out_build_still_stops_transient_unit(self):
        spec = importlib.util.spec_from_file_location(
            "glm52_campaign_timeout_cleanup",
            ROOT / "scripts/glm52_w1_affine_campaign.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        timeout = __import__("subprocess").TimeoutExpired(["sleep"], 1)
        stopped = mock.Mock(returncode=0, stdout="")
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(campaign, "ROOT_AUTHORITY", True),
            mock.patch.object(
                campaign.subprocess,
                "run",
                side_effect=[timeout, stopped],
            ) as run,
            self.assertRaises(__import__("subprocess").TimeoutExpired),
        ):
            campaign._run_checked(
                ["/usr/bin/sleep", "60"],
                cwd=Path(temporary),
                timeout=1,
                untrusted=True,
            )
        cleanup_command = run.call_args_list[1].args[0]
        self.assertEqual(cleanup_command[:2], ["/usr/bin/systemctl", "stop"])

    def test_root_child_does_not_deadlock_on_submitter_inference_lock(self):
        launcher = (
            ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
        ).read_text(encoding="utf-8")
        root_branch = launcher.split(
            "if [[ $ROOT_AUTHORITY == 1 ]]; then\n"
            "  # The immutable root submitter owns",
            1,
        )[1].split("elif [[ $RUN_AS_CURRENT_USER == 1 ]]", 1)[0]
        self.assertNotIn("/run/lock/frontier-at-home/inference.lock", root_branch)
        submitter = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("fcntl.flock(inference, fcntl.LOCK_EX)", submitter)

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

    def test_unexpected_failure_gets_a_preserved_failure_receipt(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submitter.ACTIVE_REQUEST = {
                "root": str(root),
                "request_id": "1" * 64,
                "phase": "public-randomness",
            }
            with mock.patch.object(submitter, "_quarantine_seal") as seal:
                submitter._record_failed_active_request(TimeoutError("fault"))
            receipt = (root / "receipt.json").read_text(encoding="utf-8")
            self.assertIn('"terminal_state":"FAIL"', receipt)
            self.assertIn('"failure_phase":"public-randomness"', receipt)
            seal.assert_called_once_with(root)

    def test_inference_lock_is_left_usable_by_dsv4(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "inference.lock"
            with (
                mock.patch.object(submitter, "INFERENCE_LOCK", lock),
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown") as chown,
                mock.patch.object(submitter.os, "fchmod") as chmod,
            ):
                with submitter._open_inference_lock():
                    pass
            chown.assert_called_once()
            chmod.assert_called_once_with(mock.ANY, 0o660)

    def test_lock_path_is_root_anchored_and_reboot_safe(self):
        submitter = SUBMITTER.read_text(encoding="utf-8")
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn(
            'INFERENCE_LOCK = Path("/run/lock/frontier-at-home/inference.lock")',
            submitter,
        )
        self.assertIn(
            "d /run/lock/frontier-at-home 0750 root dsv4 -",
            installer,
        )
        self.assertIn(
            "f /run/lock/frontier-at-home/inference.lock 0660 root dsv4 -",
            installer,
        )
        self.assertIn("os.fchown(descriptor, 0, identity.pw_gid)", submitter)

    def test_nonwritable_lock_directory_blocks_replacement_and_contention(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_root = Path(temporary) / "root-owned"
            lock_root.mkdir(mode=0o750)
            lock = lock_root / "inference.lock"
            lock.touch(mode=0o660)
            lock_root.chmod(0o550)
            with lock.open("a+b") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                completed = subprocess.run(
                    ["/usr/bin/flock", "-n", str(lock), "-c", "true"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                with self.assertRaises(PermissionError):
                    lock.unlink()
            lock_root.chmod(0o750)

    def test_root_worktrees_and_scorer_are_worker_traversable(self):
        campaign = (
            ROOT / "scripts/glm52_w1_affine_campaign.py"
        ).read_text(encoding="utf-8")
        self.assertIn("os.chmod(worktree_root, 0o711)", campaign)
        self.assertIn("_seal_candidate_tree(harness_source)", campaign)
        self.assertIn("os.chmod(target, 0o555)", campaign)
        self.assertIn("os.chmod(frozen, 0o555)", campaign)

    def test_quarantine_does_not_relabel_external_hardlink(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "attempt"
            root.mkdir()
            external = base / "external"
            external.write_text("preserve", encoding="utf-8")
            linked = root / "linked"
            linked.hardlink_to(external)
            with (
                mock.patch.object(submitter.os, "chown") as chown,
                mock.patch.object(submitter.os, "chmod"),
            ):
                submitter._quarantine_seal(root)
            touched_paths = {Path(call.args[0]) for call in chown.call_args_list}
            self.assertNotIn(linked, touched_paths)
            self.assertEqual(external.read_text(encoding="utf-8"), "preserve")

    def test_pass_replay_requires_published_authority_inode(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("authority_link.stat().st_ino", source)
        self.assertIn("completed_receipts[-1].stat().st_ino", source)

    def test_root_campaign_directory_is_not_dsv4_writable(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("campaign.mkdir(mode=0o700)", source)
        self.assertIn("os.chmod(campaign, 0o700)", source)
        self.assertNotIn("os.chown(campaign, 0, dsv4_identity.pw_gid)", source)
        self.assertIn("os.chmod(request_root, 0o711)", source)

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
