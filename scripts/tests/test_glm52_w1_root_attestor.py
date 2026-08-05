#!/usr/bin/env python3
"""Security contract for the one-time, root-owned GLM W1 authority."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import fcntl
import json
import os
import re
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime, timezone
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


def load_controller():
    spec = importlib.util.spec_from_file_location(
        "glm52_goal_root_authority_test", CONTROLLER
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load GLM controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootAttestorContractTests(unittest.TestCase):
    def test_python_runtime_staging_is_not_below_noexec_run(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertIn(
            "python_temporary=$(/usr/bin/mktemp -d "
            "/usr/local/libexec/.glm52-w1-python.XXXXXX)",
            source,
        )
        self.assertIn('/usr/bin/rm -rf -- "$python_temporary"', source)
        self.assertNotIn(
            "python_temporary=$harness_temporary/python-runtime", source,
        )

    def test_root_clone_scopes_safe_directory_to_exact_source_gitdir(self):
        source = INSTALLER.read_text(encoding="utf-8")
        upload_pack = (
            "/usr/bin/git -c safe.directory="
            "/home/bmarti44/spark-deepseek-v4-flash/.git upload-pack"
        )
        self.assertIn('readonly SOURCE_UPLOAD_PACK="/usr/bin/git -c safe.directory=$REPO/.git upload-pack"', source)
        self.assertIn('--upload-pack="$SOURCE_UPLOAD_PACK"', source)
        self.assertNotIn("git config --global", source)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            environment = {**os.environ, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}
            rejected = subprocess.run([
                "/usr/bin/git", "clone", "--no-local", "--no-checkout",
                str(ROOT), str(root / "without-safe-directory"),
            ], capture_output=True, text=True, check=False, env=environment)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("dubious ownership", rejected.stderr)
            accepted = subprocess.run([
                "/usr/bin/git", "-c", "core.hooksPath=/dev/null", "clone",
                "--no-local", "--no-checkout", f"--upload-pack={upload_pack}",
                str(ROOT), str(root / "exact-safe-directory"),
            ], capture_output=True, text=True, check=False, env=environment)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            cloned = subprocess.run([
                "/usr/bin/git", "-C", str(root / "exact-safe-directory"),
                "rev-parse", "HEAD",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(cloned.returncode, 0, cloned.stderr)
            self.assertEqual(cloned.stdout.strip(), subprocess.run([
                "/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD",
            ], capture_output=True, text=True, check=True).stdout.strip())

    def test_p1_reservation_and_receipt_bind_exact_cuda_backend(self):
        submitter = load_submitter()
        self.assertEqual(submitter.P1_SCORING_BACKEND, {
            "device": "cuda",
            "probe_compute": "torch-float32-weights-fp16-autocast",
        })
        candidate = "1" * 40
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_parent = root / "owner-state"
            state_root = root / "root-state"
            source_parent.mkdir()
            state_root.mkdir()
            source = source_parent / "glm52-cpu-substitution"
            source.mkdir()
            summary_payload = b'{"decision":{"verdict":"STOP_PROBE"}}\n'
            (source / "summary.json").write_bytes(summary_payload)
            with (
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown"),
                self.assertRaisesRegex(ValueError, "scoring backend"),
            ):
                submitter.publish_p1_result(
                    candidate, source.name,
                    source_parent=source_parent, state_root=state_root,
                    approval_reader=lambda: ({
                        "candidate_hash": candidate,
                        "controller_sha256": "2" * 64,
                    }, {
                        "approval_sha256": "3" * 64,
                        "approval_device": 1,
                        "approval_inode": 2,
                    }),
                    reservation_reader=lambda: {
                        "candidate_hash": candidate,
                        "output_sha256": hashlib.sha256(
                            os.fsencode(source),
                        ).hexdigest(),
                        "reservation_sha256": "4" * 64,
                        "created_epoch_ns": 5,
                        "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
                    },
                    completed_validator=lambda _root, _approval: {
                        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
                        "decision_verdict": "STOP_PROBE",
                        "scoring_backend": {
                            "device": "cpu", "probe_compute": "torch-float32",
                        },
                    },
                )

    def test_p1_result_publication_moves_into_root_authority_and_binds_manifest(self):
        submitter = load_submitter()
        candidate = "1" * 40
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_parent = root / "owner-state"
            state_root = root / "root-state"
            source_parent.mkdir()
            state_root.mkdir()
            source = source_parent / "glm52-p1-result"
            source.mkdir()
            (source / "attempt.json").write_text(
                '{"status":"COMPLETE"}\n', encoding="utf-8",
            )
            (source / "summary.json").write_text(
                '{"decision":{"verdict":"STOP_PROBE"}}\n', encoding="utf-8",
            )
            manifest = submitter._tree_manifest(source)
            digest = submitter._manifest_sha256(manifest)
            with (
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown"),
                self.assertRaisesRegex(ValueError, "reservation|completed"),
            ):
                submitter.publish_p1_result(
                    candidate, source.name,
                    source_parent=source_parent, state_root=state_root,
                    approval_reader=lambda: ({
                        "candidate_hash": candidate,
                        "controller_sha256": "2" * 64,
                    }, {
                        "approval_sha256": "3" * 64,
                        "approval_device": 1,
                        "approval_inode": 2,
                    }),
                    reservation_reader=lambda: {
                        "candidate_hash": candidate,
                        "output_sha256": hashlib.sha256(
                            os.fsencode(source),
                        ).hexdigest(),
                        "reservation_sha256": "4" * 64,
                        "created_epoch_ns": 5,
                        "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
                    },
                    completed_validator=lambda _root, _approval: (_ for _ in ()).throw(
                        ValueError("completed fixed replay failed"),
                    ),
                )

    def test_p1_publication_cannot_treat_rename_as_fd_revocation(self):
        submitter = load_submitter()
        candidate = "1" * 40
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_parent = root / "owner-state"
            state_root = root / "root-state"
            source_parent.mkdir()
            state_root.mkdir()
            source = source_parent / "glm52-valid-result"
            source.mkdir()
            summary = source / "summary.json"
            summary_payload = b'{"decision":{"verdict":"KEEP_PARETO_SEPARATE"}}\n'
            summary.write_bytes(summary_payload)
            retained = os.open(summary, os.O_RDWR)
            try:
                with (
                    mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                    mock.patch.object(submitter, "ROOT_GID", os.getgid()),
                ):
                    receipt = submitter.publish_p1_result(
                        candidate, source.name,
                        source_parent=source_parent, state_root=state_root,
                        approval_reader=lambda: ({
                            "candidate_hash": candidate,
                            "controller_sha256": "2" * 64,
                        }, {
                            "approval_sha256": "3" * 64,
                            "approval_device": 1,
                            "approval_inode": 2,
                        }),
                        reservation_reader=lambda: {
                            "candidate_hash": candidate,
                            "output_sha256": hashlib.sha256(
                                os.fsencode(source),
                            ).hexdigest(),
                            "reservation_sha256": "4" * 64,
                            "created_epoch_ns": 5,
                            "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
                        },
                        completed_validator=lambda _root, _approval: {
                            "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
                            "decision_verdict": "KEEP_PARETO_SEPARATE",
                            "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
                        },
                    )
                os.lseek(retained, 0, os.SEEK_SET)
                os.write(retained, b"FAKE")
                authoritative = Path(receipt["authoritative_root"])
                self.assertEqual(
                    (authoritative / "summary.json").read_bytes(), summary_payload,
                )
            finally:
                os.close(retained)

    def test_controller_never_reopens_root_only_authoritative_result(self):
        module_path = ROOT / "scripts" / "81_glm_union_baseline.py"
        spec = importlib.util.spec_from_file_location("p1_controller", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = inspect.getsource(module._publish_root_result)
        self.assertNotIn("_result_tree_manifest(authoritative)", source)

    def test_completed_validator_uses_only_root_owned_offline_python_runtime(self):
        submitter = load_submitter()
        self.assertEqual(
            submitter.P1_PYTHON_RUNTIME.parent,
            Path("/usr/local/libexec/glm52-w1"),
        )
        self.assertNotEqual(submitter.P1_PYTHON_RUNTIME.parent, submitter.INSTALLED_HARNESS)
        source = inspect.getsource(submitter._run_p1_completed_validator)
        self.assertIn("P1_PYTHON_RUNTIME", source)
        self.assertIn('"PYTHONNOUSERSITE": "1"', source)
        self.assertIn('"PYTHONPATH": str(P1_PYTHON_RUNTIME)', source)
        self.assertIn('"-S"', source)
        self.assertNotIn('"--device", "cpu"', source)
        self.assertIn("_python_dependency_tree_sha256", source)

    def test_python_runtime_tree_hash_rejects_package_mutation(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as raw:
            runtime = Path(raw)
            package = runtime / "numpy"
            package.mkdir()
            module = package / "__init__.py"
            module.write_bytes(b"version = 1\n")
            runtime.chmod(0o555)
            package.chmod(0o555)
            module.chmod(0o444)
            with (
                mock.patch.object(submitter, "P1_PYTHON_DEPENDENCIES", ("numpy",)),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
            ):
                expected = submitter._python_dependency_tree_sha256(runtime)
                self.assertRegex(expected, r"^[0-9a-f]{64}$")
                module.chmod(0o644)
                module.write_bytes(b"version = 2\n")
                module.chmod(0o444)
                self.assertNotEqual(
                    submitter._python_dependency_tree_sha256(runtime), expected,
                )
                package.chmod(0o755)
                (package / "redirect").symlink_to(runtime, target_is_directory=True)
                package.chmod(0o555)
                with self.assertRaisesRegex(ValueError, "directory identity"):
                    submitter._python_dependency_tree_sha256(runtime)

    def test_p1_result_publication_rejects_digest_race_but_preserves_attempt(self):
        submitter = load_submitter()
        candidate = "1" * 40
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_parent = root / "owner-state"
            state_root = root / "root-state"
            source_parent.mkdir()
            state_root.mkdir()
            source = source_parent / "glm52-p1-result"
            source.mkdir()
            (source / "attempt.json").write_text(
                '{"status":"FAILED"}\n', encoding="utf-8",
            )
            with (
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown"),
                self.assertRaisesRegex(ValueError, "completed"),
            ):
                submitter.publish_p1_result(
                        candidate, source.name,
                        source_parent=source_parent, state_root=state_root,
                        approval_reader=lambda: ({
                            "candidate_hash": candidate,
                            "controller_sha256": "2" * 64,
                        }, {
                            "approval_sha256": "3" * 64,
                            "approval_device": 1,
                            "approval_inode": 2,
                        }),
                        reservation_reader=lambda: {
                            "candidate_hash": candidate,
                            "output_sha256": hashlib.sha256(
                                os.fsencode(source),
                            ).hexdigest(),
                            "reservation_sha256": "4" * 64,
                            "created_epoch_ns": 5,
                            "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
                        },
                        completed_validator=lambda _root, _approval: (_ for _ in ()).throw(
                            ValueError("completed fixed replay failed"),
                        ),
                    )
            self.assertTrue(source.exists())
            quarantined = list((state_root / "p1-results").glob("glm52-p1-result*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].stat().st_mode & 0o777, 0o500)

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
        self.assertEqual(
            submitter.parse_request(["diagnose", sha256]),
            ("diagnose", sha256),
        )
        self.assertEqual(
            submitter.parse_request(["reserve-p1", sha1, sha256, sha256]),
            ("reserve-p1", sha1, sha256, sha256),
        )
        self.assertEqual(
            submitter.parse_request(["reserve-p1-smoke", sha1, sha256, sha256]),
            ("reserve-p1-smoke", sha1, sha256, sha256),
        )
        self.assertEqual(
            submitter.parse_request(["reserve-p1-approval-smoke", sha1, sha256, sha256]),
            ("reserve-p1-approval-smoke", sha1, sha256, sha256),
        )
        self.assertEqual(
            submitter.parse_request(["p1-authority"]),
            ("p1-authority",),
        )
        for malformed in (
            [],
            ["run", sha1, sha1],
            ["run", "../repo", sha1, sha256],
            ["run", sha1, sha1, sha256, "--command=id"],
            ["status", sha256, "extra"],
            ["diagnose", "../attempt"],
            ["diagnose", sha256, "extra"],
            ["reserve-p1", sha1],
            ["reserve-p1", "../candidate", sha256],
            ["reserve-p1", sha1, sha256],
            ["reserve-p1-smoke", sha1, sha256],
            ["reserve-p1-approval-smoke", sha1, sha256],
            ["p1-authority", "extra"],
            ["shell", sha256],
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    submitter.parse_request(malformed)

    def test_p1_reservation_is_exclusive_root_owned_and_persistent(self):
        submitter = load_submitter()
        candidate = "1" * 40
        reservation = "2" * 64
        output = "3" * 64
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            marker_root = state / "p1-baseline-heldout-v1"
            marker = marker_root / "reservation.json"
            with (
                mock.patch.object(submitter, "STATE_ROOT", state),
                mock.patch.object(submitter, "P1_RESERVATION_ROOT", marker_root),
                mock.patch.object(submitter, "P1_RESERVATION", marker),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
                mock.patch.object(
                    submitter, "_read_p1_approval",
                    return_value=({
                        "candidate_hash": candidate,
                        "controller_sha256": "5" * 64,
                    }, {
                        "approval_sha256": "6" * 64,
                        "approval_device": 7,
                        "approval_inode": 8,
                    }),
                ) as approval_reader,
                mock.patch("builtins.print") as printer,
            ):
                self.assertEqual(submitter.reserve_p1(
                    candidate, reservation, output,
                    marker_root=marker_root, marker=marker,
                ), 0)
                approval_reader.assert_called_once()
                first = json.loads(marker.read_text(encoding="utf-8"))
                self.assertEqual(first["candidate_hash"], candidate)
                self.assertEqual(first["reservation_sha256"], reservation)
                self.assertEqual(first["output_sha256"], output)
                self.assertEqual(
                    first["scoring_backend"], submitter.P1_SCORING_BACKEND,
                )
                self.assertEqual(marker.stat().st_mode & 0o777, 0o444)
                self.assertEqual(marker_root.stat().st_mode & 0o777, 0o555)
                first_inode = marker.stat().st_ino
                self.assertEqual(
                    submitter.reserve_p1(
                        "3" * 40, "4" * 64, "5" * 64,
                        marker_root=marker_root, marker=marker,
                    ), 17,
                )
                self.assertEqual(marker.stat().st_ino, first_inode)
                self.assertEqual(
                    json.loads(marker.read_text(encoding="utf-8")), first,
                )
                self.assertEqual(printer.call_count, 2)
                marker_root.chmod(0o700)
                marker.chmod(0o600)

    def test_p1_reservation_rejects_caller_selected_clean_head(self):
        """The root helper, not the caller's repository, selects the candidate."""
        submitter = load_submitter()
        approved = "1" * 40
        attacker_selected = "3" * 40
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            marker_root = state / "p1-baseline-heldout-v1"
            marker = marker_root / "reservation.json"
            approval = state / "p1-approved.json"
            approval.write_text(json.dumps({
                "schema_version": 1,
                "classification": "GLM52_P1_ROOT_APPROVED_CANDIDATE",
                "candidate_hash": approved,
                "controller_sha256": "2" * 64,
            }) + "\n", encoding="utf-8")
            approval.chmod(0o444)
            try:
                with (
                    mock.patch.object(submitter, "STATE_ROOT", state),
                    mock.patch.object(submitter, "P1_RESERVATION_ROOT", marker_root),
                    mock.patch.object(submitter, "P1_RESERVATION", marker),
                    mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                    mock.patch.object(submitter, "ROOT_GID", os.getgid()),
                    mock.patch.object(
                        submitter, "_read_p1_approval",
                        return_value=({
                            "candidate_hash": approved,
                            "controller_sha256": "2" * 64,
                        }, {
                            "approval_sha256": "5" * 64,
                            "approval_device": 6,
                            "approval_inode": 7,
                        }),
                    ),
                    self.assertRaisesRegex(ValueError, "root-approved"),
                ):
                    submitter.reserve_p1(
                        attacker_selected, "4" * 64, "5" * 64,
                        marker_root=marker_root, marker=marker,
                    )
            finally:
                if marker_root.exists():
                    marker_root.chmod(0o700)
                if marker.exists():
                    marker.chmod(0o600)
                approval.chmod(0o600)

    def test_p1_root_approval_is_exact_root_owned_and_canonical(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approval = root / "p1-approved.json"
            value = {
                "schema_version": 1,
                "classification": "GLM52_P1_ROOT_APPROVED_CANDIDATE",
                "candidate_hash": "1" * 40,
                "controller_sha256": "2" * 64,
            }
            approval.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            approval.chmod(0o444)
            with (
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
            ):
                observed, identity = submitter._read_p1_approval(approval)
                self.assertEqual(observed, value)
                self.assertRegex(identity["approval_sha256"], r"^[0-9a-f]{64}$")
                approval.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "identity"):
                    submitter._read_p1_approval(approval)

    def test_p1_reservation_reader_rejects_mutable_or_linked_marker(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "reservation.json"
            marker.write_text(json.dumps({
                "schema_version": 1,
                "classification": "GLM52_P1_PERMANENT_RESERVATION",
                "candidate_hash": "1" * 40,
                "reservation_sha256": "2" * 64,
                "output_sha256": "3" * 64,
                "created_epoch_ns": 123,
                "scoring_backend": dict(submitter.P1_SCORING_BACKEND),
            }) + "\n", encoding="utf-8")
            with (
                mock.patch.object(submitter, "P1_RESERVATION", marker),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
            ):
                marker.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "identity"):
                    submitter._read_p1_reservation(marker)
                marker.chmod(0o444)
                linked = root / "linked.json"
                os.link(marker, linked)
                with self.assertRaisesRegex(ValueError, "identity"):
                    submitter._read_p1_reservation(marker)

    def test_first_drand_round_is_strictly_after_freeze(self):
        submitter = load_submitter()
        genesis = datetime.fromtimestamp(1_595_431_050, timezone.utc)
        cases = (
            (genesis, 2),
            (datetime.fromtimestamp(1_595_431_050.1, timezone.utc), 2),
            (datetime.fromtimestamp(1_595_431_079.999, timezone.utc), 2),
            (datetime.fromtimestamp(1_595_431_080, timezone.utc), 3),
        )
        for frozen_at, expected_round in cases:
            with self.subTest(frozen_at=frozen_at):
                round_number = submitter.first_drand_round_after(
                    frozen_at.isoformat()
                )
                self.assertEqual(round_number, expected_round)
                published_at = 1_595_431_050 + (round_number - 1) * 30
                self.assertGreater(published_at, frozen_at.timestamp())

    def test_post_freeze_fetch_rejects_duplicate_randomness_keys(self):
        submitter = load_submitter()
        frozen_at = datetime.fromtimestamp(
            1_595_431_050, timezone.utc
        ).isoformat()
        payloads = (
            '{"round":1,"round":2,"randomness":"' + "a" * 64 + '"}',
            '{"round":2,"round":1,"randomness":"' + "a" * 64 + '"}',
        )
        for payload in payloads:
            with (
                self.subTest(payload=payload[:30]),
                mock.patch.object(
                    submitter,
                    "_run",
                    return_value=subprocess.CompletedProcess(
                        ["curl"], 0, payload, ""
                    ),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "drand response is invalid"
                ),
            ):
                submitter._fetch_first_post_freeze_drand(frozen_at)

    def test_campaign_validates_lineage_before_creating_fixtures_or_running_arms(self):
        source = (
            ROOT / "scripts/glm52_w1_affine_campaign.py"
        ).read_text(encoding="utf-8")
        run_body = source.split("def run(args: argparse.Namespace) -> int:", 1)[1]
        lineage_validation = run_body.index("validate_manifest_lineage(")
        self.assertLess(lineage_validation, run_body.index("_write_manifests("))
        self.assertLess(lineage_validation, run_body.index("subprocess.run("))
        self.assertIn("commit_time_fetcher=", run_body[: run_body.index("_write_manifests(")])
        submitter = SUBMITTER.read_text(encoding="utf-8")
        self.assertNotIn('"https://api.drand.sh/public/latest"', submitter)

    def test_diagnosis_reads_only_exact_sealed_failed_campaign(self):
        submitter = load_submitter()
        composite = "2" * 64
        request_id = "3" * 64
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            attempt = state / "requests" / request_id / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "campaign.log").write_text(
                "ordinary output\n"
                "glm52-w1-affine-campaign: W1 raw memory telemetry "
                "does not cover execution\n",
                encoding="utf-8",
            )
            (attempt / "receipt.json").write_text(
                '{"composite_candidate_sha256":"' + composite + '",'
                '"request_id":"' + request_id + '",'
                '"failure_phase":"campaign","terminal_state":"FAIL"}\n',
                encoding="utf-8",
            )
            (attempt / "request.json").write_text(
                json.dumps({"request_id": request_id}) + "\n",
                encoding="utf-8",
            )
            freeze = attempt / "freeze"
            freeze.mkdir()
            (freeze / "freeze.json").write_text(
                json.dumps({"composite_candidate_sha256": composite}) + "\n",
                encoding="utf-8",
            )
            large_unrelated_tree = attempt / "worktrees"
            large_unrelated_tree.mkdir()
            for number in range(1_100):
                (large_unrelated_tree / f"source-{number:04d}").write_text(
                    "frozen source\n",
                    encoding="utf-8",
                )
            paths = [attempt, *attempt.rglob("*")]
            for path in sorted(paths, reverse=True):
                path.chmod(0o500 if path.is_dir() else 0o400)
            old_time_ns = 1_700_000_000_000_000_000
            for path in paths:
                os.utime(
                    path,
                    ns=(old_time_ns, old_time_ns),
                    follow_symlinks=False,
                )
            metadata_before = {
                path.relative_to(attempt).as_posix(): (
                    path.lstat().st_atime_ns,
                    path.lstat().st_mtime_ns,
                    path.lstat().st_ctime_ns,
                    path.lstat().st_mode,
                    path.lstat().st_ino,
                )
                for path in paths
            }
            with (
                mock.patch.object(submitter, "STATE_ROOT", state),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
            ):
                diagnosis = submitter.diagnose_campaign(composite)
            self.assertEqual(diagnosis["terminal_state"], "NO_RESULT")
            self.assertEqual(diagnosis["request_id"], request_id)
            self.assertEqual(
                diagnosis["exact_error"],
                "glm52-w1-affine-campaign: W1 raw memory telemetry "
                "does not cover execution",
            )
            self.assertRegex(
                diagnosis["diagnostic_inputs_manifest_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                (attempt / "campaign.log").stat().st_mode & 0o777,
                0o400,
            )
            metadata_after = {
                path.relative_to(attempt).as_posix(): (
                    path.lstat().st_atime_ns,
                    path.lstat().st_mtime_ns,
                    path.lstat().st_ctime_ns,
                    path.lstat().st_mode,
                    path.lstat().st_ino,
                )
                for path in paths
            }
            self.assertEqual(metadata_after, metadata_before)

    def test_diagnosis_rejects_ambiguous_or_unsealed_campaign(self):
        submitter = load_submitter()
        composite = "4" * 64
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            attempt = state / "requests" / ("5" * 64) / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "campaign.log").write_text("failure\n", encoding="utf-8")
            (attempt / "receipt.json").write_text(
                '{"composite_candidate_sha256":"' + composite + '",'
                '"failure_phase":"campaign","terminal_state":"FAIL"}\n',
                encoding="utf-8",
            )
            with (
                mock.patch.object(submitter, "STATE_ROOT", state),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
            ):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    submitter.diagnose_campaign(composite)

    def test_diagnosis_rejects_two_sealed_matching_campaigns(self):
        submitter = load_submitter()
        composite = "6" * 64
        request_id = "7" * 64
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            for number in (1, 2):
                attempt = (
                    state
                    / "requests"
                    / request_id
                    / f"attempt-{number:03d}"
                )
                (attempt / "freeze").mkdir(parents=True)
                values = {
                    "receipt.json": {
                        "composite_candidate_sha256": composite,
                        "request_id": request_id,
                        "failure_phase": "campaign",
                        "terminal_state": "FAIL",
                    },
                    "request.json": {"request_id": request_id},
                    "freeze/freeze.json": {
                        "composite_candidate_sha256": composite
                    },
                }
                for relative, value in values.items():
                    (attempt / relative).write_text(
                        json.dumps(value) + "\n",
                        encoding="utf-8",
                    )
                (attempt / "campaign.log").write_text(
                    "exact failure\n",
                    encoding="utf-8",
                )
                for path in sorted(
                    [attempt, *attempt.rglob("*")],
                    reverse=True,
                ):
                    path.chmod(0o500 if path.is_dir() else 0o400)
            with (
                mock.patch.object(submitter, "STATE_ROOT", state),
                mock.patch.object(submitter, "ROOT_UID", os.getuid()),
                mock.patch.object(submitter, "ROOT_GID", os.getgid()),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                submitter.diagnose_campaign(composite)

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

    def test_root_bundle_git_ignores_system_and_global_configuration(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        run_helper = source.split("def _run(", 1)[1].split(
            "def _git_as_owner(", 1
        )[0]
        self.assertIn('"GIT_CONFIG_NOSYSTEM": "1"', run_helper)
        self.assertIn('"GIT_CONFIG_GLOBAL": "/dev/null"', run_helper)
        self.assertIn('"GIT_OPTIONAL_LOCKS": "0"', run_helper)

    def test_requested_commit_never_selects_root_executed_source(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("ROOT_EXECUTION_SURFACE", source)
        self.assertIn("_assert_root_execution_surface(", source)
        self.assertIn(
            "_bundle_clone(REPOSITORY, harness, harness_bundle, "
            "harness_repository)",
            source,
        )
        self.assertIn(
            'campaign_program = INSTALLED_HARNESS / '
            '"scripts/glm52_w1_affine_campaign.py"',
            source,
        )
        self.assertNotIn(
            'campaign_program = frozen_harness / '
            '"scripts/glm52_w1_affine_campaign.py"',
            source,
        )
        self.assertNotIn(
            "_git_root(INSTALLED_HARNESS, \"show\", f\"{harness}:{PROFILE}\")",
            source,
        )

        campaign = (
            ROOT / "scripts/glm52_w1_affine_campaign.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--harness-source"', campaign)
        self.assertIn("goal = _goal_module(SCORER)", campaign)
        self.assertNotIn("goal = _goal_module(frozen_scorer_path)", campaign)

    def test_root_execution_surface_rejects_candidate_code_changes(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / "installed"
            candidate = root / "candidate"
            for relative in submitter.ROOT_EXECUTION_SURFACE:
                for base in (installed, candidate):
                    path = base / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"fixed:{relative}\n", encoding="utf-8")
            with mock.patch.object(
                submitter, "INSTALLED_HARNESS", installed
            ):
                submitter._assert_root_execution_surface(candidate)
                changed = candidate / submitter.ROOT_EXECUTION_SURFACE[0]
                changed.write_text("owner-controlled root payload\n")
                with self.assertRaisesRegex(
                    ValueError, "root execution surface differs"
                ):
                    submitter._assert_root_execution_surface(candidate)
                changed.unlink()
                changed.symlink_to(
                    installed / submitter.ROOT_EXECUTION_SURFACE[0]
                )
                with self.assertRaisesRegex(
                    ValueError, "root execution surface is unsafe"
                ):
                    submitter._assert_root_execution_surface(candidate)

    def test_installed_scorer_resolves_later_candidate_in_sealed_repository(self):
        controller = load_controller()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / "installed"
            requested = root / "requested"
            subprocess.run(
                ["/usr/bin/git", "init", "-q", str(installed)],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(installed), "config", "user.name", "test"],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(installed),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            (installed / "authority").write_text("fixed\n", encoding="utf-8")
            subprocess.run(
                ["/usr/bin/git", "-C", str(installed), "add", "authority"],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(installed), "commit", "-qm", "A"],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "clone", "-q", str(installed), str(requested)],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(requested), "config", "user.name", "test"],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(requested),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            (requested / "metadata").write_text("new\n", encoding="utf-8")
            subprocess.run(
                ["/usr/bin/git", "-C", str(requested), "add", "metadata"],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(requested), "commit", "-qm", "B"],
                check=True,
            )
            candidate = subprocess.run(
                ["/usr/bin/git", "-C", str(requested), "rev-parse", "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            tree = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(requested),
                    "rev-parse",
                    f"{candidate}^{{tree}}",
                ],
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidate_hash": candidate,
                        "git_tree": tree,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "git tree"):
                controller.validate_source_provenance(
                    source, candidate, repository=installed
                )
            controller.validate_source_provenance(
                source, candidate, repository=requested
            )

            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "manifest.json").write_text(
                json.dumps(
                    {
                        "gate": "W1",
                        "candidate_hash": candidate,
                        "lineage": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a repository commit"):
                controller.validate_attempt(
                    attempt,
                    root_authority_pending=True,
                    source_repository=installed,
                )
            with (
                mock.patch.object(
                    controller, "validate_manifest_lineage"
                ),
                self.assertRaisesRegex(ValueError, "artifacts map"),
            ):
                controller.validate_attempt(
                    attempt,
                    root_authority_pending=True,
                    source_repository=requested,
                )

    def test_installer_closes_docker_root_equivalence_and_is_hash_pinned(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertNotIn(
            'find "$HARNESS" -type f -exec /usr/bin/chmod 0644',
            source,
        )
        self.assertNotIn(
            'find "$HARNESS" -type d -exec /usr/bin/chmod 0755',
            source,
        )
        self.assertIn(
            "/usr/bin/gpasswd -d bmarti44 docker",
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
        self.assertIn(
            'readonly APPROVAL=/usr/local/libexec/glm52-w1/p1-approved.json',
            source,
        )
        self.assertIn(
            'readonly CONTROLLER_SOURCE=scripts/81_glm_union_baseline.py',
            source,
        )
        self.assertIn(
            '/usr/bin/install -o root -g root -m 0444 "$approval_temporary" "$APPROVAL"',
            source,
        )
        self.assertIn('"controller_sha256": controller', source)
        expected = hashlib.sha256(SUBMITTER.read_bytes()).hexdigest()
        match = re.search(
            r"^readonly SUBMITTER_SHA256=([0-9a-f]{64})$",
            source,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), expected)

    def test_installer_copies_and_hash_verifies_offline_scorer_dependencies(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"(?m)^readonly PYTHON_DEPENDENCY_SHA256=[0-9a-f]{64}$",
        )
        self.assertIn("dependency_tree_sha", source)
        self.assertIn('"$PYTHON_DEPENDENCY_SOURCE/$dependency"', source)
        self.assertIn('dependency_tree_sha "$PYTHON_RUNTIME"', source)
        self.assertIn("PYTHONNOUSERSITE=1", source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("import numpy, tokenizers, torch", source)

    def test_controller_pins_the_same_root_submitter_bytes_as_installer(self):
        controller = (ROOT / "scripts/81_glm_union_baseline.py").read_text(
            encoding="utf-8",
        )
        expected = hashlib.sha256(SUBMITTER.read_bytes()).hexdigest()
        match = re.search(
            r'^FROZEN_ROOT_SUBMITTER_SHA256 = "([0-9a-f]{64})"$',
            controller,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), expected)

    def test_installer_publishes_only_the_contained_runtime_read_surface(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertIn(
            'readonly CONTAINED_RUNTIME_DIRS=(\n'
            '    "$HARNESS"\n'
            '    "$HARNESS/results"\n'
            '    "$HARNESS/results/glm52-gates"\n'
            '    "$HARNESS/results/glm52-gates/harness"\n'
            '    "$HARNESS/scripts"\n'
            ")",
            source,
        )
        self.assertIn(
            'readonly CONTAINED_RUNTIME_FILES=(\n'
            '    "$HARNESS/results/glm52-gates/harness/glm_safe_run.sh"\n'
            '    "$HARNESS/scripts/03_memory_guard.py"\n'
            ")",
            source,
        )
        self.assertIn(
            '/usr/bin/chmod 0555 "${CONTAINED_RUNTIME_DIRS[@]}"',
            source,
        )
        self.assertIn(
            '/usr/bin/chmod 0444 "${CONTAINED_RUNTIME_FILES[@]}"',
            source,
        )
        self.assertIn(
            '/usr/sbin/runuser -u dsv4 -- /usr/bin/test -x "$path" ||',
            source,
        )
        self.assertIn(
            '/usr/sbin/runuser -u dsv4 -- /usr/bin/test -r "$path" ||',
            source,
        )
        self.assertNotIn('/usr/bin/chmod -R', source)

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

    def test_installer_requires_reviewed_root_owned_staged_copy(self):
        source = INSTALLER.read_text(encoding="utf-8")
        for required in (
            "GLM52_REVIEWED_INSTALLER_SHA256",
            "installer must be executed from a reviewed root-owned staged copy",
            "/usr/bin/sha256sum -- \"$0\"",
            "/usr/bin/stat -c '%u:%g:%a:%F' -- \"$0\"",
            "0:0:500:regular file",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_staging_copy_race_yields_reviewed_bytes_or_hash_failure(self):
        reviewed = b"#!/bin/bash\nprintf 'reviewed\\n'\n" + b"#" * 65536
        replacement = b"#!/bin/bash\nprintf 'unreviewed\\n'\n" + b"!" * 65534
        expected = hashlib.sha256(reviewed).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "user-installer"
            source.write_bytes(reviewed)
            stop = threading.Event()

            def mutate():
                while not stop.is_set():
                    source.write_bytes(replacement)
                    source.write_bytes(reviewed)

            writer = threading.Thread(target=mutate)
            writer.start()
            try:
                for index in range(32):
                    staged = root / f"staged-{index}"
                    subprocess.run(
                        ["/usr/bin/install", "-m", "0500", source, staged],
                        check=True,
                    )
                    check = subprocess.run(
                        ["/usr/bin/sha256sum", "-c"],
                        input=f"{expected}  {staged}\n",
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if check.returncode == 0:
                        self.assertEqual(staged.read_bytes(), reviewed)
                    else:
                        self.assertNotEqual(
                            hashlib.sha256(staged.read_bytes()).hexdigest(), expected,
                        )
            finally:
                stop.set()
                writer.join()

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
        self.assertIn('f"safe.directory={source.resolve()}"', campaign)
        self.assertRegex(campaign, r"_trusted_git\(\s*engine_source")
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
        self.assertIn("with _hold_inference_locks():", submitter)

    def test_obsolete_fidelity_runner_cannot_be_relaunched(self):
        self.assertFalse(RUNNER.exists() and RUNNER.stat().st_mode & 0o111)
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("validate_w1_root_receipt", controller)
        self.assertIn("/var/lib/glm52-w1/by-composite", controller)

    def test_submitter_only_ingests_the_exact_p1_publication_tree(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        launcher = (
            ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("shutil.copytree", source)
        campaign = source.split("def run_campaign(", 1)[1].split(
            "\ndef show_status", 1,
        )[0]
        self.assertNotIn("/home/bmarti44/.local/state", campaign)
        publication = source.split("def publish_p1_result(", 1)[1].split(
            "\ndef _open_noatime", 1,
        )[0]
        self.assertIn("_copy_tree_root_owned(source, destination)", publication)
        self.assertIn("completed_validator(destination, approval)", publication)
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

    def test_owner_git_streams_bundle_to_root_opened_descriptor(self):
        submitter = load_submitter()
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "engine.bundle"
            destination = root / "engine-repository"

            def stream_bundle(argv, **kwargs):
                self.assertEqual(argv[-4:], ["bundle", "create", "-", "HEAD"])
                output = kwargs["stdout"]
                output.write(b"streamed bundle")
                output.flush()
                return subprocess.CompletedProcess(argv, 0, None, "")

            def root_git(argv, **_kwargs):
                stdout = (
                    f"{commit} HEAD\n"
                    if argv[1:3] == ["bundle", "list-heads"]
                    else ""
                )
                return subprocess.CompletedProcess(argv, 0, stdout, "")

            with (
                mock.patch.object(
                    submitter,
                    "_git_as_owner",
                    side_effect=AssertionError(
                        "owner Git must not create a pathname bundle"
                    ),
                ),
                mock.patch.object(
                    submitter.subprocess,
                    "run",
                    side_effect=stream_bundle,
                ),
                mock.patch.object(submitter, "_run", side_effect=root_git),
            ):
                submitter._bundle_clone(root, commit, bundle, destination)
            self.assertEqual(bundle.read_bytes(), b"streamed bundle")

    def test_inference_lock_is_left_usable_by_dsv4(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "inference.lock"
            with (
                mock.patch.object(submitter, "INFERENCE_LOCK", lock),
                mock.patch.object(
                    submitter,
                    "LEGACY_INFERENCE_LOCK",
                    Path(temporary) / "absent-legacy.lock",
                ),
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown") as chown,
                mock.patch.object(submitter.os, "fchmod") as chmod,
                mock.patch.object(submitter, "_validate_current_lock_parent"),
            ):
                with submitter._open_one_inference_lock(
                    lock, stable_parent=True
                ):
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
        self.assertIn(
            'LEGACY_INFERENCE_LOCK = Path("/run/dsv4/inference.lock")',
            submitter,
        )
        self.assertIn(
            "a pre-migration inference server is still running",
            installer,
        )

    def test_held_legacy_lock_fails_before_new_lock_creation(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            legacy = base / "legacy.lock"
            current = base / "current" / "inference.lock"
            legacy.touch()
            with legacy.open("a+b") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                with (
                    mock.patch.object(
                        submitter, "LEGACY_INFERENCE_LOCK", legacy
                    ),
                    mock.patch.object(submitter, "INFERENCE_LOCK", current),
                    mock.patch.object(submitter.os, "fchown"),
                    mock.patch.object(submitter.os, "fchmod"),
                    mock.patch.object(
                        submitter,
                        "_validate_legacy_lock_namespace",
                    ),
                    mock.patch.object(
                        submitter,
                        "_validate_current_lock_parent",
                    ),
                    self.assertRaisesRegex(
                        PermissionError, "pre-migration inference server"
                    ),
                ):
                    with submitter._hold_inference_locks():
                        self.fail("held legacy lock was accepted")
            self.assertFalse(current.exists())

    def test_submitter_holds_legacy_and_current_locks_together(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            legacy = base / "legacy" / "inference.lock"
            current = base / "current" / "inference.lock"
            legacy.parent.mkdir()
            legacy.touch()
            current.parent.mkdir()
            with (
                mock.patch.object(
                    submitter, "LEGACY_INFERENCE_LOCK", legacy
                ),
                mock.patch.object(submitter, "INFERENCE_LOCK", current),
                mock.patch.object(submitter.os, "chown"),
                mock.patch.object(submitter.os, "fchown"),
                mock.patch.object(
                    submitter,
                    "_validate_legacy_lock_namespace",
                ),
                mock.patch.object(
                    submitter,
                    "_validate_current_lock_parent",
                ),
                submitter._hold_inference_locks(),
            ):
                for path in (legacy, current):
                    completed = subprocess.run(
                        ["/usr/bin/flock", "-n", str(path), "-c", "true"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn('exec 8<>"$LEGACY_LOCK"', installer)
        self.assertIn("/usr/bin/flock -n -E 75 8", installer)

    def test_installer_converts_legacy_lock_without_following_links(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertNotIn("/usr/sbin/gpasswd", installer)
        self.assertIn("/usr/bin/gpasswd", installer)
        self.assertIn(
            "install -d -o root -g dsv4 -m 1770 /run/dsv4",
            installer,
        )
        self.assertIn("d /run/dsv4 1770 root dsv4 -", installer)
        self.assertIn(
            "f /run/dsv4/inference.lock 0660 root dsv4 -",
            installer,
        )
        self.assertIn("'RuntimeDirectory='", installer)
        self.assertIn("os.O_NOFOLLOW", installer)
        self.assertIn("except FileNotFoundError:", installer)
        self.assertIn("os.O_CREAT | os.O_EXCL", installer)
        self.assertIn("opened.st_ino != visible.st_ino", installer)
        self.assertIn('exec 8<>"$LEGACY_LOCK"', installer)
        submitter = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("stat.S_IMODE(parent.st_mode) != 0o1770", submitter)
        self.assertIn("details.st_uid != 0", submitter)

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

    def test_publication_uses_exact_per_request_attempt_not_frozen_source(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stale = output / "attempt-001"
            stale.mkdir()
            with self.assertRaisesRegex(ValueError, "exact campaign attempt"):
                submitter._select_campaign_controller_attempt(output)

    def test_publication_rejects_symlinked_campaign_attempt(self):
        submitter = load_submitter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stale = output / "stale"
            stale.mkdir()
            (output / "controller-attempt-final").symlink_to(stale)
            with self.assertRaisesRegex(ValueError, "exact campaign attempt"):
                submitter._select_campaign_controller_attempt(output)

    def test_root_campaign_leaves_final_attempt_in_per_request_output(self):
        source = (
            ROOT / "scripts/glm52_w1_affine_campaign.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'destination = output / "controller-attempt-final"',
            source,
        )
        self.assertIn("if ROOT_AUTHORITY:", source)

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
