#!/usr/bin/env python3
"""Contract for the root-owned W9 execution/publication boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/94_glm52_w9_submit.py"
INSTALLER_PATH = ROOT / "scripts/95_install_glm52_w9_runner.sh"
SPEC = importlib.util.spec_from_file_location("w9_root_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W9RootRunnerTests(unittest.TestCase):
    def test_request_parser_rejects_traversal_and_extra_authority(self) -> None:
        self.assertEqual(
            RUNNER.parse_request(["run", "/home/bmarti44/.local/state/randomness.json",
                                  "attempt-fp4-c4"]),
            ("run", "/home/bmarti44/.local/state/randomness.json", "attempt-fp4-c4"),
        )
        self.assertEqual(RUNNER.parse_request(["verify", "attempt-fp4-c4"]),
                         ("verify", "attempt-fp4-c4"))
        for request in (
            ["run", "/etc/shadow", "attempt-x"],
            ["run", "/home/bmarti44/.local/state/r.json", "../attempt-x"],
            ["run", "/home/bmarti44/.local/state/r.json", "/attempt-x"],
            ["verify", "attempt-x/child"],
            ["shell"],
        ):
            with self.assertRaises(ValueError):
                RUNNER.parse_request(request)

    def test_randomness_receipt_is_bounded_stable_owner_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            receipt = root / "receipt.json"
            receipt.write_text('{}\n', encoding="utf-8")
            with RUNNER.BoundOwnerInput(receipt, os.getuid(), 32768) as bound:
                self.assertEqual(bound.read_bytes(), b'{}\n')
            link = root / "link.json"
            link.symlink_to(receipt)
            with self.assertRaises(ValueError):
                with RUNNER.BoundOwnerInput(link, os.getuid(), 32768):
                    pass
            receipt.write_bytes(b"x" * 32769)
            with self.assertRaises(ValueError):
                with RUNNER.BoundOwnerInput(receipt, os.getuid(), 32768):
                    pass

    def test_two_fresh_replays_must_be_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir(); second.mkdir()
            for name in RUNNER.REPLAY_ARTIFACTS:
                (first / name).write_bytes((name + "\n").encode())
                (second / name).write_bytes((name + "\n").encode())
            hashes = RUNNER.compare_replays(first, second)
            self.assertEqual(set(hashes), set(RUNNER.REPLAY_ARTIFACTS))
            (second / "raw.jsonl").write_bytes(b"fabricated\n")
            with self.assertRaisesRegex(ValueError, "replay"):
                RUNNER.compare_replays(first, second)

    def test_publication_is_noreplace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"; source.mkdir()
            destination = root / "destination"
            RUNNER.rename_noreplace(source, destination)
            self.assertTrue(destination.is_dir())
            replacement = root / "replacement"; replacement.mkdir()
            with self.assertRaises(FileExistsError):
                RUNNER.rename_noreplace(replacement, destination)

    def test_fabricated_tree_without_root_attestation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = pathlib.Path(temporary) / "attempt-fake"
            attempt.mkdir(mode=0o500)
            for name in RUNNER.REPLAY_ARTIFACTS:
                (attempt / name).write_text('{}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attestation"):
                RUNNER.validate_published_attempt(
                    attempt, "attempt-fake", required_uid=os.getuid())

    def test_attestation_is_bound_to_attempt_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = pathlib.Path(temporary) / "attempt-bound"
            attempt.mkdir()
            artifacts = {}
            for name in RUNNER.REPLAY_ARTIFACTS:
                value = (name + "\n").encode()
                (attempt / name).write_bytes(value)
                artifacts[name] = RUNNER.sha256_bytes(value)
            attestation = RUNNER.build_attestation(
                attempt_name="attempt-bound", candidate_hash="a" * 40,
                review_commit="b" * 40, randomness_sha256="c" * 64,
                randomness_round=6358000, replay_sha256=artifacts,
                first_exit=0, second_exit=0,
            )
            (attempt / RUNNER.ATTESTATION_NAME).write_text(
                json.dumps(attestation, sort_keys=True) + "\n", encoding="utf-8")
            RUNNER.seal_tree(attempt, uid=os.getuid(), gid=os.getgid())
            RUNNER.validate_published_attempt(
                attempt, "attempt-bound", required_uid=os.getuid())
            with self.assertRaisesRegex(ValueError, "attempt"):
                RUNNER.validate_published_attempt(
                    attempt, "attempt-other", required_uid=os.getuid())

    def test_installer_and_runner_are_narrow_root_boundaries(self) -> None:
        installer = INSTALLER_PATH.read_text(encoding="utf-8")
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        for token in (
            "GLM52_REVIEWED_INSTALLER_SHA256", "/usr/bin/mktemp",
            "W9-fp4-falsifier-review-r254.json", "PASS_RUNTIME_ALLOWED",
            "/usr/local/libexec/glm52-w9", "/usr/local/sbin/glm52-w9-submit",
            "RENAME_NOREPLACE", "visudo", "NOPASSWD",
        ):
            self.assertIn(token, installer)
        for token in (
            "os.geteuid()", "BoundOwnerInput", "compare_replays",
            "root-attestation.json", "rename_noreplace", "seal_tree",
            "/var/lib/glm52-w9/attempts", "/usr/local/libexec/glm52-w9/repository",
            "/usr/local/libexec/glm52-w1/python", "env=FIXED_ENVIRONMENT",
        ):
            self.assertIn(token, runner)
        self.assertNotIn("shell=True", runner)


if __name__ == "__main__":
    unittest.main()
