#!/usr/bin/env python3
"""Adversarial contracts for authoritative W1 affine evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
GOAL_PATH = ROOT / "scripts/glm52_goal.py"
RUNNER_PATH = ROOT / "scripts/glm52_w1_affine_campaign.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_campaign(goal):
    harness = "b" * 40
    engine = "a" * 40
    binary = "2" * 64
    configuration = "3" * 64
    fixture = "1" * 64
    model = "6" * 64
    tokenizer = "7" * 64
    build = "8" * 64
    composite = "c" * 64
    engine_source = "d" * 64
    build_log = "e" * 64
    baseline_environment = "4" * 64
    candidate_environment = "5" * 64
    signature = "01" * 96
    randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    seed = hashlib.sha256(f"{composite}:{randomness}:W1".encode()).hexdigest()
    candidate_arm = "A" if int(seed[:2], 16) % 2 == 0 else "B"
    first = "ABBA" if int(seed[2:4], 16) % 2 == 0 else "BAAB"
    other = "BAAB" if first == "ABBA" else "ABBA"
    schedules = [first if block % 2 == 0 else other for block in range(5)]
    fixture_blocks = []
    attempts = []
    for block, schedule in enumerate(schedules):
        case_ids = [f"case-{block:02d}-{index:02d}" for index in range(20)]
        fixture_blocks.append(
            {
                "block": block,
                "manifest_sha256": f"{block + 10:x}" * 64,
                "ordered_case_ids": case_ids,
            }
        )
        for sequence, arm in enumerate(schedule):
            candidate = arm == candidate_arm
            rows = ["id\ttarget_tokens\tnll\ttarget_top1_correct"]
            rows.extend(
                f"{case_id}\t100\t{200.5 if candidate else 200.0}\t70"
                for case_id in case_ids
            )
            mode = 2 if candidate else 0
            store_rows = 2000 if candidate else 0
            changed = 1000 if candidate else 0
            evidence = {
                "launcher_log": (
                    "SAFE_RUN_DONE rc=0 killed=no "
                    f"dir=/state/attempt-{block}-{sequence}\n"
                ),
                "main_log": (
                    "2026-07-29T00:00:00.000+00:00 "
                    "cgroup_verified path=/unit memory_high=1 memory_max=2 "
                    "memory_swap_max=0 memory_oom_group=1\n"
                    f"2026-07-29T00:00:00.000+00:00 "
                    f"candidate_binary_sha256={binary}\n"
                    f"2026-07-29T00:00:00.000+00:00 "
                    f"executed_environment_sha256="
                    f"{candidate_environment if candidate else baseline_environment}\n"
                    f"2026-07-29T00:00:00.000+00:00 "
                    f"executed_candidate_verified pid={block * 10 + sequence + 1} "
                    f"start_ticks={1000 + block * 10 + sequence}\n"
                    "2026-07-29T00:00:05.000+00:00 "
                    "SAFE_RUN end rc=0 killed=no\n"
                ),
                "cmd_log": (
                    f"ds4: GLM compact cache fidelity resolved_mode={mode}\n"
                    "ds4: GLM compact cache fidelity attestation "
                    f"resolved_mode={mode} affine_store_rows={store_rows} "
                    f"affine_changed_values={changed}\n"
                ),
                "samples_log": "".join(
                    "2026-07-29T00:00:"
                    f"{sample / 4:06.3f}+00:00 "
                    "mem_avail_kb=92274688 eng_rss_kb=1 read_bytes=1\n"
                    for sample in range(21)
                ),
                "kernel_log": "-- No entries --\n",
                "quality_tsv": "\n".join(rows) + "\n",
            }
            attempt_index = block * 4 + sequence
            nonce = hashlib.sha256(
                f"{seed}:{attempt_index}:W1-witness".encode()
            ).hexdigest()
            unit = (
                f"glm52-w1-{seed[:8]}-{attempt_index:02d}-{arm}-"
                f"{9000 + attempt_index}"
            )
            message = (
                f"W1_WITNESS nonce={nonce} unit={unit} binary={binary} "
                "environment="
                f"{candidate_environment if candidate else baseline_environment} "
                f"pid={block * 10 + sequence + 1} "
                f"start_ticks={1000 + block * 10 + sequence} "
                "rc=0 killed=no "
                "cmd_sha256="
                f"{hashlib.sha256(evidence['cmd_log'].encode()).hexdigest()} "
                "samples_sha256="
                f"{hashlib.sha256(evidence['samples_log'].encode()).hexdigest()} "
                "artifact_sha256="
                f"{hashlib.sha256(evidence['quality_tsv'].encode()).hexdigest()} "
                f"artifact_identity=66306:{7000 + attempt_index}:4096"
            )
            evidence["journal_witness"] = json.dumps(
                {
                    "cursor": f"cursor-{attempt_index}",
                    "realtime_timestamp": str(1000000 + attempt_index),
                    "boot_id": "boot",
                    "invocation_id": f"{attempt_index:032x}",
                    "pid": str(8000 + attempt_index),
                    "uid": "995",
                    "cgroup": f"/user.slice/{unit}.service",
                    "user_unit": f"{unit}.service",
                    "message": message,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            attempts.append(
                {
                    "block": block,
                    "sequence": sequence,
                    "arm": arm,
                    "fixture_content_sha256_before": "9" * 64,
                    "fixture_content_sha256_after": "9" * 64,
                    "model_identity_before": "66306:1:211075856448:29203:29203:436",
                    "model_identity_after": "66306:1:211075856448:29203:29203:436",
                    "evidence": evidence,
                }
            )
    return {
        "record_type": "w1_affine_raw_campaign",
        "harness_candidate_hash": harness,
        "engine_candidate_hash": engine,
        "composite_candidate_sha256": composite,
        "seed_sha256": seed,
        "binary_sha256": binary,
        "configuration_sha256": configuration,
        "fixture_sha256": fixture,
        "fixture_content_sha256": "9" * 64,
        "model_content_sha256": model,
        "tokenizer_content_sha256": tokenizer,
        "engine_build_sha256": build,
        "engine_source_sha256": engine_source,
        "build_log_sha256": build_log,
        "baseline_environment_sha256": baseline_environment,
        "candidate_environment_sha256": candidate_environment,
        "candidate_arm": candidate_arm,
        "lineage": {
            "freeze": {
                "candidate_hash": harness,
                "frozen_at": "2026-07-29T04:56:00+00:00",
                "composite_candidate_sha256": composite,
            },
            "randomness": {
                "source": "drand-default",
                "round": 6329000,
                "randomness": randomness,
                "signature": signature,
                "obtained_at": "2026-07-29T05:00:00+00:00",
                "seed_sha256": seed,
            },
        },
        "fixture_blocks": fixture_blocks,
        "attempts": attempts,
    }


class W1AffineAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.goal = load(GOAL_PATH, "goal_w1_authority")
        cls.runner = load(RUNNER_PATH, "runner_w1_authority")

    def test_legacy_self_authored_scorer_cannot_authorize_w1(self):
        from scripts.tests.test_glm52_goal import FormulaTests

        synthetic = FormulaTests()._w1_affine_campaign()
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v1", [synthetic]
            )

    def test_raw_scorer_derives_pass_and_rejects_forged_fixture_or_noop_kernel(self):
        campaign = raw_campaign(self.goal)
        result = self.goal.score_registered_gate(
            "W1", "w1.affine-quality.v2", [campaign]
        )
        self.assertEqual(result["verdict"], "PASS")

        forged = copy.deepcopy(campaign)
        forged["attempts"][0]["evidence"]["quality_tsv"] = forged[
            "attempts"
        ][0]["evidence"]["quality_tsv"].replace("case-00-00", "forged-00")
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v2", [forged]
            )

        noop = copy.deepcopy(campaign)
        candidate = next(
            attempt
            for attempt in noop["attempts"]
            if attempt["arm"] == noop["candidate_arm"]
        )
        candidate["evidence"]["cmd_log"] = candidate["evidence"][
            "cmd_log"
        ].replace("affine_changed_values=1000", "affine_changed_values=0")
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v2", [noop]
            )

        uncovered = copy.deepcopy(campaign)
        for attempt in uncovered["attempts"]:
            attempt["evidence"]["samples_log"] = (
                "2026-07-29T00:00:00.000+00:00 "
                "mem_avail_kb=92274688 eng_rss_kb=1 read_bytes=1\n"
            )
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v2", [uncovered]
            )

        substituted = copy.deepcopy(campaign)
        for attempt in substituted["attempts"]:
            if attempt["arm"] == substituted["candidate_arm"]:
                attempt["evidence"]["quality_tsv"] = attempt["evidence"][
                    "quality_tsv"
                ].replace("\t200.5\t", "\t100.0\t")
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v2", [substituted]
            )

    def test_fabricated_drand_and_cached_model_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            beacon = Path(temporary) / "beacon.json"
            beacon.write_text(
                json.dumps(
                    {
                        "round": 1,
                        "randomness": "00" * 32,
                        "signature": "aa",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                self.runner._drand_record(beacon)

            model = Path(temporary) / "model"
            model.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            self.runner.verify_model_content(model, expected)
            model.write_bytes(b"replacement")
            with self.assertRaises(ValueError):
                self.runner.verify_model_content(model, expected)

    def test_driver_finalizes_controller_attempt_and_hash_bound_evidence(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('"manifest.json"', source)
        self.assertIn('"evidence_sha256"', source)
        self.assertIn("validate_attempt", source)
        self.assertIn("engine-build.json", source)
        self.assertIn("composite_candidate_sha256", source)
        self.assertIn("verify_frozen_candidate", source)
        self.assertIn("frozen_scorer_path", source)
        self.assertIn("make", source)
        self.assertIn("tests/test_glm_affine_int8_cuda", source)

    def test_seed_binds_the_complete_frozen_candidate(self):
        randomness = "1" * 64
        first = self.runner.confirmation_seed(randomness, "2" * 64)
        second = self.runner.confirmation_seed(randomness, "3" * 64)
        self.assertNotEqual(first, second)
        self.assertEqual(
            first,
            hashlib.sha256(f"{'2' * 64}:{randomness}:W1".encode()).hexdigest(),
        )

    def test_w1_has_a_registered_controller_runner(self):
        runner = ROOT / "scripts/glm52-runners/W1"
        self.assertTrue(runner.is_file())
        self.assertTrue(runner.stat().st_mode & 0o111)
        source = runner.read_text(encoding="utf-8")
        self.assertIn("/usr/local/sbin/glm52-w1-submit", source)
        self.assertIn(" run ", source)
        self.assertNotIn("glm52_w1_affine_campaign.py", source)

    def test_root_fixture_is_traversable_but_not_mutable_by_engine(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("os.chmod(output, 0o711)", source)
        self.assertIn("_seal_root_fixture_inputs(manifests)", source)
        self.assertIn("os.chmod(manifest, 0o444)", source)
        self.assertIn("os.chmod(manifest.parent, 0o555)", source)
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary) / "fixtures"
            fixture_dir.mkdir()
            manifests = [fixture_dir / f"block-{index}.tsv" for index in range(5)]
            for manifest in manifests:
                manifest.write_text("fixture\n", encoding="utf-8")
            with mock.patch.object(self.runner.os, "chown") as chown:
                self.runner._seal_root_fixture_inputs(manifests)
            self.assertEqual(fixture_dir.stat().st_mode & 0o777, 0o555)
            self.assertTrue(
                all(manifest.stat().st_mode & 0o777 == 0o444 for manifest in manifests)
            )
            self.assertEqual(
                chown.call_args_list,
                [mock.call(manifest, 0, 0) for manifest in manifests]
                + [mock.call(fixture_dir.resolve(), 0, 0)],
            )

    def test_root_fixture_sealing_rejects_aliases_and_mixed_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            regular = first / "block-0.tsv"
            regular.write_text("fixture\n", encoding="utf-8")
            alias = first / "block-1.tsv"
            alias.symlink_to(regular)
            with self.assertRaisesRegex(ValueError, "private regular"):
                self.runner._seal_root_fixture_inputs([alias])
            alias.unlink()
            alias.hardlink_to(regular)
            with self.assertRaisesRegex(ValueError, "private regular"):
                self.runner._seal_root_fixture_inputs([regular])
            alias.unlink()
            other = second / "block-1.tsv"
            other.write_text("fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one directory"):
                self.runner._seal_root_fixture_inputs([regular, other])

    def test_unpersisted_journal_witness_cannot_authorize(self):
        campaign = raw_campaign(self.goal)
        unavailable = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with mock.patch.object(
            self.goal.subprocess, "run", return_value=unavailable
        ):
            with self.assertRaisesRegex(ValueError, "externally persisted"):
                self.goal._verify_w1_journal_authority(campaign)

    def test_controller_bindings_reject_evidence_not_equal_to_raw_record(self):
        campaign = raw_campaign(self.goal)
        identity = campaign["attempts"][0]["model_identity_before"]
        fixture = {
            "schema_version": 1,
            "content_sha256": campaign["fixture_content_sha256"],
            "blocks": [
                {
                    **block,
                    "referenced_files": [],
                }
                for block in campaign["fixture_blocks"]
            ],
        }
        values = {
            "evidence": campaign,
            "model": {
                "schema_version": 1,
                "content_sha256": campaign["model_content_sha256"],
                "identity": identity,
            },
            "tokenizer": {
                "schema_version": 1,
                "lineage": "embedded-in-model-container",
                "content_sha256": campaign["tokenizer_content_sha256"],
            },
            "fixture": fixture,
            "diff": {
                "schema_version": 1,
                "commit": campaign["engine_candidate_hash"],
                "quality_binary_sha256": campaign["binary_sha256"],
                "status_porcelain": "",
                "cuda_test_passed": True,
                "clean_build_transcript_sha256": campaign[
                    "build_log_sha256"
                ],
                "object_sha256": {
                    "gguf-tools/quality-testing/score_official.o": "f" * 64
                },
            },
            "configuration": {
                field: campaign[field]
                for field in (
                    "harness_candidate_hash",
                    "engine_candidate_hash",
                    "composite_candidate_sha256",
                    "binary_sha256",
                    "model_content_sha256",
                    "tokenizer_content_sha256",
                    "engine_build_sha256",
                    "engine_source_sha256",
                    "build_log_sha256",
                    "fixture_sha256",
                    "fixture_content_sha256",
                    "lineage",
                )
            },
            "engine_source": {"bundle": "test"},
            "build_log": {"result": "passed"},
        }
        manifest = {
            "candidate_hash": campaign["harness_candidate_hash"],
            "binary_sha256": campaign["binary_sha256"],
            "configuration_sha256": campaign["configuration_sha256"],
            "fixture_sha256": campaign["fixture_sha256"],
            "diff_sha256": campaign["engine_build_sha256"],
            "engine_source_sha256": campaign["engine_source_sha256"],
            "build_log_sha256": campaign["build_log_sha256"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name, value in values.items():
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                paths[name] = path
            bundle = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"{campaign['engine_candidate_hash']} HEAD\n",
                stderr="",
            )
            with mock.patch.object(
                self.goal, "_verify_w1_journal_authority"
            ), mock.patch.object(self.goal.subprocess, "run", return_value=bundle):
                self.goal.validate_record_artifact_bindings(
                    "W1", manifest, [campaign], paths
                )
            forged = copy.deepcopy(campaign)
            forged["attempts"][0]["evidence"]["quality_tsv"] += "forged\n"
            with mock.patch.object(
                self.goal, "_verify_w1_journal_authority"
            ), mock.patch.object(self.goal.subprocess, "run", return_value=bundle):
                with self.assertRaises(ValueError):
                    self.goal.validate_record_artifact_bindings(
                        "W1", manifest, [forged], paths
                    )


if __name__ == "__main__":
    unittest.main()
