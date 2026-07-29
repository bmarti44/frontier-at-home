#!/usr/bin/env python3
"""Adversarial contracts for authoritative W1 affine evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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
    baseline_environment = "4" * 64
    candidate_environment = "5" * 64
    signature = "01" * 96
    randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    seed = hashlib.sha256(f"{harness}:{randomness}:W1".encode()).hexdigest()
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
                    "cgroup_verified path=/unit memory_high=1 memory_max=2 "
                    "memory_swap_max=0 memory_oom_group=1\n"
                    f"candidate_binary_sha256={binary}\n"
                    f"executed_environment_sha256="
                    f"{candidate_environment if candidate else baseline_environment}\n"
                    f"executed_candidate_verified pid={block * 10 + sequence + 1} "
                    f"start_ticks={1000 + block * 10 + sequence}\n"
                    "SAFE_RUN end rc=0 killed=no\n"
                ),
                "cmd_log": (
                    f"ds4: GLM compact cache fidelity resolved_mode={mode}\n"
                    "ds4: GLM compact cache fidelity attestation "
                    f"resolved_mode={mode} affine_store_rows={store_rows} "
                    f"affine_changed_values={changed}\n"
                ),
                "samples_log": (
                    "2026-07-29T00:00:00.000+00:00 "
                    "mem_avail_kb=92274688 eng_rss_kb=1 read_bytes=1\n"
                ),
                "kernel_log": "-- No entries --\n",
                "quality_tsv": "\n".join(rows) + "\n",
            }
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
        "seed_sha256": seed,
        "binary_sha256": binary,
        "configuration_sha256": configuration,
        "fixture_sha256": fixture,
        "fixture_content_sha256": "9" * 64,
        "model_content_sha256": model,
        "tokenizer_content_sha256": tokenizer,
        "engine_build_sha256": build,
        "baseline_environment_sha256": baseline_environment,
        "candidate_environment_sha256": candidate_environment,
        "candidate_arm": candidate_arm,
        "lineage": {
            "freeze": {
                "candidate_hash": harness,
                "frozen_at": "2026-07-29T00:00:00+00:00",
            },
            "randomness": {
                "source": "drand-default",
                "round": 6329000,
                "randomness": randomness,
                "signature": signature,
                "obtained_at": "2026-07-29T00:01:00+00:00",
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


if __name__ == "__main__":
    unittest.main()
