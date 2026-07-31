#!/usr/bin/env python3
"""Contracts for the strict W1 affine ABBA/BAAB campaign driver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/glm52_w1_affine_campaign.py"


def load_campaign():
    spec = importlib.util.spec_from_file_location("w1_affine_campaign", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load campaign module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W1AffineCampaignTests(unittest.TestCase):
    def test_root_cgroup_launcher_forwards_real_packed_flag(self):
        launcher = (
            ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            launcher,
            re.compile(r"\\bDS4_GLM_COMPACT_CACHE_AFFINE_INT8\\b"),
            "the root cgroup boundary drops the real packed-cache flag",
        )

    def test_real_packed_candidate_environment_is_exact_and_fake_arms_are_off(self):
        campaign = load_campaign()
        environment = campaign.real_candidate_environment()
        self.assertEqual(
            environment["DS4_GLM_COMPACT_CACHE_AFFINE_INT8"], "1"
        )
        self.assertNotIn(
            "DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE", environment
        )
        self.assertIn(
            "DS4_GLM_COMPACT_CACHE_AFFINE_INT8",
            campaign.FIDELITY_ENVIRONMENT_NAMES,
        )

    def test_real_storage_attestation_requires_candidate_device_writes(self):
        campaign = load_campaign()
        candidate_log = (
            "ds4: GLM compact cache storage format=affine-int8-block16\n"
            "ds4: GLM compact cache storage attestation "
            "format=affine-int8-block16 packed_store_rows=8192 "
            "packed_read_values=4194304\n"
        )
        baseline_log = (
            "ds4: GLM compact cache storage format=f32\n"
            "ds4: GLM compact cache storage attestation "
            "format=f32 packed_store_rows=0 packed_read_values=0\n"
        )
        self.assertEqual(
            campaign.parse_storage_attestation(candidate_log),
            ("affine-int8-block16", 8192, 4194304),
        )
        self.assertEqual(
            campaign.parse_storage_attestation(baseline_log),
            ("f32", 0, 0),
        )
        with self.assertRaises(ValueError):
            campaign.parse_storage_attestation(
                candidate_log.replace("packed_read_values=4194304",
                                      "packed_read_values=0")
            )

    def test_seed_controls_opaque_arm_and_alternating_counterbalance(self):
        campaign = load_campaign()
        seed = "00" + "00" + "11" * 30
        self.assertEqual(campaign.candidate_arm(seed), "A")
        self.assertEqual(
            campaign.schedules(seed),
            ("ABBA", "BAAB", "ABBA", "BAAB", "ABBA"),
        )
        self.assertEqual(sum(map(len, campaign.schedules(seed))), 20)

    def test_fixture_digest_binds_manifest_and_every_referenced_byte(self):
        campaign = load_campaign()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "prompt").write_bytes(b"prompt")
            (source / "continuation").write_bytes(b"continuation")
            (source / "response").write_bytes(b"response")
            manifest = source / "manifest.tsv"
            manifest.write_text(
                "# id\tprompt_file\tcontinuation_file\tresponse_file\n"
                "case_000\tprompt\tcontinuation\tresponse\n",
                encoding="utf-8",
            )
            before = campaign.content_complete_fixture_sha256(
                source, [manifest]
            )
            (source / "response").write_bytes(b"mutated")
            after = campaign.content_complete_fixture_sha256(
                source, [manifest]
            )
            self.assertNotEqual(before, after)

    def test_environment_digest_matches_live_wrapper_canonicalization(self):
        campaign = load_campaign()
        names = (
            "DS4_CUDA_FETCH_THREADS",
            "DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE",
        )
        values = {"DS4_CUDA_FETCH_THREADS": "6"}
        expected = hashlib.sha256(
            b"DS4_CUDA_FETCH_THREADS=6\n"
            b"DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE=<UNSET>\n"
        ).hexdigest()
        self.assertEqual(campaign.environment_sha256(names, values), expected)

    def test_randomness_json_rejects_duplicate_security_fields(self):
        campaign = load_campaign()
        values = {
            "round": 2,
            "randomness": "a" * 64,
            "signature": "b" * 192,
        }
        for field, invalid in (
            ("round", 1),
            ("randomness", "c" * 64),
            ("signature", "d" * 192),
        ):
            for first, second in ((invalid, values[field]), (values[field], invalid)):
                pairs = [
                    f'"round":{values["round"]}',
                    f'"randomness":"{values["randomness"]}"',
                    f'"signature":"{values["signature"]}"',
                    (
                        f'"{field}":{second}'
                        if field == "round"
                        else f'"{field}":"{second}"'
                    ),
                ]
                pairs[("round", "randomness", "signature").index(field)] = (
                    f'"{field}":{first}'
                    if field == "round"
                    else f'"{field}":"{first}"'
                )
                payload = "{" + ",".join(pairs) + "}"
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "drand.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        campaign._strict_json(path)

        relay_payload = (
            '{"round":1,"round":2,"randomness":"' + "a" * 64 + '",'
            '"signature":"' + "b" * 192 + '"}'
        ).encode()
        with (
            mock.patch.object(
                campaign.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["curl"], 0, relay_payload, b""
                ),
            ),
            self.assertRaisesRegex(ValueError, "duplicate"),
        ):
            campaign._authenticate_drand(values)

    def test_invalid_lineage_stops_before_all_model_work(self):
        campaign = load_campaign()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "engine"
            source.mkdir()
            (source / "gguf-tools/quality-testing").mkdir(parents=True)
            (source / "gguf-tools/quality-testing/score_official").touch()
            (source / "ds4-server").touch()
            harness = root / "harness"
            harness.mkdir()
            model = root / "model.gguf"
            model.touch()
            frozen = {
                "engine_candidate_hash": "1" * 40,
                "harness_candidate_hash": "2" * 40,
                "composite_candidate_sha256": "3" * 64,
                "frozen_at": "2026-07-29T00:00:01+00:00",
            }

            class RejectingGoal:
                @staticmethod
                def validate_manifest_lineage(*_args, **_kwargs):
                    raise ValueError(
                        "drand round was published before the candidate freeze"
                    )

            downstream = (
                "_freeze_scorer",
                "_write_manifests",
                "verify_model_content",
            )
            patches = [
                mock.patch.object(campaign, name)
                for name in downstream
            ]
            started = [patch.start() for patch in patches]
            self.addCleanup(
                lambda: [patch.stop() for patch in reversed(patches)]
            )
            with (
                mock.patch.object(
                    campaign,
                    "verify_frozen_candidate",
                    return_value=(frozen, harness, source, model),
                ),
                mock.patch.object(
                    campaign,
                    "_authenticate_drand",
                    return_value={
                        "round": 1,
                        "randomness": "4" * 64,
                        "signature": "5" * 192,
                        "obtained_at": "2026-07-29T00:00:02+00:00",
                    },
                ),
                mock.patch.object(
                    campaign,
                    "_drand_record",
                    return_value={
                        "round": 1,
                        "randomness": "4" * 64,
                        "signature": "5" * 192,
                    },
                ),
                mock.patch.object(
                    campaign,
                    "confirmation_seed",
                    return_value="6" * 64,
                ),
                mock.patch.object(
                    campaign,
                    "_goal_module",
                    return_value=RejectingGoal(),
                ),
                mock.patch.object(campaign, "sha256_file", return_value="7" * 64),
                self.assertRaisesRegex(
                    ValueError, "published before the candidate freeze"
                ),
            ):
                campaign.run(
                    SimpleNamespace(
                        freeze_dir=root / "freeze",
                        drand_json=root / "drand.json",
                        output=root / "output",
                    )
                )
            for mocked in started:
                mocked.assert_not_called()

    def test_attestation_requires_one_matching_start_and_exit_record(self):
        campaign = load_campaign()
        log = (
            "ds4: GLM compact cache fidelity resolved_mode=2\n"
            "ds4: GLM compact cache fidelity attestation "
            "resolved_mode=2 affine_store_rows=480 affine_changed_values=240\n"
        )
        self.assertEqual(campaign.parse_attestation(log), (2, 480, 240))
        with self.assertRaises(ValueError):
            campaign.parse_attestation(log + log)
        with self.assertRaises(ValueError):
            campaign.parse_attestation(
                log.replace("resolved_mode=2 affine_store_rows", "resolved_mode=0 affine_store_rows")
            )

    def test_driver_supports_root_authority_and_keeps_memory_containment(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("glm_cgroup_run.sh", source)
        self.assertIn('"GLM_SAFE_RUN_AS_CURRENT_USER": "0" if ROOT_AUTHORITY else "1"', source)
        self.assertIn('"GLM_W1_ROOT_AUTHORITY": "1"', source)
        self.assertIn('"GLM_SAFE_KILL_FLOOR_GIB": "40"', source)
        self.assertIn('"GLM_SAFE_MIN_START_GIB": "110"', source)
        self.assertIn('"/run/dsv4/ds4-engine.lock"', source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)


if __name__ == "__main__":
    unittest.main()
