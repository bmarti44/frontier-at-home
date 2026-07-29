#!/usr/bin/env python3
"""Contracts for the strict W1 affine ABBA/BAAB campaign driver."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


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
