#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "w4_serving_scorer", ROOT / "scripts/101_score_w4_serving_campaign.py")
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
SHA = hashlib.sha256(b"fixture").hexdigest()


class W4ServingCampaignScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedules = ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA"]
        self.rows = []
        for block, schedule in enumerate(self.schedules):
            for position, letter in enumerate(schedule):
                arm = "off" if letter == "A" else "on"
                seconds = 10.0 if arm == "off" else 8.0
                self.rows.append({
                    "block": block, "position": position, "arm": arm,
                    "run_id": f"b{block}p{position}", "binary_sha256": SHA,
                    "model_sha256": SHA, "common_config_sha256": SHA,
                    "request_sha256": SHA, "topk_cub": int(arm == "on"),
                    "request_start_ns": 1_000_000_000,
                    "response_complete_ns": 1_000_000_000 + int(seconds * 1e9),
                    "prompt_tokens": 18_000, "cached_tokens": 0,
                    "cache_write_tokens": 18_000,
                    "response_semantic_sha256": SHA,
                    "final_logits_sha256": SHA, "logit_sequence_sha256": SHA,
                    "executed_environment_sha256": SHA,
                    "topk_marker_count": int(arm == "on"),
                    "server_fresh": True,
                    "safety": {
                        "containment_rc": 0,
                        "minimum_mem_available_kib": 20 * 1024 * 1024,
                        "swap_growth_bytes": 0, "cgroup_max_delta": 0,
                        "cgroup_oom_delta": 0, "cgroup_oom_kill_delta": 0,
                        "xid_count": 0, "surviving_descendants": 0,
                    },
                })
        self.microgate = {
            "block_a_ms": [4.9] * 5, "block_b_ms": [1.6] * 5,
            "selected_ids_sha256": SHA,
            "speedup_lower_95": 4.9 / 1.6,
            "required_speedup_lower_95": 2.0, "verdict": "PASS",
        }

    def score(self):
        return SCORER.score_campaign_rows(
            self.rows, self.schedules, self.microgate)

    def test_valid_campaign_passes_fixed_formula(self) -> None:
        result = self.score()
        self.assertEqual(result["verdict"], "PASS")
        self.assertAlmostEqual(result["observed"]["prefill_speedup_lower_95"],
                               1.25)

    def test_prefill_below_five_percent_fails(self) -> None:
        for row in self.rows:
            if row["arm"] == "on":
                row["response_complete_ns"] = 10_900_000_000
        result = self.score()
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["prefill_speedup"])

    def test_missing_duplicate_reordered_or_unequal_fixture_fails(self) -> None:
        variants = [self.rows[:-1], self.rows + [copy.deepcopy(self.rows[-1])]]
        swapped = copy.deepcopy(self.rows)
        swapped[0]["position"], swapped[1]["position"] = 1, 0
        variants.append(swapped)
        unequal = copy.deepcopy(self.rows)
        unequal[-1]["request_sha256"] = "1" * 64
        variants.append(unequal)
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.assertEqual(
                    SCORER.score_campaign_rows(rows, self.schedules,
                                               self.microgate)["verdict"],
                    "FAIL")

    def test_nonfinite_nonpositive_or_short_timing_fails(self) -> None:
        for value in (math.nan, math.inf, 1_000_000_000, 999_999_999):
            rows = copy.deepcopy(self.rows)
            rows[0]["response_complete_ns"] = value
            with self.subTest(value=value):
                self.assertEqual(
                    SCORER.score_campaign_rows(rows, self.schedules,
                                               self.microgate)["verdict"],
                    "FAIL")

    def test_cache_flag_marker_logit_and_safety_mutations_fail(self) -> None:
        for name in ("cache", "flag", "marker", "logit", "oom", "memory"):
            rows = copy.deepcopy(self.rows)
            if name == "cache": rows[0]["cached_tokens"] = 1
            if name == "flag": rows[0]["topk_cub"] = 1
            if name == "marker": rows[0]["topk_marker_count"] = 1
            if name == "logit": rows[0]["final_logits_sha256"] = "2" * 64
            if name == "oom": rows[0]["safety"]["cgroup_oom_kill_delta"] = 1
            if name == "memory": rows[0]["safety"]["minimum_mem_available_kib"] = 1
            with self.subTest(name=name):
                self.assertEqual(
                    SCORER.score_campaign_rows(rows, self.schedules,
                                               self.microgate)["verdict"],
                    "FAIL")

    def test_microgate_malformed_or_regressed_fails(self) -> None:
        for mutation in ("timing", "speed", "verdict"):
            micro = copy.deepcopy(self.microgate)
            if mutation == "timing": micro["block_b_ms"][0] = 0
            if mutation == "speed": micro["speedup_lower_95"] = 1.0
            if mutation == "verdict": micro["verdict"] = "FAIL"
            with self.subTest(mutation=mutation):
                self.assertEqual(
                    SCORER.score_campaign_rows(self.rows, self.schedules,
                                               micro)["verdict"], "FAIL")

    def test_authoritative_replay_rejects_synthetic_rows_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schedules.json").write_text(json.dumps(self.schedules))
            (root / "microgate-summary.json").write_text(json.dumps(self.microgate))
            (root / "raw.jsonl").write_text("".join(
                json.dumps(row) + "\n" for row in self.rows))
            result = SCORER.score_run_dir(root)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("manifest", result["failure"])

    def test_stable_reader_rejects_symlink_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"public evidence")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                SCORER._read_stable(link)

    def test_snapshot_parsing_cannot_reopen_replaced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "summary.json"
            artifact.write_bytes(b'{"verdict":"PASS"}\n')
            snapshot = SCORER._snapshot_files(root, {"summary.json"})
            artifact.rename(root / "summary.original")
            artifact.write_bytes(b'{"verdict":"FAIL"}\n')
            self.assertEqual(snapshot["summary.json"][0], b'{"verdict":"PASS"}\n')
            with self.assertRaises(SCORER.InvalidCampaign):
                SCORER._verify_snapshot_unchanged(root, snapshot, {"summary.json"})

    def test_runner_and_base_execute_from_retained_candidate_bytes(self) -> None:
        runner_path = ROOT / "scripts/102_run_w4_serving_campaign.py"
        base_path = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
        runner_bytes = runner_path.read_bytes()
        base_bytes = base_path.read_bytes()
        with mock.patch.object(SCORER.importlib.util, "spec_from_file_location",
                               side_effect=AssertionError("filesystem import reopened")):
            module = SCORER._load_runner_from_bytes(
                runner_bytes, base_bytes, runner_path)
        self.assertTrue(callable(module.parse_arm))
        self.assertEqual(module.BASE.__name__, "w7_campaign_base")

    def test_authoritative_runner_replay_rejects_malformed_extra_sync_trace(self) -> None:
        runner_path = ROOT / "scripts/102_run_w4_serving_campaign.py"
        base_path = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
        runner = SCORER._load_runner_from_bytes(
            runner_path.read_bytes(), base_path.read_bytes(), runner_path)
        first = ("ds4: GLM sync start=0 prompt=19772 suffix=19772 checkpoint=0 "
                 "dense_len=0 ctx_cap=8192 dense_fit=0 resume_min=4 dense_gap=0 "
                 "indexed_keep=0 indexed_batch=1 batch_ffn=1")
        second = ("ds4: GLM sync start=19772 prompt=19783 suffix=11 checkpoint=19772 "
                  "dense_len=0 ctx_cap=8192 dense_fit=0 resume_min=4 dense_gap=1 "
                  "indexed_keep=1 indexed_batch=1 batch_ffn=1")
        mutated = first + "\n" + second + "\n" + second + " hidden=1"
        with self.assertRaisesRegex(runner.CampaignError, "malformed sync trace"):
            runner.validate_novel_sync_trace(mutated, 19_783)
        unicode_segment = (first.replace("prompt=19772", "prompt=1977٢", 1)
                           + "\n" + second)
        with self.assertRaisesRegex(runner.CampaignError, "malformed sync trace"):
            runner.validate_novel_sync_trace(unicode_segment, 19_783)


if __name__ == "__main__":
    unittest.main()
