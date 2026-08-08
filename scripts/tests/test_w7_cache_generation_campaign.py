#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/90_score_w7_cache_generation_campaign.py"
SPEC = importlib.util.spec_from_file_location("w7_cache_campaign", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SCHEDULES = ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA"]
TOKEN_IDS = list(range(128))
SHA = {
    "binary": "1" * 64,
    "model": "2" * 64,
    "config": "3" * 64,
    "request": "4" * 64,
    "output": hashlib.sha256(
        json.dumps(TOKEN_IDS, separators=(",", ":")).encode("ascii")
    ).hexdigest(),
    "logits": "6" * 64,
    "logit_sequence": "7" * 64,
}


def good_rows() -> list[dict[str, object]]:
    rows = []
    for block, schedule in enumerate(SCHEDULES):
        for position, letter in enumerate(schedule):
            arm = "off" if letter == "A" else "on"
            request_start = 10_000_000_000 * (1 + block * 4 + position)
            ttft_ns = 2_000_000_000 if arm == "off" else 1_800_000_000
            step_ns = 500_000_000 if arm == "off" else 476_190_476
            first = request_start + ttft_ns
            timestamps = [first + token * step_ns for token in range(128)]
            rows.append({
                "block": block,
                "position": position,
                "arm": arm,
                "run_id": f"b{block}-p{position}",
                "binary_sha256": SHA["binary"],
                "model_sha256": SHA["model"],
                "common_config_sha256": SHA["config"],
                "request_sha256": SHA["request"],
                "stable_remap": 1 if arm == "on" else 0,
                "request_start_ns": request_start,
                "token_timestamps_ns": timestamps,
                "output_token_ids": TOKEN_IDS.copy(),
                "output_sha256": SHA["output"],
                "generated_text_sha256": "9" * 64,
                "generated_text_bytes": 128,
                "final_logits_sha256": SHA["logits"],
                "logit_sequence_sha256": SHA["logit_sequence"],
                "server_fresh": True,
                "safety": {
                    "containment_rc": 0,
                    "minimum_mem_available_kb": 49_000_000,
                    "swap_growth_bytes": 0,
                    "cgroup_max_delta": 0,
                    "cgroup_oom_delta": 0,
                    "cgroup_oom_kill_delta": 0,
                    "xid_count": 0,
                    "surviving_descendants": 0,
                    "false_generation_flushes": 0 if arm == "on" else 374,
                },
            })
    return rows


class W7CacheGenerationCampaignTest(unittest.TestCase):
    def test_fixed_formula_accepts_passing_five_block_campaign(self) -> None:
        result = MODULE.score_campaign_rows(good_rows(), SCHEDULES)
        self.assertEqual(result["verdict"], "PASS")
        self.assertLessEqual(result["observed"]["ttft_ratio_upper_95"], 0.95)
        self.assertGreaterEqual(result["observed"]["decode_ratio_lower_95"], 1.0)

    def test_rejects_schedule_fixture_and_identity_mutations(self) -> None:
        mutations = []
        bad = good_rows(); bad[0]["arm"] = "on"; mutations.append(bad)
        bad = good_rows(); bad[1]["request_sha256"] = "7" * 64; mutations.append(bad)
        bad = good_rows(); bad[2]["binary_sha256"] = "8" * 64; mutations.append(bad)
        bad = good_rows(); bad[3]["run_id"] = bad[2]["run_id"]; mutations.append(bad)
        for rows in mutations:
            self.assertEqual(MODULE.score_campaign_rows(rows, SCHEDULES)["verdict"], "FAIL")

    def test_rejects_output_logit_short_timing_and_safety_mutations(self) -> None:
        mutations = []
        bad = good_rows(); bad[1]["output_sha256"] = "7" * 64; mutations.append(bad)
        bad = good_rows(); bad[1]["final_logits_sha256"] = "7" * 64; mutations.append(bad)
        bad = good_rows(); bad[1]["logit_sequence_sha256"] = "8" * 64; mutations.append(bad)
        bad = good_rows(); bad[1]["generated_text_sha256"] = "8" * 64; mutations.append(bad)
        bad = good_rows(); bad[1]["output_token_ids"][0] = 999; mutations.append(bad)
        bad = good_rows(); bad[1]["token_timestamps_ns"] = bad[1]["token_timestamps_ns"][:127]; mutations.append(bad)
        bad = good_rows(); bad[1]["safety"]["cgroup_oom_kill_delta"] = 1; mutations.append(bad)
        bad = good_rows(); bad[1]["server_fresh"] = False; mutations.append(bad)
        for rows in mutations:
            self.assertEqual(MODULE.score_campaign_rows(rows, SCHEDULES)["verdict"], "FAIL")

    def test_rejects_ttft_and_decode_bounds(self) -> None:
        bad_ttft = good_rows()
        for row in bad_ttft:
            if row["arm"] == "on":
                first = row["request_start_ns"] + 1_940_000_000
                step = 476_190_476
                row["token_timestamps_ns"] = [first + token * step for token in range(128)]
        self.assertEqual(MODULE.score_campaign_rows(bad_ttft, SCHEDULES)["verdict"], "FAIL")

        bad_decode = good_rows()
        for row in bad_decode:
            if row["arm"] == "on":
                first = row["token_timestamps_ns"][0]
                row["token_timestamps_ns"] = [first + token * 526_315_789 for token in range(128)]
        self.assertEqual(MODULE.score_campaign_rows(bad_decode, SCHEDULES)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
