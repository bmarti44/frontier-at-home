#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_module("dsv4_cold_scorer", ROOT / "scripts/94_score_dsv4_cold_load.py")
RUNNER = load_module("dsv4_cold_runner", ROOT / "scripts/95_run_dsv4_cold_load.py")
MODEL_BYTES = 96_832_507_552
SHA = {name: char * 64 for name, char in {
    "candidate": "1", "runner": "2", "scorer": "3", "model": "4",
    "config": "5", "bundle": "6", "semantic": "7", "logit": "8",
    "randomness": "9", "receipt": "a",
}.items()}


def expected_schedules(randomness_hex: str) -> list[str]:
    seed = bytes.fromhex(randomness_hex)
    domain = b"frontier-at-home/dsv4-cold-load/v1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0 else "BAAB"
        for block in range(5)
    ]


def manifest() -> dict[str, object]:
    randomness = SHA["randomness"]
    return {
        "schema_version": 1,
        "candidate_hash": SHA["candidate"],
        "runner_sha256": SHA["runner"],
        "scorer_sha256": SHA["scorer"],
        "model_sha256": SHA["model"],
        "configuration_sha256": SHA["config"],
        "runtime_bundle_sha256": SHA["bundle"],
        "model_bytes": MODEL_BYTES,
        "randomness": {
            "value": randomness,
            "receipt_sha256": SHA["receipt"],
        },
        "schedules": expected_schedules(randomness),
    }


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    schedules = expected_schedules(SHA["randomness"])
    for block, schedule in enumerate(schedules):
        for position, letter in enumerate(schedule):
            arm = "off" if letter == "A" else "on"
            launch = (block * 4 + position + 1) * 100_000_000_000
            ready_seconds = 90.0 if arm == "off" else 18.0
            tensor_seconds = 80.0 if arm == "off" else 16.0
            result.append({
                "schema_version": 1,
                "block": block,
                "position": position,
                "arm": arm,
                "run_id": f"b{block}-p{position}",
                "candidate_hash": SHA["candidate"],
                "model_sha256": SHA["model"],
                "configuration_sha256": SHA["config"],
                "runtime_bundle_sha256": SHA["bundle"],
                "process_launch_monotonic_ns": launch,
                "health_ready_monotonic_ns": launch + int(ready_seconds * 1e9),
                "tensor_load_start_monotonic_ns": launch + 1_000_000_000,
                "tensor_load_end_monotonic_ns": launch + 1_000_000_000 + int(tensor_seconds * 1e9),
                "server_pid": 10_000 + block * 4 + position,
                "server_start_ticks": 20_000 + block * 4 + position,
                "server_fresh": True,
                "physical_read_bytes": MODEL_BYTES,
                "cache_resident_bytes_before": 0,
                "direct_shard_count": 3 if arm == "on" else 0,
                "direct_required": arm == "on",
                "semantic_sha256": SHA["semantic"],
                "first_token_logit_sha256": SHA["logit"],
                "authenticated_health": True,
                "authenticated_completion": True,
                "unauthenticated_rejected": True,
                "minimum_mem_available_kb": 50 * 1024 * 1024,
                "swap_growth_bytes": 0,
                "cgroup_oom_delta": 0,
                "cgroup_oom_kill_delta": 0,
                "xid_count": 0,
                "surviving_descendants": 0,
                "containment_rc": 0,
            })
    return result


class Dsv4ColdLoadCampaignTests(unittest.TestCase):
    def test_randomness_derives_five_balanced_blocks(self) -> None:
        schedules = RUNNER.arm_schedule(SHA["randomness"])
        self.assertEqual(schedules, expected_schedules(SHA["randomness"]))
        self.assertEqual(RUNNER.arm_schedule(SHA["randomness"]), schedules)
        self.assertNotEqual(RUNNER.arm_schedule("b" * 64), schedules)

    def test_accepts_matched_safe_fast_campaign(self) -> None:
        result = SCORER.score_campaign(manifest(), rows())
        self.assertEqual(result["verdict"], "PASS")
        self.assertLessEqual(result["observed"]["ready_ratio_upper_95"], 0.5)
        self.assertLessEqual(result["observed"]["on_tensor_seconds_upper_95"], 20.393206603359758)

    def test_rejects_incomplete_duplicate_unmatched_or_stale_rows(self) -> None:
        mutations = []
        bad = rows()[:-1]; mutations.append(bad)
        bad = rows(); bad[1]["run_id"] = bad[0]["run_id"]; mutations.append(bad)
        bad = rows(); bad[2]["model_sha256"] = "f" * 64; mutations.append(bad)
        bad = rows(); bad[3]["runtime_bundle_sha256"] = "e" * 64; mutations.append(bad)
        bad = rows(); bad[4]["arm"] = "on" if bad[4]["arm"] == "off" else "off"; mutations.append(bad)
        for raw in mutations:
            self.assertEqual(SCORER.score_campaign(manifest(), raw)["verdict"], "FAIL")

    def test_rejects_warm_fallback_unsafe_or_unobserved_arms(self) -> None:
        fields = {
            "cache_resident_bytes_before": 2 * 1024**3,
            "physical_read_bytes": MODEL_BYTES // 2,
            "server_fresh": False,
            "minimum_mem_available_kb": 9 * 1024 * 1024,
            "swap_growth_bytes": 4096,
            "cgroup_oom_kill_delta": 1,
            "xid_count": 1,
            "surviving_descendants": 1,
            "containment_rc": 1,
        }
        for key, value in fields.items():
            bad = rows(); bad[1][key] = value
            self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL", key)
        bad = rows(); on = next(row for row in bad if row["arm"] == "on"); on["direct_shard_count"] = 2
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); off = next(row for row in bad if row["arm"] == "off"); off["direct_shard_count"] = 1
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")

    def test_rejects_auth_semantic_and_timing_failures(self) -> None:
        for key in ("authenticated_health", "authenticated_completion", "unauthenticated_rejected"):
            bad = rows(); bad[0][key] = False
            self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["semantic_sha256"] = "c" * 64
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["first_token_logit_sha256"] = "d" * 64
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["health_ready_monotonic_ns"] = bad[1]["process_launch_monotonic_ns"]
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")

    def test_rejects_performance_regression_and_nonfinite_values(self) -> None:
        bad = rows()
        for row in bad:
            if row["arm"] == "on":
                row["health_ready_monotonic_ns"] = row["process_launch_monotonic_ns"] + 60_000_000_000
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[0]["physical_read_bytes"] = float("nan")
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
