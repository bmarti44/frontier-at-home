#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/92_score_w7_evict_store_probe.py"
SPEC = importlib.util.spec_from_file_location("w7_evict_store_scorer", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(arm: str) -> dict[str, object]:
    start = 1_000_000_000
    first = start + (2_000_000_000 if arm == "off" else 1_300_000_000)
    return {
        "arm": arm,
        "position": 0 if arm == "off" else 1,
        "run_id": f"run-{arm}",
        "binary_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "common_config_sha256": "3" * 64,
        "request_sha256": "4" * 64,
        "diagnostic_skip": 0 if arm == "off" else 1,
        "request_start_ns": start,
        "token_timestamps_ns": [first + index * 500_000_000 for index in range(128)],
        "output_token_ids": list(range(128)),
        "output_sha256": hashlib.sha256(
            json.dumps(list(range(128)), separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "generated_text_sha256": "6" * 64,
        "generated_text_bytes": 80,
        "logit_sha256s": ["7" * 64, "8" * 64, "9" * 64],
        "selected_checkpoint_tokens": 5044,
        "checkpoint_id": "token-text:9e5ba8aa0b75e6c618f68d9834ef541c44cd4b42",
        "evict_store_count": 1 if arm == "off" else 0,
        "skip_marker_count": 0 if arm == "off" else 1,
        "activation_marker_count": 0 if arm == "off" else 1,
        "server_fresh": True,
        "safety": {
            "containment_rc": 0,
            "minimum_mem_available_kb": 48_000_000,
            "swap_growth_bytes": 0,
            "cgroup_max_delta": 0,
            "cgroup_oom_delta": 0,
            "cgroup_oom_kill_delta": 0,
            "xid_count": 0,
            "surviving_descendants": 0,
        },
    }


class W7EvictStoreProbeScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [row("off"), row("on")]

    def test_valid_probe_passes_fixed_formula(self) -> None:
        result = MODULE.score_probe_rows(self.rows, ["off", "on"])
        self.assertEqual(result["verdict"], "PASS", result)
        self.assertGreaterEqual(result["observed"]["warm_append_seconds_saved"], 0.5)
        self.assertGreaterEqual(result["observed"]["decode_ratio"], 0.99)

    def assert_fails(self, rows: list[dict[str, object]], order=None) -> None:
        result = MODULE.score_probe_rows(rows, order or ["off", "on"])
        self.assertEqual(result["verdict"], "FAIL", result)

    def test_rejects_marker_and_checkpoint_mutations(self) -> None:
        for field, value in (
            ("evict_store_count", 1),
            ("skip_marker_count", 0),
            ("activation_marker_count", 0),
            ("selected_checkpoint_tokens", 5043),
        ):
            mutated = copy.deepcopy(self.rows)
            mutated[1][field] = value
            self.assert_fails(mutated)

    def test_rejects_unequal_outputs_logits_or_fixtures(self) -> None:
        for field, value in (
            ("output_token_ids", [999] * 128),
            ("generated_text_sha256", "a" * 64),
            ("logit_sha256s", ["a" * 64, "8" * 64, "9" * 64]),
            ("request_sha256", "b" * 64),
            ("binary_sha256", "c" * 64),
        ):
            mutated = copy.deepcopy(self.rows)
            mutated[1][field] = value
            self.assert_fails(mutated)

    def test_rejects_distinct_valid_checkpoint_identity(self) -> None:
        mutated = copy.deepcopy(self.rows)
        mutated[1]["checkpoint_id"] = "token-text:" + "a" * 40
        self.assert_fails(mutated)

    def test_rejects_short_or_unsafe_rows(self) -> None:
        mutated = copy.deepcopy(self.rows)
        mutated[1]["token_timestamps_ns"] = mutated[1]["token_timestamps_ns"][:127]
        self.assert_fails(mutated)
        for field, value in (
            ("minimum_mem_available_kb", 10 * 1024 * 1024 - 1),
            ("swap_growth_bytes", 1),
            ("xid_count", 1),
            ("surviving_descendants", 1),
        ):
            mutated = copy.deepcopy(self.rows)
            mutated[1]["safety"][field] = value
            self.assert_fails(mutated)

    def test_rejects_insufficient_perf(self) -> None:
        mutated = copy.deepcopy(self.rows)
        mutated[1]["token_timestamps_ns"] = [
            mutated[1]["request_start_ns"] + 1_800_000_000 + index * 510_000_000
            for index in range(128)
        ]
        self.assert_fails(mutated)

    def test_rejects_missing_duplicate_or_order_mismatch(self) -> None:
        self.assert_fails(self.rows[:1])
        self.assert_fails([self.rows[0], copy.deepcopy(self.rows[0])])
        self.assert_fails(self.rows, ["on", "off"])


if __name__ == "__main__":
    unittest.main()
