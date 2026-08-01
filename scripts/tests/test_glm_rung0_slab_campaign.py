#!/usr/bin/env python3
"""Contracts for the minimal GLM Rung 0.1 slab campaign."""

from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/70_glm_rung0_slab_campaign.py"
SPEC = importlib.util.spec_from_file_location("glm_rung0_slab_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class Rung0SlabCampaignTests(unittest.TestCase):
    def passing_records(self):
        records = []
        for block, sequence, arm in CAMPAIGN.arm_schedule():
            mode = "off" if arm == "A" else "on"
            step = 100_000_000 if mode == "off" else 80_000_000
            reps = []
            for rep in range(2):
                reps.append(
                    {
                        "valid": True,
                        "request_sha256": "d" * 64,
                        "generated_reasoning_sha256": "e" * 64,
                        "generated_content_sha256": "f" * 64,
                        "client_completion_tokens": 128,
                        "sse_token_timestamps_ns": [
                            1_000_000_000 + index * step for index in range(128)
                        ],
                        "token_ids": list(range(128)),
                        "ttft_s": 1.0 if mode == "off" else 1.02,
                        "client_prompt_tokens": 256,
                    }
                )
            records.append(
                {
                    "schema_version": 1,
                    "block": block,
                    "sequence": sequence,
                    "arm": arm,
                    "mode": mode,
                    "server_instance_id": f"server-{block}-{sequence}",
                    "binary_sha256": "a" * 64,
                    "configuration_sha256": ("b" if mode == "off" else "c")
                    * 64,
                    "fixture_sha256": "d" * 64,
                    "suite_valid": True,
                    "reps": reps,
                    "engine": {
                        "slab_mode": mode,
                        "slab_reads": 0 if mode == "off" else 20,
                        "slab_peak_qd": 0 if mode == "off" else 8,
                        "access_stream_sha256": "1" * 64,
                        "arena_pin_ok": True,
                        "trace_lines": 0,
                    },
                    "external_io": {
                        "read_bytes_delta": 1000,
                        "elapsed_seconds": 1.0,
                        "peak_read_qd": 1 if mode == "off" else 8,
                        "sample_count": 20,
                    },
                    "safety": {
                        "minimum_available_gib": 18.0,
                        "cgroup_high_events": 0,
                        "cgroup_max_events": 0,
                        "cgroup_oom_events": 0,
                        "cgroup_swap_bytes": 0,
                        "xid": False,
                        "survivors": [],
                        "failures": [],
                    },
                }
            )
        return records

    @staticmethod
    def passing_nll():
        return {
            "case_count": 100,
            "token_weighted_delta_nll": 0.0,
            "top1_loss_pp": 0.0,
            "deterministic": True,
        }

    def test_schedule_is_five_fresh_abba_baab_blocks(self):
        schedule = CAMPAIGN.arm_schedule()
        self.assertEqual(len(schedule), 20)
        for block in range(5):
            group = [row for row in schedule if row[0] == block]
            self.assertEqual([row[1] for row in group], list(range(4)))
            self.assertEqual(
                "".join(row[2] for row in group),
                "ABBA" if block % 2 == 0 else "BAAB",
            )

    def test_timed_arms_differ_only_by_slab_identity(self):
        off = CAMPAIGN.canonical_engine_environment("off")
        on = CAMPAIGN.canonical_engine_environment("on")
        slab = {
            "DS4_CUDA_EXPERT_SLAB_PATH",
            "DS4_CUDA_EXPERT_SLAB_SHA256",
            "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256",
        }
        self.assertEqual(set(on) - set(off), slab)
        self.assertEqual(
            {key: value for key, value in on.items() if key not in slab}, off
        )
        self.assertEqual(off["DS4_CUDA_FETCH_THREADS"], "8")
        self.assertNotIn("DS4_CUDA_EXPERT_SLAB_TRACE", on)

    def test_fixed_scorer_accepts_complete_lossless_campaign(self):
        result = CAMPAIGN.score_campaign(self.passing_records(), self.passing_nll())
        self.assertEqual(result["verdict"], "PASS")
        self.assertGreater(result["decode_ratio_lower_95"], 1.0)
        self.assertLessEqual(result["warm_ttft_ratio_upper_95"], 1.05)

    def test_fixed_scorer_rejects_false_success_mutations(self):
        mutations = {
            "identical arms": lambda rows: rows[1].update(
                configuration_sha256="b" * 64
            ),
            "zero slab reads": lambda rows: rows[1]["engine"].update(
                slab_reads=0
            ),
            "output mismatch": lambda rows: rows[1]["reps"][0].update(
                generated_content_sha256="0" * 64
            ),
            "short output": lambda rows: rows[1]["reps"][0].update(
                sse_token_timestamps_ns=[1, 2]
            ),
            "missing io": lambda rows: rows[1]["external_io"].update(
                sample_count=0
            ),
            "stale server": lambda rows: rows[1].update(
                server_instance_id=rows[0]["server_instance_id"]
            ),
            "safety event": lambda rows: rows[1]["safety"].update(
                cgroup_high_events=1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                rows = copy.deepcopy(self.passing_records())
                mutate(rows)
                with self.assertRaises(ValueError):
                    CAMPAIGN.score_campaign(rows, self.passing_nll())

    def test_fixed_scorer_requires_exact_zero_nll_for_lossless_transport(self):
        nll = self.passing_nll()
        nll["token_weighted_delta_nll"] = 1e-9
        with self.assertRaises(ValueError):
            CAMPAIGN.score_campaign(self.passing_records(), nll)


if __name__ == "__main__":
    unittest.main()
