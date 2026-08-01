#!/usr/bin/env python3
"""Contracts for the minimal GLM Rung 0.1 slab campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/70_glm_rung0_slab_campaign.py"
SPEC = importlib.util.spec_from_file_location("glm_rung0_slab_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class Rung0SlabCampaignTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
