#!/usr/bin/env python3
"""Production-launcher contract for the default-off DSV4 cold-load arm."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "21_serve_llamacpp.sh"
SWITCH = ROOT / "scripts" / "52_engine_switch.sh"
PROFILE = ROOT / "configs" / "profiles" / "dsv4-1m-fast.env"
PLAN = ROOT / "results" / "dsv4-cold-load" / "plan.json"


class Dsv4ColdLoadContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.launcher = LAUNCHER.read_text(encoding="utf-8")

    def test_direct_io_is_exact_logged_default_off_boolean(self) -> None:
        self.assertIn("direct_io=${DSV4_DIRECT_IO:-0}", self.launcher)
        self.assertIn(
            "DSV4_DIRECT_IO must be 0 or 1",
            self.launcher,
        )
        self.assertIn("DSV4 cold-load mode: direct-io", self.launcher)

    def test_direct_io_requires_no_mmap_and_binary_support(self) -> None:
        self.assertIn(
            "DSV4_DIRECT_IO=1 requires DSV4_NO_MMAP=1",
            self.launcher,
        )
        self.assertIn(
            "llama-server lacks required --direct-io-required support",
            self.launcher,
        )

    def test_direct_io_reaches_only_the_model_load_command(self) -> None:
        self.assertIn(
            "(( direct_io == 0 )) || server_command+=(--direct-io-required)",
            self.launcher,
        )
        self.assertIn(
            "(( direct_io != 0 )) || server_command+=(--no-direct-io)",
            self.launcher,
        )
        self.assertNotIn("DSV4_DIRECT_IO", PROFILE.read_text(encoding="utf-8"))
        self.assertNotIn("DSV4_DIRECT_IO", SWITCH.read_text(encoding="utf-8"))

    def test_acceptance_formula_is_frozen(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        acceptance = plan["acceptance"]
        self.assertIn("fresh_server_blocks", acceptance)
        self.assertEqual(acceptance["fresh_server_blocks"], 5)
        self.assertEqual(acceptance["maximum_candidate_ready_seconds"], 30.0)
        self.assertEqual(acceptance["minimum_fio_fraction"], 0.5)
        expected = (
            plan["model_bytes"]
            / 1_000_000_000
            / plan["fio_reference"]["conservative_sustained_gb_s"]
        )
        self.assertAlmostEqual(
            acceptance["reference_seconds_at_full_fio"], expected, places=12
        )
        self.assertAlmostEqual(
            acceptance["maximum_tensor_load_seconds_at_minimum_fio_fraction"],
            expected / acceptance["minimum_fio_fraction"],
            places=12,
        )
        self.assertTrue(acceptance["exact_replay_first_token_match"])
        self.assertTrue(acceptance["agent_gate_required"])


if __name__ == "__main__":
    unittest.main()
