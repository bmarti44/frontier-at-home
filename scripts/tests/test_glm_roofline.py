#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "36_glm_roofline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("glm_roofline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RooflineTests(unittest.TestCase):
    def test_optimistic_bound_still_rejects_parity(self):
        roofline = load_module()
        report = roofline.compute_roofline(
            dsv4_decode_tok_s=11.417,
            clean_intervals_ms=[473.75] * 159,
            loader_groups_ms=[[243.61 / 75] * 75 for _ in range(159)],
            moe_groups_ms=[[101.84 / 75] * 75 for _ in range(159)],
            bandwidth_gb_s=273.0,
        )
        self.assertEqual(report["decision"], "NO_GO")
        self.assertLess(report["roofline"]["optimistic_short_context_ratio"], 0.80)

    def test_rejects_missing_layer_coverage(self):
        roofline = load_module()
        with self.assertRaises(ValueError):
            roofline.compute_roofline(
                dsv4_decode_tok_s=11.0,
                clean_intervals_ms=[400.0] * 2,
                loader_groups_ms=[[1.0]],
                moe_groups_ms=[[1.0]],
                bandwidth_gb_s=273.0,
            )


if __name__ == "__main__":
    unittest.main()
