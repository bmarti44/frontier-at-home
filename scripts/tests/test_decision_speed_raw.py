#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "34_decision.py"


def load_module():
    spec = importlib.util.spec_from_file_location("decision_speed_raw", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DecisionRawTimingTests(unittest.TestCase):
    def test_validates_raw_timestamps_instead_of_retokenized_text(self):
        decision = load_module()
        rep = {
            "completion_tokens": 200,
            "server_completion_tokens": 200,
            "client_completion_tokens": 204,
            "ttft_s": 1.0,
            "decode_tok_s": 2.0,
            "timing_source": "server_raw_token_log",
            "token_timestamps_ns": list(range(1, 201)),
            "token_ids": list(range(200)),
        }
        self.assertTrue(decision.recompute_speed_rep_validity(rep))
        rep["token_timestamps_ns"][64] = rep["token_timestamps_ns"][63]
        self.assertFalse(decision.recompute_speed_rep_validity(rep))


if __name__ == "__main__":
    unittest.main()
