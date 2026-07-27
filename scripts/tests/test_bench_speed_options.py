#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "30_bench_speed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_speed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchOptionTests(unittest.TestCase):
    def test_decisive_subset_options(self):
        bench = load_module()
        argv = [
            str(SCRIPT),
            "--base-url",
            "http://127.0.0.1:8011",
            "--out",
            "/tmp/out.json",
            "--stack-label",
            "test",
            "--context-levels",
            "0",
            "--max-tokens",
            "160",
            "--min-completion-tokens",
            "128",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bench.parse_args()
        self.assertEqual(args.context_levels, (0,))
        self.assertEqual(args.max_tokens, 160)
        self.assertEqual(args.min_completion_tokens, 128)


if __name__ == "__main__":
    unittest.main()
