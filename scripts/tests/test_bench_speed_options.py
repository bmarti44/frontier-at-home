#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
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
            "--seed",
            "123",
            "--model-id",
            "glm-5.2",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bench.parse_args()
        self.assertEqual(args.context_levels, (0,))
        self.assertEqual(args.max_tokens, 160)
        self.assertEqual(args.min_completion_tokens, 128)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.model_id, "glm-5.2")

    def test_any_invalid_or_missing_rep_fails_the_cell(self):
        bench = load_module()
        self.assertTrue(bench.reps_are_complete([{"valid": True}], 1))
        self.assertFalse(bench.reps_are_complete([{"valid": False}], 1))
        self.assertFalse(bench.reps_are_complete([], 1))
        self.assertFalse(bench.reps_are_complete([{"valid": True}], 2))

    def test_observable_output_not_server_usage_controls_decode_validity(self):
        bench = load_module()
        self.assertEqual(bench.observable_output_errors(157, 157, 128), [])
        self.assertTrue(bench.observable_output_errors(157, 160, 128))
        self.assertTrue(bench.observable_output_errors(127, 127, 128))

    def test_parses_strict_raw_token_timestamps(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "noise\n"
                "DS4_TOKEN_TIMING request=chatcmpl-a index=1 monotonic_ns=100 token=7\n"
                "DS4_TOKEN_TIMING request=chatcmpl-a index=2 monotonic_ns=250 token=8\n",
                encoding="utf-8",
            )
            timing = bench.read_token_timing(
                path,
                0,
                expected_request="chatcmpl-a",
                expected_count=2,
            )
        self.assertEqual(timing["request"], "chatcmpl-a")
        self.assertEqual(timing["indices"], [1, 2])
        self.assertEqual(timing["monotonic_ns"], [100, 250])
        self.assertEqual(timing["token_ids"], [7, 8])

    def test_rejects_missing_or_nonmonotonic_raw_timestamps(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "DS4_TOKEN_TIMING request=x index=1 monotonic_ns=200 token=7\n"
                "DS4_TOKEN_TIMING request=x index=3 monotonic_ns=100 token=8\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                bench.read_token_timing(
                    path,
                    0,
                    expected_request="x",
                    expected_count=2,
                )

    def test_raw_token_timing_rejects_malformed_identity_and_count(self):
        bench = load_module()
        mutations = (
            (
                "DS4_TOKEN_TIMING malformed\n",
                "malformed",
            ),
            (
                "DS4_TOKEN_TIMING request=other index=1 monotonic_ns=100 token=7\n",
                "request",
            ),
            (
                "DS4_TOKEN_TIMING request=chatcmpl-a index=1 monotonic_ns=100 token=7\n",
                "count",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            for contents, error in mutations:
                with self.subTest(error=error):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, error):
                        bench.read_token_timing(
                            path,
                            0,
                            expected_request="chatcmpl-a",
                            expected_count=2,
                        )


if __name__ == "__main__":
    unittest.main()
