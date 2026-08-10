#!/usr/bin/env python3
"""The request timeout must scale with the generation budget and be recorded.

`REQUEST_TIMEOUT_S` was a module constant of 300 seconds. That was invisible at the
budgets the pre-0731 baselines used, and became the binding limit as soon as the
thinking contract raised max_tokens:

    thinking @2048   0 timeouts, 33/253 truncated at the token ceiling
    thinking @8192   0 truncated, 20/253 killed by the 300 s client timeout

Raising the budget did not remove the limit, it moved it from the token ceiling to
the wall clock. At the measured ~19 tok/s decode rate an 8192-token generation
needs roughly 430 s, so the constant guaranteed that the longest-reasoning items --
exactly the ones the larger budget was meant to capture -- could never complete.

A timed-out item is scored incorrect, so this silently depresses the score of
whichever arm reasons longest. It must therefore be (a) settable, so an arm can be
run with the clock slack its budget requires, and (b) in the config digest, so an
arm run with 300 s cannot be compared against one run with 900 s without the
difference being visible.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "31_bench_accuracy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_accuracy_to", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse(module, *extra: str):
    argv = [
        "31_bench_accuracy.py",
        "--base-url", "http://127.0.0.1:8013",
        "--out", "/dev/null",
        "--stack-label", "unit-test",
        "--suite", "gsm8k",
        "--split", "dev",
        "--transcripts-dir", "/dev/null",
        *extra,
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return module.parse_args()
    finally:
        sys.argv = saved


class RequestTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_request_timeout_is_settable(self):
        args = parse(self.module, "--request-timeout", "900")
        self.assertEqual(args.request_timeout, 900)

    def test_default_preserves_the_published_baseline_behaviour(self):
        args = parse(self.module)
        self.assertEqual(args.request_timeout, 300)

    def test_nonpositive_timeout_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse(self.module, "--request-timeout", "0")

    def test_config_digest_distinguishes_the_request_timeout(self):
        module = self.module
        module.load_config_evidence = lambda _p: ("a" * 64, ["b" * 64], [])
        module.load_harness_manifest_line = lambda: "harness-line"

        short = parse(module, "--request-timeout", "300", "--config-evidence", "/dev/null")
        long = parse(module, "--request-timeout", "900", "--config-evidence", "/dev/null")

        short_digest, short_payload, _ = module.derive_config_digest(short)
        long_digest, long_payload, _ = module.derive_config_digest(long)

        self.assertEqual(short_payload["request_timeout_s"], 300)
        self.assertEqual(long_payload["request_timeout_s"], 900)
        self.assertNotEqual(
            short_digest,
            long_digest,
            "an arm whose long-reasoning items were killed by the clock must not "
            "share a config digest with one that let them finish",
        )


if __name__ == "__main__":
    unittest.main()
