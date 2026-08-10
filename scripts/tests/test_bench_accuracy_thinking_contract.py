#!/usr/bin/env python3
"""The thinking contract must be the default and must be visible in the digest.

DeepSeek-V4-Flash-0731 is a reasoning model. Measured on MMLU-Pro dev, it scores
201/253 with the thinking rendering and 178/253 with the chat rendering, so the
chat rendering silently discards ~9 points of the model's capability. Thinking is
therefore the operating contract for this model, and the harness default must
match the contract the endpoint actually serves rather than the contract the
superseded baselines were measured under.

Two independent properties are asserted here:

1. `--thinking-mode` defaults to `thinking`. A caller who passes nothing gets the
   contract the model is served under.
2. The rendering mode participates in the config digest. The digest keys the
   holdout ledger (`spend_holdout_budget`), so if two renderings collapse to one
   digest, a chat run and a thinking run over the same suite are indistinguishable
   to every consumer that compares by digest -- which is exactly the silent
   cross-contract comparison the flag exists to prevent.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "31_bench_accuracy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_accuracy", SCRIPT)
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


class ThinkingContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_thinking_mode_defaults_to_thinking(self):
        args = parse(self.module)
        self.assertEqual(args.thinking_mode, "thinking")

    def test_chat_rendering_remains_selectable(self):
        args = parse(self.module, "--thinking-mode", "chat")
        self.assertEqual(args.thinking_mode, "chat")

    def test_config_digest_distinguishes_the_rendering_mode(self):
        module = self.module
        evidence = [
            {"path": "x", "sha256": "0" * 64},
        ]

        def fake_load_config_evidence(_paths):
            return "a" * 64, ["b" * 64], evidence

        module.load_config_evidence = fake_load_config_evidence
        module.load_harness_manifest_line = lambda: "harness-line"

        chat = parse(module, "--thinking-mode", "chat", "--config-evidence", "/dev/null")
        thinking = parse(module, "--thinking-mode", "thinking", "--config-evidence", "/dev/null")

        chat_digest, chat_payload, _ = module.derive_config_digest(chat)
        think_digest, think_payload, _ = module.derive_config_digest(thinking)

        self.assertIn("thinking_mode", chat_payload)
        self.assertEqual(chat_payload["thinking_mode"], "chat")
        self.assertEqual(think_payload["thinking_mode"], "thinking")
        self.assertNotEqual(
            chat_digest,
            think_digest,
            "chat and thinking renderings collapse to one config digest; "
            "holdout-ledger keys and every digest-keyed comparison would treat "
            "two different contracts as the same run",
        )


if __name__ == "__main__":
    unittest.main()
