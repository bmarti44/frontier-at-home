#!/usr/bin/env python3
"""Golden and harness-selection tests for the Qwen3.8 chat encoder."""

import importlib.util
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "31_bench_accuracy.py"
DSV4_ENCODER_PATH = (
    ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_dsv4.py"
)
QWEN38_ENCODER_PATH = (
    ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_qwen38.py"
)
TOKENIZER_CONFIG_PATH = Path(
    "/home/bmarti44/models/qwen3.8-27b/tokenizer_config.json"
)

REASONING_INSTRUCTIONS = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
MESSAGES = [
    {"role": "system", "content": "Follow the instructions."},
    {"role": "user", "content": "What is 2 + 2?"},
]
THINKING_GOLDEN = (
    f"<|im_start|>system\n{REASONING_INSTRUCTIONS}\n\n"
    "Follow the instructions.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
CHAT_GOLDEN = (
    "<|im_start|>system\nFollow the instructions.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_benchmark(module, *extra: str):
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


class Qwen38EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_qwen38_test", QWEN38_ENCODER_PATH)

    def test_system_user_thinking_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="thinking"),
            THINKING_GOLDEN,
        )

    def test_system_user_chat_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="chat"),
            CHAT_GOLDEN,
        )

    def test_context_affects_rendering_but_is_not_reemitted(self):
        rendered = self.encoder.encode_messages(
            [{"role": "user", "content": "New question"}],
            thinking_mode="chat",
            context=[{"role": "user", "content": "Prior question"}],
        )
        self.assertEqual(
            rendered,
            "<|im_start|>user\nNew question<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_encoder_default_preserves_dsv4_behavior(self):
        benchmark = load_module("bench_accuracy_qwen38_test", BENCHMARK_PATH)
        official_dsv4 = load_module("encoding_dsv4_reference", DSV4_ENCODER_PATH)
        args = parse_benchmark(benchmark)

        self.assertEqual(args.encoder, "dsv4")
        selected = benchmark.load_encoder(args.encoder)
        row = {"question": "What is 1 + 1?"}
        rendered, rendering = benchmark.render_item(
            "gsm8k", row, selected, args.thinking_mode, args.encoder
        )
        content = (
            "What is 1 + 1?\n\nThink briefly if needed, then end with the final "
            "numeric answer on its own line in the form: Answer: <number>"
        )
        self.assertEqual(
            rendered,
            official_dsv4.encode_messages(
                [{"role": "user", "content": content}], thinking_mode="thinking"
            ),
        )
        self.assertEqual(rendering, "official-encoder-thinking")

    def test_config_digest_distinguishes_encoder_choice(self):
        benchmark = load_module("bench_accuracy_qwen38_digest_test", BENCHMARK_PATH)
        evidence = [{"path": "x", "sha256": "0" * 64}]
        benchmark.load_config_evidence = lambda _paths: (
            "a" * 64,
            ["b" * 64],
            evidence,
        )
        benchmark.load_harness_manifest_line = lambda: "harness-line"

        dsv4 = parse_benchmark(
            benchmark, "--encoder", "dsv4", "--config-evidence", "/dev/null"
        )
        qwen38 = parse_benchmark(
            benchmark, "--encoder", "qwen38", "--config-evidence", "/dev/null"
        )
        dsv4_digest, dsv4_payload, _ = benchmark.derive_config_digest(dsv4)
        qwen38_digest, qwen38_payload, _ = benchmark.derive_config_digest(qwen38)

        self.assertEqual(dsv4_payload["encoder"], "dsv4")
        self.assertEqual(qwen38_payload["encoder"], "qwen38")
        self.assertNotEqual(dsv4_digest, qwen38_digest)

    def test_reasoning_effort_accepts_each_encoder_contract(self):
        benchmark = load_module("bench_accuracy_effort_accept_test", BENCHMARK_PATH)
        self.assertEqual(
            parse_benchmark(
                benchmark, "--encoder", "dsv4", "--reasoning-effort", "max"
            ).reasoning_effort,
            "max",
        )
        self.assertEqual(
            parse_benchmark(
                benchmark, "--encoder", "qwen38", "--reasoning-effort", "low"
            ).reasoning_effort,
            "low",
        )

    def test_reasoning_effort_rejects_cross_encoder_values_early(self):
        benchmark = load_module("bench_accuracy_effort_reject_test", BENCHMARK_PATH)
        for encoder, effort in (("dsv4", "low"), ("qwen38", "max")):
            with self.subTest(encoder=encoder, effort=effort):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    parse_benchmark(
                        benchmark,
                        "--encoder", encoder,
                        "--reasoning-effort", effort,
                    )
                message = stderr.getvalue()
                self.assertIn(f"encoder '{encoder}' does not support", message)
                self.assertIn(f"'{effort}'", message)

    def test_matches_pinned_jinja_for_golden_cases(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 is not installed")
        if not TOKENIZER_CONFIG_PATH.is_file():
            self.skipTest(f"pinned tokenizer config is unavailable: {TOKENIZER_CONFIG_PATH}")

        config = json.loads(TOKENIZER_CONFIG_PATH.read_text(encoding="utf-8"))
        template = jinja2.Environment().from_string(config["chat_template"])
        for mode in ("thinking", "chat"):
            with self.subTest(mode=mode):
                expected = template.render(
                    messages=MESSAGES,
                    add_generation_prompt=True,
                    enable_thinking=mode == "thinking",
                )
                actual = self.encoder.encode_messages(MESSAGES, thinking_mode=mode)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
