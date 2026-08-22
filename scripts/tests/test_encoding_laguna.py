#!/usr/bin/env python3
"""Golden and harness-selection tests for the Laguna S 2.1 chat encoder."""

import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "31_bench_accuracy.py"
LAGUNA_ENCODER_PATH = (
    ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_laguna.py"
)
LAGUNA_TEMPLATE_PATH = Path(
    os.environ.get(
        "LAGUNA_CHAT_TEMPLATE_PATH",
        "/home/bmarti44/models/laguna-s-2.1/poolside/chat_template.jinja",
    )
)

MESSAGES = [
    {"role": "system", "content": "Follow the instructions."},
    {"role": "user", "content": "What is 2 + 2?"},
]
MAX_GOLDEN = (
    "〈|EOS|〉<system>Follow the instructions.</system>\n"
    "<user>What is 2 + 2?</user>\n"
    "<assistant><think>"
)
OFF_GOLDEN = (
    "〈|EOS|〉<system>Follow the instructions.</system>\n"
    "<user>What is 2 + 2?</user>\n"
    "<assistant></think>"
)
MULTI_TURN = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "First question"},
    {
        "role": "assistant",
        "reasoning_content": "Earlier reasoning",
        "content": "Earlier answer",
    },
    {"role": "user", "content": "Follow-up"},
]


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


class LagunaEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_laguna_test", LAGUNA_ENCODER_PATH)

    def test_system_user_max_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(
                MESSAGES, thinking_mode="thinking", reasoning_effort="max"
            ),
            MAX_GOLDEN,
        )

    def test_system_user_default_is_max_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="thinking"),
            MAX_GOLDEN,
        )

    def test_system_user_off_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(
                MESSAGES, thinking_mode="thinking", reasoning_effort="off"
            ),
            OFF_GOLDEN,
        )

    def test_chat_mode_is_off_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="chat"),
            OFF_GOLDEN,
        )

    def test_multi_turn_max_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(
                MULTI_TURN, thinking_mode="thinking", reasoning_effort="max"
            ),
            "〈|EOS|〉<system>Be concise.</system>\n"
            "<user>First question</user>\n"
            "<assistant><think>Earlier reasoning</think>Earlier answer</assistant>\n"
            "<user>Follow-up</user>\n"
            "<assistant><think>",
        )

    def test_multi_turn_off_golden(self):
        self.assertEqual(
            self.encoder.encode_messages(
                MULTI_TURN, thinking_mode="thinking", reasoning_effort="off"
            ),
            "〈|EOS|〉<system>Be concise.</system>\n"
            "<user>First question</user>\n"
            "<assistant></think>Earlier answer</assistant>\n"
            "<user>Follow-up</user>\n"
            "<assistant></think>",
        )

    def test_off_drops_reasoning_after_last_user(self):
        messages = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "reasoning": "must not survive",
                "content": "Answer",
            },
        ]
        self.assertEqual(
            self.encoder.encode_messages(
                messages, thinking_mode="thinking", reasoning_effort="off"
            ),
            f"〈|EOS|〉<system>{self.encoder.DEFAULT_SYSTEM_MESSAGE}</system>\n"
            "<user>Question</user>\n"
            "<assistant></think>Answer</assistant>\n"
            "<assistant></think>",
        )

    def test_context_continuation_is_exact_official_template_suffix(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 is not installed")
        if not LAGUNA_TEMPLATE_PATH.is_file():
            self.skipTest(
                f"Laguna chat template is unavailable: {LAGUNA_TEMPLATE_PATH}"
            )

        template = self._load_official_template(jinja2)
        context = [
            {"role": "system", "content": "Split exactly."},
            {"role": "user", "content": "Prior question"},
            {
                "role": "assistant",
                "reasoning_content": "Prior reasoning",
                "content": "Prior answer",
            },
        ]
        messages = [{"role": "user", "content": "New question"}]
        render_args = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
        prefix = template.render(
            messages=context, add_generation_prompt=False, **render_args
        )
        full = template.render(
            messages=context + messages,
            add_generation_prompt=True,
            **render_args,
        )
        self.assertTrue(full.startswith(prefix))

        suffix = self.encoder.encode_messages(
            messages,
            thinking_mode="thinking",
            context=context,
            reasoning_effort="off",
        )
        self.assertEqual(prefix + suffix, full)
        self.assertFalse(suffix.startswith(self.encoder.EOS))
        self.assertNotIn("<system>", suffix)

    def test_benchmark_accepts_laguna_efforts(self):
        benchmark = load_module("bench_accuracy_laguna_accept_test", BENCHMARK_PATH)
        for effort in ("off", "max"):
            with self.subTest(effort=effort):
                args = parse_benchmark(
                    benchmark, "--encoder", "laguna", "--reasoning-effort", effort
                )
                self.assertEqual(args.reasoning_effort, effort)

    def test_benchmark_rejects_other_efforts(self):
        benchmark = load_module("bench_accuracy_laguna_reject_test", BENCHMARK_PATH)
        for effort in ("low", "medium", "xhigh", "high"):
            with self.subTest(effort=effort):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    parse_benchmark(
                        benchmark,
                        "--encoder", "laguna",
                        "--reasoning-effort", effort,
                    )
                self.assertIn("encoder 'laguna' does not support", stderr.getvalue())

    @staticmethod
    def _load_official_template(jinja2):
        source = LAGUNA_TEMPLATE_PATH.read_text(encoding="utf-8")
        # Hugging Face registers ``generation`` as an output-tracking extension.
        # It emits no text, so an always-true block is equivalent for this check.
        source = source.replace("{%- generation -%}", "{%- if true -%}")
        source = source.replace("{%- endgeneration -%}", "{%- endif -%}")
        return jinja2.Environment().from_string(source)

    def test_matches_official_jinja_matrix(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 is not installed")
        if not LAGUNA_TEMPLATE_PATH.is_file():
            self.skipTest(
                f"Laguna chat template is unavailable: {LAGUNA_TEMPLATE_PATH}"
            )

        template = self._load_official_template(jinja2)
        cases = [
            (
                "default-system",
                [{"role": "user", "content": "Hello"}],
                "off",
                True,
            ),
            (
                "empty-system-off",
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Hello"},
                ],
                "off",
                True,
            ),
            (
                "empty-system-thinking-emits-block",
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Hello"},
                ],
                "max",
                True,
            ),
            (
                "preserve-mid-conversation-reasoning",
                [
                    {"role": "user", "content": "First"},
                    {
                        "role": "assistant",
                        "reasoning_content": "kept",
                        "content": "Answer",
                    },
                    {"role": "user", "content": "Second"},
                ],
                "off",
                False,
            ),
            (
                "drop-assistant-after-last-user-reasoning",
                [
                    {"role": "user", "content": "Only question"},
                    {
                        "role": "assistant",
                        "reasoning": "dropped",
                        "content": "Answer",
                    },
                ],
                "off",
                True,
            ),
            (
                "reasoning-precedes-reasoning-content",
                [
                    {"role": "user", "content": "First"},
                    {
                        "role": "assistant",
                        "reasoning": "preferred",
                        "reasoning_content": "fallback only",
                        "content": "First answer",
                    },
                    {"role": "user", "content": "Second"},
                    {
                        "role": "assistant",
                        "reasoning_content": "fallback",
                        "content": "Second answer",
                    },
                    {"role": "user", "content": "Third"},
                ],
                "off",
                False,
            ),
            (
                "system-trailing-whitespace-rstrip",
                [
                    {"role": "system", "content": "Keep leading  \n\t  "},
                    {"role": "user", "content": "Hello"},
                ],
                "off",
                True,
            ),
        ]
        for name, messages, effort, drop_thinking in cases:
            with self.subTest(name=name):
                expected = template.render(
                    messages=messages,
                    add_generation_prompt=True,
                    enable_thinking=effort == "max",
                    preserve_thinking=not drop_thinking,
                )
                actual = self.encoder.encode_messages(
                    messages,
                    thinking_mode="thinking",
                    reasoning_effort=effort,
                    drop_thinking=drop_thinking,
                )
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()


class LagunaBosTextTests(unittest.TestCase):
    """add_default_bos_token=False omits the literal leading EOS glyph.

    The Laguna GGUF sets add_bos_token=true (BOS == EOS == token 2), so
    llama.cpp adds the token at /v1/completions tokenization; emitting the
    glyph too would double it (G1 smoke finding, 2026-08-21).
    """

    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_laguna_bos_test", LAGUNA_ENCODER_PATH)

    def test_default_keeps_the_template_faithful_prefix(self):
        rendered = self.encoder.encode_messages(
            [{"role": "user", "content": "Hi"}], thinking_mode="thinking"
        )
        self.assertTrue(rendered.startswith(self.encoder.EOS))

    def test_false_omits_only_the_leading_glyph(self):
        messages = [
            {"role": "system", "content": "Terse."},
            {"role": "user", "content": "Hi"},
        ]
        with_bos = self.encoder.encode_messages(
            messages, thinking_mode="thinking"
        )
        without_bos = self.encoder.encode_messages(
            messages, thinking_mode="thinking", add_default_bos_token=False
        )
        self.assertEqual(self.encoder.EOS + without_bos, with_bos)

    def test_context_continuation_never_emits_the_glyph(self):
        rendered = self.encoder.encode_messages(
            [{"role": "user", "content": "More"}],
            thinking_mode="thinking",
            context=[{"role": "user", "content": "Hi"}],
            add_default_bos_token=True,
        )
        self.assertFalse(rendered.startswith(self.encoder.EOS))
