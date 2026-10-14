#!/usr/bin/env python3
"""Golden, template-fidelity and harness tests for the MiMo-V2.6 encoder.

MiMo-V2.6-Flash's official template (identical in the RL and MOPD repos up to
a trailing newline) has no BOS, no reasoning-effort levels and no separator
between turns. Thinking is on unless ``enable_thinking`` is false, which
renders an empty ``<think></think>`` after the assistant header. Every past
assistant turn carries ``<think>{reasoning_content}</think>`` (empty when
absent), so the encoder ignores ``drop_thinking``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tests.template_fidelity import (  # noqa: E402
    FidelityAdapter,
    assert_context_split,
    assert_matrix,
    set_generation_tails,
)

ENCODER_PATH = ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_mimo26.py"
BENCHMARK_PATH = ROOT / "scripts" / "31_bench_accuracy.py"
KIT_PATH = ROOT / "scripts" / "94_qualify_profile.py"
TEMPLATE_PATH = Path(os.environ.get(
    "MIMO26_CHAT_TEMPLATE_PATH",
    "/home/bmarti44/models/mimo-v2.6-flash-mopd/chat_template.jinja",
))

MESSAGES = [
    {"role": "system", "content": "Follow the instructions."},
    {"role": "user", "content": "What is 2 + 2?"},
]
THINKING_GOLDEN = (
    "<|im_start|>system\nFollow the instructions.<|im_end|>"
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>"
    "<|im_start|>assistant\n"
)
CHAT_GOLDEN = THINKING_GOLDEN + "<think></think>"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MiMo26Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_mimo26_test", ENCODER_PATH)

    def test_thinking_and_chat_renderings(self) -> None:
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="thinking"),
            THINKING_GOLDEN)
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="chat"),
            CHAT_GOLDEN)

    def test_history_keeps_reasoning_and_content_verbatim(self) -> None:
        messages = [
            {"role": "user", "content": " hi "},
            {"role": "assistant", "content": " ok ", "reasoning_content": " r "},
            {"role": "assistant", "content": "x", "reasoning": "ignored field"},
            {"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]},
        ]
        for drop in (True, False):
            with self.subTest(drop_thinking=drop):
                self.assertEqual(
                    self.encoder.encode_messages(messages, thinking_mode="thinking",
                                                 drop_thinking=drop),
                    "<|im_start|>user\n hi <|im_end|>"
                    "<|im_start|>assistant\n<think> r </think> ok <|im_end|>"
                    "<|im_start|>assistant\n<think></think>x<|im_end|>"
                    "<|im_start|>user\nab<|im_end|>"
                    "<|im_start|>assistant\n")

    def test_bos_flag_and_effort_and_unsupported_inputs(self) -> None:
        enc = self.encoder.encode_messages
        self.assertEqual(enc(MESSAGES, "thinking", add_default_bos_token=True),
                         THINKING_GOLDEN)
        for effort in ("low", "high", "max", "xhigh"):
            with self.subTest(effort=effort):
                with self.assertRaises(ValueError):
                    enc(MESSAGES, "thinking", reasoning_effort=effort)
        for bad in (
            [],
            [{"role": "assistant", "content": "x",
              "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}],
            [{"role": "user", "content": "x", "tools": [{"name": "f"}]}],
            [{"role": "user", "content": [{"type": "image_url", "image_url": "u"}]}],
            [{"role": "user", "content": 5}],
        ):
            with self.subTest(messages=bad):
                with self.assertRaises(ValueError):
                    enc(bad, "thinking")
        with self.assertRaises(ValueError):
            enc(MESSAGES, "sometimes")

    def test_context_renders_only_the_suffix(self) -> None:
        context = [MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "4"}]
        suffix = self.encoder.encode_messages(
            [{"role": "user", "content": "And 3 + 3?"}], "chat", context=context)
        self.assertEqual(
            suffix,
            "<|im_start|>user\nAnd 3 + 3?<|im_end|><|im_start|>assistant\n<think></think>")


class MiMo26TemplateFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jinja2
        except ImportError:
            raise unittest.SkipTest("jinja2 is not installed")
        if not TEMPLATE_PATH.is_file():
            raise unittest.SkipTest(f"MiMo chat template is unavailable: {TEMPLATE_PATH}")
        cls.jinja2 = jinja2
        cls.encoder = load_module("encoding_mimo26_fidelity", ENCODER_PATH)
        set_generation_tails(["<|im_start|>assistant\n<think></think>",
                              "<|im_start|>assistant\n"])
        cls.adapter = FidelityAdapter(
            encode=cls._encode, official=cls._official, template_path=TEMPLATE_PATH)

    @classmethod
    def _encode(cls, case):
        return cls.encoder.encode_messages(
            list(case.messages),
            thinking_mode="thinking" if case.thinking else "chat",
            context=list(case.context) or None,
            drop_thinking=case.drop_thinking,
        )

    @staticmethod
    def _official(template, case):
        return template.render(messages=case.full_messages(),
                               add_generation_prompt=True,
                               enable_thinking=case.thinking)

    def test_standard_matrix_is_byte_exact(self) -> None:
        assert_matrix(self, self.adapter, self.jinja2)

    def test_context_split_reconstructs_the_full_render(self) -> None:
        assert_context_split(self, self.adapter, self.jinja2)


class MiMo26Harness(unittest.TestCase):
    def test_bench_registers_mimo26_without_bos_or_effort_levels(self) -> None:
        bench = load_module("bench_accuracy_mimo26", BENCHMARK_PATH)
        self.assertEqual(bench.ENCODER_PATHS["mimo26"], ENCODER_PATH)
        self.assertEqual(bench.ENCODER_REASONING_EFFORTS["mimo26"], frozenset())
        self.assertIs(bench.ENCODER_EMIT_BOS_TEXT["mimo26"], False)

    def test_kit_maps_the_catalog_slug_to_mimo26(self) -> None:
        kit = load_module("qualify_profile_mimo26", KIT_PATH)
        self.assertEqual(kit.MODEL_ENCODERS["mimo-v2.6-flash"], "mimo26")


if __name__ == "__main__":
    unittest.main()
