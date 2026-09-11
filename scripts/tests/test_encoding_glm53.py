#!/usr/bin/env python3
"""Golden, harness-selection, and template-fidelity tests for the GLM-5.3 encoder.

The fidelity part reuses the shared matrix in scripts/tests/template_fidelity.py
against the chat template shipped with the served weights. GLM's template has
no ``enable_thinking`` switch: it always ends the generation prompt with
``<|assistant|><think>``, and the model's own non-thinking form is
``<think></think>`` (how the template writes an assistant turn without
reasoning). The adapter therefore appends ``</think>`` to the official render
for chat-mode cases, which is exactly the encoder's documented contract.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

from scripts.tests.template_fidelity import (
    FidelityAdapter,
    assert_context_split,
    assert_matrix,
    set_generation_tails,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "31_bench_accuracy.py"
ENCODER_PATH = ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_glm53.py"
TEMPLATE_PATH = Path(
    os.environ.get(
        "GLM53_CHAT_TEMPLATE_PATH",
        "/home/bmarti44/models/glm-5.3-flash/k2-densek4-mtp/chat_template.jinja",
    )
)

MESSAGES = [
    {"role": "system", "content": "Follow the instructions."},
    {"role": "user", "content": "What is 2 + 2?"},
]
THINKING_GOLDEN = (
    "[gMASK]<sop><|system|>Reasoning Effort: Max"
    "<|system|>Follow the instructions."
    "<|user|>What is 2 + 2?"
    "<|assistant|><think>"
)
CHAT_GOLDEN = THINKING_GOLDEN + "</think>"
LOW_GOLDEN = THINKING_GOLDEN.replace("Reasoning Effort: Max", "Reasoning Effort: Low")


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


class Glm53EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_glm53_test", ENCODER_PATH)
        cls.bench = load_module("bench_accuracy_glm53_test", BENCHMARK_PATH)

    def test_system_user_thinking_golden(self):
        self.assertEqual(self.encoder.encode_messages(MESSAGES, thinking_mode="thinking"), THINKING_GOLDEN)

    def test_system_user_chat_golden(self):
        self.assertEqual(self.encoder.encode_messages(MESSAGES, thinking_mode="chat"), CHAT_GOLDEN)

    def test_low_effort_renders_the_system_line(self):
        self.assertEqual(
            self.encoder.encode_messages(MESSAGES, thinking_mode="thinking", reasoning_effort="low"),
            LOW_GOLDEN,
        )

    def test_bos_text_can_be_omitted(self):
        rendered = self.encoder.encode_messages(MESSAGES, thinking_mode="thinking", add_default_bos_token=False)
        self.assertEqual(rendered, THINKING_GOLDEN[len("[gMASK]<sop>"):])

    def test_context_affects_rendering_but_is_not_reemitted(self):
        context = [MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "4"}]
        suffix = self.encoder.encode_messages(
            [{"role": "user", "content": "And 3 + 3?"}], thinking_mode="thinking", context=context,
        )
        self.assertEqual(suffix, "<|user|>And 3 + 3?<|assistant|><think>")

    def test_drop_thinking_keeps_reasoning_only_after_last_user(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "r1"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d", "reasoning_content": "r2"},
        ]
        dropped = self.encoder.encode_messages(messages, thinking_mode="thinking")
        self.assertIn("<|assistant|><think></think>b", dropped)
        self.assertIn("<|assistant|><think>r2</think>d", dropped)
        kept = self.encoder.encode_messages(messages, thinking_mode="thinking", drop_thinking=False)
        self.assertIn("<|assistant|><think>r1</think>b", kept)

    def test_rejects_tools_and_media(self):
        with self.assertRaises(ValueError):
            self.encoder.encode_messages(
                [{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}, {"role": "user", "content": "z"}],
                thinking_mode="thinking",
            )
        with self.assertRaises(ValueError):
            self.encoder.encode_messages(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
                thinking_mode="thinking",
            )

    def test_rejects_unknown_effort(self):
        with self.assertRaises(ValueError):
            self.encoder.encode_messages(MESSAGES, thinking_mode="thinking", reasoning_effort="xhigh")

    def test_benchmark_registers_glm53_with_its_effort_contract(self):
        self.assertEqual(self.bench.ENCODER_PATHS["glm53"], ENCODER_PATH)
        self.assertEqual(self.bench.ENCODER_REASONING_EFFORTS["glm53"], frozenset(("low", "high", "max")))
        self.assertTrue(self.bench.ENCODER_EMIT_BOS_TEXT["glm53"])
        for effort in ("low", "high", "max"):
            args = parse_benchmark(self.bench, "--encoder", "glm53", "--reasoning-effort", effort)
            self.assertEqual(args.reasoning_effort, effort)

    def test_benchmark_rejects_cross_encoder_efforts(self):
        for effort in ("xhigh", "medium", "off"):
            with self.subTest(effort=effort), self.assertRaises(SystemExit):
                parse_benchmark(self.bench, "--encoder", "glm53", "--reasoning-effort", effort)

    def test_render_item_uses_the_glm_template(self):
        rendered, label = self.bench.render_item(
            "gsm8k", {"question": "1+1?"}, self.encoder, "thinking", "glm53", "low",
        )
        self.assertTrue(rendered.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>1+1?"))
        self.assertTrue(rendered.endswith("<|assistant|><think>"))
        self.assertEqual(label, "official-encoder-glm53-thinking-effort-low")


class Glm53TemplateFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jinja2
        except ImportError:
            raise unittest.SkipTest("jinja2 is not installed")
        if not TEMPLATE_PATH.is_file():
            raise unittest.SkipTest(f"GLM-5.3 chat template is unavailable: {TEMPLATE_PATH}")
        # The template uses {% break %}: the harness compiles with a bare
        # Environment(), so hand it one that enables loop controls (as the
        # Hugging Face / vLLM renderer does).
        cls.jinja2 = types.SimpleNamespace(
            Environment=lambda: jinja2.Environment(extensions=["jinja2.ext.loopcontrols"]),
        )
        cls.encoder = load_module("encoding_glm53_fidelity", ENCODER_PATH)
        set_generation_tails(["<|assistant|><think></think>", "<|assistant|><think>"])
        cls.adapter = FidelityAdapter(
            encode=cls._encode, official=cls._official, template_path=TEMPLATE_PATH,
        )

    @classmethod
    def _encode(cls, case):
        return cls.encoder.encode_messages(
            list(case.messages),
            thinking_mode="thinking" if case.thinking else "chat",
            context=list(case.context) or None,
            drop_thinking=case.drop_thinking,
            reasoning_effort="max",
        )

    @classmethod
    def _official(cls, template, case):
        rendered = template.render(
            messages=case.full_messages(), add_generation_prompt=True, clear_thinking=case.drop_thinking,
        )
        # GLM has no enable_thinking switch; its non-thinking form closes the
        # think block immediately.
        return rendered if case.thinking else rendered + "</think>"

    def test_standard_matrix_is_byte_exact(self):
        assert_matrix(self, self.adapter, self.jinja2)

    def test_context_split_reconstruction(self):
        assert_context_split(self, self.adapter, self.jinja2)

    def test_effort_line_matches_the_template(self):
        template = self.jinja2.Environment().from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
        for effort in ("low", "high", None):
            with self.subTest(effort=effort):
                self.assertEqual(
                    self.encoder.encode_messages(MESSAGES, thinking_mode="thinking", reasoning_effort=effort),
                    template.render(messages=MESSAGES, add_generation_prompt=True, reasoning_effort=effort),
                )


if __name__ == "__main__":
    unittest.main()
