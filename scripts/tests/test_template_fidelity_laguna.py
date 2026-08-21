#!/usr/bin/env python3
"""Laguna S 2.1 wiring for the shared template-fidelity harness.

This complements test_encoding_laguna.py (golden strings and benchmark
wiring): here the shared standard matrix in template_fidelity.py is the
source of the fixtures, so future hardening of that matrix automatically
applies to Laguna as well.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

from scripts.tests.template_fidelity import (
    FidelityAdapter,
    assert_context_split,
    assert_matrix,
    set_generation_tails,
)

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = (
    ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_laguna.py"
)
TEMPLATE_PATH = Path(
    os.environ.get(
        "LAGUNA_CHAT_TEMPLATE_PATH",
        "/home/bmarti44/models/laguna-s-2.1/poolside/chat_template.jinja",
    )
)


def _load_encoder():
    spec = importlib.util.spec_from_file_location(
        "encoding_laguna_fidelity", ENCODER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _neutralize_generation_extension(source: str) -> str:
    # Hugging Face registers ``generation`` as an output-tracking extension.
    # It emits no text, so an always-true block is equivalent here.
    source = source.replace("{%- generation -%}", "{%- if true -%}")
    return source.replace("{%- endgeneration -%}", "{%- endif -%}")


class LagunaTemplateFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jinja2
        except ImportError:
            raise unittest.SkipTest("jinja2 is not installed")
        if not TEMPLATE_PATH.is_file():
            raise unittest.SkipTest(
                f"Laguna chat template is unavailable: {TEMPLATE_PATH}"
            )
        cls.jinja2 = jinja2
        cls.encoder = _load_encoder()
        set_generation_tails(
            ["<assistant><think>", "<assistant></think>"]
        )
        cls.adapter = FidelityAdapter(
            encode=cls._encode,
            official=cls._official,
            template_path=TEMPLATE_PATH,
            preprocess=_neutralize_generation_extension,
        )

    @classmethod
    def _encode(cls, case):
        return cls.encoder.encode_messages(
            list(case.messages),
            thinking_mode="thinking" if case.thinking else "chat",
            context=list(case.context) or None,
            drop_thinking=case.drop_thinking,
            reasoning_effort="max" if case.thinking else "off",
        )

    @classmethod
    def _official(cls, template, case):
        return template.render(
            messages=case.full_messages(),
            add_generation_prompt=True,
            enable_thinking=case.thinking,
            preserve_thinking=not case.drop_thinking,
        )

    def test_standard_matrix_is_byte_exact(self):
        assert_matrix(self, self.adapter, self.jinja2)

    def test_context_split_reconstruction(self):
        assert_context_split(self, self.adapter, self.jinja2)


if __name__ == "__main__":
    unittest.main()
