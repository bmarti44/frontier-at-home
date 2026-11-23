#!/usr/bin/env python3
"""Golden, harness-selection, and template-fidelity tests for the Inkling 2 encoder.

The fidelity part renders the official ``chat_template.jinja`` shipped with
the weights the way Hugging Face (and therefore vLLM) does: a sandboxed
environment with ``trim_blocks``/``lstrip_blocks``, loop controls, the
``raise_exception`` global, and a ``tojson`` filter that takes
``sort_keys``/``separators`` and does not escape HTML. The template's
``{% generation %}`` block only tracks assistant spans, so it is replaced by an
always-true block. The template file's digest is pinned through
``weights/inkling-2-mini/manifest.json``; a different template fails instead
of silently passing.

Inkling's template has no generation prompt: it ends after the last message
(or the thinking-effort line), and the model writes ``<|message_model|>``
itself. ``thinking_mode='chat'`` is the template's ``none`` effort (level 0).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path

from scripts.tests.template_fidelity import (
    FidelityAdapter,
    FidelityCase,
    assert_context_split,
    assert_matrix,
    load_template,
    set_generation_tails,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "31_bench_accuracy.py"
KIT_PATH = ROOT / "scripts" / "94_qualify_profile.py"
ENCODER_PATH = ROOT / "vendor" / "official-encoding" / "encoding" / "encoding_inkling2.py"
WEIGHTS_MANIFEST = ROOT / "weights" / "inkling-2-mini" / "manifest.json"
TEMPLATE_PATH = Path(
    os.environ.get(
        "INKLING2_CHAT_TEMPLATE_PATH",
        "/home/bmarti44/models/inkling-2-mini/chat_template.jinja",
    )
)

SYS = "<|message_system|>"
USER = "<|message_user|>"
MODEL = "<|message_model|>"
TOOL = "<|message_tool|>"
TEXT = "<|content_text|>"
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
IMAGE = "<|content_image|><|unused_200054|>"


def effort_line(level: str) -> str:
    return f"{SYS}{TEXT}Thinking effort level: {level}{END}"


MESSAGES = [
    {"role": "system", "content": "Follow the instructions."},
    {"role": "user", "content": "What is 2 + 2?"},
]
THINKING_GOLDEN = (
    f"{SYS}{TEXT}Follow the instructions.{END}"
    + effort_line("0.9")
    + f"{USER}{TEXT}What is 2 + 2?{END}"
)
CHAT_GOLDEN = THINKING_GOLDEN.replace(effort_line("0.9"), effort_line("0"))
EFFORT_LEVELS = {
    "none": "0",
    "minimal": "0.1",
    "low": "0.2",
    "medium": "0.7",
    "high": "0.9",
    "xhigh": "0.99",
    "max": "0.99",
}
PUBLIC_EFFORTS = frozenset(("low", "medium", "high", "xhigh", "max"))

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city, in °C.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
        },
    },
}
BARE_TOOL = {"name": "ping", "parameters": {"type": "object", "properties": {}}}
TOOL_CALL_TURN = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Need the weather first.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"unit": "c", "city": "Zürich", "opts": {"z": 1, "a": [1, 2.5, None, True]}},
            },
        }
    ],
}
TOOL_RESULT_BY_ID = {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 7}'}
TOOL_RESULT_BY_NAME = {"role": "tool", "name": "get_weather", "content": "7 degrees"}


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


class Inkling2EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module("encoding_inkling2_test", ENCODER_PATH)
        cls.bench = load_module("bench_accuracy_inkling2_test", BENCHMARK_PATH)

    def encode(self, messages, **kwargs):
        kwargs.setdefault("thinking_mode", "thinking")
        return self.encoder.encode_messages(messages, **kwargs)

    def test_system_user_thinking_golden(self):
        self.assertEqual(self.encode(MESSAGES), THINKING_GOLDEN)

    def test_chat_mode_is_the_templates_none_effort(self):
        self.assertEqual(self.encode(MESSAGES, thinking_mode="chat"), CHAT_GOLDEN)
        # Chat mode ignores the effort, as the other encoders do.
        self.assertEqual(
            self.encode(MESSAGES, thinking_mode="chat", reasoning_effort="low"), CHAT_GOLDEN
        )

    def test_every_template_effort_name_renders_its_level(self):
        for effort, level in EFFORT_LEVELS.items():
            with self.subTest(effort=effort):
                self.assertEqual(
                    self.encode(MESSAGES, reasoning_effort=effort),
                    THINKING_GOLDEN.replace(effort_line("0.9"), effort_line(level)),
                )

    def test_effort_line_precedes_the_first_non_system_message(self):
        rendered = self.encode([{"role": "user", "content": "hi"}], reasoning_effort="low")
        self.assertEqual(rendered, effort_line("0.2") + f"{USER}{TEXT}hi{END}")

    def test_system_only_conversation_ends_with_the_effort_line(self):
        rendered = self.encode([{"role": "system", "content": "a"}, {"role": "system", "content": "b"}])
        self.assertEqual(rendered, f"{SYS}{TEXT}a{END}{SYS}{TEXT}b{END}" + effort_line("0.9"))

    def test_no_generation_prompt_and_no_bos_text(self):
        with_bos = self.encode(MESSAGES, add_default_bos_token=True)
        without_bos = self.encode(MESSAGES, add_default_bos_token=False)
        self.assertEqual(with_bos, without_bos)
        self.assertFalse(with_bos.endswith(MODEL))
        self.assertTrue(with_bos.endswith(END))

    def test_tool_declarations_are_sorted_compact_json(self):
        rendered = self.encode([{"role": "user", "content": "x"}], tools=[WEATHER_TOOL, BARE_TOOL])
        specs = [
            {
                "description": "Weather for a city, in °C.",
                "name": "get_weather",
                "parameters": WEATHER_TOOL["function"]["parameters"],
                "type": "function",
            },
            {"description": "", "name": "ping", "parameters": BARE_TOOL["parameters"], "type": "function"},
        ]
        declared = json.dumps(specs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            rendered,
            f"{SYS}tool_declare<|content_xml|>{declared}{END}" + effort_line("0.9") + f"{USER}{TEXT}x{END}",
        )

    def test_assistant_reasoning_tool_calls_and_tool_results(self):
        messages = [
            {"role": "user", "content": "Weather?"},
            TOOL_CALL_TURN,
            TOOL_RESULT_BY_ID,
            TOOL_RESULT_BY_NAME,
            {"role": "assistant", "content": "7 degrees."},
        ]
        call = (
            '{"name":"get_weather","args":{"city":"Zürich",'
            '"opts":{"z":1,"a":[1,2.5,null,true]},"unit":"c"}}'
        )
        self.assertEqual(
            self.encode(messages, reasoning_effort="low"),
            effort_line("0.2")
            + f"{USER}{TEXT}Weather?{END}"
            + f"{MODEL}<|content_thinking|>Need the weather first.{END}"
            + f"{MODEL}get_weather<|content_invoke_tool_json|>{call}{END}{EOS}"
            + f'{TOOL}get_weather{TEXT}{{"temp": 7}}{END}'
            + f"{TOOL}get_weather{TEXT}7 degrees{END}"
            + f"{MODEL}{TEXT}7 degrees.{END}{EOS}",
        )

    def test_reasoning_content_is_always_kept(self):
        # The official template has no clear/preserve-thinking switch.
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "r1"},
            {"role": "user", "content": "c"},
        ]
        dropped = self.encode(messages, drop_thinking=True)
        self.assertEqual(dropped, self.encode(messages, drop_thinking=False))
        self.assertIn(f"{MODEL}<|content_thinking|>r1{END}{MODEL}{TEXT}b{END}{EOS}", dropped)

    def test_list_content_text_and_image_parts(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Look:"},
                {"type": "image_url", "image_url": {"url": "data:,"}},
                {"type": "image"},
                "plain",
            ],
        }]
        self.assertEqual(
            self.encode(messages),
            effort_line("0.9")
            + f"{USER}{TEXT}Look:{END}{USER}{IMAGE}{END}{USER}{IMAGE}{END}{USER}{TEXT}plain{END}",
        )

    def test_context_affects_rendering_but_is_not_reemitted(self):
        context = [MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "4"}]
        suffix = self.encode([{"role": "user", "content": "And 3 + 3?"}], context=context, tools=[BARE_TOOL])
        self.assertEqual(suffix, f"{USER}{TEXT}And 3 + 3?{END}")

    def test_tool_header_lookup_sees_the_context(self):
        context = [{"role": "user", "content": "Weather?"}, TOOL_CALL_TURN]
        suffix = self.encode([TOOL_RESULT_BY_ID], context=context)
        self.assertEqual(suffix, f'{TOOL}get_weather{TEXT}{{"temp": 7}}{END}')

    def test_rejects_what_the_template_rejects(self):
        bad = {
            "unknown effort": ([{"role": "user", "content": "x"}], {"reasoning_effort": "turbo"}),
            "unknown role": ([{"role": "developer", "content": "x"}], {}),
            "assistant image": ([{"role": "assistant", "content": [{"type": "image"}]}], {}),
            "audio": ([{"role": "user", "content": [{"type": "input_audio"}]}], {}),
            "video": ([{"role": "user", "content": [{"type": "video"}]}], {}),
            "string arguments": ([{"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": '{"a": 1}'}}]}], {}),
            "mismatched assistant name": ([{"role": "assistant", "name": "g", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": {}}}]}], {}),
        }
        for label, (messages, kwargs) in bad.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.encode(messages, **kwargs)
        with self.assertRaises(ValueError):
            self.encode(MESSAGES, thinking_mode="reasoning")

    def test_benchmark_registers_inkling2_with_its_effort_contract(self):
        self.assertEqual(self.bench.ENCODER_PATHS["inkling2"], ENCODER_PATH)
        self.assertEqual(self.bench.ENCODER_REASONING_EFFORTS["inkling2"], PUBLIC_EFFORTS)
        self.assertFalse(self.bench.ENCODER_EMIT_BOS_TEXT["inkling2"])
        for effort in sorted(PUBLIC_EFFORTS):
            args = parse_benchmark(self.bench, "--encoder", "inkling2", "--reasoning-effort", effort)
            self.assertEqual(args.reasoning_effort, effort)
        with self.assertRaises(SystemExit):
            parse_benchmark(self.bench, "--encoder", "inkling2", "--reasoning-effort", "off")

    def test_render_item_uses_the_inkling_template(self):
        rendered, label = self.bench.render_item(
            "gsm8k", {"question": "1+1?"}, self.encoder, "thinking", "inkling2", "low",
        )
        self.assertTrue(rendered.startswith(effort_line("0.2") + f"{USER}{TEXT}1+1?"))
        self.assertTrue(rendered.endswith(f"Answer: <number>{END}"))
        self.assertEqual(label, "official-encoder-inkling2-thinking-effort-low")

    def test_qualification_kit_maps_the_model_to_the_encoder(self):
        kit = load_module("qualify_profile_inkling2_test", KIT_PATH)
        self.assertEqual(kit.MODEL_ENCODERS["inkling-2-mini"], "inkling2")


def _hf_jinja():
    """jinja2 shim whose Environment() mirrors transformers' chat renderer."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def environment():
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        return env

    return types.SimpleNamespace(Environment=environment), jinja2


def _neutralize_generation_extension(source: str) -> str:
    # Hugging Face's AssistantTracker only records assistant spans; it emits
    # no text, so an always-true block renders the same bytes.
    source = source.replace("{%- generation -%}", "{%- if true -%}")
    return source.replace("{%- endgeneration -%}", "{%- endif -%}")


class Inkling2TemplateFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.jinja2, cls.real_jinja2 = _hf_jinja()
        except ImportError:
            raise unittest.SkipTest("jinja2 is not installed")
        if not TEMPLATE_PATH.is_file():
            raise unittest.SkipTest(f"Inkling 2 chat template is unavailable: {TEMPLATE_PATH}")
        cls.encoder = load_module("encoding_inkling2_fidelity", ENCODER_PATH)
        # The only "generation tail" is the trailing effort line of a render
        # that has no non-system message yet; the full render moves it before
        # the first non-system message.
        set_generation_tails([effort_line(level) for level in set(EFFORT_LEVELS.values())])
        cls.adapter = FidelityAdapter(
            encode=cls._encode,
            official=cls._official,
            template_path=TEMPLATE_PATH,
            preprocess=_neutralize_generation_extension,
        )
        cls.template = load_template(cls.jinja2, TEMPLATE_PATH, _neutralize_generation_extension)

    @classmethod
    def _encode(cls, case):
        return cls.encoder.encode_messages(
            list(case.messages),
            thinking_mode="thinking" if case.thinking else "chat",
            context=list(case.context) or None,
            drop_thinking=case.drop_thinking,
        )

    @classmethod
    def _official(cls, template, case):
        return template.render(
            messages=case.full_messages(),
            reasoning_effort=None if case.thinking else "none",
        )

    def official(self, messages, **kwargs):
        return self.template.render(messages=messages, **kwargs)

    def test_template_is_the_pinned_official_file(self):
        manifest = json.loads(WEIGHTS_MANIFEST.read_text(encoding="utf-8"))
        pinned = {entry["name"]: entry["sha256"] for entry in manifest["files"]}
        digest = hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, pinned["chat_template.jinja"])

    def test_standard_matrix_is_byte_exact(self):
        assert_matrix(self, self.adapter, self.jinja2)

    def test_context_split_reconstruction(self):
        assert_context_split(self, self.adapter, self.jinja2)

    def test_system_only_context_split_strips_the_trailing_effort_line(self):
        context = [{"role": "system", "content": "A."}]
        messages = [{"role": "system", "content": "B."}, {"role": "user", "content": "Q"}]
        full = self.encoder.encode_messages(context + messages, thinking_mode="thinking")
        prefix = self.encoder.encode_messages(context, thinking_mode="thinking")
        suffix = self.encoder.encode_messages(messages, thinking_mode="thinking", context=context)
        self.assertTrue(prefix.endswith(effort_line("0.9")))
        self.assertEqual(prefix[: -len(effort_line("0.9"))] + suffix, full)
        self.assertEqual(full, self.official(context + messages))

    def test_every_effort_matches_the_template(self):
        for effort in [None, *EFFORT_LEVELS, " low ", 0, 0.0, 0.5, 0.99]:
            with self.subTest(effort=effort):
                self.assertEqual(
                    self.encoder.encode_messages(MESSAGES, thinking_mode="thinking", reasoning_effort=effort),
                    self.official(MESSAGES, reasoning_effort=effort),
                )

    def test_tools_images_and_tool_turns_match_the_template(self):
        system = {"role": "system", "content": "Use tools."}
        user = {"role": "user", "content": "Weather in Zürich?"}
        cases = {
            "tools-declared": ([system, user], [WEATHER_TOOL, BARE_TOOL]),
            "tool-round-trip": (
                [system, user, TOOL_CALL_TURN, TOOL_RESULT_BY_ID, TOOL_RESULT_BY_NAME,
                 {"role": "assistant", "content": "7 degrees."}],
                [WEATHER_TOOL],
            ),
            "tool-call-with-text-and-name": (
                [user, {"role": "assistant", "name": "get_weather", "content": "Checking.",
                        "tool_calls": [{"id": "c2", "function": {"name": "get_weather", "arguments": None}}]},
                 {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "ok"},
                                                                    {"type": "input_image"}]}],
                None,
            ),
            "unmatched-tool-call-id": ([user, {"role": "tool", "tool_call_id": "nope", "content": "x"}], None),
            "assistant-empty-and-none-content": (
                [user, {"role": "assistant", "content": ""}, {"role": "assistant", "content": None},
                 {"role": "assistant", "content": "", "reasoning_content": "only thinking"}],
                None,
            ),
            "mixed-parts": (
                [{"role": "user", "content": [
                    "s", {"type": "input_text", "text": "t"}, {"text": "untyped"},
                    {"type": "text", "text": None}, {"type": "image_url", "image_url": {"url": "x"}}]}],
                None,
            ),
            "empty-user-and-late-system": (
                [{"role": "user", "content": ""}, {"role": "system", "content": "late"}], None,
            ),
            "system-only": ([system, {"role": "system", "content": ""}], None),
            "empty-tools-list": ([user], []),
        }
        for label, (messages, tools) in cases.items():
            for effort in (None, "low"):
                with self.subTest(case=label, effort=effort):
                    self.assertEqual(
                        self.encoder.encode_messages(
                            messages, thinking_mode="thinking", reasoning_effort=effort, tools=tools),
                        self.official(messages, tools=tools, reasoning_effort=effort),
                    )

    def test_encoder_and_template_reject_the_same_inputs(self):
        cases = {
            "unknown effort": ([{"role": "user", "content": "x"}], {"reasoning_effort": "turbo"}),
            "effort above range": ([{"role": "user", "content": "x"}], {"reasoning_effort": 1.5}),
            "unknown role": ([{"role": "developer", "content": "x"}], {}),
            "assistant image": ([{"role": "assistant", "content": [{"type": "image"}]}], {}),
            "audio": ([{"role": "user", "content": [{"type": "audio_url"}]}], {}),
            "video": ([{"role": "user", "content": [{"type": "video_url"}]}], {}),
            "string arguments": ([{"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": '{"a": 1}'}}]}], {}),
            "list arguments": ([{"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": [1]}}]}], {}),
            "nameless call": ([{"role": "assistant", "content": "", "tool_calls": [
                {"function": {"arguments": {}}}]}], {}),
        }
        for label, (messages, kwargs) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(self.real_jinja2.exceptions.TemplateError):
                    self.official(messages, **kwargs)
                with self.assertRaises(ValueError):
                    self.encoder.encode_messages(messages, thinking_mode="thinking", **kwargs)


if __name__ == "__main__":
    unittest.main()
