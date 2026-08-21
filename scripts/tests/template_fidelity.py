#!/usr/bin/env python3
"""Reusable byte-exactness harness for plain-Python chat-template encoders.

Every model integrated into this repo ships a plain-Python renderer under
vendor/official-encoding/encoding/ that must reproduce the official Jinja
chat template byte-for-byte. The per-model test files kept re-inventing the
fixture matrix — and the narrow first drafts repeatedly produced false
greens (the Laguna encoder passed 9/9 of its own tests while diverging from
the template on three inputs). This module owns the standard conversation
matrix once; a model's test file supplies only two callables.

Usage in a model test file::

    from scripts.tests.template_fidelity import (
        FidelityAdapter, standard_matrix, assert_context_split,
    )

    adapter = FidelityAdapter(
        encode=lambda case: encoder.encode_messages(
            case.messages, thinking_mode=case.thinking_mode, ...),
        official=lambda template, case: template.render(
            messages=case.full_messages(), enable_thinking=..., ...),
        template_path=Path(os.environ.get("X_CHAT_TEMPLATE_PATH", DEFAULT)),
    )
    for case in standard_matrix():
        ...compare adapter.encode(case) with adapter.official(...)

The matrix deliberately includes the cases that have caught real bugs:
default/empty system messages, thinking on/off, reasoning preservation
after the last user turn, drop_thinking=False, whitespace edges, and
context-split reconstruction (prefix + suffix == full render).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class FidelityCase:
    """One conversation in the standard matrix.

    ``context`` holds messages the encoder is told were already rendered
    (incremental encoding); ``messages`` are the new messages. Templates
    always see the full conversation via :meth:`full_messages`.
    """

    case_id: str
    messages: List[Dict[str, Any]]
    context: List[Dict[str, Any]] = field(default_factory=list)
    thinking: bool = True
    drop_thinking: bool = True

    def full_messages(self) -> List[Dict[str, Any]]:
        return list(self.context) + list(self.messages)


@dataclass(frozen=True)
class FidelityAdapter:
    """Bridges one model's encoder and its official Jinja template.

    ``encode``: FidelityCase -> str, calling the plain-Python encoder with
    the case's context/thinking/drop_thinking mapped to the encoder's own
    argument names.
    ``official``: (compiled jinja template, FidelityCase) -> str, rendering
    the official template for the FULL conversation with the model's own
    template variables.
    ``template_path``: the official chat template file (usually inside the
    downloaded weights tree; point an env var at it in the test file).
    """

    encode: Callable[[FidelityCase], str]
    official: Callable[[Any, FidelityCase], str]
    template_path: Path
    # Optional source rewrite before compilation — e.g. neutralizing the
    # Hugging Face ``{%- generation -%}`` extension, which plain Jinja
    # cannot parse and which emits no text.
    preprocess: Optional[Callable[[str], str]] = None


def load_template(jinja2_module: Any, template_path: Path,
                  preprocess: Optional[Callable[[str], str]] = None) -> Any:
    """Compile the official template with a plain Jinja environment."""
    source = template_path.read_text(encoding="utf-8")
    if preprocess is not None:
        source = preprocess(source)
    return jinja2_module.Environment().from_string(source)


def standard_matrix() -> List[FidelityCase]:
    """The conversation matrix every encoder must survive byte-exactly."""
    system = {"role": "system", "content": "You are a terse coding aide."}
    user_a = {"role": "user", "content": "Write a fizzbuzz in Python."}
    user_b = {"role": "user", "content": "Now golf it."}
    assistant_plain = {"role": "assistant", "content": "print(1)"}
    assistant_reasoning = {
        "role": "assistant",
        "content": "print(2)",
        "reasoning": "the user wants shorter code",
    }
    assistant_reasoning_content = {
        "role": "assistant",
        "content": "print(3)",
        "reasoning_content": "alternate reasoning field",
    }
    cases = [
        FidelityCase("system-user-thinking", [system, user_a]),
        FidelityCase("system-user-off", [system, user_a], thinking=False),
        FidelityCase("default-system", [user_a]),
        FidelityCase("default-system-off", [user_a], thinking=False),
        FidelityCase(
            "empty-system-optout",
            [{"role": "system", "content": ""}, user_a],
            thinking=False,
        ),
        FidelityCase(
            "empty-system-thinking",
            [{"role": "system", "content": ""}, user_a],
        ),
        FidelityCase(
            "whitespace-system",
            [{"role": "system", "content": "Trailing space.  \n"}, user_a],
        ),
        FidelityCase(
            "multiturn-drop-thinking",
            [system, user_a, assistant_reasoning, user_b],
            thinking=False,
        ),
        FidelityCase(
            "multiturn-preserve-thinking",
            [system, user_a, assistant_reasoning, user_b],
            thinking=False,
            drop_thinking=False,
        ),
        FidelityCase(
            "assistant-after-last-user-off",
            [system, user_a, assistant_reasoning],
            thinking=False,
        ),
        FidelityCase(
            "reasoning-field-precedence",
            [system, user_a, assistant_reasoning_content, user_b],
            drop_thinking=False,
        ),
        FidelityCase(
            "multiturn-thinking",
            [system, user_a, assistant_plain, user_b],
        ),
    ]
    return cases


def context_split_cases() -> List[FidelityCase]:
    """Cases where ``context`` was already rendered; the encoder must emit
    only the byte suffix relative to rendering the full conversation."""
    system = {"role": "system", "content": "You are a terse coding aide."}
    user_a = {"role": "user", "content": "Write a fizzbuzz in Python."}
    assistant_plain = {"role": "assistant", "content": "print(1)"}
    user_b = {"role": "user", "content": "Now golf it."}
    return [
        FidelityCase(
            "context-split-basic",
            [user_b],
            context=[system, user_a, assistant_plain],
        ),
        FidelityCase(
            "context-split-off",
            [user_b],
            context=[system, user_a, assistant_plain],
            thinking=False,
        ),
    ]


def assert_matrix(test_case: Any, adapter: FidelityAdapter,
                  jinja2_module: Any) -> None:
    """Run the standard matrix byte-exactly. Call from a unittest method."""
    template = load_template(jinja2_module, adapter.template_path, adapter.preprocess)
    for case in standard_matrix():
        with test_case.subTest(case=case.case_id):
            test_case.assertEqual(
                adapter.encode(case),
                adapter.official(template, case),
                f"encoder diverges from official template on {case.case_id}",
            )


def assert_context_split(test_case: Any, adapter: FidelityAdapter,
                         jinja2_module: Any) -> None:
    """prefix(context alone) + suffix(incremental) must equal the full
    render for every context-split case."""
    template = load_template(jinja2_module, adapter.template_path, adapter.preprocess)
    for case in context_split_cases():
        with test_case.subTest(case=case.case_id):
            full_case = FidelityCase(
                case.case_id + "-full",
                case.full_messages(),
                thinking=case.thinking,
                drop_thinking=case.drop_thinking,
            )
            full = adapter.encode(full_case)
            prefix_case = FidelityCase(
                case.case_id + "-prefix",
                case.context,
                thinking=case.thinking,
                drop_thinking=case.drop_thinking,
            )
            suffix = adapter.encode(case)
            prefix = adapter.encode(prefix_case)
            test_case.assertTrue(
                full.startswith(_strip_generation_tail(prefix)),
                f"{case.case_id}: prefix render is not a prefix of the "
                f"full render",
            )
            test_case.assertEqual(
                _strip_generation_tail(prefix) + suffix,
                full,
                f"{case.case_id}: context-split render diverges from the "
                f"full render",
            )
            test_case.assertEqual(
                full,
                adapter.official(template, full_case),
                f"{case.case_id}: full render diverges from the official "
                f"template",
            )


# The generation prompt an encoder appends after the last message (for
# Laguna: "<assistant>" + THINK marker). Adapters whose encoders use a
# different tail should override via set_generation_tails().
_GENERATION_TAILS: List[str] = []


def set_generation_tails(tails: List[str]) -> None:
    global _GENERATION_TAILS
    _GENERATION_TAILS = list(tails)


def _strip_generation_tail(rendered: str) -> str:
    for tail in sorted(_GENERATION_TAILS, key=len, reverse=True):
        if tail and rendered.endswith(tail):
            return rendered[: -len(tail)]
    return rendered
