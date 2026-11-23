"""Plain-Python text renderer for the official Inkling 2 chat template.

The public entry point intentionally matches ``encoding_dsv4.encode_messages``
and adds an optional ``tools`` list. It reproduces ``chat_template.jinja``
shipped with thinkingmachines/Inkling-2-Mini (revision b85c3cfa) byte for
byte, as transformers and vLLM render it.

Template contract:

- Every message is one or more ``<|message_<role>|>[name]<|content_*|>...
  <|end_message|>`` blocks; assistant messages end with
  ``<|content_model_end_sampling|>`` (the EOS token).
- One ``Thinking effort level: <n>`` system line is emitted before the first
  non-system message, or after the last message when there is none. The
  levels are none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9 (the
  default), xhigh 0.99, and max 0.99; a number in [0, 0.99] is also accepted.
- Tools are declared once at the very start as
  ``<|message_system|>tool_declare<|content_xml|>`` + sorted compact JSON.
- There is no generation prompt and no BOS text: the prompt ends after the
  last message (or the effort line) and the model writes
  ``<|message_model|>`` itself. ``add_default_bos_token`` is accepted for
  interface compatibility and changes nothing.
- ``reasoning_content`` is always rendered; the template has no
  clear-thinking switch, so ``drop_thinking`` changes nothing. A
  ``reasoning`` key is not read.

Thinking contract: ``thinking_mode='thinking'`` renders ``reasoning_effort``
(default high, 0.9). ``thinking_mode='chat'`` renders the template's ``none``
level (0) and, like the DSV4, Qwen and Laguna encoders, ignores the effort;
an unknown effort is still rejected.

Context: messages in ``context`` were already rendered and are not
re-emitted (nor are the tool declarations). A context with no non-system
message ends with the trailing effort line, which the full render moves in
front of the first non-system message; strip it like a generation prompt
before appending the suffix.

Inputs the template rejects (unknown roles or efforts, images outside user
and tool messages, audio, video, tool-call arguments that are not an object)
raise ``ValueError``.
"""

import json
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional


ROLE_TOKENS = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}
MODEL = ROLE_TOKENS["assistant"]
SYSTEM = ROLE_TOKENS["system"]
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_IMAGE = "<|content_image|><|unused_200054|>"
CONTENT_XML = "<|content_xml|>"
INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

EFFORT_LEVELS = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}
DEFAULT_EFFORT_LEVEL = 0.9
TEXT_PART_TYPES = ("text", "input_text")
IMAGE_PART_TYPES = ("image", "input_image", "image_url")
AUDIO_PART_TYPES = ("audio", "input_audio", "audio_url")


def _effort_level(reasoning_effort: Any) -> float:
    """The template's ``emit_thinking_effort`` resolution."""
    if reasoning_effort is None:
        return DEFAULT_EFFORT_LEVEL
    if isinstance(reasoning_effort, str):
        key = reasoning_effort.strip()
        if key not in EFFORT_LEVELS:
            raise ValueError(
                f"Unknown reasoning_effort: {reasoning_effort}. Supported names are "
                + ", ".join(EFFORT_LEVELS)
                + " (default high)."
            )
        return EFFORT_LEVELS[key]
    if isinstance(reasoning_effort, bool) or not isinstance(reasoning_effort, (int, float)):
        raise ValueError(f"Unknown reasoning_effort: {reasoning_effort!r}")
    level = float(reasoning_effort)
    if math.isnan(level) or level < 0.0 or level > 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return level


def _effort_line(level: float) -> str:
    rendered = "0" if level == 0.0 else str(level)
    return f"{SYSTEM}{CONTENT_TEXT}Thinking effort level: {rendered}{END_MESSAGE}"


def _tojson(value: Any, **kwargs: Any) -> str:
    """transformers' ``tojson`` filter: no HTML escaping, no ASCII escaping."""
    return json.dumps(value, ensure_ascii=False, **kwargs)


def _tool_declarations(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""
    specs = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError("Each tool must be an object.")
        function = tool["function"] if "function" in tool else tool
        if not isinstance(function, Mapping) or "name" not in function:
            raise ValueError("Each tool needs a function name.")
        specs.append({
            "description": function.get("description") or "",
            "name": function["name"],
            "parameters": function.get("parameters") or {},
            "type": tool.get("type") or "function",
        })
    declared = _tojson(specs, sort_keys=True, separators=(",", ":"))
    return f"{SYSTEM}tool_declare{CONTENT_XML}{declared}{END_MESSAGE}"


def _header(message: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    role = message["role"]
    header = ROLE_TOKENS[role]
    if role == "tool":
        if message.get("name"):
            header += str(message["name"])
        elif message.get("tool_call_id"):
            # Every matching call in the whole conversation appends its name.
            for previous in messages:
                if previous.get("role") != "assistant" or not previous.get("tool_calls"):
                    continue
                for call in previous["tool_calls"]:
                    if not isinstance(call, Mapping) or "id" not in call:
                        continue
                    if call["id"] != message["tool_call_id"]:
                        continue
                    if "function" not in call:
                        raise ValueError("Tool call has no function object.")
                    function = call["function"]
                    if isinstance(function, Mapping) and "name" in function:
                        header += str(function["name"])
    elif role == "assistant" and message.get("name"):
        header += str(message["name"])
    return header


def _render_parts(role: str, header: str, content: Any) -> str:
    if not isinstance(content, (list, tuple)):
        raise ValueError("Unexpected content type.")
    rendered = ""
    for part in content:
        if isinstance(part, str):
            rendered += f"{header}{CONTENT_TEXT}{part}{END_MESSAGE}"
            continue
        if not isinstance(part, Mapping):
            raise ValueError("Unexpected item type in content.")
        if "type" not in part or part["type"] in TEXT_PART_TYPES:
            text = part.get("text")
            text = text if isinstance(text, str) else ""
            rendered += f"{header}{CONTENT_TEXT}{text}{END_MESSAGE}"
        elif part["type"] in IMAGE_PART_TYPES:
            if role not in ("user", "tool"):
                raise ValueError("Image content is only supported for user and tool messages")
            rendered += f"{header}{CONTENT_IMAGE}{END_MESSAGE}"
        elif part["type"] in AUDIO_PART_TYPES:
            raise ValueError("Inkling 2 does not accept audio input")
        else:
            raise ValueError(f"Unsupported content part type: {part['type']}")
    return rendered


def _render_tool_calls(message: Dict[str, Any]) -> str:
    rendered = ""
    for call in message["tool_calls"]:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError("Tool call has no function object.")
        name = function.get("name")
        if not isinstance(name, str):
            raise ValueError("tool call function name must be a string")
        if message.get("name") and message["name"] != name:
            raise ValueError(f"assistant name {message['name']} does not match its tool call {name}")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            raise ValueError(
                "tool call arguments must be a parsed object, not a JSON string; canonicalize upstream"
            )
        if not isinstance(arguments, Mapping):
            raise ValueError("tool call arguments must be an object")
        # Top-level keys are sorted; nested values keep their own key order.
        items = ",".join(
            _tojson(key) + ":" + _tojson(value, separators=(",", ":"))
            for key, value in sorted(arguments.items(), key=lambda item: item[0])
        )
        rendered += (
            f"{MODEL}{name}{INVOKE_TOOL_JSON}"
            + '{"name":' + _tojson(name, sort_keys=True, separators=(",", ":"))
            + ',"args":{' + items + "}}" + END_MESSAGE
        )
    return rendered


def _render_message(message: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    role = message["role"]
    header = _header(message, messages)
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    is_assistant = role == "assistant"
    rendered = ""
    if is_assistant and reasoning:
        rendered += f"{header}{CONTENT_THINKING}{reasoning}{END_MESSAGE}"
    if isinstance(content, str) and (
        content or not (is_assistant and (message.get("tool_calls") or reasoning))
    ):
        rendered += f"{header}{CONTENT_TEXT}{content}{END_MESSAGE}"
    elif content:
        rendered += _render_parts(role, header, content)
    if is_assistant and message.get("tool_calls"):
        rendered += _render_tool_calls(message)
    if is_assistant:
        rendered += END_SAMPLING
    return rendered


def _render_messages(
    messages: List[Dict[str, Any]],
    *,
    thinking_mode: str,
    reasoning_effort: Any,
    tools: Optional[List[Dict[str, Any]]],
    start_index: int = 0,
) -> str:
    if not messages:
        raise ValueError("No messages provided.")
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")
    level = _effort_level(reasoning_effort)
    if thinking_mode == "chat":
        level = EFFORT_LEVELS["none"]
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Each message must be an object.")
        if message.get("role") not in ROLE_TOKENS:
            raise ValueError(f"Unknown message role: {message.get('role')}")

    prompt = _tool_declarations(tools) if start_index == 0 else ""
    effort_emitted = False
    for index, message in enumerate(messages):
        if not effort_emitted and message["role"] != "system":
            effort_emitted = True
            if index >= start_index:
                prompt += _effort_line(level)
        rendered = _render_message(message, messages)
        if index >= start_index:
            prompt += rendered
    if not effort_emitted:
        prompt += _effort_line(level)
    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Encode messages with the official Inkling 2 chat template.

    ``thinking_mode`` is ``'thinking'`` (render ``reasoning_effort``, default
    ``high``) or ``'chat'`` (the template's ``none`` level). Efforts are the
    template's names (none, minimal, low, medium, high, xhigh, max) or a
    number in [0, 0.99]. ``tools`` are declared at the start of the prompt.
    ``drop_thinking`` and ``add_default_bos_token`` are accepted for
    interface compatibility and change nothing: the template keeps every
    ``reasoning_content`` and emits no BOS. Context messages are treated as
    preceding conversation messages and are not re-emitted.
    """
    del drop_thinking, add_default_bos_token
    context_messages = list(context or [])
    full_messages = context_messages + list(messages)
    return _render_messages(
        full_messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        tools=tools,
        start_index=len(context_messages),
    )
