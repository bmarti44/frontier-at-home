"""Plain-Python text renderer for the official Laguna S 2.1 chat template.

The public entry point intentionally matches ``encoding_dsv4.encode_messages``.
Laguna's template emits its EOS prefix for a complete render. Incremental
renders with ``context`` omit that already-rendered prefix. The
``add_default_bos_token`` argument is accepted for interface compatibility but
does not otherwise alter the rendered text.
"""

from typing import Any, Dict, List, Optional


EOS = "〈|EOS|〉"
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language conversations."
)
THINK_START = "<think>"
THINK_END = "</think>"


def _render_content(content: Any) -> str:
    """Match the template's text-only ``content is string`` branch."""
    return content if isinstance(content, str) else ""


def _resolve_reasoning_effort(reasoning_effort: Optional[str]) -> str:
    resolved = "max" if reasoning_effort is None else reasoning_effort
    if resolved not in ("off", "max"):
        raise ValueError(
            f"Unexpected reasoning effort {reasoning_effort}. Supported types are "
            "max (default) and off."
        )
    return resolved


def _render_messages(
    messages: List[Dict[str, Any]],
    *,
    thinking_mode: str,
    drop_thinking: bool,
    reasoning_effort: Optional[str],
    start_index: int = 0,
) -> str:
    if not messages:
        raise ValueError("No messages provided.")
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")

    effort = _resolve_reasoning_effort(reasoning_effort)
    enable_thinking = thinking_mode == "thinking" and effort == "max"

    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("No user query found in messages.")

    prompt = EOS if start_index == 0 else ""
    message_start = 0
    if start_index == 0:
        system_message = DEFAULT_SYSTEM_MESSAGE
        if messages[0].get("role") == "system":
            system_message = _render_content(messages[0].get("content"))
            message_start = 1
        has_system = bool(system_message and system_message.strip())
        if has_system or enable_thinking:
            prompt += "<system>"
            if has_system:
                prompt += system_message.rstrip()
            prompt += "</system>\n"

    for index, message in enumerate(messages):
        if index < max(start_index, message_start):
            continue
        role = message.get("role")
        content = _render_content(message.get("content"))
        if role == "user":
            prompt += f"<user>{content}</user>\n"
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                raise ValueError(
                    "Tool calls are not supported by the text-only Laguna encoder"
                )
            reasoning_value = message.get("reasoning")
            if not isinstance(reasoning_value, str):
                reasoning_value = message.get("reasoning_content")
            reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
            preserve_thinking = enable_thinking or not drop_thinking
            prompt += "<assistant>"
            if preserve_thinking:
                prompt += f"{THINK_START}{reasoning}{THINK_END}"
            else:
                prompt += THINK_END
            prompt += f"{content}</assistant>\n"
            continue
        if role == "system":
            prompt += f"<system>{content}</system>\n"
            continue
        if role == "tool":
            raise ValueError(
                "Tool messages are not supported by the text-only Laguna encoder"
            )
        raise ValueError("Unexpected message role.")

    prompt += "<assistant>"
    prompt += THINK_START if enable_thinking else THINK_END
    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode messages using the official Laguna S 2.1 text chat template.

    ``reasoning_effort`` accepts ``"max"`` (the default) and ``"off"``.
    ``thinking_mode='chat'`` also selects the non-thinking rendering. Context
    messages are treated as preceding conversation messages, matching the other
    repository encoders, and are not emitted again. The official template's EOS
    prefix is unconditional for a complete render, irrespective of
    ``add_default_bos_token``. It is omitted from an incremental continuation
    when ``context`` supplies the already-rendered prefix.
    """
    del add_default_bos_token
    context_messages = list(context or [])
    full_messages = context_messages + list(messages)
    return _render_messages(
        full_messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
        start_index=len(context_messages),
    )
