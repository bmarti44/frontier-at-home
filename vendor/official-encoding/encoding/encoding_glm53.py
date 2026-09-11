"""Plain-Python text renderer for the GLM-5.3-Flash chat template.

The public entry point intentionally matches ``encoding_dsv4.encode_messages``.
The template (``chat_template.jinja`` shipped with the weights) always opens
with the literal ``[gMASK]<sop>`` and a ``Reasoning Effort`` system line, so
``add_default_bos_token=False`` omits only that leading ``[gMASK]<sop>`` for a
serving path whose tokenizer adds it itself (the served pack's tokenizer adds
no BOS, so the harness keeps the literal text).

Thinking contract: the official template always ends the generation prompt
with ``<|assistant|><think>``. ``thinking_mode='chat'`` renders the model's
own non-thinking form, ``<|assistant|><think></think>``, which is how the
template writes a prior assistant turn that carried no reasoning.

Text-only: tools, tool calls, tool messages, images, video, and audio are
outside this encoder's surface and raise.
"""

from typing import Any, Dict, List, Optional


BOS_TEXT = "[gMASK]<sop>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
THINK_START = "<think>"
THINK_END = "</think>"


def _render_content(content: Any) -> str:
    """Match the template's ``visible_text`` macro for text content."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        rendered: list[str] = []
        for item in content:
            if isinstance(item, str):
                rendered.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                rendered.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") in (
                "image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio",
            ):
                raise ValueError("Media content is not supported by the text-only GLM-5.3 encoder")
            else:
                raise ValueError("Unexpected item type in content.")
        return "".join(rendered)
    return str(content)


def _resolve_reasoning_effort(reasoning_effort: Optional[str]) -> str:
    """The template accepts ``low`` and ``high``; anything else renders ``Max``."""
    if reasoning_effort is None or reasoning_effort == "max":
        return "max"
    if reasoning_effort in ("low", "high"):
        return reasoning_effort
    raise ValueError(
        f"Unexpected reasoning effort {reasoning_effort}. Supported types are "
        "max (default), high, and low."
    )


def _split_reasoning(message: Dict[str, Any]) -> tuple[Optional[str], str]:
    """Return (reasoning_content or None, visible content) like the template."""
    content = _render_content(message.get("content"))
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content, content
    if THINK_END in content:
        reasoning = content.split(THINK_END)[0].split(THINK_START)[-1]
        return reasoning, content.split(THINK_END)[-1]
    return None, content


def _render_messages(
    messages: List[Dict[str, Any]],
    *,
    thinking_mode: str,
    drop_thinking: bool,
    reasoning_effort: Optional[str],
    start_index: int = 0,
    emit_bos_text: bool = True,
) -> str:
    if not messages:
        raise ValueError("No messages provided.")
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("No user query found in messages.")

    effort = _resolve_reasoning_effort(reasoning_effort)
    last_user_index = max(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )

    prompt = ""
    if start_index == 0:
        if emit_bos_text:
            prompt += BOS_TEXT
        prompt += f"{SYSTEM}Reasoning Effort: {effort.capitalize()}"

    for index, message in enumerate(messages):
        if index < start_index:
            continue
        role = message.get("role")
        if role == "user":
            prompt += USER + _render_content(message.get("content"))
            continue
        if role == "system":
            prompt += SYSTEM + _render_content(message.get("content"))
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                raise ValueError("Tool calls are not supported by the text-only GLM-5.3 encoder")
            reasoning, content = _split_reasoning(message)
            keep_reasoning = (not drop_thinking or index > last_user_index) and reasoning is not None
            prompt += ASSISTANT
            prompt += f"{THINK_START}{reasoning}{THINK_END}" if keep_reasoning else f"{THINK_START}{THINK_END}"
            if content.strip():
                prompt += content.strip()
            continue
        if role == "tool":
            raise ValueError("Tool messages are not supported by the text-only GLM-5.3 encoder")
        raise ValueError("Unexpected message role.")

    prompt += ASSISTANT + THINK_START
    if thinking_mode == "chat":
        prompt += THINK_END
    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode messages using the GLM-5.3-Flash text chat template.

    ``thinking_mode='thinking'`` leaves the template's ``<think>`` open for
    the model to reason; ``'chat'`` closes it immediately (the template's own
    non-thinking assistant form). ``drop_thinking`` is the template's
    ``clear_thinking``: prior assistant reasoning is dropped up to the last
    user turn. ``reasoning_effort`` is ``low``, ``high``, or ``max``
    (default), rendered into the leading ``Reasoning Effort`` system line.
    Context messages are treated as preceding conversation messages and are
    not re-emitted.
    """
    context_messages = list(context or [])
    full_messages = context_messages + list(messages)
    return _render_messages(
        full_messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
        start_index=len(context_messages),
        emit_bos_text=add_default_bos_token,
    )
