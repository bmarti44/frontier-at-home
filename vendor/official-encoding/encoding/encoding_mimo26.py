"""Plain-Python text renderer for the official MiMo-V2.6 chat template.

The public entry point matches ``encoding_dsv4.encode_messages``. MiMo's
template (XiaomiMiMo/MiMo-V2.6-Flash-MOPD ``chat_template.jinja``; the RL
repo's differs only by a trailing newline) renders, with no separator between
turns:

* every non-assistant message as ``<|im_start|>{role}\\n{content}<|im_end|>``;
* every assistant message as
  ``<|im_start|>assistant\\n<think>{reasoning_content}</think>{content}<|im_end|>``,
  keeping each turn's reasoning (empty when absent), so ``drop_thinking`` has
  no effect;
* the generation prompt ``<|im_start|>assistant\\n``, followed by
  ``<think></think>`` only when thinking is off.

Content is emitted verbatim (no stripping). The template has no BOS and no
reasoning-effort levels. Tools, tool calls and image/audio/video parts are
outside this text-only encoder and raise.
"""

from typing import Any, Dict, List, Optional

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
MEDIA_KEYS = ("image", "image_url", "video", "video_url", "audio", "audio_url",
              "input_audio")


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        rendered: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                rendered.append(str(part))
                continue
            if part.get("type") in MEDIA_KEYS or any(key in part for key in MEDIA_KEYS):
                raise ValueError("Media parts are not supported by the text-only MiMo encoder")
            if "text" in part:
                rendered.append(str(part["text"]))
        return "".join(rendered)
    raise ValueError("Unexpected content type.")


def _render_message(message: Dict[str, Any]) -> str:
    if message.get("tools"):
        raise ValueError("Tools are not supported by the text-only MiMo encoder")
    role = message.get("role")
    content = _render_content(message.get("content"))
    if role == "assistant":
        if message.get("tool_calls"):
            raise ValueError("Tool calls are not supported by the text-only MiMo encoder")
        reasoning = message.get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) else ""
        return f"{IM_START}assistant\n{THINK_START}{reasoning}{THINK_END}{content}{IM_END}"
    if not isinstance(role, str) or not role:
        raise ValueError("Unexpected message role.")
    return f"{IM_START}{role}\n{content}{IM_END}"


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode messages with the official MiMo-V2.6 text chat template.

    ``thinking_mode='thinking'`` is the template default (``enable_thinking``
    unset or true); ``'chat'`` is ``enable_thinking=false``. Context messages
    were already rendered: only the new messages and the generation prompt are
    returned, which the template's per-message rendering makes exact.
    """
    del drop_thinking, add_default_bos_token
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")
    if reasoning_effort is not None:
        raise ValueError("MiMo-V2.6 has no reasoning-effort levels; omit reasoning_effort")
    if not messages:
        raise ValueError("No messages provided.")
    for message in list(context or []):
        _render_message(message)  # same validation as the rendered messages
    prompt = "".join(_render_message(message) for message in messages)
    prompt += f"{IM_START}assistant\n"
    if thinking_mode == "chat":
        prompt += f"{THINK_START}{THINK_END}"
    return prompt
