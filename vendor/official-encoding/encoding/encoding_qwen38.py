"""Plain-Python text renderer for the official Qwen3.8-27B chat template.

The public entry point intentionally matches ``encoding_dsv4.encode_messages``.
Qwen's template has no BOS prefix, so ``add_default_bos_token`` is accepted for
interface compatibility but does not alter the rendered text.
"""

from typing import Any, Dict, List, Optional


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"

REASONING_INSTRUCTIONS_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
REASONING_INSTRUCTIONS_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration."
)


def _render_content(content: Any) -> str:
    """Render the text-only branches of the official Jinja macro."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        rendered: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Unexpected item type in content.")
            if "text" in item:
                rendered.append(str(item["text"]))
            elif any(key in item for key in ("image", "image_url", "video")) or item.get(
                "type"
            ) in ("image", "video"):
                # The benchmark is text-only; vision branches are intentionally
                # outside this encoder's supported surface.
                continue
            else:
                raise ValueError("Unexpected item type in content.")
        return "".join(rendered)
    raise ValueError("Unexpected content type.")


def _resolve_reasoning_effort(reasoning_effort: Optional[str]) -> str:
    # Preserve the DSV4 caller contract while mapping its effort vocabulary onto
    # the values accepted by the Qwen Jinja template.
    aliases = {None: "xhigh", "max": "xhigh", "high": "medium"}
    resolved = aliases.get(reasoning_effort, reasoning_effort)
    if resolved not in ("xhigh", "medium", "low"):
        raise ValueError(
            f"Unexpected reasoning effort {reasoning_effort}. Supported types are "
            "xhigh (default), medium, and low."
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

    reasoning_instructions = ""
    if thinking_mode == "thinking":
        effort = _resolve_reasoning_effort(reasoning_effort)
        if effort == "xhigh":
            reasoning_instructions = REASONING_INSTRUCTIONS_XHIGH
        elif effort == "low":
            reasoning_instructions = REASONING_INSTRUCTIONS_LOW

    last_query_index = -1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user":
            content = _render_content(message.get("content")).strip()
            if not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            ):
                last_query_index = index
                break
    if last_query_index < 0:
        raise ValueError("No user query found in messages.")

    for index, message in enumerate(messages):
        if message.get("role") == "system" and index != 0:
            raise ValueError("System message must be at the beginning.")

    prompt = ""
    first = messages[0]
    if start_index == 0 and first.get("role") == "system":
        system_content = _render_content(first.get("content")).strip()
        combined = (
            reasoning_instructions
            + ("\n\n" if reasoning_instructions and system_content else "")
            + system_content
        )
        if combined:
            prompt += f"{IM_START}system\n{combined}{IM_END}\n"
    elif start_index == 0 and reasoning_instructions:
        prompt += f"{IM_START}system\n{reasoning_instructions}{IM_END}\n"

    for index, message in enumerate(messages):
        if index < start_index:
            continue
        role = message.get("role")
        content = _render_content(message.get("content")).strip()
        if role == "system":
            continue
        if role == "user":
            prompt += f"{IM_START}user\n{content}{IM_END}\n"
            continue
        if role == "assistant":
            reasoning_content = message.get("reasoning_content")
            reasoning = reasoning_content.strip() if isinstance(reasoning_content, str) else ""
            preserve_thinking = not drop_thinking or index > last_query_index
            prompt += f"{IM_START}assistant\n"
            if preserve_thinking:
                prompt += f"{THINK_START}\n{reasoning}\n{THINK_END}\n\n"
            prompt += content
            if message.get("tool_calls"):
                raise ValueError("Tool calls are not supported by the text-only Qwen3.8 encoder")
            prompt += f"{IM_END}\n"
            continue
        if role == "tool":
            raise ValueError("Tool messages are not supported by the text-only Qwen3.8 encoder")
        raise ValueError("Unexpected message role.")

    prompt += f"{IM_START}assistant\n"
    if thinking_mode == "chat":
        prompt += f"{THINK_START}\n\n{THINK_END}\n\n"
    else:
        prompt += f"{THINK_START}\n"
    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode messages using the official Qwen3.8-27B text chat template.

    ``thinking_mode='thinking'`` corresponds to Jinja
    ``enable_thinking=true``; ``'chat'`` corresponds to false. Context messages
    are treated as preceding conversation messages. Qwen's official template
    does not emit a BOS token, irrespective of ``add_default_bos_token``.
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
