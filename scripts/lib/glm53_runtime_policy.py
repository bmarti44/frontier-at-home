"""GLM request policy, installed into the frozen runtime and selected at startup.

No model imports, token-path hooks, environment reads, or diagnostic I/O.
The processor patch additionally bounds the actual video frame sampler.
"""
import json
import math

MAX_BODY_BYTES = 16 * 1024 * 1024


def parse_body(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    def constant(value):
        raise ValueError("nonfinite JSON number")

    body = json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=constant)
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    return body


def validate_request(body, *, chat=True):
    for field in ("mm_processor_kwargs", "media_io_kwargs", "multi_modal_data",
                  "multi_modal_uuids", "prompt_embeds"):
        if body.get(field) not in (None, {}):
            raise ValueError("media processing overrides are unavailable for this profile")
    for field in ("n", "best_of"):
        if field in body and (type(body[field]) is not int or body[field] != 1):
            raise ValueError("this profile allows one completion per request")
    if not chat:
        prompt = body.get("prompt")
        if not (isinstance(prompt, str) or
                (isinstance(prompt, list) and prompt and
                 all(type(token) is int and token >= 0 for token in prompt))):
            raise ValueError("completions require one text prompt or token sequence")
        return
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    images = videos = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid message")
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("invalid message content")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("invalid content part")
            kind = part.get("type")
            # vLLM permits extra fields and switches parsing when uuid exists.
            # A closed shape binds our counted modality to its parsed modality.
            allowed = {"text": {"type", "text"},
                       "refusal": {"type", "refusal"},
                       "image_url": {"type", "image_url"},
                       "video_url": {"type", "video_url"}}
            if not isinstance(kind, str) or kind not in allowed or set(part) != allowed[kind]:
                raise ValueError("unsupported or ambiguous content part")
            if kind in ("image_url", "video_url"):
                value = part[kind]
                keys = {"url", "detail"} if kind == "image_url" else {"url"}
                if (not isinstance(value, dict) or set(value) - keys or
                        not isinstance(value.get("url"), str) or not value["url"]):
                    raise ValueError("media requires a URL object without overrides")
            if kind == "image_url":
                images += 1
            elif kind == "video_url":
                videos += 1
            elif kind not in ("text", "refusal"):
                raise ValueError("unsupported media content type")
    if images > 4 or videos > 1 or (images and videos):
        raise ValueError("each request allows up to four images OR one video")


class MediaPolicyMiddleware:
    """Reject before vLLM fetches media; replay accepted bytes unchanged."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope.get("path"), scope.get("method")
        if method == "GET" and path in ("/health", "/v1/models", "/metrics", "/version"):
            return await self.app(scope, receive, send)
        if method != "POST" or path not in ("/v1/chat/completions", "/v1/completions"):
            return await self.reject(send, 404, "endpoint unavailable for this profile")
        raw = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            if event["type"] != "http.request":
                return await self.reject(send, 400, "invalid request stream")
            raw.extend(event.get("body", b""))
            if len(raw) > MAX_BODY_BYTES:
                return await self.reject(send, 413, "request body exceeds 16 MiB")
            if not event.get("more_body", False):
                break
        try:
            validate_request(parse_body(raw), chat=path == "/v1/chat/completions")
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            return await self.reject(send, 400, str(exc))
        pending = True
        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(raw), "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)

    @staticmethod
    async def reject(send, status, message):
        payload = json.dumps({"error": {"message": message,
                             "type": "invalid_request_error", "code": status}}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": payload})


class SplitTerminalSseMiddleware:
    """Emit llama.cpp-shaped SSE terminal events from vLLM streams.

    vLLM attaches ``finish_reason`` to the last content delta. Every harness in
    this repository (30_bench_speed.py, 32_golden_tests.py, the matched GLM-5.2
    campaign that byte-freezes them) expects the llama.cpp shape: content deltas
    carry no finish_reason, and the terminal event carries an empty delta. This
    middleware splits one such vLLM event into those two events, byte-for-byte
    otherwise, so the frozen harnesses measure GLM unchanged. Non-SSE responses
    and events without both a finish_reason and a non-empty delta pass through.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        streaming = False
        buffer = bytearray()

        async def wrapped_send(event):
            nonlocal streaming
            if event["type"] == "http.response.start":
                ctype = b""
                for name, value in event.get("headers", []):
                    if name.lower() == b"content-type":
                        ctype = value
                streaming = ctype.lower().startswith(b"text/event-stream")
                return await send(event)
            if event["type"] != "http.response.body" or not streaming:
                return await send(event)
            buffer.extend(event.get("body", b""))
            more = event.get("more_body", False)
            out = bytearray()
            while True:
                cut = buffer.find(b"\n\n")
                if cut < 0:
                    break
                out.extend(split_terminal_event(bytes(buffer[:cut])) + b"\n\n")
                del buffer[:cut + 2]
            if not more:
                out.extend(buffer)
                buffer.clear()
            if out or not more:
                await send({"type": "http.response.body", "body": bytes(out), "more_body": more})

        return await self.app(scope, receive, wrapped_send)


def split_terminal_event(raw):
    """Return ``raw`` unchanged, or two events if a finish_reason carries a delta."""
    if not raw.startswith(b"data:"):
        return raw
    data = raw[5:].strip()
    if data == b"[DONE]":
        return raw
    try:
        event = json.loads(data)
    except ValueError:
        return raw
    choices = event.get("choices") if isinstance(event, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return raw
    choice = choices[0]
    delta = choice.get("delta")
    if choice.get("finish_reason") is None or not isinstance(delta, dict):
        return raw
    if not any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
        return raw
    first = dict(event)
    first.pop("usage", None)
    first["choices"] = [{**choice, "finish_reason": None}]
    second = dict(event)
    second["choices"] = [{**choice, "delta": {}}]
    encode = lambda e: b"data: " + json.dumps(e, separators=(",", ":"), ensure_ascii=False).encode()
    return encode(first) + b"\n\n" + encode(second)
