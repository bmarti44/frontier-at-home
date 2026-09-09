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
