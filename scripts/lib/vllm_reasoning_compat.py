"""llama.cpp-shaped reasoning field for vLLM >=0.31 responses, selected at startup.

vLLM 0.31 names the reasoning field ``reasoning``; every harness in this
repository (30_bench_speed.py and the other byte-frozen benches) reads
llama.cpp's ``reasoning_content``. This middleware renames the key in chat
``delta`` and ``message`` objects of SSE and JSON responses, byte-for-byte
otherwise. List it before SplitTerminalSseMiddleware: vLLM adds ``--middleware``
classes in order and the last one is outermost.

No model imports, token-path hooks, environment reads, or diagnostic I/O.
"""
import json

_KEY = b'"reasoning"'


class ReasoningContentMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        mode = None
        start = None
        buffer = bytearray()

        async def wrapped_send(event):
            nonlocal mode, start
            if event["type"] == "http.response.start":
                ctype = b""
                for name, value in event.get("headers", []):
                    if name.lower() == b"content-type":
                        ctype = value.lower()
                if ctype.startswith(b"text/event-stream"):
                    mode = "sse"
                elif ctype.startswith(b"application/json"):
                    mode, start = "json", event
                    return None
                return await send(event)
            if event["type"] != "http.response.body" or mode is None:
                return await send(event)
            buffer.extend(event.get("body", b""))
            more = event.get("more_body", False)
            if mode == "json":
                if more:
                    return None
                body = rename_document(bytes(buffer))
                buffer.clear()
                headers = [(n, str(len(body)).encode() if n.lower() == b"content-length" else v)
                           for n, v in start.get("headers", [])]
                await send({**start, "headers": headers})
                return await send({"type": "http.response.body", "body": body, "more_body": False})
            out = bytearray()
            while True:
                cut = buffer.find(b"\n\n")
                if cut < 0:
                    break
                out.extend(rename_event(bytes(buffer[:cut])) + b"\n\n")
                del buffer[:cut + 2]
            if not more:
                out.extend(buffer)
                buffer.clear()
            if out or not more:
                await send({"type": "http.response.body", "body": bytes(out), "more_body": more})

        return await self.app(scope, receive, wrapped_send)


def _encode(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()


def _rename(doc, field):
    """Rename ``reasoning`` in each choice's ``field``; return the new doc or None."""
    choices = doc.get("choices") if isinstance(doc, dict) else None
    if not isinstance(choices, list):
        return None
    changed = False
    renamed = []
    for choice in choices:
        part = choice.get(field) if isinstance(choice, dict) else None
        if isinstance(part, dict) and "reasoning" in part and "reasoning_content" not in part:
            part = {("reasoning_content" if k == "reasoning" else k): v for k, v in part.items()}
            choice = {**choice, field: part}
            changed = True
        renamed.append(choice)
    return {**doc, "choices": renamed} if changed else None


def rename_event(raw):
    """Return one SSE event with ``delta.reasoning`` renamed, else ``raw`` unchanged."""
    if not raw.startswith(b"data:") or _KEY not in raw:
        return raw
    try:
        doc = json.loads(raw[5:].strip())
    except ValueError:
        return raw
    out = _rename(doc, "delta")
    return raw if out is None else b"data: " + _encode(out)


def rename_document(raw):
    """Return a JSON body with ``message.reasoning`` renamed, else ``raw`` unchanged."""
    if _KEY not in raw:
        return raw
    try:
        doc = json.loads(raw)
    except ValueError:
        return raw
    out = _rename(doc, "message")
    return raw if out is None else _encode(out)
