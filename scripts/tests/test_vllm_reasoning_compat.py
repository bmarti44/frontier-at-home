"""vLLM >=0.31 names the reasoning field ``reasoning``; the repository harnesses
(30_bench_speed.py and the other byte-frozen benches) read llama.cpp's
``reasoning_content``. The opt-in middleware renames the key in responses only.
"""
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))


async def run(middlewares, chunks, content_type=b"text/event-stream", length=True):
    """Drive ``middlewares`` (outermost first) around a fake engine; return (start, body)."""
    sent = []
    async def send(event):
        sent.append(event)
    async def engine(scope, recv, sender):
        headers = [(b"content-type", content_type)]
        if length:
            headers.append((b"content-length", str(sum(map(len, chunks))).encode()))
        await sender({"type": "http.response.start", "status": 200, "headers": headers})
        for i, chunk in enumerate(chunks):
            await sender({"type": "http.response.body", "body": chunk,
                          "more_body": i < len(chunks) - 1})
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    app = engine
    for cls in reversed(middlewares):
        app = cls(app)
    await app({"type": "http", "method": "POST", "path": "/v1/chat/completions"}, receive, send)
    assert not sent[-1].get("more_body", False)
    start = next(e for e in sent if e["type"] == "http.response.start")
    return start, b"".join(e["body"] for e in sent if e["type"] == "http.response.body")


def event(delta, finish=None):
    e = {"id": "x", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return b"data: " + json.dumps(e, separators=(",", ":")).encode() + b"\n\n"


def events(body):
    return [json.loads(l[5:]) for l in body.split(b"\n\n") if l.startswith(b"data: {")]


class ReasoningContentRename(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.compat = importlib.import_module("vllm_reasoning_compat")
        self.policy = importlib.import_module("glm53_runtime_policy")

    async def test_stream_reasoning_deltas_become_reasoning_content(self):
        raw = (event({"role": "assistant", "reasoning": "th"}) + event({"reasoning": "ink"})
               + event({"content": "ok"}) + event({}, finish="stop") + b"data: [DONE]\n\n")
        _, body = await run([self.compat.ReasoningContentMiddleware], [raw[:30], raw[30:90], raw[90:]])
        deltas = [e["choices"][0]["delta"] for e in events(body)]
        self.assertEqual(deltas, [{"role": "assistant", "reasoning_content": "th"},
                                  {"reasoning_content": "ink"}, {"content": "ok"}, {}])
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))

    async def test_events_without_reasoning_pass_through_byte_identically(self):
        raw = event({"content": "Hi"}) + event({}, finish="stop") + b"data: [DONE]\n\n"
        _, body = await run([self.compat.ReasoningContentMiddleware], [raw[:7], raw[7:]])
        self.assertEqual(body, raw)
        both = event({"reasoning": "a", "reasoning_content": "b"})
        _, body = await run([self.compat.ReasoningContentMiddleware], [both])
        self.assertEqual(body, both)

    async def test_inner_rename_lets_the_terminal_split_see_reasoning(self):
        # vLLM adds --middleware in order and the last one is outermost, so the
        # profile lists the rename first and the split second.
        raw = event({"reasoning": "r"}, finish="length") + b"data: [DONE]\n\n"
        _, body = await run([self.policy.SplitTerminalSseMiddleware,
                             self.compat.ReasoningContentMiddleware], [raw])
        got = [(e["choices"][0]["delta"], e["choices"][0]["finish_reason"]) for e in events(body)]
        self.assertEqual(got, [({"reasoning_content": "r"}, None), ({}, "length")])

    async def test_non_stream_message_is_renamed_and_length_corrected(self):
        doc = {"choices": [{"index": 0, "message": {"role": "assistant", "reasoning": "why",
                                                    "content": "x"}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 3}}
        blob = json.dumps(doc).encode()
        start, body = await run([self.compat.ReasoningContentMiddleware],
                                [blob[:10], blob[10:]], content_type=b"application/json")
        out = json.loads(body)
        self.assertEqual(out["choices"][0]["message"],
                         {"role": "assistant", "reasoning_content": "why", "content": "x"})
        self.assertEqual(out["usage"], doc["usage"])
        self.assertEqual(dict(start["headers"])[b"content-length"], str(len(body)).encode())

    async def test_non_chat_and_unparseable_json_pass_through(self):
        for blob in (b'{"data":[{"id":"default"}]}', b'{"choices":', b'{"choices":[{"message":{"content":"x"}}]}'):
            with self.subTest(blob=blob):
                start, body = await run([self.compat.ReasoningContentMiddleware], [blob],
                                        content_type=b"application/json")
                self.assertEqual(body, blob)
                self.assertEqual(dict(start["headers"])[b"content-length"], str(len(blob)).encode())


if __name__ == "__main__":
    unittest.main()
