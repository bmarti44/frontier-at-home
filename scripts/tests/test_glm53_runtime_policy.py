"""Request admission must reject invalid media before invoking the engine.

ASGI tests use tiny synthetic payloads, never model qualification evidence.
"""
import asyncio
import importlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


class MediaAdmission(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, payload, path="/v1/chat/completions", chunked=False):
        policy = importlib.import_module("glm53_runtime_policy")
        called, sent = [], []
        raw = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        chunks = [raw[:5], raw[5:]] if chunked else [raw]
        async def receive():
            if chunks:
                part = chunks.pop(0)
                return {"type": "http.request", "body": part, "more_body": bool(chunks)}
            return {"type": "http.disconnect"}
        async def send(event):
            sent.append(event)
        async def engine(scope, recv, sender):
            body = await recv()
            called.append(body["body"])
            await sender({"type": "http.response.start", "status": 200, "headers": []})
            await sender({"type": "http.response.body", "body": b"ok"})
        await policy.MediaPolicyMiddleware(engine)(
            {"type": "http", "method": "POST", "path": path}, receive, send)
        return called, sent[0]["status"]

    def request(self, images=0, videos=0):
        parts = [{"type": "text", "text": "Describe."}]
        parts += [{"type": "image_url", "image_url": {"url": "https://example.invalid/i"}}] * images
        parts += [{"type": "video_url", "video_url": {"url": "https://example.invalid/v"}}] * videos
        return {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": parts}]}

    async def test_valid_text_images_or_video_reaches_engine_byte_identically(self):
        for images, videos in ((0, 0), (4, 0), (0, 1)):
            request = self.request(images, videos)
            request["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
            called, status = await self.invoke(request, chunked=True)
            self.assertEqual(status, 200)
            self.assertEqual(called, [json.dumps(request).encode()])

    async def test_limits_apply_across_all_messages(self):
        for images, videos in ((5, 0), (0, 2), (1, 1)):
            called, status = await self.invoke(self.request(images, videos))
            self.assertEqual((called, status), ([], 400))
        request = self.request(4, 0)
        request["messages"] += self.request(1, 0)["messages"]
        self.assertEqual(await self.invoke(request), ([], 400))

    async def test_processor_overrides_and_multiple_completions_are_rejected(self):
        for key, value in (("mm_processor_kwargs", {"max_frames": 2048}),
                           ("media_io_kwargs", {"video": {"num_frames": 2048}}),
                           ("n", 2), ("best_of", 2)):
            request = self.request(0, 1)
            request[key] = value
            self.assertEqual(await self.invoke(request), ([], 400))

    async def test_malformed_duplicate_or_oversized_body_fails_before_engine(self):
        for body in (b'{}{}', b'{"messages": [], "messages": []}',
                     b'{"messages": [], "temperature": 1e999}'):
            self.assertEqual(await self.invoke(body), ([], 400))
        self.assertEqual(await self.invoke(b" " * (16 * 1024 * 1024 + 1)), ([], 413))

    async def test_alternative_generation_routes_cannot_bypass_media_policy(self):
        for path in ("/v1/responses", "/invocations", "/pooling", "/v1/chat/completions/"):
            self.assertEqual(await self.invoke(self.request(5), path), ([], 404))

    async def test_disguised_and_cross_type_media_never_reaches_engine(self):
        # Pinned vLLM honors direct media keys when uuid is present, even if
        # the part declares type=text. Counted type must equal parsed type.
        for kind in ("text", "refusal", "image_url", "video_url"):
            for hidden in ("image_url", "video_url", "image_embeds", "audio_embeds"):
                part = {"type": kind, "uuid": "synthetic-uuid", hidden: "https://example.invalid/media"}
                if kind in ("text", "refusal"):
                    part[kind] = "Describe."
                request = self.request(4, 0)
                request["messages"][0]["content"].append(part)
                self.assertEqual(await self.invoke(request), ([], 400))
        request = self.request()
        request["messages"][0]["content"][0]["video_url"] = "https://example.invalid/media"
        self.assertEqual(await self.invoke(request), ([], 400))


if __name__ == "__main__":
    unittest.main()
