#!/usr/bin/env python3
"""Probe a profile's declared media maximum: N images at WxH and one video of F frames.

Synthetic fixtures (solid-colour images, a square moving right across F frames)
give answers that can be checked without a judge. One request carries all the
images, one carries the video, and one carries images+1 to prove the limit is
enforced (HTTP 400 from the admission middleware). Writes <out>/summary.json,
<out>/raw.jsonl and the fixtures' sha256; never edits a bundle in place.

Exit 0 on PASS, 1 on FAIL, 2 on usage/transport error.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COLOURS = [("red", (255, 0, 0)), ("green", (0, 160, 0)), ("blue", (0, 0, 255)), ("yellow", (255, 230, 0)),
           ("purple", (140, 0, 200)), ("orange", (255, 140, 0)), ("white", (255, 255, 255)), ("black", (0, 0, 0))]


def solid_png(rgb: tuple[int, int, int], size: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


def moving_square_mp4(frames: int, size: int, path: Path) -> bytes:
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (size, size))
    side = size // 4
    for i in range(frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = int((size - side) * i / max(1, frames - 1))
        frame[size // 2 - side // 2: size // 2 + side // 2, x: x + side] = 255
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def data_url(mime: str, blob: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(blob).decode()


def score_colours(text: str, expected: list[str]) -> tuple[bool, list[int]]:
    """All expected colours named, in order of first mention."""
    low = text.lower()
    positions = [low.find(c) for c in expected]
    return all(p >= 0 for p in positions) and positions == sorted(positions), positions


def score_direction(text: str) -> bool:
    low = text.lower()
    return ("right" in low) and not any(w in low for w in ("to the left", "leftward", "moves left", "moving left"))


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float) -> None:
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout

    def chat(self, content: list[dict], max_tokens: int) -> tuple[int, str, float]:
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                   "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": True}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
                return resp.status, body["choices"][0]["message"].get("content") or "", time.perf_counter() - started
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode("utf-8", "replace")[:500], time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack-label", required=True)
    ap.add_argument("--api-key-file")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--video-frames", type=int, default=16)
    ap.add_argument("--video-size", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--request-timeout", type=float, default=600)
    args = ap.parse_args()
    if args.images > len(COLOURS):
        print(f"at most {len(COLOURS)} images supported", file=sys.stderr)
        return 2
    out = Path(args.out)
    if (out / "summary.json").exists() or (out / "raw.jsonl").exists():
        print(f"refusing to overwrite evidence in {out}", file=sys.stderr)
        return 2
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    client = Client(args.base_url, args.model, key, args.request_timeout)

    fixtures: dict[str, str] = {}
    images = []
    for i in range(args.images + 1):
        name, rgb = COLOURS[i]
        blob = solid_png(rgb, args.image_size)
        (out / "fixtures" / f"image-{i}-{name}.png").write_bytes(blob)
        fixtures[f"image-{i}-{name}.png"] = hashlib.sha256(blob).hexdigest()
        images.append((name, blob))
    video = moving_square_mp4(args.video_frames, args.video_size, out / "fixtures" / "video.mp4")
    fixtures["video.mp4"] = hashlib.sha256(video).hexdigest()

    raw = out / "raw.jsonl"
    rows = []

    def run(name: str, content: list[dict], max_tokens: int):
        status, text, wall = client.chat(content, max_tokens)
        row = {"case": name, "status": status, "wall_s": round(wall, 3), "text": text}
        rows.append(row)
        with raw.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return status, text

    expected = [n for n, _ in images[: args.images]]
    content = [{"type": "image_url", "image_url": {"url": data_url("image/png", b)}} for _, b in images[: args.images]]
    content.append({"type": "text", "text": f"There are {args.images} images, each a single solid colour. "
                    "Name the colour of each image in order, one word each, comma-separated."})
    status, text = run("images-at-max", content, args.max_tokens)
    images_ok, positions = (status == 200) and score_colours(text, expected), None
    if status == 200:
        images_ok, positions = score_colours(text, expected)

    content = [{"type": "image_url", "image_url": {"url": data_url("image/png", b)}} for _, b in images]
    content.append({"type": "text", "text": "Name the colour of each image."})
    status_over, _ = run("images-over-max", content, 32)

    content = [{"type": "video_url", "video_url": {"url": data_url("video/mp4", video)}},
               {"type": "text", "text": "A white square moves across the frames. In which direction does it move: "
                "left, right, up or down? Answer with one word."}]
    status_v, text_v = run("video-at-max", content, args.max_tokens)
    video_ok = status_v == 200 and score_direction(text_v)

    checks = {"images_at_max_answered": status == 200, "images_at_max_colours_in_order": bool(images_ok),
              "images_over_max_rejected_400": status_over == 400,
              "video_at_max_answered": status_v == 200, "video_direction_right": bool(video_ok)}
    summary = {"kind": "media-max", "schema_version": 1, "stack_label": args.stack_label, "model": args.model,
               "declared": {"images": args.images, "image_size": args.image_size,
                            "video_frames": args.video_frames, "video_size": args.video_size},
               "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL",
               "colour_positions": positions, "fixtures_sha256": fixtures,
               "cases": [{k: v for k, v in r.items()} for r in rows], "unix": time.time()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"verdict": summary["verdict"], "checks": checks}))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
