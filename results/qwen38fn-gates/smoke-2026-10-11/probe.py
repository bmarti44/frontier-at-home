#!/usr/bin/env python3
"""Phase-1 smoke probes (DIAGNOSTIC): greedy identity inputs and SSE granularity.

Usage: probe.py --base-url http://127.0.0.1:8017 --out <file.json>
Writes one JSON document: per-prompt greedy text + sha256, and per-stream
content-chunk vs completion-token counts (block streaming shows up as
completion_tokens / content_chunks well above 1).
"""
import argparse
import hashlib
import json
import time
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists in O(n). Code only.",
    "Explain in three sentences what a race condition is.",
    "Return a JSON object with keys name, version, deps for a tiny npm package.",
]


def post(base, body, stream=False):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def greedy(base, prompt):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 256, "seed": 0}
    with post(base, body) as r:
        doc = json.load(r)
    msg = doc["choices"][0]["message"]
    text = (msg.get("reasoning") or msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    return {"prompt": prompt, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "completion_tokens": doc["usage"]["completion_tokens"], "text": text}


def granularity(base, prompt):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 256, "stream": True,
            "stream_options": {"include_usage": True}}
    chunks, usage, stamps = 0, None, []
    with post(base, body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    chunks += 1
                    stamps.append(time.monotonic())
    toks = (usage or {}).get("completion_tokens")
    return {"prompt": prompt, "text_chunks": chunks, "completion_tokens": toks,
            "tokens_per_chunk": (toks / chunks) if toks and chunks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"greedy": [greedy(a.base_url, p) for p in PROMPTS],
           "granularity": [granularity(a.base_url, p) for p in PROMPTS]}
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"greedy": [g["sha256"][:12] for g in out["greedy"]],
                      "tokens_per_chunk": [g["tokens_per_chunk"] for g in out["granularity"]]}))


if __name__ == "__main__":
    main()
