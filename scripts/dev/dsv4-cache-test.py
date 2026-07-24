#!/usr/bin/env python3
"""Deterministic prefix-cache verification against the loopback llama.cpp engine.

Three requests:
  A: long fixed prefix + question 1  -> expect full prefill
  B: identical to A                  -> expect ~0 tokens prefilled
  C: same prefix + question 2        -> expect only the tail prefilled
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:8011"

with open("/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt") as f:
    fixture = f.read()

# fixture is ~40657 tokens for the full file; take ~46% for ~19K tokens
prefix = fixture[: int(len(fixture) * 0.46)]

def ask(question, max_tokens=32):
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Reference document follows.\n" + prefix},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    wall = time.monotonic() - t0
    usage = resp.get("usage", {})
    timings = resp.get("timings", {})
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timings": {k: round(v, 2) if isinstance(v, float) else v
                     for k, v in timings.items()} if timings else None,
    }

print("A (cold, full prefill):", json.dumps(ask("Summarize the first paragraph in one sentence.")))
print("B (identical repeat):  ", json.dumps(ask("Summarize the first paragraph in one sentence.")))
print("C (same prefix, new q):", json.dumps(ask("What is the last topic discussed? One sentence.")))
