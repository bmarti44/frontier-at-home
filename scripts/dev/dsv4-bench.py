#!/usr/bin/env python3
"""Decode/prefill benchmark: prose + structured workloads, short and long context."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8011"
label = sys.argv[1] if len(sys.argv) > 1 else "run"

with open("/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt") as f:
    fixture = f.read()
prefix = fixture[: int(len(fixture) * 0.46)]

def chat(messages, max_tokens):
    body = {"model": "d", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    wall = time.monotonic() - t0
    t = resp.get("timings", {})
    out = resp["choices"][0]["message"].get("content") or ""
    return {"wall": round(wall, 1), "prompt_n": t.get("prompt_n"),
            "cache_n": t.get("cache_n"),
            "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
            "decode_tps": round(t.get("predicted_per_second") or 0, 2),
            "gen_n": t.get("predicted_n"),
            "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
            "text_head": out[:70].replace("\n", " ")}

PROSE = "Write a flowing two-paragraph essay about the history of shipbuilding."
CODE = ("Write a Python class Inventory with methods add_item, remove_item, "
        "get_total, and a JSON export method. Include docstrings.")

r1 = chat([{"role": "user", "content": PROSE}], 256)
print(f"[{label}] short-ctx prose : {json.dumps(r1)}")
r2 = chat([{"role": "user", "content": CODE}], 256)
print(f"[{label}] short-ctx code  : {json.dumps(r2)}")
r3 = chat([{"role": "system", "content": "Reference document follows.\n" + prefix},
           {"role": "user", "content": "Summarize the document's first half in detail."}], 256)
print(f"[{label}] 19K prefill+gen : {json.dumps(r3)}")
r4 = chat([{"role": "system", "content": "Reference document follows.\n" + prefix},
           {"role": "user", "content": CODE}], 256)
print(f"[{label}] 19K-ctx code    : {json.dumps(r4)}")
