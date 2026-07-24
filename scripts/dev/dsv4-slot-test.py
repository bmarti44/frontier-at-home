#!/usr/bin/env python3
"""Slot save/restore round-trip verification."""
import json, time, urllib.request

BASE = "http://127.0.0.1:8011"

with open("/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt") as f:
    fixture = f.read()
prefix = fixture[: int(len(fixture) * 0.46)]

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return time.monotonic() - t0, resp

def ask(question):
    wall, resp = post("/v1/chat/completions", {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Reference document follows.\n" + prefix},
            {"role": "user", "content": question},
        ],
        "max_tokens": 16, "temperature": 0, "stream": False,
    })
    t = resp.get("timings", {})
    return {"wall_s": round(wall, 2), "cache_n": t.get("cache_n"),
            "prompt_n": t.get("prompt_n"),
            "prefill_tok_s": round(t.get("prompt_per_second", 0), 1)}

print("1. prefill prefix:      ", json.dumps(ask("Say READY.")))
wall, resp = post("/slots/0?action=save", {"filename": "agent-prefix.bin"})
print(f"2. save slot:            wall={wall:.2f}s", json.dumps(resp))
print("3. clobber with new pfx:", json.dumps(ask("Ignore prior context. 2+2?")) if False else "skipped-inline")
# clobber: genuinely different prompt so slot KV is overwritten
wall2, resp2 = post("/v1/chat/completions", {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Unrelated: name three colors." * 50}],
    "max_tokens": 8, "temperature": 0, "stream": False})
t2 = resp2.get("timings", {})
print(f"3. clobber slot:         wall={wall2:.2f}s cache_n={t2.get('cache_n')} prompt_n={t2.get('prompt_n')}")
wall3, resp3 = post("/slots/0?action=restore", {"filename": "agent-prefix.bin"})
print(f"4. restore slot:         wall={wall3:.2f}s", json.dumps(resp3))
print("5. repeat original ask: ", json.dumps(ask("Say READY.")))
