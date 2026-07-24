#!/usr/bin/env python3
"""ds4 benchmark: same workloads as dsv4-bench, wall-clock based if no timings."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8012"
label = sys.argv[1] if len(sys.argv) > 1 else "ds4"

with open("/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt") as f:
    fixture = f.read()
prefix = fixture[: int(len(fixture) * 0.46)]

code = "\n".join(
    f'''def process_record_{i}(record, options):
    """Validate and transform record batch {i}."""
    if record.get("status") != "active":
        return None
    result = {{"id": record["id"], "batch": {i}, "score": record.get("score", 0) * 1.5}}
    if options.get("verbose"):
        print(f"processed record {{record['id']}} in batch {i}")
    return result''' for i in range(20))

def chat(tag, messages, max_tokens):
    body = {"model": "d", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.load(r)
    except Exception as e:
        print(f"[{label}] {tag}: ERROR {e}")
        return
    wall = time.monotonic() - t0
    u = resp.get("usage", {})
    t = resp.get("timings") or {}
    gen = u.get("completion_tokens") or t.get("predicted_n")
    line = {"wall_s": round(wall, 1), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": gen,
            "approx_decode_tps": round(gen / wall, 2) if gen else None}
    if t:
        line["timings"] = {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in t.items()
                           if k in ("cache_n","prompt_n","prompt_per_second",
                                     "predicted_per_second","draft_n","draft_n_accepted")}
    out = resp["choices"][0]["message"].get("content") or ""
    line["text_head"] = out[:60].replace("\n", " ")
    print(f"[{label}] {tag}: {json.dumps(line)}")

chat("short prose ", [{"role": "user", "content": "Write a flowing two-paragraph essay about the history of shipbuilding."}], 256)
chat("short code  ", [{"role": "user", "content": "Write a Python class Inventory with methods add_item, remove_item, get_total, and a JSON export method. Include docstrings."}], 256)
chat("echo-edit   ", [{"role": "user", "content": "Here is a Python file:\n```python\n" + code + "\n```\nReproduce the file exactly, changing only every function's docstring to end with ' (v2)'. Output the full file."}], 1024)
chat("json-tools  ", [{"role": "user", "content": "Output a JSON array of 30 tool-call objects, each exactly like "
     '{"type":"function","name":"read_file","arguments":{"path":"/tmp/file_N.txt","encoding":"utf-8","max_bytes":65536}} '
     "with N from 1 to 30. Output only JSON."}], 1024)
chat("19K prefill ", [{"role": "system", "content": "Reference document follows.\n" + prefix},
                      {"role": "user", "content": "Summarize the document's first half in detail."}], 256)
