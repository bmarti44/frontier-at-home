#!/usr/bin/env python3
"""Test ngram speculation on context-echoing workloads (the agentic case)."""
import json, time, urllib.request

BASE = "http://127.0.0.1:8011"

code = "\n".join(
    f'''def process_record_{i}(record, options):
    """Validate and transform record batch {i}."""
    if record.get("status") != "active":
        return None
    result = {{"id": record["id"], "batch": {i}, "score": record.get("score", 0) * 1.5}}
    if options.get("verbose"):
        print(f"processed record {{record['id']}} in batch {i}")
    return result''' for i in range(20))

def run(label, messages, max_tokens):
    body = {"model": "d", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    t = resp.get("timings", {})
    print(f"[{label}] wall={time.monotonic()-t0:.1f}s decode_tps={t.get('predicted_per_second'):.2f} "
          f"gen_n={t.get('predicted_n')} draft_n={t.get('draft_n')} draft_acc={t.get('draft_n_accepted')}")

run("echo-edit", [
    {"role": "user", "content": "Here is a Python file:\n```python\n" + code +
     "\n```\nReproduce the file exactly, changing only every function's docstring to end with ' (v2)'. Output the full file."}], 1024)
run("json-tools", [
    {"role": "user", "content": "Output a JSON array of 30 tool-call objects, each exactly like "
     '{"type":"function","name":"read_file","arguments":{"path":"/tmp/file_N.txt","encoding":"utf-8","max_bytes":65536}} '
     "with N from 1 to 30. Output only JSON."}], 1024)
