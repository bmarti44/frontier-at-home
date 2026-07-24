#!/usr/bin/env python3
"""ds4 v0.4.2 gauntlet: speed, spec-burst survival, prefix cache, deep context."""
import json, time, urllib.request

BASE = "http://127.0.0.1:8022"
with open("/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt") as f:
    fixture = f.read()

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
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.load(r)
    except Exception as e:
        print(f"[{tag}] CRASH/{type(e).__name__}: {e}")
        return None
    wall = time.monotonic() - t0
    t = resp.get("timings") or {}
    out = ((resp["choices"][0]["message"].get("content") or "")[:50]).replace("\n", " ")
    keep = {k: (round(v,2) if isinstance(v,float) else v) for k,v in t.items()}
    mem = open("/proc/meminfo").readline().split()
    print(f"[{tag}] wall={wall:.1f}s timings={json.dumps(keep)} | {out}")
    return resp

P = fixture[: int(len(fixture) * 0.46)]   # ~19K tokens
DEEP = fixture[: int(len(fixture) * 0.86)] # ~35K tokens (past old 28K envelope)

chat("1 prose        ", [{"role":"user","content":"Write a flowing two-paragraph essay about the history of shipbuilding."}], 256)
chat("2 code         ", [{"role":"user","content":"Write a Python class Inventory with add_item, remove_item, get_total, and a JSON export method. Docstrings."}], 256)
chat("3 echo-edit    ", [{"role":"user","content":"Here is a Python file:\n```python\n"+code+"\n```\nReproduce the file exactly, changing only every docstring to end with ' (v2)'. Output the full file."}], 1024)
chat("4 json-killer-1", [{"role":"user","content":"Output a JSON array of 30 tool-call objects, each exactly like "
     '{"type":"function","name":"read_file","arguments":{"path":"/tmp/file_N.txt","encoding":"utf-8","max_bytes":65536}} '
     "with N from 1 to 30. Output only JSON."}], 1024)
chat("5 json-killer-2", [{"role":"user","content":"Output a JSON array of 40 objects like "
     '{"id":N,"status":"ok","payload":{"path":"/data/item_N.bin","bytes":4096,"flags":["read","cache","sync"]}} '
     "for N 1 to 40. Only JSON."}], 1024)
chat("6 19K cold     ", [{"role":"system","content":"Reference document follows.\n"+P},{"role":"user","content":"Summarize the first half in detail."}], 256)
chat("7 19K warm     ", [{"role":"system","content":"Reference document follows.\n"+P},{"role":"user","content":"Summarize the first half in detail."}], 256)
chat("8 19K new-tail ", [{"role":"system","content":"Reference document follows.\n"+P},{"role":"user","content":"What topic closes the document? One sentence."}], 256)
chat("9 35K deep     ", [{"role":"system","content":"Reference document follows.\n"+DEEP},{"role":"user","content":"Summarize the final section briefly."}], 128)
