#!/usr/bin/env python3
"""Code-shaped decode probe (DIAGNOSTIC, reported in the lever screen; gates only DSpark's code clause).

The repo speed bench times prose. DSpark drafts accept far more tokens per step
on code than on prose, so this probe streams three greedy code-writing
requests and times decode as (completion_tokens - K) / (t_last_event -
t_first_event): completion_tokens is the server's usage count (reasoning plus
content), and K, the most tokens one SSE event may carry, bounds what arrived
before the first event. A stream averaging more than K tokens per event is
invalid, as in scripts/30_bench_speed_blocks.py.

Usage: code_decode.py --base-url http://127.0.0.1:8018 --out code.json --max-tokens-per-event 8
"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "Write a complete Python module implementing an LRU cache class with get, put, "
    "delete and resize methods, type hints and docstrings, followed by unittest tests.",
    "Write a TypeScript function that parses an ISO-8601 duration string such as P3DT4H12M "
    "into total seconds, with input validation, and a set of Jest tests.",
    "Write a Rust implementation of a bounded multi-producer single-consumer ring buffer "
    "using atomics, with comments explaining the memory ordering.",
]
MIN_TOKENS = 200


def one(base, prompt, max_tokens, k, timeout):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "seed": 0, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, usage = [], None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    stamps.append(time.monotonic() - t0)
    tokens = (usage or {}).get("completion_tokens")
    row = {"completion_tokens": tokens, "events": len(stamps), "ttft_s": stamps[0] if stamps else None}
    errors = []
    if not tokens or tokens < MIN_TOKENS:
        errors.append(f"completion_tokens {tokens} < {MIN_TOKENS}")
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        errors.append("fewer than two timed events")
    elif tokens and tokens / len(stamps) > k:
        errors.append(f"{tokens / len(stamps):.2f} tokens per event > K={k}")
    row["errors"] = errors
    row["decode_tok_s"] = None if errors else (tokens - k) / (stamps[-1] - stamps[0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-tokens-per-event", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    rows = []
    for prompt in PROMPTS:
        row = one(a.base_url, prompt, a.max_tokens, a.max_tokens_per_event, a.timeout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rates = [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None]
    doc = {"method": "(completion_tokens - K) / (t_last_event - t_first_event), K = max tokens per event",
           "max_tokens_per_event": a.max_tokens_per_event, "rows": rows,
           "valid": len(rates) == len(PROMPTS),
           "median_decode_tok_s": statistics.median(rates) if len(rates) == len(PROMPTS) else None}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}))


if __name__ == "__main__":
    main()
