#!/usr/bin/env python3
"""Cold-length prefill probe (DIAGNOSTIC).

Sends prompts of distinct, previously unseen lengths, each with a unique random
prefix so prefix caching cannot hit, and records time to the first streamed
event and the server-reported prompt tokens. On SM12x, vLLM's Inkling 2 MoE
recompiles Triton kernels for new batch shapes (vLLM PR #69288); a warmed
fixed-length benchmark hides that cost, this probe exposes it.

Usage: cold_ttft.py --base-url http://127.0.0.1:8018 --out cold.json [--words 2200,5600,...]
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu "
         "river stone cloud ember maple quartz signal vector lantern harbor meadow copper").split()


def one(base, model, words, seed, timeout):
    rng = random.Random(seed)
    body_text = " ".join(rng.choice(WORDS) for _ in range(words))
    prompt = f"[{seed:08x}-{rng.getrandbits(64):016x}] Summarize in one word:\n{body_text}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if first is None and ev.get("choices"):
                first = time.monotonic() - t0
            if ev.get("usage"):
                usage = ev["usage"]
    total = time.monotonic() - t0
    pt = (usage or {}).get("prompt_tokens")
    ttft = first if first is not None else total
    return {"seed": seed, "words": words, "prompt_tokens": pt, "ttft_s": ttft, "total_s": total,
            "prefill_tok_per_s": (pt / ttft) if pt and ttft else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", default="2213,5677,9311,14149,21017,27953")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x1A2B)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    rows = []
    for i, w in enumerate(int(x) for x in a.words.split(",")):
        row = one(a.base_url, a.model, w, a.seed + i, a.timeout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    tot_tok = sum(r["prompt_tokens"] or 0 for r in rows)
    tot_t = sum(r["ttft_s"] for r in rows)
    doc = {"rows": rows, "sum_prompt_tokens": tot_tok, "sum_ttft_s": tot_t,
           "aggregate_prefill_tok_per_s": tot_tok / tot_t if tot_t else None}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}))


if __name__ == "__main__":
    main()
