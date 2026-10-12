#!/usr/bin/env python3
"""Reported, not gating (PREREG "Reported, not gating"): decode on CJK output.

Candidate 10's draft (L2) can propose only token ids below 65,536, which covers
most English, code and JSON tokens but not most Chinese or Japanese ones. Run 2
measures English prose only. This streams the same greedy prompts (English,
Chinese, Japanese; thinking off) to one server and reports per language the
block-aware decode (completion tokens - K) / (last event - first event),
median of 3 reps after 1 warm-up, with the share of the output's token ids at
or above 65,536 (the output re-tokenized by the model's tokenizer). Run once
against candidate 10 (run 1) and once against production candidate 9, it gives
each engine's CJK/English decode ratio.

Usage: cjk_decode.py --base-url URL --tokenizer tokenizer.json --k 4 --out FILE
"""
import argparse
import json
import statistics
import time
import urllib.request

from tokenizers import Tokenizer

DRAFT_VOCAB = 65_536
MAX_TOKENS, REPS = 384, 3
PROMPTS = {
    "en": "Write a detailed essay of at least 800 words on the history of the Yangtze River, its geography "
          "and its influence on Chinese culture. Answer in English only.",
    "zh": "请用中文写一篇至少800字的文章，详细介绍长江的历史、地理以及它对中国文化的影响。只用中文回答。",
    "ja": "日本語で、富士山の歴史、地理、そして日本文化への影響について、少なくとも800字の詳しい文章を書いてください。"
          "日本語だけで答えてください。",
}


def stream(base: str, prompt: str) -> dict:
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": MAX_TOKENS, "min_tokens": MAX_TOKENS, "ignore_eos": True, "seed": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning") or d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    stamps.append(time.monotonic())
                    text.append(piece)
    return {"stamps": stamps, "text": "".join(text), "tokens": (usage or {}).get("completion_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--k", type=int, required=True, help="max tokens per SSE event (1 + draft length)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    out = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "k": a.k, "languages": {}}
    for lang, prompt in PROMPTS.items():
        reps = []
        for i in range(REPS + 1):
            r = stream(a.base_url, f"[{lang}-{i}-{time.time_ns()}] {prompt}")
            ids = tok.encode(r["text"], add_special_tokens=False).ids
            elapsed = r["stamps"][-1] - r["stamps"][0] if len(r["stamps"]) > 1 else 0.0
            ok = r["tokens"] == MAX_TOKENS and elapsed > 0
            reps.append({"warmup": i == 0, "completion_tokens": r["tokens"], "events": len(r["stamps"]),
                         "decode_tok_s": (r["tokens"] - a.k) / elapsed if ok else None,
                         "high_id_share": sum(t >= DRAFT_VOCAB for t in ids) / len(ids) if ids else None,
                         "text_head": r["text"][:120]})
        measured = [x["decode_tok_s"] for x in reps if not x["warmup"]]
        out["languages"][lang] = {
            "median_decode_tok_s": statistics.median(measured) if all(measured) else None,
            "median_high_id_share": statistics.median(x["high_id_share"] or 0.0 for x in reps if not x["warmup"]),
            "reps": reps}
    en = out["languages"]["en"]["median_decode_tok_s"]
    for doc in out["languages"].values():
        doc["ratio_to_en"] = doc["median_decode_tok_s"] / en if en and doc["median_decode_tok_s"] else None
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: {m: v[m] for m in ("median_decode_tok_s", "median_high_id_share", "ratio_to_en")}
                      for k, v in out["languages"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
