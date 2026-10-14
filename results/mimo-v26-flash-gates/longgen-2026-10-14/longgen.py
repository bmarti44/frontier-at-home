#!/usr/bin/env python3
"""Long-generation diagnostic for mimo26flash (PREREG.md here; DIAGNOSTIC).

run:     send the three fixed fixtures to one server and record the raw responses.
verdict: score every arm's record with the fixed rule and write summary.json.

Every request goes to /v1/completions, as the holdout's did. The chat fixtures
are rendered by the bench's own mimo26 encoder in thinking mode, temperature 0,
seed 42.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = "mimo-v2.6-flash"
ARMS = ("a", "b", "c")
REACH_TOKENS = 4600  # prompt + completion must pass the 4,096 boundary with margin

COUNT_RAW_PROMPT = "".join(f"{i}\n" for i in range(1, 201))
COUNT_CHAT = ("Write the integers from 1 to 3000 in increasing order, one per line, "
              "with no other text.")
ESSAY_CHAT = ("Write a very long, detailed technical essay (at least 6,000 words) explaining how a "
              "modern out-of-order CPU core executes instructions, from fetch to retirement. "
              "Use section headings.")
FIXTURES = (
    # name, kind, content, first expected integer (None: no exact check), max_tokens
    ("count-raw", "raw", COUNT_RAW_PROMPT, 201, 6144),
    ("count-chat", "chat", COUNT_CHAT, 1, 8192),
    ("essay-chat", "chat", ESSAY_CHAT, None, 8192),
)
# Degeneration: one word repeated 8+ times in a row, or one character 60+ times in a row.
WORD_RUN = re.compile(r"\b(\w+)(?:\s+\1\b){7,}")
CHAR_RUN = re.compile(r"(.)\1{59,}", re.S)


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "31_bench_accuracy.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = bench
    spec.loader.exec_module(bench)
    return bench


def render(bench, encoder, content: str) -> str:
    rendered = encoder.encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="thinking",
        reasoning_effort=None,
        add_default_bos_token=bench.ENCODER_EMIT_BOS_TEXT["mimo26"],
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("the mimo26 encoder returned an invalid prompt")
    return rendered


def post(base_url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        document = json.loads(response.read())
    return document, time.monotonic() - started


def run(args) -> int:
    bench = load_bench()
    encoder = bench.load_encoder("mimo26")
    record = {"kind": "mimo26-longgen-arm", "arm": args.arm, "base_url": args.base_url,
              "started_unix": time.time(), "fixtures": []}
    for name, kind, content, _, max_tokens in FIXTURES:
        prompt = content if kind == "raw" else render(bench, encoder, content)
        payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": 0, "seed": 42, "stream": False}
        entry = {"name": name, "kind": kind, "max_tokens": max_tokens,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            document, elapsed = post(args.base_url, payload, args.request_timeout)
            choice = document["choices"][0]
            entry.update(text=choice.get("text", ""), finish_reason=choice.get("finish_reason"),
                         usage=document.get("usage"), elapsed_s=elapsed)
        except Exception as error:  # recorded, scored as an error
            entry["error"] = f"{type(error).__name__}: {error}"
        record["fixtures"].append(entry)
        print(f"{args.arm} {name}: finish={entry.get('finish_reason')} usage={entry.get('usage')} "
              f"elapsed={entry.get('elapsed_s', 0):.0f}s error={entry.get('error')}", flush=True)
    record["ended_unix"] = time.time()
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0 if all("error" not in f for f in record["fixtures"]) else 1


def first_sequence_error(text: str, start: int) -> tuple[int | None, int]:
    """Char offset of the first line that is not the next integer, and how many lines were right.

    A last line cut by the token limit counts as right when it is a prefix of the expected one.
    """
    offset = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        expected = str(start + index)
        last = index == len(lines) - 1
        if line.strip() != expected and not (last and expected.startswith(line.strip())):
            return offset, index
        offset += len(line) + 1
    return None, len(lines)


def score_fixture(entry: dict, first_int: int | None) -> dict:
    out = {"name": entry["name"]}
    if "error" in entry:
        return {**out, "error": entry["error"], "reached": False, "broken": None}
    text = entry["text"]
    usage = entry.get("usage") or {}
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return {**out, "error": f"usage missing: {usage}", "reached": False, "broken": None}
    out.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
               finish_reason=entry.get("finish_reason"),
               decode_tok_s=completion_tokens / entry["elapsed_s"] if entry.get("elapsed_s") else None)
    out["reached"] = prompt_tokens + completion_tokens >= REACH_TOKENS
    onsets = []
    for pattern in (WORD_RUN, CHAR_RUN):
        match = pattern.search(text)
        if match:
            onsets.append(match.start())
    out["degenerate_char"] = min(onsets) if onsets else None
    body = text
    if entry["kind"] == "chat":
        cut = text.find("</think>")
        out["think_closed"] = cut >= 0
        body = text[cut + len("</think>"):].lstrip("\n") if cut >= 0 else ""
        body_offset = len(text) - len(body) if cut >= 0 else None
    else:
        body_offset = 0
    if first_int is not None:
        if body_offset is None:
            out.update(sequence_error_char=None, sequence_lines_right=0, sequence_checked=False)
        else:
            error, right = first_sequence_error(body, first_int)
            out.update(sequence_error_char=None if error is None else body_offset + error,
                       sequence_lines_right=right, sequence_checked=True)
    breaks = [c for c in (out["degenerate_char"], out.get("sequence_error_char")) if c is not None]
    out["broken"] = bool(breaks)
    if breaks:
        # Approximate position from the fixture's mean characters per token (labeled approximate).
        chars_per_token = max(len(text), 1) / completion_tokens
        out["break_char"] = min(breaks)
        out["break_abs_token_approx"] = prompt_tokens + int(min(breaks) / chars_per_token)
    return out


def verdict(args) -> int:
    expected = {name: first for name, _, _, first, _ in FIXTURES}
    arms = {}
    problems = []
    for arm in ARMS:
        path = Path(args.dir) / f"arm-{arm}.json"
        if not path.is_file():
            problems.append(f"arm {arm}: no record")
            continue
        record = json.loads(path.read_text())
        names = [f.get("name") for f in record.get("fixtures", [])]
        if record.get("arm") != arm or names != list(expected):
            problems.append(f"arm {arm}: malformed record (arm {record.get('arm')!r}, fixtures {names})")
            continue
        arms[arm] = [score_fixture(f, expected[f["name"]]) for f in record["fixtures"]]
    def informative(arm):  # fixtures that reached the boundary with no request error
        return {s["name"]: s for s in arms.get(arm, []) if s.get("reached") and s.get("broken") is not None}
    result = {"kind": "mimo26-longgen-verdict", "label": "DIAGNOSTIC", "arms": arms, "problems": problems}
    a = informative("a")
    broken_in_a = sorted(n for n, s in a.items() if s["broken"])
    result["reproduced_in_a"] = bool(broken_in_a)
    if problems:
        result["verdict"] = "NO_RESULT"
        result["reason"] = "; ".join(problems)
    elif not broken_in_a:
        result["verdict"] = "NOT_REPRODUCED"
        result["reason"] = "no fixture that reached 4,600 tokens broke in arm A"
    else:
        def clean(arm):
            arm_scores = informative(arm)
            return all(n in arm_scores and not arm_scores[n]["broken"] for n in broken_in_a)
        b_clean, c_clean = clean("b"), clean("c")
        result.update(broken_in_a=broken_in_a, b_clean=b_clean, c_clean=c_clean)
        result["verdict"] = {
            (True, False): "OUTPUT_CHUNKING",
            (False, True): "DRAFTER",
            (True, True): "CHUNKING_AND_DRAFTER",
            (False, False): "NEITHER",
        }[(b_clean, c_clean)]
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--request-timeout", type=float, default=1500)
    v = sub.add_parser("verdict")
    v.add_argument("--dir", default=str(HERE))
    v.add_argument("--out", default=str(HERE / "summary.json"))
    args = parser.parse_args()
    return run(args) if args.cmd == "run" else verdict(args)


if __name__ == "__main__":
    sys.exit(main())
