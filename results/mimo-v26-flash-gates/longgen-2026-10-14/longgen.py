#!/usr/bin/env python3
"""Long-generation diagnostic for mimo26flash (PREREG.md here; DIAGNOSTIC).

run:     send the fixed fixtures to one server and record the raw responses,
         rewriting the record after each fixture.
verdict: score every arm's record with the fixed rule and write summary.json.

Every request goes to /v1/completions, as the holdout's did, at temperature 0,
seed 42. The chat fixtures are rendered by the bench's own mimo26 encoder in
thinking mode. The two holdout controls are re-rendered by the bench's own
render_item and must hash to the prompt the holdout sent.
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
HOLDOUT_TRANSCRIPTS = REPO / "results/mimo-v26-flash-gates/holdout-2026-10-13/accuracy/transcripts/mmlu-pro"
MODEL = "mimo-v2.6-flash"
ARMS = ("a", "b", "c")
REACH_TOKENS = 4600   # prompt + completion must pass the 4,096 boundary with margin
EARLY_TOKENS = 3500   # a break before this is reported, never attributed (review A, M2)
BEYOND_TOKENS = 500   # a clean arm must run this far past arm a's break (review A, M1)

COUNT_RAW_PROMPT = "".join(f"{i}\n" for i in range(1, 201))
COUNT_CHAT = ("Write the integers from 1 to 3000 in increasing order, one per line, "
              "with no other text.")
ESSAY_CHAT = ("Write a very long, detailed technical essay (at least 6,000 words) explaining how a "
              "modern out-of-order CPU core executes instructions, from fetch to retirement. "
              "Use section headings.")
# name, kind, content, first expected integer (None: no exact check), max_tokens, primary
FIXTURES = (
    ("count-raw", "raw", COUNT_RAW_PROMPT, 201, 6144, True),
    ("count-chat", "chat", COUNT_CHAT, 1, 6144, True),
    # Holdout MMLU-Pro answers already scored and recorded; replayed here only as
    # a reproducer of the breakdown (review A, M4). Nothing is scored for accuracy.
    ("holdout-06889", "holdout", "06889-7058", None, 5000, True),
    ("holdout-08349", "holdout", "08349-8536", None, 5000, True),
    ("essay-chat", "chat", ESSAY_CHAT, None, 6144, False),
)
# Hard degeneration signals (review A, M3): one word 5+ times in a row, one
# character 60+ times in a row, or a script an English answer never uses. CJK
# is excluded: MiMo thinks in Chinese at times. On the holdout these flag 26 of
# MiMo's 247 MMLU-Pro answers and none of Flash-Next's 247.
WORD_RUN = re.compile(r"\b(\w+)(?:\s+\1\b){4,}")
CHAR_RUN = re.compile(r"(.)\1{59,}", re.S)
STRAY_SCRIPT = re.compile(r"[ऀ-ॿ฀-๿가-힯؀-ۿ]")
FENCE = re.compile(r"^\s*```[\w-]*\s*$")


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "31_bench_accuracy.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = bench
    spec.loader.exec_module(bench)
    return bench


def render_chat(bench, encoder, content: str) -> str:
    rendered = encoder.encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="thinking",
        reasoning_effort=None,
        add_default_bos_token=bench.ENCODER_EMIT_BOS_TEXT["mimo26"],
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("the mimo26 encoder returned an invalid prompt")
    return rendered


def render_holdout(bench, encoder, rows, transcript_name: str) -> str:
    transcript = json.loads((HOLDOUT_TRANSCRIPTS / f"{transcript_name}.json").read_text())
    rendered, label = bench.render_item("mmlu-pro", rows[transcript["index"]], encoder,
                                        thinking_mode="thinking", encoder_name="mimo26", reasoning_effort=None)
    if label != transcript["rendering"] or \
            hashlib.sha256(rendered.encode()).hexdigest() != transcript["rendered_prompt_sha256"]:
        raise RuntimeError(f"{transcript_name}: the re-rendered prompt is not the one the holdout sent")
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
    rows = bench.load_jsonl("mmlu-pro", bench.load_pins()[0])
    prompts = {}
    for name, kind, content, _, _, _ in FIXTURES:  # render everything before the first request
        prompts[name] = (content if kind == "raw" else render_chat(bench, encoder, content) if kind == "chat"
                         else render_holdout(bench, encoder, rows, content))
    out = Path(args.out)
    record = {"kind": "mimo26-longgen-arm", "arm": args.arm, "base_url": args.base_url,
              "started_unix": time.time(), "fixtures": []}
    for name, kind, _, _, max_tokens, _ in FIXTURES:
        prompt = prompts[name]
        payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": 0, "seed": 42, "stream": False}
        entry = {"name": name, "kind": kind, "max_tokens": max_tokens,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            document, elapsed = post(args.base_url, payload, args.request_timeout)
            choice = document["choices"][0]
            entry.update(text=choice.get("text", ""), finish_reason=choice.get("finish_reason"),
                         usage=document.get("usage"), elapsed_s=elapsed)
        except Exception as error:  # recorded; scored inconclusive
            entry["error"] = f"{type(error).__name__}: {error}"
        record["fixtures"].append(entry)
        # A cut arm keeps every fixture it finished (review A, L1).
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"{args.arm} {name}: finish={entry.get('finish_reason')} usage={entry.get('usage')} "
              f"elapsed={entry.get('elapsed_s', 0):.0f}s error={entry.get('error')}", flush=True)
    record["ended_unix"] = time.time()
    out.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if all("error" not in f for f in record["fixtures"]) else 1


def sequence_error(body: str, start: int) -> int | None:
    """Char offset in body of the first line that is not the next integer, or None.

    A leading code fence and blank lines, a closing fence, and a last line cut by
    the token limit (a prefix of the expected one) are allowed (review A, M2).
    """
    lines = body.split("\n")
    offset = 0
    index = 0
    while index < len(lines) and (not lines[index].strip() or FENCE.match(lines[index])):
        offset += len(lines[index]) + 1
        index += 1
    expected = start
    for position in range(index, len(lines)):
        line = lines[position].strip()
        last = position == len(lines) - 1
        if line == str(expected):
            expected += 1
        elif last and str(expected).startswith(line):
            pass
        elif FENCE.match(lines[position]) or (not line and all(not rest.strip() or FENCE.match(rest)
                                                                   for rest in lines[position:])):
            return None  # the closing fence or trailing blank lines end the list
        else:
            return offset
        offset += len(lines[position]) + 1
    return None


def score_fixture(entry: dict, first_int: int | None) -> dict:
    out = {"name": entry["name"]}
    if "error" in entry:
        return {**out, "state": "inconclusive", "why": entry["error"]}
    text = entry["text"]
    usage = entry.get("usage") or {}
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return {**out, "state": "inconclusive", "why": f"usage missing: {usage}"}
    chars_per_token = max(len(text), 1) / completion_tokens
    def position(char: int) -> int:  # approximate absolute token position (labeled approximate)
        return prompt_tokens + int(char / chars_per_token)
    out.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
               end_abs_tokens=prompt_tokens + completion_tokens, finish_reason=entry.get("finish_reason"),
               decode_tok_s=completion_tokens / entry["elapsed_s"] if entry.get("elapsed_s") else None)
    signals = {}
    for label, pattern in (("word_run", WORD_RUN), ("char_run", CHAR_RUN), ("stray_script", STRAY_SCRIPT)):
        match = pattern.search(text)
        if match:
            signals[label] = match.start()
    if first_int is not None:
        cut = text.find("</think>") if entry["kind"] == "chat" else -1
        if entry["kind"] == "chat" and cut < 0:
            out["sequence"] = "no </think>"
        else:
            body_start = cut + len("</think>") if cut >= 0 else 0
            error = sequence_error(text[body_start:], first_int)
            out["sequence"] = "exact" if error is None else "error"
            if error is not None:
                signals["sequence"] = body_start + error
    out["signals_abs_approx"] = {k: position(v) for k, v in signals.items()}
    if signals:
        first = min(signals.values())
        out["break_abs_approx"] = position(first)
        out["state"] = "early" if position(first) < EARLY_TOKENS else "broken"
    elif first_int is not None and out.get("sequence") != "exact":
        out["state"] = "inconclusive"
        out["why"] = out.get("sequence")
    elif out["end_abs_tokens"] >= REACH_TOKENS:
        out["state"] = "clean"
    else:
        out["state"] = "inconclusive"
        out["why"] = f"ended at {out['end_abs_tokens']} tokens, before {REACH_TOKENS}"
    return out


def divergence(text_a: str, text_b: str) -> int | None:
    """First differing character between two arms' outputs, or None if one is a prefix of the other."""
    for index, (x, y) in enumerate(zip(text_a, text_b)):
        if x != y:
            return index
    return None


def verdict(args) -> int:
    spec = {name: (first, primary) for name, _, _, first, _, primary in FIXTURES}
    names = [name for name, *_ in FIXTURES]
    arms, records, problems = {}, {}, []
    for arm in ARMS:
        path = Path(args.dir) / f"arm-{arm}.json"
        if not path.is_file():
            problems.append(f"arm {arm}: no record")
            continue
        record = json.loads(path.read_text())
        got = [f.get("name") for f in record.get("fixtures", [])]
        if record.get("arm") != arm or got != names[:len(got)] or not got:
            problems.append(f"arm {arm}: malformed record (arm {record.get('arm')!r}, fixtures {got})")
            continue
        records[arm] = {f["name"]: f for f in record["fixtures"]}
        arms[arm] = {f["name"]: score_fixture(f, spec[f["name"]][0]) for f in record["fixtures"]}
    result = {"kind": "mimo26-longgen-verdict", "label": "DIAGNOSTIC", "arms": arms, "problems": problems}
    # Arms a and b share kernels and the checkpoint grid; they should match until a requeue (review A, L2).
    if "a" in records and "b" in records:
        result["a_b_divergence"] = {}
        for name in names:
            fa, fb = records["a"].get(name), records["b"].get(name)
            if fa and fb and "text" in fa and "text" in fb:
                char = divergence(fa["text"], fb["text"])
                score = arms["a"][name]
                approx = None
                if char is not None and score.get("completion_tokens"):
                    approx = score["prompt_tokens"] + int(char / (max(len(fa["text"]), 1) / score["completion_tokens"]))
                result["a_b_divergence"][name] = {"char": char, "abs_token_approx": approx}
    if problems and "a" not in arms:
        result.update(verdict="NO_RESULT", reason="; ".join(problems))
        return finish(result, args.out)
    broken_in_a = sorted(n for n, s in arms["a"].items() if spec[n][1] and s["state"] == "broken")
    result["broken_in_a"] = broken_in_a
    result["reproduced_in_a"] = bool(broken_in_a)
    if problems:
        result.update(verdict="NO_RESULT", reason="; ".join(problems))
    elif not broken_in_a:
        result.update(verdict="NOT_REPRODUCED", reason="no primary fixture broke at or after "
                      f"{EARLY_TOKENS} tokens in arm a")
    else:
        def arm_state(arm: str) -> str:
            states = []
            for name in broken_in_a:
                s = arms[arm].get(name, {"state": "inconclusive"})
                need = arms["a"][name]["break_abs_approx"] + BEYOND_TOKENS
                if s["state"] == "broken":
                    states.append("broken")
                elif s["state"] == "clean" and s["end_abs_tokens"] >= need:
                    states.append("clean")
                else:
                    states.append("inconclusive")
            if "broken" in states:
                return "broken"
            return "clean" if all(x == "clean" for x in states) else "inconclusive"
        b, c = arm_state("b"), arm_state("c")
        result.update(b_state=b, c_state=c)
        if "inconclusive" in (b, c):
            result.update(verdict="INCONCLUSIVE", reason=f"arm b {b}, arm c {c}")
        else:
            result["verdict"] = {
                ("clean", "broken"): "OUTPUT_CHUNKING",
                ("broken", "clean"): "DRAFTER",
                ("clean", "clean"): "CHUNKING_AND_DRAFTER",
                ("broken", "broken"): "NEITHER",
            }[(b, c)]
    return finish(result, args.out)


def finish(result: dict, out: str) -> int:
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("arms",)}, indent=1))
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
