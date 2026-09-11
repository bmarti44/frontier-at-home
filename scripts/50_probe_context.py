#!/usr/bin/env python3
"""Engine-agnostic full-context fill gate for an OpenAI-compatible server.

Fills every slot with a prompt that tokenizes to exactly --tokens-per-slot
tokens, streams all slots concurrently, checks needle retrieval at fixed
relative positions (three present codes in order plus a literal end marker as the
negative control), samples MemAvailable while the
requests run, and writes manifest.json / raw.jsonl / summary.json to --out.

Run with an interpreter that can import ``tokenizers`` (both /usr/bin/python3
and the native-runtime venv qualify on the Spark host). Exit 0 PASS, 1 FAIL,
2 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEMINFO_PATH = Path("/proc/meminfo")
NEEDLE_FRACTIONS = (0.02, 0.20, 0.45, 0.70, 0.90, 0.98)
# Slot i asks for needle TARGET_ORDER[i % 6]; slots 0 and 1 cover 2% and 98%.
TARGET_ORDER = (0, 5, 2, 3, 1, 4)
SALT_EVERY_TOKENS = 4000
SHORT_TOKENS = 4096
ANIMALS = (
    "otter", "heron", "badger", "lynx", "falcon", "walrus", "gecko", "bison",
    "marten", "puffin", "ibex", "jackal", "koala", "ocelot", "tapir", "wombat",
)
PAD_WORDS = ("and", "the", "of", "to", "in", "on", "for", "with", "at", "by")
CODE_RE = re.compile(r"(?<!\d)\d{8}(?!\d)")
SLOT_SUMMARY_KEYS = ("slot", "http_status", "error", "finish_reason", "client_prompt_tokens", "server_prompt_tokens",
                     "prompt_token_delta", "generation_tokens", "ttft_s", "wall_s", "content", "needle_correct",
                     "controls_ok", "slot_ok")


class UsageError(Exception):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class HfTokenizer:
    """Thin wrapper so tests can substitute any object with encode(text)->list[int]."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


def load_tokenizer(path: Path, expected_sha256: str) -> Any:
    if not path.is_file():
        raise UsageError(f"tokenizer file missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise UsageError(f"tokenizer sha256 mismatch: {actual} != {expected_sha256}")
    try:
        import tokenizers  # noqa: F401,PLC0415
    except ImportError as error:
        raise UsageError(
            "the 'tokenizers' package is not importable; run this script with an "
            "interpreter that has it (e.g. /usr/bin/python3 or the native-runtime venv)"
        ) from error
    return HfTokenizer(path)


# ----------------------------------------------------------------- prompts
def build_slot_prompt(
    tokenizer: Any, paragraphs: list[str], target_tokens: int, seed: int, slot: int
) -> dict[str, Any]:
    """Deterministic prompt for one slot whose user content is exactly target_tokens."""
    if target_tokens < 512:
        raise UsageError("tokens-per-slot must be >= 512")
    rng = random.Random(f"{seed}:{slot}")
    animals = rng.sample(ANIMALS, len(NEEDLE_FRACTIONS) + 2)
    needle_animals, controls = animals[:6], animals[6:]
    codes = [f"{rng.randrange(10**7, 10**8):08d}" for _ in needle_animals]
    target_index = TARGET_ORDER[slot % len(TARGET_ORDER)]
    header = (
        f"Context probe slot {slot}. Read the records below carefully; some lines "
        "state a secret code for an animal. Answer the question at the very end.\n\n"
    )
    # Ask for three codes that all exist (every asked needle is present) plus a
    # literal end marker. Asking for an absent animal's code makes reasoning
    # models search a 250K prompt indefinitely instead of answering; the
    # negative control is instead "no code other than the three asked".
    asked_indices = [target_index, (target_index + 2) % 6, (target_index + 4) % 6]
    asked_animals = [needle_animals[i] for i in asked_indices]
    footer = (
        "\n\nQuestion: Return exactly one line containing four comma-separated fields. "
        f"The first three fields must be the 8-digit secret codes for {asked_animals[0]}, "
        f"{asked_animals[1]} and {asked_animals[2]}, in that order. The fourth field must be "
        "the literal NO_EXTRA_RECORD. Do not include labels, explanations or other text. "
        "Do not invent any other code."
    )
    fixed = len(tokenizer.encode(header)) + len(tokenizer.encode(footer))
    body_budget = target_tokens - fixed - 32  # leave room for the exact-size pad
    if body_budget < 200:
        raise UsageError("tokens-per-slot too small for header, footer and needles")
    order = list(range(len(paragraphs)))
    rng.shuffle(order)
    para_tokens = {i: len(tokenizer.encode(paragraphs[i] + "\n\n")) for i in order}
    segments: list[str] = []
    running, next_needle, next_salt, cursor = 0, 0, SALT_EVERY_TOKENS, 0
    while running < body_budget:
        if next_needle < 6 and running >= NEEDLE_FRACTIONS[next_needle] * target_tokens:
            line = f"The secret code for {needle_animals[next_needle]} is {codes[next_needle]}.\n\n"
            segments.append(line)
            running, next_needle = running + len(tokenizer.encode(line)), next_needle + 1
            continue
        if running >= next_salt:
            salt = f"slot salt {rng.getrandbits(64):016x}\n\n"
            segments.append(salt)
            running, next_salt = running + len(tokenizer.encode(salt)), next_salt + SALT_EVERY_TOKENS
            continue
        index = order[cursor % len(order)]
        cursor += 1
        segments.append(paragraphs[index] + "\n\n")
        running += para_tokens[index]
    while next_needle < 6:  # late needles (90%/98%) may fall past the body budget
        segments.append(f"The secret code for {needle_animals[next_needle]} is {codes[next_needle]}.\n\n")
        next_needle += 1
    pad: list[str] = []
    for _ in range(512):
        text = header + "".join(segments) + "pad: " + " ".join(pad) + footer
        count = len(tokenizer.encode(text))
        if count == target_tokens:
            break
        if count > target_tokens:
            if pad:
                pad.pop()
            else:  # drop the last filler segment, never a needle line
                filler = [i for i, seg in enumerate(segments) if not seg.startswith("The secret code")]
                if not filler:
                    raise UsageError("cannot trim prompt to target token count")
                segments.pop(filler[-1])
        else:
            pad.extend(rng.choice(PAD_WORDS) for _ in range(max(1, (target_tokens - count) // 2)))
    else:
        raise UsageError(f"slot {slot}: prompt did not converge to {target_tokens} tokens")
    needles = []
    for animal, code in zip(needle_animals, codes):
        line = f"The secret code for {animal} is {code}."
        char_offset = text.index(line)
        token_offset = len(tokenizer.encode(text[:char_offset]))
        needles.append({"animal": animal, "code": code, "char_offset": char_offset,
                        "token_offset": token_offset, "relative_position": token_offset / target_tokens})
    return {"slot": slot, "text": text, "client_prompt_tokens": count, "needles": needles, "controls": controls,
            "target_animal": needle_animals[target_index], "target_code": codes[target_index],
            "target_needle_index": target_index, "asked_animals": asked_animals,
            "asked_codes": [codes[i] for i in asked_indices], "asked_needle_indices": asked_indices}


END_MARKER = "NO_EXTRA_RECORD"


def score_answer(content: str, asked_codes: list[str]) -> dict[str, Any]:
    """All asked codes present in order; no other 8-digit code; the end marker present."""
    found = CODE_RE.findall(content)
    positions = [content.find(code) for code in asked_codes]
    in_order = all(p >= 0 for p in positions) and positions == sorted(positions)
    return {"codes_found": found, "needle_correct": in_order,
            "controls_ok": END_MARKER in content and set(found) <= set(asked_codes)}


# ---------------------------------------------------------------- transport
def blank_row() -> dict[str, Any]:
    return {"started_unix": time.time(), "http_status": None, "error": None, "content": "", "reasoning": "",
            "usage": None, "finish_reason": None, "token_timestamps_ns": [], "ttft_s": None, "wall_s": None}


class Client:
    def __init__(self, base_url: str, api_key: str | None, timeout_s: float, extra_body: dict) -> None:
        self.base_url, self.api_key, self.timeout_s, self.extra_body = base_url.rstrip("/"), api_key, timeout_s, extra_body

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def models(self) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + "/v1/models", headers=self.headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return {"status": response.status, "body": json.loads(response.read())}
        except Exception as error:  # noqa: BLE001
            return {"status": None, "error": str(error)}

    def stream_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body_dict = dict(self.extra_body)
        body_dict.update(payload)
        body_dict["stream"] = True
        body_dict["stream_options"] = {"include_usage": True}
        body = json.dumps(body_dict, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body,
                                         headers=self.headers(), method="POST")
        row = blank_row()
        row.update(request_sha256=hashlib.sha256(body).hexdigest(), request_bytes=len(body))
        started_ns = time.perf_counter_ns()
        try:
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout_s)
            except urllib.error.HTTPError as error:
                row["http_status"] = error.code
                row["error"] = error.read().decode("utf-8", errors="replace")[:500]
                return row
            with response:
                row["http_status"] = response.status
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if isinstance(event.get("usage"), dict):
                        row["usage"] = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        fragment = delta.get("content") or ""
                        row["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if fragment:
                            row["content"] += fragment
                            row["token_timestamps_ns"].append(time.perf_counter_ns())
                        if choice.get("finish_reason"):
                            row["finish_reason"] = choice["finish_reason"]
        except Exception as error:  # noqa: BLE001
            row["error"] = f"{type(error).__name__}: {error}"
        ended_ns = time.perf_counter_ns()
        stamps = row["token_timestamps_ns"]
        row.update(ttft_s=(stamps[0] - started_ns) / 1e9 if stamps else None,
                   wall_s=(ended_ns - started_ns) / 1e9, ended_unix=time.time())
        return row


class MemorySampler(threading.Thread):
    def __init__(self, interval_s: float = 1.0, meminfo: Path = MEMINFO_PATH) -> None:
        super().__init__(daemon=True)
        self.interval_s, self.meminfo, self.stop = interval_s, meminfo, threading.Event()
        self.samples: list[dict[str, Any]] = []

    @staticmethod
    def read_meminfo(path: Path) -> dict[str, int]:
        values = {}
        for line in path.read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0])
        return values

    def run(self) -> None:
        while True:
            try:
                info = self.read_meminfo(self.meminfo)
                self.samples.append({"unix": time.time(), "mem_available_kib": info["MemAvailable"]})
            except (OSError, KeyError, ValueError) as error:
                self.samples.append({"unix": time.time(), "error": str(error)})
            if self.stop.wait(self.interval_s):
                return

    def summary(self, floor_gib: float) -> dict[str, Any]:
        good = [s for s in self.samples if "mem_available_kib" in s]
        if not good:
            return {"samples": len(self.samples), "min_mem_available_gib": None, "memory_floor_pass": False}
        lowest = min(good, key=lambda s: s["mem_available_kib"])
        gib = [s["mem_available_kib"] / 2**20 for s in good]
        return {
            "samples": len(self.samples),
            "min_mem_available_gib": min(gib),
            "median_mem_available_gib": statistics.median(gib),
            "min_at_unix": lowest["unix"],
            "memory_floor_gib": floor_gib,
            "memory_floor_pass": min(gib) >= floor_gib,
        }


# -------------------------------------------------------------------- phases
def run_phase(
    phase: str, client: Client, tokenizer: Any, paragraphs: list[str], args: argparse.Namespace,
    tokens: int, raw: Any, health: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prompts = [build_slot_prompt(tokenizer, paragraphs, tokens, args.seed, s) for s in range(args.slots)]
    results: list[dict[str, Any] | None] = [None] * args.slots
    barrier = threading.Barrier(args.slots)

    def worker(prompt: dict[str, Any]) -> None:
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt["text"]}],
                   "temperature": 0, "seed": args.seed, "max_tokens": args.max_tokens}
        try:
            barrier.wait(timeout=120)
            row = client.stream_chat(payload)
        except Exception as error:  # noqa: BLE001
            row = blank_row()
            row["error"] = f"{type(error).__name__}: {error}"
        server_tokens = (row["usage"] or {}).get("prompt_tokens")
        row.update({
            "kind": "request", "phase": phase, "slot": prompt["slot"],
            "client_prompt_tokens": prompt["client_prompt_tokens"],
            "server_prompt_tokens": server_tokens,
            "prompt_token_delta": None if server_tokens is None else server_tokens - prompt["client_prompt_tokens"],
            "generation_tokens": (row["usage"] or {}).get("completion_tokens"),
            "target_animal": prompt["target_animal"], "target_code": prompt["target_code"],
            "asked_animals": prompt["asked_animals"], "asked_codes": prompt["asked_codes"],
            "asked_needle_indices": prompt["asked_needle_indices"], "controls": prompt["controls"],
        })
        row.update(score_answer(row["content"], prompt["asked_codes"]))
        delta = row["prompt_token_delta"]
        row["slot_ok"] = bool(
            row["http_status"] == 200 and row["error"] is None and row["finish_reason"]
            and delta is not None and 0 <= delta <= 512 and row["needle_correct"] and row["controls_ok"]
        )
        results[prompt["slot"]] = row

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in prompts]
    stop = threading.Event()

    def health_loop() -> None:
        while health is not None:
            health.append({"unix": time.time(), **client.models()})
            if stop.wait(10):
                return

    health_thread = threading.Thread(target=health_loop, daemon=True)
    for thread in (health_thread, *threads):
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    health_thread.join()
    rows = [r if r is not None else dict(blank_row(), error="worker produced no row") for r in results]
    for row, prompt in zip(rows, prompts):
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        prompt.pop("text")
    raw.flush()
    return [{"prompt": p, "result": r} for p, r in zip(prompts, rows)]


def host_facts() -> dict[str, Any]:
    uname = platform.uname()
    facts: dict[str, Any] = {"uname": uname._asdict(), "nproc": os.cpu_count()}
    try:
        facts["mem_total_kib"] = MemorySampler.read_meminfo(MEMINFO_PATH)["MemTotal"]
    except (OSError, KeyError, ValueError):
        facts["mem_total_kib"] = None
    return facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    for name in ("--base-url", "--model", "--stack-label", "--tokenizer-sha256"):
        add(name, required=True)
    add("--out", type=Path, required=True)
    add("--tokenizer-path", type=Path, required=True)
    add("--slots", type=int, default=4)
    add("--tokens-per-slot", type=int, default=250128)
    add("--fixture", type=Path, default=ROOT / "fixtures" / "ctx-32k.txt")
    add("--seed", type=int, default=42)
    add("--max-tokens", type=int, default=512,
        help="generation budget; reasoning models spend part of it thinking before the three answer lines")
    add("--request-timeout", type=float, default=3600)
    add("--api-key-file", type=Path)
    add("--mem-floor-gib", type=float, default=10)
    add("--extra-body", default="{}", help="JSON object merged into every request body")
    add("--profile-id")
    add("--phase", choices=("short", "full", "both"), default="both")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.slots < 1:
            raise UsageError("--slots must be >= 1")
        extra_body = json.loads(args.extra_body)
        if not isinstance(extra_body, dict):
            raise UsageError("--extra-body must be a JSON object")
        if not args.fixture.is_file():
            raise UsageError(f"fixture missing: {args.fixture}")
        api_key = args.api_key_file.read_text().strip() if args.api_key_file else None
        tokenizer = load_tokenizer(args.tokenizer_path, args.tokenizer_sha256)
    except (UsageError, OSError, json.JSONDecodeError) as error:
        print(f"usage error: {error}", file=sys.stderr)
        return 2
    paragraphs = [p.strip() for p in args.fixture.read_text().split("\n\n") if p.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.base_url, api_key, args.request_timeout, extra_body)
    manifest: dict[str, Any] = {
        "script_sha256": sha256_file(Path(__file__)),
        "fixture_sha256": sha256_file(args.fixture),
        "tokenizer_sha256": args.tokenizer_sha256.lower(),
        "base_url": args.base_url, "model": args.model, "stack_label": args.stack_label,
        "profile_id": args.profile_id,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "models_response": client.models(), "host": host_facts(), "start_unix": time.time(),
        "needle_fractions": NEEDLE_FRACTIONS, "phases": {},
    }
    phases: dict[str, list[dict[str, Any]]] = {}
    memory: dict[str, Any] = {}
    health: list[dict[str, Any]] = []
    with (args.out / "raw.jsonl").open("w") as raw:
        try:
            if args.phase in ("short", "both"):
                phases["short"] = run_phase("short", client, tokenizer, paragraphs, args, SHORT_TOKENS, raw)
            short_ok = all(c["result"]["slot_ok"] for c in phases.get("short", []))
            if args.phase in ("full", "both") and short_ok:
                sampler = MemorySampler()
                sampler.start()
                wall_started = time.perf_counter()
                try:
                    phases["full"] = run_phase(
                        "full", client, tokenizer, paragraphs, args, args.tokens_per_slot, raw, health
                    )
                finally:
                    sampler.stop.set()
                    sampler.join()
                memory = sampler.summary(args.mem_floor_gib)
                memory["total_wall_s"] = time.perf_counter() - wall_started
                raw.write(json.dumps({"kind": "memory_samples", "phase": "full", "samples": sampler.samples,
                                      "health": health}, allow_nan=False) + "\n")
        except UsageError as error:
            print(f"usage error: {error}", file=sys.stderr)
            return 2
    for name, cases in phases.items():
        manifest["phases"][name] = [c["prompt"] for c in cases]
    manifest["end_unix"] = time.time()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    full = [c["result"] for c in phases.get("full", [])]
    total_tokens = sum(r["server_prompt_tokens"] or 0 for r in full)
    checks: dict[str, bool] = {"short_phase_ok": short_ok}
    if args.phase != "short":
        checks |= {
        "full_phase_ran": bool(full) and len(full) == args.slots,
        "all_slots_ok": bool(full) and all(r["slot_ok"] for r in full),
        "needles_all_correct": bool(full) and all(r["needle_correct"] for r in full),
        "controls_ok": bool(full) and all(r["controls_ok"] for r in full),
        "memory_floor_pass": bool(memory.get("memory_floor_pass")),
        "total_tokens_ok": total_tokens >= args.slots * args.tokens_per_slot,
        }
    ttfts = [r["ttft_s"] for r in full if r["ttft_s"] is not None]
    summary = {
        "scope": ("short sanity round only; NOT a full-context qualification" if args.phase == "short"
                  else f"direct fill of {args.slots} slots x {args.tokens_per_slot} tokens"),
        "stack_label": args.stack_label, "model": args.model, "profile_id": args.profile_id,
        "slots": args.slots, "tokens_per_slot": args.tokens_per_slot, "total_tokens": total_tokens,
        "all_slots_ok": checks.get("all_slots_ok", False),
        "needles_correct": sum(1 for r in full if r["needle_correct"]), "needles_total": len(full),
        "controls_ok": checks.get("controls_ok", False),
        "min_mem_available_gib": memory.get("min_mem_available_gib"),
        "median_mem_available_gib": memory.get("median_mem_available_gib"),
        "min_mem_at_unix": memory.get("min_at_unix"),
        "memory_floor_gib": args.mem_floor_gib, "memory_floor_pass": checks.get("memory_floor_pass", False),
        "max_ttft_s": max(ttfts) if ttfts else None, "total_wall_s": memory.get("total_wall_s"),
        "per_slot": [{k: r[k] for k in SLOT_SUMMARY_KEYS} for r in full],
        "checks": checks,
        "formulas": {
            "total_tokens": "sum(usage.prompt_tokens over full-phase slots)",
            "slot_ok": "http_status==200 and finish_reason and 0<=server-client<=512 and needle_correct and controls_ok",
            "needle_correct": "the three asked 8-digit codes all appear in content in the asked order (regex (?<!\\d)\\d{8}(?!\\d))",
            "controls_ok": "'NO_EXTRA_RECORD' in content and every 8-digit code found is one of the asked codes",
            "memory_floor_pass": "min(MemAvailable)/2^20 >= mem_floor_gib over 1 s samples during the full phase",
            "ttft_s": "(first content fragment perf_counter - request start perf_counter)",
            "verdict": "PASS iff all checks true",
        },
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("verdict", "total_tokens", "needles_correct", "needles_total",
                                              "min_mem_available_gib", "max_ttft_s")}), flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
