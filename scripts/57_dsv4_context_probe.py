#!/usr/bin/env python3
"""Send one exact-size, multi-position retrieval probe to a DSV4 server."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "vendor" / "official-encoding" / "tokenizer.json"
TOKENIZER_SHA256 = (
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8"
    "392edb4e894fd10cc0d70b4957c8633cf"
)
FILLER_PATH = ROOT / "fixtures" / "ctx-32k.txt"
RECORD_PATTERN = re.compile(r"RECORD_[A-Z]+_[0-9a-z]+")
ENGINE_PROGRESS_PATTERN = re.compile(
    r"task\s+(?P<task>[0-9]+)\s+\|\s+prompt processing,\s+"
    r"n_tokens\s*=\s*(?P<tokens>[0-9]+)"
)


def load_tokenizer() -> Any:
    from tokenizers import Tokenizer

    raw = TOKENIZER_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != TOKENIZER_SHA256:
        raise RuntimeError(
            f"tokenizer hash mismatch: expected={TOKENIZER_SHA256}, actual={actual}"
        )
    return Tokenizer.from_str(raw.decode("utf-8", errors="strict"))


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def exact_prefix(tokenizer: Any, text: str, target: int) -> str:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) < target:
        raise RuntimeError(f"fixture has {len(encoding.ids)} tokens; need {target}")
    end = encoding.offsets[target - 1][1]
    result = text[:end]
    actual = token_count(tokenizer, result)
    if actual != target:
        raise RuntimeError(f"exact prefix has {actual} tokens; expected {target}")
    return result


def derive(seed_sha256: str, label: str) -> str:
    return hashlib.sha256(
        f"dsv4-context.v1:{seed_sha256}:{label}".encode()
    ).hexdigest()


def build_fixture(tokenizer: Any, target: int, seed_sha256: str) -> dict[str, Any]:
    if target < 1024:
        raise ValueError("target must be at least 1024 tokens")
    if not re.fullmatch(r"[0-9a-f]{64}", seed_sha256):
        raise ValueError("seed must be a lowercase SHA-256 digest")
    filler = FILLER_PATH.read_text(encoding="utf-8")
    filler_tokens = token_count(tokenizer, filler)
    repetitions = target // filler_tokens + 2
    base = (filler + "\n\n") * repetitions
    encoding = tokenizer.encode(base, add_special_tokens=False)
    labels = ("ALPHA", "BRAVO", "CHARLIE")
    fractions = (0.12, 0.50, 0.82)
    inserts: list[tuple[int, str, str, str]] = []
    for index, (label, fraction) in enumerate(zip(labels, fractions)):
        token_index = int(target * fraction)
        char_index = encoding.offsets[token_index][0]
        value = f"RECORD_{label}_{derive(seed_sha256, f'record:{index}')[:16]}"
        line = f"\n\nAUDIT RECORD {index + 1}: {value}\n\n"
        inserts.append((char_index, line, f"needle-{index}", value))
    text = base
    for char_index, line, _, _ in reversed(inserts):
        text = text[:char_index] + line + text[char_index:]
    text = exact_prefix(tokenizer, text, target)

    records = []
    for _, _, case_id, value in inserts:
        marker_at = text.find(value)
        if marker_at < 0:
            raise RuntimeError(f"fixture lost {case_id}")
        position = token_count(tokenizer, text[:marker_at])
        records.append(
            {
                "case_id": case_id,
                "position": position,
                "value": value,
                "expected_sha256": hashlib.sha256(value.encode()).hexdigest(),
            }
        )
    positions = [record["position"] for record in records]
    if not (
        positions[0] <= target // 4
        and target // 4 < positions[1] < 3 * target // 4
        and positions[2] >= 3 * target // 4
    ):
        raise RuntimeError(f"retrieval positions lack required coverage: {positions}")
    return {
        "text": text,
        "records": records,
        "fixture_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def validate_retrieval(output: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [record["value"] for record in records]
    observed = RECORD_PATTERN.findall(output)
    checks = {
        "all_expected_once": all(output.count(value) == 1 for value in expected),
        "ordered": observed == expected,
        "negative_control": "NO_EXTRA_RECORD" in output,
        "no_unexpected_record": set(observed) == set(expected),
    }
    return {
        "checks": checks,
        "observed": observed,
        "pass": all(checks.values()),
    }


def validate_completion(
    *,
    content: str,
    reasoning_content: str,
    finish_reason: str | None,
    done: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score only the user-visible final answer and reject truncation."""
    retrieval = validate_retrieval(content, records)
    checks = {
        "stream_complete": done is True,
        "finish_reason_stop": finish_reason == "stop",
        "final_content_present": bool(content),
        "retrieval_in_final_content": retrieval["pass"],
    }
    return {
        "checks": checks,
        "retrieval": retrieval,
        "reasoning_content_sha256": hashlib.sha256(
            reasoning_content.encode()
        ).hexdigest(),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "pass": all(checks.values()),
    }


def parse_engine_progress(log: str, *, task_id: int | None = None) -> dict[str, Any]:
    """Derive evaluated tokens from process-linked server progress events."""
    rows = [
        (int(match.group("task")), int(match.group("tokens")))
        for match in ENGINE_PROGRESS_PATTERN.finditer(log)
    ]
    if task_id is not None:
        rows = [row for row in rows if row[0] == task_id]
    if not rows:
        raise RuntimeError("engine progress evidence is missing")
    task_ids = {row[0] for row in rows}
    if len(task_ids) != 1:
        raise RuntimeError("engine progress evidence contains multiple tasks")
    counts = [row[1] for row in rows]
    if any(right <= left for left, right in zip(counts, counts[1:])):
        raise RuntimeError("engine progress token counts are not increasing")
    return {
        "task_id": next(iter(task_ids)),
        "evaluated_tokens": counts[-1],
        "progress_events": len(counts),
        "first_evaluated_tokens": counts[0],
    }


def require_token_count_agreement(
    *, requested_tokens: int, usage_tokens: int, engine_tokens: int
) -> None:
    """Fail unless independent engine progress reaches the requested target."""
    if engine_tokens < requested_tokens:
        raise RuntimeError(
            f"engine progress did not reach target: {engine_tokens} < "
            f"{requested_tokens}"
        )
    if usage_tokens < requested_tokens:
        raise RuntimeError(
            f"server usage did not reach target: {usage_tokens} < "
            f"{requested_tokens}"
        )
    # Chat templating adds a small suffix/prefix, but independent progress and
    # response usage must still describe the same request.
    if abs(engine_tokens - usage_tokens) > 64:
        raise RuntimeError(
            "engine progress and usage token counts disagree: "
            f"{engine_tokens} != {usage_tokens}"
        )
def stream_completion(
    base_url: str, payload: dict[str, Any], api_key: str | None
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.time()
    monotonic_started = time.monotonic()
    content: list[str] = []
    reasoning_content: list[str] = []
    timestamps: list[float] = []
    usage = None
    finish_reason = None
    response_ids: set[str] = set()
    done = False
    try:
        response = urllib.request.urlopen(request, timeout=3600)
    except urllib.error.HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    with response:
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise RuntimeError(f"malformed SSE line: {line[:200]!r}")
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            response_id = event.get("id")
            if isinstance(response_id, str) and response_id:
                response_ids.add(response_id)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                reason = choice.get("finish_reason")
                if reason is not None:
                    finish_reason = reason
                delta = choice.get("delta", {})
                for field, destination in (
                    ("reasoning_content", reasoning_content),
                    ("content", content),
                ):
                    fragment = delta.get(field)
                    if fragment:
                        if not isinstance(fragment, str):
                            raise RuntimeError(f"{field} fragment is not text")
                        destination.append(fragment)
                        timestamps.append(time.time())
    return {
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "started_at_seconds": started,
        "finished_at_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - monotonic_started,
        "response_ids": sorted(response_ids),
        "done": done,
        "finish_reason": finish_reason,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning_content),
        "event_timestamps": timestamps,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--context-cap", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "context_cap": args.context_cap,
        "target_tokens": args.target_tokens,
        "seed_sha256": args.seed_sha256,
        "pass": False,
    }
    try:
        if not 1 <= args.target_tokens < args.context_cap:
            raise ValueError("target tokens must be positive and below context cap")
        tokenizer = load_tokenizer()
        fixture = build_fixture(tokenizer, args.target_tokens, args.seed_sha256)
        values = [record["value"] for record in fixture["records"]]
        prompt = (
            fixture["text"]
            + "\n\nReturn the values from the three numbered AUDIT RECORD lines "
            "in numeric order, separated by commas. Then append the literal "
            "NO_EXTRA_RECORD. Do not invent any other record."
        )
        api_key = None
        if args.api_key_file:
            api_key = args.api_key_file.read_text(encoding="utf-8").strip()
            if not api_key:
                raise ValueError("API key is empty")
        response = stream_completion(
            args.base_url,
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "temperature": 0,
                "seed": int(args.seed_sha256[:8], 16),
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            api_key,
        )
        usage = response["usage"]
        if not isinstance(usage, dict):
            raise RuntimeError("completion omitted usage")
        processed = usage.get("prompt_tokens")
        completed = usage.get("completion_tokens")
        if not isinstance(processed, int) or isinstance(processed, bool):
            raise RuntimeError("prompt token count is invalid")
        if not isinstance(completed, int) or isinstance(completed, bool):
            raise RuntimeError("completion token count is invalid")
        completion = validate_completion(
            content=response["content"],
            reasoning_content=response["reasoning_content"],
            finish_reason=response["finish_reason"],
            done=response["done"],
            records=fixture["records"],
        )
        checks = {
            "server_processed_target": processed >= args.target_tokens,
            "within_context_cap": processed <= args.context_cap,
            "response_identity": len(response["response_ids"]) == 1,
            "completion_token_timestamps": (
                completed > 0
                and len(response["event_timestamps"]) == completed
            ),
            "strict_completion": completion["pass"],
        }
        result.update(
            {
                "fixture_sha256": fixture["fixture_sha256"],
                "tokenizer_sha256": TOKENIZER_SHA256,
                "records": fixture["records"],
                "expected_values": values,
                "processed_tokens": processed,
                "completed_output_tokens": completed,
                "response": response,
                "completion": completion,
                "truncated": response["finish_reason"] != "stop",
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
