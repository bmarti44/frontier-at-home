#!/usr/bin/env python3
"""Measure streaming prefill and decode speed on an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUEST_TIMEOUT_S = 300
SEED = 42
# Top cell 28672: candidate A's engine envelope caps single prompts near 30K
# at ctx=32768 (lazy session graph); identical cells for both stacks.
CONTEXT_LEVELS = (0, 4096, 16384, 28672)
MAX_TOKENS = 256
MIN_VALID_COMPLETION_TOKENS = 200
REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = REPO_ROOT / "vendor" / "official-encoding" / "tokenizer.json"
FIXTURE_PATH = REPO_ROOT / "fixtures" / "ctx-32k.txt"
PREAMBLE_WORDS = (
    "amber", "anchor", "apricot", "atlas", "basil", "beacon", "birch",
    "canyon", "cedar", "cinder", "cobalt", "comet", "coral", "delta",
    "ember", "falcon", "fern", "fjord", "flint", "garden", "granite",
    "harbor", "hazel", "indigo", "island", "juniper", "lantern", "lilac",
    "maple", "marble", "meadow", "meteor", "moss", "nectar", "oasis",
    "olive", "onyx", "orchid", "pebble", "pine", "quartz", "raven",
    "river", "saffron", "silver", "spruce", "summit", "thistle", "valley",
    "violet", "willow", "zephyr",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="server root URL")
    parser.add_argument("--api-key-file", type=Path, help="file containing bearer token")
    parser.add_argument("--out", required=True, type=Path, help="results JSON path")
    parser.add_argument("--stack-label", required=True, help="stack name recorded in output")
    parser.add_argument("--reps", type=int, default=5, help="measured reps per context (default: 5)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="warmup reps to run and discard before each cell (default: 0)",
    )
    parser.add_argument(
        "--ignore-eos-supported",
        action="store_true",
        help="send the llama.cpp ignore_eos extension",
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every request body (per-stack mode control)",
    )
    parser.add_argument(
        "--context-levels",
        default=",".join(str(value) for value in CONTEXT_LEVELS),
        help="comma-separated exact fixture-token context levels",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"maximum generated tokens (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=MIN_VALID_COMPLETION_TOKENS,
        help=f"minimum valid generated tokens (default: {MIN_VALID_COMPLETION_TOKENS})",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="fixture and sampling seed")
    parser.add_argument(
        "--model-id",
        help="exact model id to select when /v1/models exposes multiple aliases",
    )
    parser.add_argument(
        "--token-timing-log",
        type=Path,
        help="server log containing DS4_TOKEN_TIMING records for exact decode timing",
    )
    args = parser.parse_args()
    if args.extra_body is not None:
        args.extra_body = json.loads(args.extra_body)
        if not isinstance(args.extra_body, dict):
            parser.error("--extra-body must be a JSON object")
    if args.reps <= 0:
        parser.error("--reps must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    try:
        args.context_levels = tuple(
            int(value) for value in args.context_levels.split(",") if value != ""
        )
    except ValueError:
        parser.error("--context-levels must contain comma-separated integers")
    if not args.context_levels or any(value < 0 for value in args.context_levels):
        parser.error("--context-levels must contain non-negative integers")
    if args.max_tokens < 128:
        parser.error("--max-tokens must be at least 128 for decode timing")
    if not 128 <= args.min_completion_tokens <= args.max_tokens:
        parser.error("--min-completion-tokens must be between 128 and --max-tokens")
    args.base_url = args.base_url.rstrip("/")
    if not args.base_url:
        parser.error("--base-url must not be empty")
    return args


def load_api_key(path: Path | None) -> str | None:
    if path is None:
        return None
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def load_tokenizer() -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("tokenizers is required; install requirements-harness.txt") from error
    if not TOKENIZER_PATH.is_file():
        raise RuntimeError(f"pinned tokenizer is missing: {TOKENIZER_PATH}")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def prefix_with_exact_tokens(tokenizer: Any, text: str, target: int) -> str:
    if target < 0:
        raise ValueError("token target must be non-negative")
    if target == 0:
        return ""
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) < target:
        raise RuntimeError(f"fixture has fewer than {target} tokens")
    end = encoding.offsets[target - 1][1]
    prefix = text[:end]
    actual = token_count(tokenizer, prefix)
    if actual != target:
        raise RuntimeError(
            f"tokenizer offset produced {actual} tokens instead of {target}"
        )
    return prefix


class Client:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        # Merged before harness-critical keys; cannot override them.
        self.extra_body = dict(extra_body or {})

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_model(self, requested: str | None = None) -> str:
        request = urllib.request.Request(
            self.base_url + "/v1/models", headers=self.headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            raise RuntimeError(
                f"GET /v1/models returned HTTP {error.code}: {raw[:500]!r}"
            ) from error
        if status != 200:
            raise RuntimeError(f"GET /v1/models returned HTTP {status}: {raw[:500]!r}")
        try:
            document = json.loads(raw)
            data = document["data"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"invalid models response: {raw[:500]!r}") from error
        if not isinstance(data, list) or not data:
            count = len(data) if isinstance(data, list) else "non-list"
            raise RuntimeError(f"expected at least one model, received {count}")
        models = [
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ]
        if len(models) != len(data):
            raise RuntimeError("one or more model ids are missing or invalid")
        if requested is not None:
            if requested not in models:
                raise RuntimeError(
                    f"requested model {requested!r} is absent; available={models!r}"
                )
            return requested
        if len(models) != 1:
            raise RuntimeError(
                f"multiple model aliases require --model-id; available={models!r}"
            )
        return models[0]

    def stream_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = dict(self.extra_body)
        request_payload.update(payload)
        request_payload["stream"] = True
        request_payload["stream_options"] = {"include_usage": True}
        body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
        headers = self.headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        request_started = time.perf_counter()
        try:
            response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"stream returned HTTP {error.code}: {raw[:500]!r}") from error

        first_content_at: float | None = None
        last_content_at: float | None = None
        generated_parts: list[str] = []
        token_timestamps: list[float] = []
        usage: dict[str, Any] | None = None
        done = False
        data_chunks = 0
        response_ids: set[str] = set()
        try:
            with response:
                if response.status != 200:
                    raise RuntimeError(f"stream returned HTTP {response.status}")
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="strict").strip()
                    if not line or line.startswith(":"):
                        continue
                    if done:
                        raise RuntimeError("received SSE data after [DONE]")
                    if not line.startswith("data:"):
                        raise RuntimeError(f"unexpected SSE line: {line[:200]!r}")
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    data_chunks += 1
                    event = json.loads(data)
                    response_id = event.get("id")
                    if response_id is not None:
                        if not isinstance(response_id, str) or not response_id:
                            raise RuntimeError("SSE response id is invalid")
                        response_ids.add(response_id)
                    event_usage = event.get("usage")
                    if event_usage is not None:
                        if not isinstance(event_usage, dict):
                            raise RuntimeError("SSE usage is not an object")
                        usage = event_usage
                    choices = event.get("choices", [])
                    if not isinstance(choices, list):
                        raise RuntimeError("SSE choices is not a list")
                    for choice in choices:
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            raise RuntimeError("SSE delta is not an object")
                        fragments: list[str] = []
                        for field in ("reasoning_content", "content"):
                            fragment = delta.get(field)
                            if fragment is not None and not isinstance(fragment, str):
                                raise RuntimeError(f"{field} delta is not a string")
                            if fragment:
                                fragments.append(fragment)
                        if fragments:
                            now = time.perf_counter()
                            generated_parts.extend(fragments)
                            token_timestamps.append(now)
                            if first_content_at is None:
                                first_content_at = now
                            last_content_at = now
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid SSE stream: {error}") from error
        if len(response_ids) != 1:
            raise RuntimeError(
                f"SSE stream has ambiguous response ids: {sorted(response_ids)!r}"
            )

        return {
            "response_id": next(iter(response_ids)),
            "request_started": request_started,
            "first_content_at": first_content_at,
            "last_content_at": last_content_at,
            "generated_text": "".join(generated_parts),
            "usage": usage,
            "done": done,
            "data_chunks": data_chunks,
            "token_timestamps": token_timestamps,
        }


def make_preamble(tokenizer: Any, unique_id: int, seed: int = SEED) -> str:
    rng = random.Random(seed + unique_id * 1_000_003)
    words = [f"benchmark-{unique_id:06d}"]
    words.extend(rng.choice(PREAMBLE_WORDS) for _ in range(80))
    source = "Preamble " + " ".join(words) + "."
    preamble = prefix_with_exact_tokens(tokenizer, source, 32)
    actual = token_count(tokenizer, preamble)
    if actual != 32:
        raise RuntimeError(f"preamble has {actual} tokens instead of 32")
    return preamble


def gpu_snapshot() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    output = completed.stdout.strip()
    return output or None


def invalid_rep(error: str) -> dict[str, Any]:
    return {
        "ttft_s": None,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "completion_tokens": None,
        "valid": False,
        "error": error,
    }


def reps_are_complete(reps: list[dict[str, Any]], expected_reps: int) -> bool:
    """Fail closed: every requested measurement must exist and be valid."""
    return (
        expected_reps > 0
        and len(reps) == expected_reps
        and all(rep.get("valid") is True for rep in reps)
    )


def observable_output_errors(
    client_completion_tokens: int,
    event_completion_tokens: int,
    minimum_tokens: int,
) -> list[str]:
    """Validate counts measured by the client, without trusting server usage."""
    reasons: list[str] = []
    if event_completion_tokens < minimum_tokens:
        reasons.append(
            f"early stop: {event_completion_tokens} timestamped tokens, "
            f"minimum is {minimum_tokens}"
        )
    if client_completion_tokens != event_completion_tokens:
        reasons.append(
            "timestamp/client token mismatch: "
            f"events={event_completion_tokens}, client={client_completion_tokens}"
        )
    return reasons


TOKEN_TIMING_RE = re.compile(
    r"^DS4_TOKEN_TIMING request=(\S+) index=(\d+) "
    r"monotonic_ns=(\d+) token=(-?\d+)$"
)


def read_token_timing(
    path: Path,
    offset: int,
    *,
    expected_request: str,
    expected_count: int,
) -> dict[str, Any]:
    """Read and validate raw per-token records appended after *offset*."""
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read()
    records: list[tuple[str, int, int, int]] = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        match = TOKEN_TIMING_RE.fullmatch(line)
        if match:
            records.append(
                (
                    match.group(1),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )
            )
        elif line.startswith("DS4_TOKEN_TIMING"):
            raise RuntimeError(f"malformed DS4_TOKEN_TIMING record: {line[:200]!r}")
    if not records:
        raise RuntimeError(f"no DS4_TOKEN_TIMING records appended to {path}")
    if len(records) != expected_count:
        raise RuntimeError(
            "token timing count mismatch: "
            f"timed={len(records)}, expected={expected_count}"
        )
    requests = {record[0] for record in records}
    if len(requests) != 1:
        raise RuntimeError(f"token timing contains multiple requests: {sorted(requests)}")
    if requests != {expected_request}:
        raise RuntimeError(
            "token timing request mismatch: "
            f"timed={sorted(requests)!r}, expected={expected_request!r}"
        )
    indices = [record[1] for record in records]
    expected = list(range(1, len(records) + 1))
    if indices != expected:
        raise RuntimeError(f"token timing indices are incomplete: {indices[:8]}...")
    monotonic_ns = [record[2] for record in records]
    if any(value <= 0 for value in monotonic_ns):
        raise RuntimeError("token timing timestamps must be positive")
    if any(later <= earlier for earlier, later in zip(monotonic_ns, monotonic_ns[1:])):
        raise RuntimeError("token timing timestamps are not strictly increasing")
    return {
        "request": records[0][0],
        "indices": indices,
        "monotonic_ns": monotonic_ns,
        "token_ids": [record[3] for record in records],
    }


def raw_visible_output_errors(
    tokenizer: Any,
    token_ids: list[int],
    generated_text: str,
) -> list[str]:
    """Bind server-side token records to the exact bytes observed by the client."""
    reasons: list[str] = []
    try:
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    except Exception as error:
        return [f"cannot determine frozen tokenizer vocabulary: {error}"]
    if (
        not isinstance(vocab_size, int)
        or vocab_size <= 0
        or any(
            not isinstance(token, int) or token < 0 or token >= vocab_size
            for token in token_ids
        )
    ):
        return ["raw timing contains a token outside the frozen tokenizer vocabulary"]
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as error:
        return [f"cannot decode raw timing tokens: {error}"]

    # Thinking mode may place the opening marker in the assistant prefix (so it
    # is absent here) or generate it. The OpenAI stream deliberately hides both
    # framing markers while preserving the reasoning and content bytes in order.
    visible = decoded
    if visible.startswith("<think>"):
        visible = visible[len("<think>") :]
    visible = visible.replace("</think>", "", 1)
    if visible != generated_text:
        reasons.append(
            "raw token/client byte mismatch: "
            f"decoded_bytes={len(visible.encode('utf-8'))}, "
            f"client_bytes={len(generated_text.encode('utf-8'))}"
        )
    return reasons


def raw_timing_envelope_errors(
    raw_elapsed_s: float,
    client_first_content_at: float,
    client_last_content_at: float,
) -> list[str]:
    """Reject raw clocks inconsistent with an independent client wall interval."""
    client_elapsed_s = client_last_content_at - client_first_content_at
    if client_elapsed_s <= 0:
        return [f"non-positive client content interval: {client_elapsed_s}"]
    ratio = raw_elapsed_s / client_elapsed_s
    if not 0.75 <= ratio <= 1.25:
        return [
            "raw/client decode interval mismatch: "
            f"raw={raw_elapsed_s}, client={client_elapsed_s}, ratio={ratio}"
        ]
    return []


def run_rep(
    client: Client,
    tokenizer: Any,
    model: str,
    fixture_slice: str,
    context_tokens: int,
    unique_id: int,
    ignore_eos_supported: bool,
    max_tokens: int = MAX_TOKENS,
    min_completion_tokens: int = MIN_VALID_COMPLETION_TOKENS,
    seed: int = SEED,
    token_timing_log: Path | None = None,
) -> dict[str, Any]:
    try:
        preamble = make_preamble(tokenizer, unique_id, seed)
        prompt = preamble + "\n\n" + fixture_slice + "\n\nContinue this text naturally, writing at least 600 more words without stopping."
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
        }
        if ignore_eos_supported:
            payload["ignore_eos"] = True
        timing_offset = (
            token_timing_log.stat().st_size if token_timing_log is not None else None
        )
        stream = client.stream_chat(payload)
        if not stream["done"]:
            return invalid_rep("SSE stream did not terminate with [DONE]")
        if stream["first_content_at"] is None or stream["last_content_at"] is None:
            return invalid_rep("SSE stream produced no content chunks")
        usage = stream["usage"]
        if not isinstance(usage, dict):
            return invalid_rep("SSE stream did not include a usage object")
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            return invalid_rep(f"invalid usage.completion_tokens: {completion_tokens!r}")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            return invalid_rep(f"invalid usage.prompt_tokens: {prompt_tokens!r}")
        client_completion_tokens = token_count(tokenizer, stream["generated_text"])
        sse_token_timestamps = stream["token_timestamps"]
        event_completion_tokens = len(sse_token_timestamps)
        if completion_tokens == 0:
            return invalid_rep("server reported zero completion tokens")
        raw_timing = None
        if token_timing_log is not None:
            raw_timing = read_token_timing(
                token_timing_log,
                timing_offset,
                expected_request=stream["response_id"],
                expected_count=completion_tokens,
            )
            token_timestamps = [
                value / 1_000_000_000 for value in raw_timing["monotonic_ns"]
            ]
            reasons = []
            if len(token_timestamps) < min_completion_tokens:
                reasons.append(
                    f"early stop: {len(token_timestamps)} raw timed tokens, "
                    f"minimum is {min_completion_tokens}"
                )
            if len(token_timestamps) != completion_tokens:
                reasons.append(
                    "raw timing/server completion mismatch: "
                    f"timed={len(token_timestamps)}, server={completion_tokens}"
                )
            reasons.extend(
                raw_visible_output_errors(
                    tokenizer,
                    raw_timing["token_ids"],
                    stream["generated_text"],
                )
            )
            timing_source = "server_raw_token_log"
        else:
            token_timestamps = sse_token_timestamps
            reasons = observable_output_errors(
                client_completion_tokens,
                len(token_timestamps),
                min_completion_tokens,
            )
            timing_source = "sse_content_events"
        timed_completion_tokens = len(token_timestamps)
        ttft_s = stream["first_content_at"] - stream["request_started"]
        decode_elapsed_s = token_timestamps[-1] - token_timestamps[0]
        if raw_timing is not None:
            reasons.extend(
                raw_timing_envelope_errors(
                    decode_elapsed_s,
                    stream["first_content_at"],
                    stream["last_content_at"],
                )
            )
        decode_tok_s = (
            (timed_completion_tokens - 1) / decode_elapsed_s
            if decode_elapsed_s > 0
            else None
        )
        prefill_tok_s = prompt_tokens / ttft_s if ttft_s > 0 else None
        if ttft_s <= 0:
            reasons.append(f"non-positive TTFT: {ttft_s}")
        if decode_elapsed_s <= 0:
            reasons.append(f"non-positive decode interval: {decode_elapsed_s}")
        if reasons:
            rep = invalid_rep("; ".join(reasons))
            rep.update(
                {
                    "ttft_s": ttft_s,
                    "decode_tok_s": decode_tok_s,
                    "prefill_tok_s": prefill_tok_s,
                    "completion_tokens": timed_completion_tokens,
                    "server_completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "client_completion_tokens": client_completion_tokens,
                    "event_completion_tokens": event_completion_tokens,
                    "timing_source": timing_source,
                    "token_timestamps_ns": (
                        raw_timing["monotonic_ns"] if raw_timing is not None else None
                    ),
                    "token_ids": raw_timing["token_ids"] if raw_timing is not None else None,
                    "client_fixture_tokens": context_tokens,
                    "data_chunks": stream["data_chunks"],
                }
            )
            return rep
        return {
            "ttft_s": ttft_s,
            "decode_tok_s": decode_tok_s,
            "prefill_tok_s": prefill_tok_s,
            "completion_tokens": timed_completion_tokens,
            "server_completion_tokens": completion_tokens,
            "valid": True,
            "prompt_tokens": prompt_tokens,
            "client_completion_tokens": client_completion_tokens,
            "event_completion_tokens": event_completion_tokens,
            "timing_source": timing_source,
            "token_timestamps_ns": (
                raw_timing["monotonic_ns"] if raw_timing is not None else None
            ),
            "token_ids": raw_timing["token_ids"] if raw_timing is not None else None,
            "client_fixture_tokens": context_tokens,
            "data_chunks": stream["data_chunks"],
        }
    except Exception as error:
        return invalid_rep(f"{type(error).__name__}: {error}")


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def iqr(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    result: dict[str, Any] = {
        "metadata": {
            "stack_label": args.stack_label,
            "base_url": args.base_url,
            "started_at": started_at,
            "finished_at": None,
            "reps": args.reps,
            "warmup_reps": args.warmup,
            "ignore_eos_supported": args.ignore_eos_supported,
            "max_tokens": args.max_tokens,
            "min_completion_tokens": args.min_completion_tokens,
            "temperature": 0,
            "seed": args.seed,
            "prefill_rate_label": "incl. queue+setup",
            "iqr_method": "inclusive quartiles",
        },
        "cells": [],
        "suite_valid": False,
    }
    try:
        api_key = load_api_key(args.api_key_file)
        tokenizer = load_tokenizer()
        fixture = FIXTURE_PATH.read_text(encoding="utf-8")
        fixture_total_tokens = token_count(tokenizer, fixture)
        if fixture_total_tokens < max(args.context_levels):
            raise RuntimeError(
                f"fixture has {fixture_total_tokens} tokens; {max(args.context_levels)} required"
            )
        fixture_slices = {
            level: prefix_with_exact_tokens(tokenizer, fixture, level)
            for level in args.context_levels
        }
        client = Client(args.base_url, api_key, args.extra_body)
        model = client.get_model(args.model_id)
        result["metadata"]["model"] = model
        result["metadata"]["fixture_path"] = str(FIXTURE_PATH.relative_to(REPO_ROOT))
        result["metadata"]["fixture_total_tokens"] = fixture_total_tokens

        any_cell_failed = False
        unique_id = 0
        for cell_index, level in enumerate(args.context_levels):
            cell: dict[str, Any] = {
                "ctx_tokens": level,
                "reps": [],
                "median_decode": None,
                "iqr_decode": None,
                "median_ttft": None,
                "gpu_before": gpu_snapshot(),
                "gpu_after": None,
            }
            for warmup_index in range(args.warmup):
                run_rep(
                    client,
                    tokenizer,
                    model,
                    fixture_slices[level],
                    level,
                    unique_id,
                    args.ignore_eos_supported,
                    args.max_tokens,
                    args.min_completion_tokens,
                    args.seed,
                    args.token_timing_log,
                )
                unique_id += 1
                if warmup_index + 1 < args.warmup or args.reps > 0:
                    time.sleep(2)

            for rep_index in range(args.reps):
                rep = run_rep(
                    client,
                    tokenizer,
                    model,
                    fixture_slices[level],
                    level,
                    unique_id,
                    args.ignore_eos_supported,
                    args.max_tokens,
                    args.min_completion_tokens,
                    args.seed,
                    args.token_timing_log,
                )
                unique_id += 1
                cell["reps"].append(rep)
                if rep_index + 1 < args.reps:
                    time.sleep(2)

            cell["gpu_after"] = gpu_snapshot()
            valid_reps = [rep for rep in cell["reps"] if rep["valid"]]
            decode_values = [rep["decode_tok_s"] for rep in valid_reps]
            ttft_values = [rep["ttft_s"] for rep in valid_reps]
            cell["median_decode"] = median(decode_values)
            cell["iqr_decode"] = iqr(decode_values)
            cell["median_ttft"] = median(ttft_values)
            invalid_count = len(cell["reps"]) - len(valid_reps)
            cell["invalid_reps"] = invalid_count
            cell["valid"] = reps_are_complete(cell["reps"], args.reps)
            if not cell["valid"]:
                any_cell_failed = True
            result["cells"].append(cell)
            if cell_index + 1 < len(args.context_levels):
                time.sleep(2)

        result["suite_valid"] = not any_cell_failed
        result["metadata"]["finished_at"] = utc_now()
        write_result(args.out, result)
        return 0 if result["suite_valid"] else 1
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["metadata"]["finished_at"] = utc_now()
        write_result(args.out, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
