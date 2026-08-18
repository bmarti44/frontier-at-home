#!/usr/bin/env python3
"""Score a pinned MMMU vision evalset through OpenAI chat completions."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "evalsets" / "mmmu-val-100" / "cases.jsonl"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ANSWER_PATTERNS = (
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=|-)?\s*\(?\s*([A-Z])\s*\)?", re.I),
    re.compile(r"\\boxed\s*\{\s*([A-Z])\s*\}", re.I),
)
AMBIGUOUS_TAIL_RE = re.compile(
    r"^\s*(?:[,;/]|\b(?:or|and)\b)\s*(?:option\s*)?\(?\s*([A-Z])\b",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="server root URL")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", required=True, type=Path, help="results directory")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--thinking-mode",
        choices=("chat", "thinking"),
        default="thinking",
        help="select the server chat template's thinking behavior",
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--request-timeout", type=int, default=300)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if not args.base_url:
        parser.error("--base-url must not be empty")
    if not 1 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 1 and 32768")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.request_timeout <= 7200:
        parser.error("--request-timeout must be between 1 and 7200 seconds")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(cases_path: Path) -> dict[str, Any]:
    """Fail closed unless the entire evalset directory matches pins.json."""
    cases_path = cases_path.resolve()
    root = cases_path.parent
    pins_path = root / "pins.json"
    try:
        pins = json.loads(pins_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read pins manifest {pins_path}: {error}") from error
    if not isinstance(pins, dict) or pins.get("schema_version") != 1:
        raise RuntimeError("invalid pins.json schema")
    if pins.get("dataset") != "MMMU/MMMU" or pins.get("split") != "validation":
        raise RuntimeError("pins.json dataset/split mismatch")
    revision = pins.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("pins.json has no immutable dataset revision")
    files = pins.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("pins.json files must be a nonempty object")

    expected_paths: set[str] = set()
    for relative, entry in files.items():
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise RuntimeError(f"unsafe pinned path: {relative!r}")
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid pin entry for {relative}")
        expected_hash = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise RuntimeError(f"invalid SHA-256 pin for {relative}")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise RuntimeError(f"invalid byte-size pin for {relative}")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"pinned file missing or not regular: {relative}")
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"byte-size mismatch for {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: pinned={expected_hash} actual={actual_hash}"
            )
        expected_paths.add(relative)

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != pins_path
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise RuntimeError(f"pins file inventory mismatch: missing={missing} extra={extra}")
    case_relative = cases_path.relative_to(root).as_posix()
    if case_relative not in expected_paths:
        raise RuntimeError(f"cases file is not pinned: {case_relative}")
    return pins


def load_cases(cases_path: Path, pins: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cases line {line_number} is invalid JSON: {error}") from error
        if not isinstance(row, dict) or set(row) != {"id", "question", "options", "answer", "image"}:
            raise RuntimeError(f"cases line {line_number} has invalid fields")
        options = row["options"]
        answer = row["answer"]
        if (
            not isinstance(row["id"], str)
            or not isinstance(row["question"], str)
            or not isinstance(options, list)
            or not 2 <= len(options) <= 26
            or any(not isinstance(option, str) or not option for option in options)
            or not isinstance(answer, str)
            or not re.fullmatch(r"[A-Z]", answer)
            or ord(answer) - ord("A") >= len(options)
            or not isinstance(row["image"], str)
            or row["image"] not in pins["files"]
            or not row["image"].startswith("images/")
        ):
            raise RuntimeError(f"cases line {line_number} is invalid")
        rows.append(row)
    if len(rows) != pins.get("rows"):
        raise RuntimeError(
            f"cases row count mismatch: pins={pins.get('rows')!r} actual={len(rows)}"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("cases contain duplicate ids")
    return rows


def extract_answer_letter(text: str, option_count: int = 26) -> str | None:
    """Extract a deliberate final choice without treating arbitrary prose as one."""
    if not isinstance(text, str) or not 1 <= option_count <= 26:
        return None
    valid = {chr(ord("A") + index) for index in range(option_count)}
    anchored: list[tuple[int, int, str]] = []
    for pattern in ANSWER_PATTERNS:
        anchored.extend(
            (match.start(), match.end(), match.group(1).upper())
            for match in pattern.finditer(text)
        )
    anchored = [match for match in anchored if match[2] in valid]
    if anchored:
        _, end, letter = max(anchored)
        line_tail = text[end:].splitlines()[0] if text[end:] else ""
        ambiguous = AMBIGUOUS_TAIL_RE.match(line_tail)
        if ambiguous and ambiguous.group(1).upper() in valid:
            return None
        return letter
    stripped = text.strip()
    standalone = re.fullmatch(
        r"(?:\*\*|__|`)?\s*\(?\s*([A-Z])\s*\)?\s*[.!]?\s*(?:\*\*|__|`)?",
        stripped,
        re.I,
    )
    if standalone and standalone.group(1).upper() in valid:
        return standalone.group(1).upper()
    # Some servers wrap the requested single-letter answer in a final line.
    last_nonempty = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    final_line = re.fullmatch(r"(?:\*\*|__|`)?\(?([A-Z])\)?[.!]?(?:\*\*|__|`)?", last_nonempty, re.I)
    if final_line and final_line.group(1).upper() in valid:
        return final_line.group(1).upper()
    return None


def load_api_key(path: Path | None) -> str | None:
    if path is None:
        return None
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key file is empty: {path}")
    return key


class Client:
    def __init__(self, base_url: str, api_key: str | None, timeout: int) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_model(self) -> tuple[str, Any]:
        request = urllib.request.Request(
            self.base_url + "/v1/models", headers=self.headers(), method="GET"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
            status = response.status
        try:
            document = json.loads(raw)
            data = document["data"]
            model = data[0]["id"] if len(data) == 1 else None
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError("invalid /v1/models response") from error
        if status != 200 or not isinstance(model, str) or not model:
            raise RuntimeError(f"invalid /v1/models response (HTTP {status})")
        return model, document

    def chat(self, payload: dict[str, Any]) -> tuple[str, str, str, Any, float]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = self.headers()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"chat completions HTTP {error.code}: {detail}") from error
        elapsed = time.perf_counter() - started
        try:
            document = json.loads(raw)
            choice = document["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            finish_reason = choice["finish_reason"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError("invalid chat completions response") from error
        content = "" if content is None else content
        reasoning = "" if reasoning is None else reasoning
        if (
            status != 200
            or not isinstance(content, str)
            or not isinstance(reasoning, str)
            or not isinstance(finish_reason, str)
        ):
            raise RuntimeError(f"invalid chat completions response (HTTP {status})")
        return content, reasoning, finish_reason, document, elapsed


def render_question(row: dict[str, Any]) -> str:
    lines = [row["question"], ""]
    lines.extend(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(row["options"])
    )
    lines.append("\nAnswer with the single letter of the correct option.")
    return "\n".join(lines)


def image_data_url(path: Path, image_bytes: bytes) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg"}:
        raise RuntimeError(f"unsupported image type: {path.name}")
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        pins = verify_pins(args.cases)
        rows = load_cases(args.cases.resolve(), pins)
        if args.limit is not None:
            rows = rows[: args.limit]
        if not rows:
            raise RuntimeError("no cases selected")
        client = Client(args.base_url, load_api_key(args.api_key_file), args.request_timeout)
        model, models_response = client.get_model()
        args.out.mkdir(parents=True, exist_ok=True)
        transcript_path = args.out / "transcripts.jsonl"
        summary_path = args.out / "summary.json"
        if transcript_path.exists() or summary_path.exists():
            raise RuntimeError(f"result files already exist in {args.out}")
        started_at = utc_now()
        correct = 0
        invalid = 0
        errors = 0
        with transcript_path.open("x", encoding="utf-8") as transcripts:
            for position, row in enumerate(rows):
                image_path = args.cases.resolve().parent / row["image"]
                image_bytes = image_path.read_bytes()
                sent_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                prompt = render_question(row)
                wire_payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Solve the visual multiple-choice question. Return only one uppercase option letter.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_url(image_path, image_bytes)},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": args.thinking_mode == "thinking"},
                }
                content = ""
                reasoning = ""
                response: Any = None
                elapsed: float | None = None
                finish_reason: str | None = None
                parsed: str | None = None
                error_text: str | None = None
                try:
                    content, reasoning, finish_reason, response, elapsed = client.chat(wire_payload)
                    if finish_reason == "stop":
                        parsed = extract_answer_letter(content, len(row["options"]))
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"
                    errors += 1
                scored_correct = parsed == row["answer"]
                correct += int(scored_correct)
                invalid += int(parsed is None)
                recorded_payload = copy.deepcopy(wire_payload)
                image_url = recorded_payload["messages"][1]["content"][0]["image_url"]["url"]
                data_url_prefix = image_url.split(",", 1)[0]
                recorded_payload["messages"][1]["content"][0]["image_url"]["url"] = (
                    f"{data_url_prefix},<REDACTED: exact sent image bytes; "
                    f"sha256={sent_image_sha256}>"
                )
                transcript = {
                    "position": position,
                    "id": row["id"],
                    "expected": row["answer"],
                    "parsed": parsed,
                    "scored_correct": scored_correct,
                    "request": recorded_payload,
                    "response": response,
                    "finish_reason": finish_reason,
                    "content": content,
                    "reasoning_content": reasoning,
                    "elapsed_s": elapsed,
                    "error": error_text,
                    "sent_image": {
                        "path": row["image"],
                        "bytes": len(image_bytes),
                        "sha256": sent_image_sha256,
                        "payload_redaction": "request.messages[1].content[0].image_url.url base64 body only",
                    },
                }
                transcripts.write(json.dumps(transcript, ensure_ascii=False) + "\n")
                transcripts.flush()
                print(
                    f"[{position + 1}/{len(rows)}] id={row['id']} correct={scored_correct} parsed={parsed}",
                    flush=True,
                )
        finished_at = utc_now()
        summary = {
            "ok": errors == 0,
            "suite": "mmmu-val-100",
            "model": model,
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "invalid_count": invalid,
            "error_count": errors,
            "started_at": started_at,
            "finished_at": finished_at,
            "cases": str(args.cases.resolve()),
            "pins_sha256": sha256_file(args.cases.resolve().parent / "pins.json"),
            "dataset": pins["dataset"],
            "dataset_revision": pins["revision"],
            "models_response": models_response,
            "transcripts": str(transcript_path.resolve()),
            "generation": {
                "endpoint": "/v1/chat/completions",
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "thinking_mode": args.thinking_mode,
                "request_timeout_s": args.request_timeout,
            },
        }
        write_json(summary_path, summary)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"38_bench_vision.py: {error}", file=sys.stderr)
        return 1
    ok = errors == 0
    print(json.dumps({"ok": ok, "summary": str(summary_path), "transcripts": str(transcript_path)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
