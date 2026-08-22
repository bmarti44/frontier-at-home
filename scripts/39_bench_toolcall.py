#!/usr/bin/env python3
"""Run the pinned deterministic tool-call probe through chat completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "evalsets" / "toolcall" / "cases.jsonl"
DEFAULT_PINS = DEFAULT_CASES.parent / "pins.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8016/v1")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable containing the bearer token",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="sampling temperature; this deterministic suite requires 0",
    )
    parser.add_argument("--out", required=True, type=Path, help="result JSON path")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if not args.base_url:
        parser.error("--base-url must not be empty")
    if not args.model.strip():
        parser.error("--model must not be empty")
    if not 1 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 1 and 32768")
    if args.temperature != 0:
        parser.error("--temperature must be 0 for deterministic scoring")
    args.temperature = 0
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(document: Any) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_pins(
    cases_path: Path = DEFAULT_CASES, pins_path: Path = DEFAULT_PINS
) -> dict[str, Any]:
    try:
        pins = json.loads(pins_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read pins manifest {pins_path}: {error}") from error
    if not isinstance(pins, dict) or pins.get("schema_version") != 1:
        raise RuntimeError("invalid pins.json schema")
    if pins.get("suite") != "toolcall-v1" or pins.get("created") != "2026-08-20":
        raise RuntimeError("pins.json suite/created mismatch")
    expected = pins.get("sha256")
    count = pins.get("case_count")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise RuntimeError("pins.json has an invalid cases SHA-256")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise RuntimeError("pins.json has an invalid case count")
    actual = sha256_file(cases_path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for cases.jsonl: pinned={expected} actual={actual}"
        )
    return pins


def _valid_schema(schema: Any) -> bool:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return False
    properties = schema.get("properties")
    required = schema.get("required", [])
    return (
        isinstance(properties, dict)
        and isinstance(required, list)
        and all(isinstance(key, str) and key in properties for key in required)
    )


def valid_openai_tool(tool: Any) -> bool:
    if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
        return False
    function = tool.get("function")
    return (
        tool["type"] == "function"
        and isinstance(function, dict)
        and set(function) == {"name", "description", "parameters"}
        and isinstance(function["name"], str)
        and bool(function["name"])
        and isinstance(function["description"], str)
        and bool(function["description"])
        and _valid_schema(function["parameters"])
    )


def load_cases(cases_path: Path, pins: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        cases_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cases line {line_number} is invalid JSON: {error}") from error
        if not isinstance(row, dict) or set(row) != {"id", "messages", "tools", "expect"}:
            raise RuntimeError(f"cases line {line_number} has invalid fields")
        expect = row["expect"]
        valid_expect = isinstance(expect, dict) and expect.get("type") in {
            "tool_call",
            "text",
        }
        if (
            not isinstance(row["id"], str)
            or not row["id"]
            or not isinstance(row["messages"], list)
            or not row["messages"]
            or not isinstance(row["tools"], list)
            or not row["tools"]
            or not all(valid_openai_tool(tool) for tool in row["tools"])
            or not valid_expect
        ):
            raise RuntimeError(f"cases line {line_number} is invalid")
        if expect["type"] == "tool_call" and (
            not isinstance(expect.get("name"), str)
            or not isinstance(expect.get("required_args", {}), dict)
        ):
            raise RuntimeError(f"cases line {line_number} has invalid tool-call expectation")
        if expect["type"] == "text" and expect.get("forbidden_call") is not True:
            raise RuntimeError(f"cases line {line_number} must forbid tool calls")
        rows.append(row)
    if len(rows) != pins["case_count"]:
        raise RuntimeError(
            f"cases row count mismatch: pins={pins['case_count']} actual={len(rows)}"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("cases contain duplicate ids")
    return rows


def deep_subset(expected: Any, actual: Any) -> bool:
    """Match expected dictionary keys recursively; arrays remain exact values."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and deep_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            deep_subset(left, right) for left, right in zip(expected, actual)
        )
    return type(expected) is type(actual) and expected == actual


def score_response(case: dict[str, Any], response: Any) -> dict[str, Any]:
    """Score only choices[0]; for calls, score its first tool call."""
    try:
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return {"passed": False, "reason": "missing first response choice"}
    if not isinstance(tool_calls, list):
        return {"passed": False, "reason": "tool_calls is not a list"}

    expect = case["expect"]
    if expect["type"] == "text":
        passed = len(tool_calls) == 0
        return {
            "passed": passed,
            "reason": "no tool calls" if passed else "unexpected tool call",
            "actual": {"tool_call_count": len(tool_calls)},
        }
    if not tool_calls:
        return {"passed": False, "reason": "expected a tool call"}
    try:
        function = tool_calls[0]["function"]
        name = function["name"]
        encoded_args = function["arguments"]
    except (KeyError, TypeError):
        return {"passed": False, "reason": "malformed first tool call"}
    if not isinstance(encoded_args, str):
        return {"passed": False, "reason": "tool arguments are not a JSON string"}
    try:
        arguments = json.loads(encoded_args)
    except json.JSONDecodeError:
        return {"passed": False, "reason": "tool arguments are invalid JSON"}
    actual = {"name": name, "arguments": arguments, "tool_call_count": len(tool_calls)}
    if name != expect["name"]:
        return {"passed": False, "reason": "wrong function name", "actual": actual}
    if not deep_subset(expect.get("required_args", {}), arguments):
        return {"passed": False, "reason": "required arguments do not match", "actual": actual}
    return {"passed": True, "reason": "matched", "actual": actual}


class Client:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"chat completions HTTP {error.code}: {detail}") from error
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("invalid chat completions response JSON") from error
        if status != 200 or not isinstance(document, dict):
            raise RuntimeError(f"invalid chat completions response (HTTP {status})")
        return document


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        pins = verify_pins()
        cases = load_cases(DEFAULT_CASES, pins)
        request_params = {
            "max_tokens": args.max_tokens,
            "model": args.model,
            "temperature": args.temperature,
        }
        client = Client(args.base_url, os.environ.get(args.api_key_env))
        per_case = []
        passed = 0
        for case in cases:
            payload = {
                **request_params,
                "messages": case["messages"],
                "tools": case["tools"],
            }
            try:
                response = client.chat(payload)
                scored = score_response(case, response)
                error = None
            except Exception as exc:
                response = None
                scored = {"passed": False, "reason": "request failed"}
                error = f"{type(exc).__name__}: {exc}"
            passed += int(scored["passed"])
            per_case.append(
                {
                    "id": case["id"],
                    **scored,
                    "expected": case["expect"],
                    "response": response,
                    "error": error,
                }
            )
            print(f"[{len(per_case)}/{len(cases)}] id={case['id']} passed={scored['passed']}", flush=True)
        result = {
            "suite": "toolcall-v1",
            "pins_sha256": sha256_file(DEFAULT_PINS),
            "model": args.model,
            "base_url": args.base_url,
            "config_digest": canonical_digest(request_params),
            "per_case": per_case,
            "passed": passed,
            "total": len(cases),
            "score": passed / len(cases),
        }
        write_json(args.out, result)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"39_bench_toolcall.py: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "out": str(args.out), "score": result["score"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
