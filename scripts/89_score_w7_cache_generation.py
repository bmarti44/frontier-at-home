#!/usr/bin/env python3
"""Score the W7 stable-model expert-cache generation invariant.

This is deliberately a log scorer, not an engine self-reported verdict.  A
single server process that has begun accepting requests must not announce a
model-generation change while serving a fixed model.  The later paired gate
still owns byte identity and performance acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


LISTEN = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
CACHE_ENABLED = "CUDA persistent expert cache enabled:"
INDEXED_RESUME = "GLM sync branch=indexed_resume"
PROMPT_DONE = " prompt done "
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"
PROMPT_START_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt start$"
)
PROMPT_DONE_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt done [0-9]+(?:\.[0-9]+)?s$"
)
FATAL_MARKERS = (
    "CUDA GLM prefill failed",
    "forward_token failed",
    "out of memory",
    "fatal error",
    " Xid ",
    "request timeout",
)


def score_text(
    text: str,
    *,
    http_status: str,
    response_text: str,
    containment_rc: str,
    containment_stdout: str,
) -> dict[str, object]:
    lines = text.splitlines()
    listen = next((i for i, line in enumerate(lines) if LISTEN in line), None)
    shutdown = next(
        (i for i, line in enumerate(lines) if SHUTDOWN in line and (listen is None or i > listen)),
        len(lines),
    )
    window = lines[listen + 1 : shutdown] if listen is not None else []
    cache_enabled_count = sum(CACHE_ENABLED in line for line in window)
    request_windows: list[dict[str, object]] = []
    starts = [i for i, line in enumerate(window) if PROMPT_START_RE.search(line)]
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(window)
        lines_for_request = window[start:end]
        start_match = PROMPT_START_RE.search(lines_for_request[0])
        assert start_match is not None
        done_matches = [
            match
            for line in lines_for_request
            if (match := PROMPT_DONE_RE.search(line)) is not None
        ]
        request_windows.append(
            {
                "prompt_tokens": int(start_match.group(2)),
                "indexed_resume_count": sum(INDEXED_RESUME in line for line in lines_for_request),
                "matching_prompt_done_count": sum(
                    match.groups() == start_match.groups() for match in done_matches
                ),
            }
        )
    indexed_completed = [
        item
        for item in request_windows
        if item["indexed_resume_count"] == 1
        and item["matching_prompt_done_count"] == 1
    ]
    response_ok = False
    response_prompt_tokens = None
    try:
        response = json.loads(response_text)
        usage = response["usage"]
        choices = response["choices"]
        response_prompt_tokens = usage["prompt_tokens"]
        response_ok = (
            type(response_prompt_tokens) is int
            and isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") is not None
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        response_ok = False
    request_bound = (
        len(indexed_completed) == 1
        and response_ok
        and indexed_completed[0]["prompt_tokens"] == response_prompt_tokens
    )
    false_flush_count = sum(FALSE_FLUSH in line for line in window)
    fatal_markers = [marker for marker in FATAL_MARKERS if any(marker in line for line in window)]
    clean_containment = (
        containment_rc.strip() == "0"
        and re.fullmatch(
            r"SAFE_RUN_DONE rc=0 killed=no dir=/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+\n?",
            containment_stdout,
        )
        is not None
    )
    checks = {
        "listener_observed": listen is not None,
        "persistent_cache_enabled_once": cache_enabled_count == 1,
        "exactly_one_bound_indexed_resume_completed": request_bound,
        "http_status_200": http_status.strip() == "200",
        "response_schema_valid": response_ok,
        "clean_containment": clean_containment,
        "fatal_markers_absent": not fatal_markers,
        "stable_model_generation_flushes_zero": false_flush_count == 0,
    }
    return {
        "formula": "PASS iff all checks are true; generation-change flushes are counted strictly after listener readiness and before shutdown",
        "observed": {
            "cache_enabled_count": cache_enabled_count,
            "request_window_count": len(request_windows),
            "bound_indexed_resume_count": len(indexed_completed),
            "response_prompt_tokens": response_prompt_tokens,
            "fatal_markers": fatal_markers,
            "false_generation_flush_count": false_flush_count,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--http-status", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--containment-rc", type=Path, required=True)
    parser.add_argument("--containment-stdout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_text(
        args.server_log.read_text(encoding="utf-8", errors="strict"),
        http_status=args.http_status.read_text(encoding="utf-8", errors="strict"),
        response_text=args.response.read_text(encoding="utf-8", errors="strict"),
        containment_rc=args.containment_rc.read_text(encoding="utf-8", errors="strict"),
        containment_stdout=args.containment_stdout.read_text(encoding="utf-8", errors="strict"),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
