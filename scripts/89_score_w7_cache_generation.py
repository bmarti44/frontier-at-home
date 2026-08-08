#!/usr/bin/env python3
"""Score the W7 stable-model expert-cache generation invariant.

This is deliberately a log scorer, not an engine self-reported verdict.  A
single server process that has begun accepting requests must not announce a
model-generation change while serving a fixed model.  The later paired gate
still owns byte identity and performance acceptance.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import re


LISTEN = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
CACHE_ENABLED = "CUDA persistent expert cache enabled:"
INDEXED_RESUME = "GLM sync branch=indexed_resume"
PROMPT_DONE = " prompt done "
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"
STABLE_REMAP = "CUDA stable model remap enabled generation="
STABLE_REMAP_RE = re.compile(r"CUDA stable model remap enabled generation=([1-9][0-9]*)$")
PROMPT_START_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt start$"
)
PROMPT_DONE_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt done [0-9]+(?:\.[0-9]+)?s$"
)
FATAL_PATTERNS = (
    re.compile(r"CUDA GLM prefill failed", re.IGNORECASE),
    re.compile(
        r"(?:CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
        r"CUDA.{0,160}(?:allocation failed|out of memory))",
        re.IGNORECASE,
    ),
    re.compile(r"NV_ERR_NO_MEMORY", re.IGNORECASE),
    re.compile(r"forward_token failed", re.IGNORECASE),
    re.compile(r"(?:out of memory|oom-kill|killed process)", re.IGNORECASE),
    re.compile(r"\bFATAL\b", re.IGNORECASE),
    re.compile(r"(?:NVRM.*)?\bXid\b", re.IGNORECASE),
    re.compile(r"request timeout", re.IGNORECASE),
)


def score_text(
    text: str,
    *,
    http_status: str,
    response_text: str,
    containment_rc: str,
    containment_stdout: str,
    mode: str = "on",
    child_exit_text: str = '{"shutdown_requested":true,"forced_kill":false,"exit_status":0}',
    safety_main_text: str = "",
    expected_binary_sha256: str = "",
    expected_environment_sha256: str = "",
    model_identity_text: str = "",
    expected_model_sha256: str = "",
    expected_model_bytes: int = 0,
) -> dict[str, object]:
    lines = text.splitlines()
    listen = next((i for i, line in enumerate(lines) if LISTEN in line), None)
    shutdown_indexes = [
        i for i, line in enumerate(lines) if SHUTDOWN in line and (listen is None or i > listen)
    ]
    shutdown = shutdown_indexes[0] if len(shutdown_indexes) == 1 else None
    window = lines[listen + 1 : shutdown] if listen is not None and shutdown is not None else []
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
                "context_start": int(start_match.group(1)),
                "prompt_tokens": int(start_match.group(2)),
                "suffix_tokens": int(start_match.group(3)),
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
    response_tuple = None
    try:
        response = json.loads(response_text)
        usage = response["usage"]
        choices = response["choices"]
        prompt_details = usage["prompt_tokens_details"]
        response_tuple = (
            prompt_details["cached_tokens"],
            usage["prompt_tokens"],
            prompt_details["cache_write_tokens"],
        )
        response_ok = (
            isinstance(response, dict)
            and "error" not in response
            and isinstance(usage, dict)
            and isinstance(prompt_details, dict)
            and isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            and type(choices[0].get("text")) is str
            and choices[0]["text"] == ""
            and choices[0].get("finish_reason") == "length"
            and all(
                type(value) is int
                for value in (
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    prompt_details.get("cached_tokens"),
                    prompt_details.get("cache_write_tokens"),
                )
            )
            and usage["completion_tokens"] == 0
            and usage["total_tokens"] == usage["prompt_tokens"]
            and prompt_details["cached_tokens"] + prompt_details["cache_write_tokens"]
            == usage["prompt_tokens"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        response_ok = False
    matching_response_windows = [
        item
        for item in request_windows
        if response_ok
        and (
            item["context_start"],
            item["prompt_tokens"],
            item["suffix_tokens"],
        )
        == response_tuple
    ]
    request_bound = (
        len(indexed_completed) == 1
        and len(matching_response_windows) == 1
        and matching_response_windows[0] is indexed_completed[0]
    )
    false_flush_count = sum(FALSE_FLUSH in line for line in window)
    activation_generations = [
        int(match.group(1))
        for line in lines[: listen + 1 if listen is not None else 0]
        if (match := STABLE_REMAP_RE.search(line)) is not None
    ]
    activation_ok = (
        mode == "on" and len(activation_generations) == 1
    ) or (
        mode == "off" and not activation_generations and STABLE_REMAP not in text
    )
    try:
        child_exit = json.loads(child_exit_text)
    except (json.JSONDecodeError, TypeError):
        child_exit = None
    child_exit_ok = (
        type(child_exit) is dict
        and set(child_exit) == {"shutdown_requested", "forced_kill", "exit_status"}
        and child_exit.get("shutdown_requested") is True
        and child_exit.get("forced_kill") is False
        and type(child_exit.get("exit_status")) is int
        and child_exit.get("exit_status") == 0
    )
    try:
        model_identity = json.loads(model_identity_text)
    except (json.JSONDecodeError, TypeError):
        model_identity = None
    model_identity_ok = (
        type(model_identity) is dict
        and set(model_identity) == {"bytes", "device", "inode", "sha256", "executed_path"}
        and type(model_identity.get("bytes")) is int
        and model_identity.get("bytes") == expected_model_bytes
        and type(model_identity.get("device")) is int
        and type(model_identity.get("inode")) is int
        and model_identity.get("device", -1) >= 0
        and model_identity.get("inode", 0) > 0
        and model_identity.get("sha256") == expected_model_sha256
        and type(model_identity.get("executed_path")) is str
        and re.fullmatch(r"/proc/[1-9][0-9]*/fd/[0-9]+", model_identity["executed_path"])
        is not None
    )
    fatal_markers = sorted(
        {
            match.group(0)
            for pattern in FATAL_PATTERNS
            for line in window
            if (match := pattern.search(line)) is not None
        }
    )
    clean_containment = (
        containment_rc.strip() == "0"
        and re.fullmatch(
            r"SAFE_RUN_DONE rc=0 killed=no dir=/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+\n?",
            containment_stdout,
        )
        is not None
    )
    safety_lines = safety_main_text.splitlines()
    safe_start_lines = [line for line in safety_lines if "SAFE_RUN start " in line]
    cgroup_lines = [line for line in safety_lines if "cgroup_verified " in line]
    binary_lines = [line for line in safety_lines if "executed_candidate_verified " in line]
    environment_lines = [line for line in safety_lines if "executed_environment_allowlist=" in line]
    clean_exit_lines = [line for line in safety_lines if "executed candidate was verified alive at least once;" in line]
    safe_end_lines = [line for line in safety_lines if "SAFE_RUN end " in line]
    safety_checks = {
        "safe_start_settings_bound": len(safe_start_lines) == 1
        and all(marker in safe_start_lines[0] for marker in (
            "vlimit_kb=419430400", "kill_floor_gib=24", "min_start_gib=110", "timeout_s=2400"
        )),
        "cgroup_settings_bound": len(cgroup_lines) == 1
        and all(marker in cgroup_lines[0] for marker in (
            "memory_high=83751862272", "memory_max=85899345920",
            "memory_swap_max=0", "memory_oom_group=1",
        )),
        "executed_binary_bound": bool(expected_binary_sha256)
        and len(binary_lines) == 1
        and f"executed_binary_sha256={expected_binary_sha256}" in binary_lines[0],
        "executed_environment_bound": bool(expected_environment_sha256)
        and len(environment_lines) == 1
        and "executed_environment_allowlist=DS4_CUDA_STABLE_MODEL_REMAP" in environment_lines[0]
        and f"executed_environment_sha256={expected_environment_sha256}" in environment_lines[0],
        "safe_wrapper_clean_exit": len(clean_exit_lines) == 1
        and len(safe_end_lines) == 1
        and "SAFE_RUN end rc=0 killed=no" in safe_end_lines[0]
        and not any("FATAL" in line or "KILL_FLOOR breached" in line for line in safety_lines),
    }
    checks = {
        "listener_observed": listen is not None,
        "shutdown_observed_once": len(shutdown_indexes) == 1,
        "stable_remap_activation_matches_arm": activation_ok,
        "child_exit_clean": child_exit_ok,
        "executed_model_identity_bound": model_identity_ok,
        "persistent_cache_enabled_once": cache_enabled_count == 1,
        "exactly_one_bound_indexed_resume_completed": request_bound,
        "http_status_200": http_status.strip() == "200",
        "response_schema_valid": response_ok,
        "clean_containment": clean_containment,
        **safety_checks,
        "fatal_markers_absent": not fatal_markers,
        "stable_model_generation_flushes_zero": false_flush_count == 0,
    }
    return {
        "formula": "PASS iff all checks are true; generation-change flushes are counted strictly after listener readiness and before shutdown",
        "observed": {
            "cache_enabled_count": cache_enabled_count,
            "request_window_count": len(request_windows),
            "bound_indexed_resume_count": len(indexed_completed),
            "response_request_tuple": response_tuple,
            "matching_response_window_count": len(matching_response_windows),
            "fatal_markers": fatal_markers,
            "false_generation_flush_count": false_flush_count,
            "stable_remap_activation_generations": activation_generations,
            "shutdown_marker_count": len(shutdown_indexes),
            "child_exit": child_exit,
            "model_identity": model_identity,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def write_exclusive(path: Path, rendered: str) -> None:
    """Publish a result once; never follow or replace an existing path."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        payload = rendered.encode("utf-8")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--http-status", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--containment-rc", type=Path, required=True)
    parser.add_argument("--containment-stdout", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--child-exit", type=Path, required=True)
    parser.add_argument("--safety-main", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--expected-environment-sha256", required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-model-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_text(
        args.server_log.read_text(encoding="utf-8", errors="strict"),
        http_status=args.http_status.read_text(encoding="utf-8", errors="strict"),
        response_text=args.response.read_text(encoding="utf-8", errors="strict"),
        containment_rc=args.containment_rc.read_text(encoding="utf-8", errors="strict"),
        containment_stdout=args.containment_stdout.read_text(encoding="utf-8", errors="strict"),
        mode=args.mode,
        child_exit_text=args.child_exit.read_text(encoding="utf-8", errors="strict"),
        safety_main_text=args.safety_main.read_text(encoding="utf-8", errors="strict"),
        expected_binary_sha256=args.expected_binary_sha256,
        expected_environment_sha256=args.expected_environment_sha256,
        model_identity_text=args.model_identity.read_text(encoding="utf-8", errors="strict"),
        expected_model_sha256=args.expected_model_sha256,
        expected_model_bytes=args.expected_model_bytes,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_exclusive(args.output, rendered)
    else:
        print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
