#!/usr/bin/env python3
"""Run a contained R0b union-trace OFF/ON qualification.

The default mode preserves the qualified short single-batch check.  The explicit
high-row mode additionally requires exact contiguous coverage across multiple
indexed-prefill chunks, including a full 2,048-row chunk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
from typing import Any
import unicodedata

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
FREEZE = ROOT / "results/glm52-gates/R0b-union-trace-smoke-freeze.json"
RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-trace-smoke-randomness.json"
CORPUS_FREEZE = ROOT / "results/glm52-gates/R0b-union-corpus-runtime-freeze.json"
CORPUS_RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-corpus-runtime-randomness.json"
QUALITY_SPLIT_PLAN = ROOT / "results/glm52-gates/R0b-union-p0-split-plan.json"
QUALITY_FREEZE = ROOT / "results/glm52-gates/R0b-union-quality-runtime-freeze.json"
QUALITY_RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-quality-runtime-randomness.json"
SHARED_PATH = ROOT / "scripts/73_run_glm_shared_router_probe.py"
SCORER_PATH = ROOT / "scripts/75_glm_union_trace_score.py"
FROZEN_RUNTIME_DEPENDENCIES = (
    "scripts/73_run_glm_shared_router_probe.py",
    "scripts/30_bench_speed.py",
    "scripts/glm52_goal.py",
    "scripts/03_memory_guard.py",
    "results/glm52-gates/harness/glm_safe_run.sh",
    "fixtures/ctx-32k.txt",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load("shared_router_runner", SHARED_PATH)
TRACE_SCORER = _load("union_trace_scorer", SCORER_PATH)
BENCH_CLIENT = _load("union_trace_bench_client", SHARED.BENCH)
TRACE_LAYER = 4
MIN_PROMPT_TOKENS = 512
MAX_CONTEXT_LEVEL = 8192
TRACE_BYTES_PER_TOKEN_LAYER = (6144 + 256 + 256 + 8 + 256) * 4
TRACE_DISK_RESERVE_BYTES = 20 * 1024**3
CORPUS_CACHE_EXPERTS = "32GB"
CORPUS_CUDA_CACHE_GB = "48"
CORPUS_MEMORY_HIGH_GIB = "71"
TRACE_NAMES = ",".join((
    "glm_indexed_ffn_norm",
    "glm_indexed_router_logits",
    "glm_indexed_router_selected",
    "glm_indexed_router_bias",
    "glm_indexed_router_probs",
))
ENV_NAMES = sorted(set(SHARED.ENV_NAMES) | {
    "DS4_GLM_SYNC_TRACE",
    "DS4_METAL_GRAPH_DUMP_PREFIX",
    "DS4_METAL_GRAPH_DUMP_NAME",
    "DS4_METAL_GRAPH_DUMP_LAYER",
    "DS4_GLM_UNION_TRACE_CORPUS",
    "DS4_GLM_STREAMING_TOKEN_PREFILL_MAX",
    "DS4_JSON_REPLACE_INVALID_UTF8",
})
SYNC_RE = re.compile(
    r"ds4: GLM sync branch=full_indexed pos=(\d+) chunk=(\d+) logits=\d+"
)
CUDA_CACHE_PREFIX = "ds4: CUDA persistent expert cache enabled: "
CUDA_CACHE_RE = re.compile(
    re.escape(CUDA_CACHE_PREFIX) + r"([1-9][0-9]*) slots x "
    r"([0-9]+(?:\.[0-9]+)?) MiB = ([0-9]+(?:\.[0-9]+)?) GiB "
    r"\(fixed arena\)"
)
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30
QUALITY_MAX_TOKENS = 8
QUALITY_FIXTURE_RELATIVE = Path(
    "gguf-tools/quality-testing/data/glm52-openrouter-100"
)
QUALITY_DISK_MAX_TOKENS = 512
QUALITY_REQUEST_COUNT = 100
QUALITY_FIXTURE_CONTENT_SHA256 = "49483fb172f700357d14167cfd9a69c686caa4e3b7889a41754bb4ba00584b0a"
UTF8_NORMALIZATION_LOG = "ds4-server: invalid UTF-8 model bytes normalized to U+FFFD"
UTF8_REGRESSION_CASE_ID = "case_021"
UTF8_REGRESSION_SEED = 805105121
UTF8_REGRESSION_TOKEN_PREFIX = (8507, 228, 35457, 11, 323, 279, 1008)
UTF8_REGRESSION_VISIBLE_PREFIX = "宆\uFFFD, and the other"


def render_quality_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("quality prompt must be text")
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: High"
        "<|system|>You are a helpful assistant"
        f"<|user|>{prompt}<|assistant|><think>"
    )


def quality_request_payload(prompt: str, seed: int) -> dict[str, Any]:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("quality request seed is invalid")
    return {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": QUALITY_MAX_TOKENS,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
    }


def quality_wire_body(prompt: str, seed: int) -> bytes:
    payload = quality_request_payload(prompt, seed)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _quality_manifest_rows(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            tuple(row)
            for row in csv.reader(
                (line for line in stream if not line.startswith("#")), delimiter="\t",
            )
            if row
        ]
    if any(len(row) != 4 for row in rows):
        raise ValueError("quality fixture manifest is malformed")
    return rows  # type: ignore[return-value]


def _normalized_text_sha256(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="strict")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_quality_case_ledger(
    candidate: Path, *, seed: int,
    plan_path: Path = QUALITY_SPLIT_PLAN,
    tokenizer_path: Path = SHARED.TOKENIZER,
) -> dict[str, Any]:
    """Build the frozen, independently tokenized 100-case request ledger."""
    candidate = candidate.resolve(strict=True)
    plan = SHARED.strict_json(plan_path)
    if (plan.get("schema_version") != 2 or
            plan.get("classification") != "PREREGISTERED" or
            plan.get("scope") != "glm52_union_probe_case_grouped_splits"):
        raise ValueError("quality split plan schema is invalid")
    fixture = candidate / QUALITY_FIXTURE_RELATIVE
    master = fixture / "manifest.tsv"
    if (not fixture.is_dir() or fixture.is_symlink() or
            SHARED.sha256(master) != plan.get("fixture_sha256")):
        raise ValueError("quality master fixture changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    holdout = set(plan["full_precision_hidden_holdout"]["case_ids"])
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    exact_hashes = {kind: set() for kind in ("prompt", "continuation", "response")}
    normalized_hashes = {kind: set() for kind in ("prompt", "continuation")}
    content_digest = hashlib.sha256()
    role_names = ("train", "calibration", "test")
    for split in role_names:
        split_record = plan["splits"][split]
        observed_count = 0
        for manifest_record in split_record["block_manifests"]:
            relative_manifest = str(manifest_record["path"])
            manifest = (ROOT / relative_manifest).resolve(strict=True)
            if (not manifest.is_relative_to(ROOT) or
                    SHARED.sha256(manifest) != manifest_record["sha256"]):
                raise ValueError("quality block manifest changed")
            manifest_bytes = manifest.read_bytes()
            relative_bytes = relative_manifest.encode("utf-8")
            content_digest.update(len(relative_bytes).to_bytes(8, "big"))
            content_digest.update(relative_bytes)
            content_digest.update(len(manifest_bytes).to_bytes(8, "big"))
            content_digest.update(manifest_bytes)
            for case_id, prompt_relative, continuation_relative, response_relative in _quality_manifest_rows(manifest):
                if case_id in seen_ids:
                    raise ValueError("quality case is duplicated across splits")
                seen_ids.add(case_id)
                observed_count += 1
                raw_by_kind: dict[str, bytes] = {}
                for kind, relative in (
                    ("prompt", prompt_relative),
                    ("continuation", continuation_relative),
                    ("response", response_relative),
                ):
                    path = (candidate / relative).resolve(strict=True)
                    if not path.is_relative_to(candidate) or not path.is_file() or path.is_symlink():
                        raise ValueError("quality fixture path is unsafe")
                    raw = path.read_bytes()
                    raw_by_kind[kind] = raw
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in exact_hashes[kind]:
                        raise ValueError("quality content is duplicated")
                    exact_hashes[kind].add(digest)
                    if kind != "response":
                        normalized = _normalized_text_sha256(raw)
                        if normalized in normalized_hashes[kind]:
                            raise ValueError("normalized quality content is duplicated")
                        normalized_hashes[kind].add(normalized)
                    relative_bytes = relative.encode("utf-8")
                    content_digest.update(len(relative_bytes).to_bytes(8, "big"))
                    content_digest.update(relative_bytes)
                    content_digest.update(len(raw).to_bytes(8, "big"))
                    content_digest.update(raw)
                prompt = raw_by_kind["prompt"].decode("utf-8", errors="strict")
                rendered = render_quality_prompt(prompt)
                token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
                if not 1 <= len(token_ids) <= QUALITY_DISK_MAX_TOKENS:
                    raise ValueError("quality prompt token count exceeds the frozen bound")
                case_seed = (seed + int(case_id.removeprefix("case_"))) % 2147483647
                cases.append({
                    "case_id": case_id,
                    "group_id": case_id,
                    "split": (
                        "train-precision-diagnostic" if case_id in holdout else
                        "train-fit" if split == "train" else split
                    ),
                    "prompt_sha256": hashlib.sha256(raw_by_kind["prompt"]).hexdigest(),
                    "continuation_sha256": hashlib.sha256(raw_by_kind["continuation"]).hexdigest(),
                    "response_sha256": hashlib.sha256(raw_by_kind["response"]).hexdigest(),
                    "seed": case_seed,
                    "token_ids": token_ids,
                    "expected_prompt_tokens": len(token_ids),
                    "request_sha256": hashlib.sha256(quality_wire_body(prompt, case_seed)).hexdigest(),
                    "prompt": prompt,
                })
        if observed_count != split_record["quality_cases"]:
            raise ValueError("quality split count changed")
    if (len(cases) != QUALITY_REQUEST_COUNT or len(seen_ids) != QUALITY_REQUEST_COUNT or
            len([case for case in cases if case["split"] == "train-fit"]) != 55 or
            len([case for case in cases if case["split"] == "train-precision-diagnostic"]) != 5):
        raise ValueError("quality case ledger is incomplete")
    random.Random(seed).shuffle(cases)
    for request_id, case in enumerate(cases, 1):
        case["request_id"] = request_id
    public_cases = [
        {key: value for key, value in case.items() if key != "prompt"}
        for case in cases
    ]
    fixture_content_sha256 = content_digest.hexdigest()
    if fixture_content_sha256 != QUALITY_FIXTURE_CONTENT_SHA256:
        raise ValueError("quality fixture content differs from the preregistered freeze")
    return {
        "schema_version": 1,
        "split_plan_sha256": SHARED.sha256(plan_path),
        "fixture_content_sha256": fixture_content_sha256,
        "tokenizer_sha256": SHARED.sha256(tokenizer_path),
        "seed": seed,
        "total_expected_prompt_tokens": sum(case["expected_prompt_tokens"] for case in cases),
        "expected_token_layer_events": 75 * sum(case["expected_prompt_tokens"] for case in cases),
        "cases": public_cases,
        "_prompts": {case["case_id"]: case["prompt"] for case in cases},
    }


def quality_probe_ledger(
    bundle: dict[str, Any], *, case_id: str | None = None,
) -> dict[str, Any]:
    """Select one frozen case for a safety or exact-regression probe."""
    cases = bundle.get("cases")
    prompts = bundle.get("_prompts")
    if (not isinstance(cases, list) or not cases or not isinstance(cases[0], dict) or
            not isinstance(prompts, dict)):
        raise ValueError("quality probe source ledger is invalid")
    selected = cases[0] if case_id is None else next(
        (case for case in cases if case.get("case_id") == case_id), None,
    )
    if selected is None:
        raise ValueError("quality probe case is not present in the frozen ledger")
    first = dict(selected)
    case_id = first.get("case_id")
    tokens = first.get("expected_prompt_tokens")
    if (not isinstance(case_id, str) or case_id not in prompts or
            not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0):
        raise ValueError("quality probe case is invalid")
    first["request_id"] = 1
    if case_id == UTF8_REGRESSION_CASE_ID:
        first["seed"] = UTF8_REGRESSION_SEED
        first["request_sha256"] = hashlib.sha256(
            quality_wire_body(prompts[case_id], UTF8_REGRESSION_SEED)
        ).hexdigest()
        first["utf8_regression_expected"] = True
    return {
        **{key: value for key, value in bundle.items()
           if key not in {"cases", "_prompts", "total_expected_prompt_tokens",
                          "expected_token_layer_events"}},
        "total_expected_prompt_tokens": tokens,
        "expected_token_layer_events": 75 * tokens,
        "cases": [first],
        "_prompts": {case_id: prompts[case_id]},
    }


def quality_window_indices(rows: list[dict[str, Any]], *, horizon: int) -> list[list[int]]:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("quality horizon must be positive")
    windows: list[list[int]] = []
    start = 0
    while start < len(rows):
        first = rows[start]
        required = (first.get("case_id"), first.get("split"), first.get("layer"))
        end = start + 1
        while end < len(rows):
            current = rows[end]
            identity = (current.get("case_id"), current.get("split"), current.get("layer"))
            if identity != required or current.get("position") != rows[end - 1].get("position") + 1:
                break
            end += 1
        for offset in range(start, end - horizon):
            windows.append(list(range(offset, offset + horizon + 1)))
        start = end
    return windows


def _exact_chunks(chunks: Any, expected_tokens: int) -> bool:
    if not isinstance(chunks, list) or not chunks:
        return False
    if not all(
        isinstance(row, list) and len(row) == 2 and
        all(isinstance(value, int) and not isinstance(value, bool) for value in row) and
        row[0] >= 0 and row[1] > 0
        for row in chunks
    ):
        return False
    return (
        chunks[0][0] == 0 and
        all(chunks[index][0] == chunks[index - 1][0] + chunks[index - 1][1]
            for index in range(1, len(chunks))) and
        chunks[-1][0] + chunks[-1][1] == expected_tokens
    )


def quality_capture_verdict(
    ledger: list[dict[str, Any]],
    off_requests: list[dict[str, Any]],
    on_requests: list[dict[str, Any]],
    trace_score: dict[str, Any],
    off_containment: dict[str, Any],
    on_containment: dict[str, Any],
) -> dict[str, Any]:
    """Apply the fixed case/request/token/layer coverage formula."""
    ledger_valid = (
        isinstance(ledger, list) and bool(ledger) and
        all(isinstance(row, dict) for row in ledger) and
        [row.get("request_id") for row in ledger] == list(range(1, len(ledger) + 1)) and
        len({row.get("case_id") for row in ledger}) == len(ledger) and
        len({row.get("request_sha256") for row in ledger}) == len(ledger) and
        all(
            isinstance(row.get("expected_prompt_tokens"), int) and
            not isinstance(row.get("expected_prompt_tokens"), bool) and
            1 <= row["expected_prompt_tokens"] <= 512 and
            isinstance(row.get("token_ids"), list) and
            len(row["token_ids"]) == row["expected_prompt_tokens"]
            for row in ledger
        )
    )
    identity_keys = (
        "case_id", "group_id", "split", "request_id", "request_sha256",
    )
    output_keys = (
        "completion_tokens", "generated_reasoning_sha256",
        "generated_reasoning_bytes", "generated_content_sha256",
        "generated_content_bytes", "token_ids", "finish_reason",
        "utf8_regression_reproduced",
    )

    def output_valid(observed: dict[str, Any]) -> bool:
        return (
            observed.get("completion_tokens") == QUALITY_MAX_TOKENS and
            observed.get("finish_reason") == "length" and
            isinstance(observed.get("token_ids"), list) and
            len(observed["token_ids"]) == QUALITY_MAX_TOKENS and
            all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in observed["token_ids"]) and
            all(isinstance(observed.get(key), str) and
                re.fullmatch(r"[0-9a-f]{64}", observed[key]) is not None
                for key in ("generated_reasoning_sha256", "generated_content_sha256")) and
            all(isinstance(observed.get(key), int) and
                not isinstance(observed[key], bool) and observed[key] >= 0
                for key in ("generated_reasoning_bytes", "generated_content_bytes")) and
            observed["generated_reasoning_bytes"] + observed["generated_content_bytes"] > 0
        )

    def arm_matches(requests: Any) -> bool:
        if not ledger_valid or not isinstance(requests, list) or len(requests) != len(ledger):
            return False
        for expected, observed in zip(ledger, requests):
            if (not isinstance(observed, dict) or
                    any(observed.get(key) != expected.get(key) for key in identity_keys) or
                    observed.get("prompt_tokens") != expected["expected_prompt_tokens"] or
                    not _exact_chunks(observed.get("full_indexed_chunks"),
                                      expected["expected_prompt_tokens"])):
                return False
        return True

    off_matches = arm_matches(off_requests)
    on_matches = arm_matches(on_requests)
    outputs_match = (
        off_matches and on_matches and
        all(output_valid(left) and output_valid(right) and
            all(left.get(key) == right.get(key) for key in output_keys)
            for left, right in zip(off_requests, on_requests))
    )
    regression_reproduced = (
        off_matches and on_matches and
        all(
            not expected.get("utf8_regression_expected") or
            (left.get("utf8_regression_reproduced") is True and
             right.get("utf8_regression_reproduced") is True)
            for expected, left, right in zip(ledger, off_requests, on_requests)
        )
    )
    expected_events = (
        75 * sum(row["expected_prompt_tokens"] for row in ledger)
        if ledger_valid else -1
    )
    checks = {
        "ledger_complete": ledger_valid,
        "off_exact_coverage": off_matches,
        "on_exact_coverage": on_matches,
        "per_case_output_identity": outputs_match,
        "exact_utf8_regression_reproduced": regression_reproduced,
        "scorer_passed": trace_score.get("verdict") == "PASS",
        "exact_request_count": trace_score.get("requests") == len(ledger),
        "exact_token_layer_events": trace_score.get("token_layer_events") == expected_events,
        "containment_clean": (
            off_containment.get("clean") is True and on_containment.get("clean") is True
        ),
    }
    return {
        "expected_token_layer_events": expected_events,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def quality_arm_identity_checks(
    off: dict[str, Any], on: dict[str, Any], trace_score: dict[str, Any],
    expected: dict[str, str],
) -> dict[str, bool]:
    """Bind distinct quality arms to the frozen identity and scored trace."""
    common = (
        "binary_sha256", "model_sha256", "tokenizer_sha256", "fixture_sha256",
        "split_plan_sha256", "configuration_sha256",
    )
    return {
        "distinct_arm_modes": off.get("mode") == "off" and on.get("mode") == "on",
        "frozen_arm_identity": all(
            off.get(key) == expected.get(key) and on.get(key) == expected.get(key)
            for key in common
        ),
        "mode_environment_identity": (
            off.get("environment_sha256") == expected.get("off_environment_sha256") and
            on.get("environment_sha256") == expected.get("on_environment_sha256")
        ),
        "off_trace_absent": off.get("trace_files") == 0 and off.get("trace_bytes") == 0,
        "on_trace_matches_scorer": (
            trace_score.get("verdict") == "PASS" and
            isinstance(trace_score.get("artifacts"), list) and
            len(trace_score["artifacts"]) > 0 and
            on.get("trace_files") == len(trace_score["artifacts"]) and
            on.get("trace_bytes") == trace_score.get("total_bytes")
        ),
    }


FINAL_ARTIFACT_RE = re.compile(
    r"final_artifact_verified path=(\S+) sha256=([0-9a-f]{64}) "
    r"device_inode=([0-9]+):([0-9]+):([0-9]+)"
)


def verify_final_artifact_receipts(
    containment: dict[str, Any], expected_paths: list[Path],
) -> None:
    """Require current artifacts to match the wrapper's post-exit receipts."""
    crash = Path(str(containment.get("crash_directory", ""))).resolve(strict=True)
    main = crash / "main.log"
    if SHARED.sha256(main) != containment.get("main_sha256"):
        raise ValueError("containment main log changed after verification")
    receipts: dict[str, tuple[str, int, int, int]] = {}
    for match in FINAL_ARTIFACT_RE.finditer(main.read_text(encoding="utf-8", errors="strict")):
        path = str(Path(match.group(1)).resolve())
        if path in receipts:
            raise ValueError("duplicate final artifact receipt")
        receipts[path] = (
            match.group(2), int(match.group(3)), int(match.group(4)), int(match.group(5)),
        )
    expected = {str(path.resolve()) for path in expected_paths}
    if set(receipts) != expected:
        raise ValueError("final artifact receipt set differs")
    for text_path, (digest, device, inode, size) in receipts.items():
        path = Path(text_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("final artifact is absent or unsafe")
        stat = path.stat()
        if ((stat.st_dev, stat.st_ino, stat.st_size) != (device, inode, size) or
                SHARED.sha256(path) != digest):
            raise ValueError("final artifact differs from containment receipt")


def quality_raw_visible_output_errors(
    tokenizer: Tokenizer, token_ids: list[int],
    generated_reasoning: str, generated_content: str,
) -> list[str]:
    """Bind raw token IDs to exact client bytes without assuming canonical BPE."""
    try:
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    except Exception as error:
        return [f"cannot determine frozen tokenizer vocabulary: {error}"]
    if (not isinstance(vocab_size, int) or vocab_size <= 0 or
            any(not isinstance(token, int) or isinstance(token, bool) or
                token < 0 or token >= vocab_size for token in token_ids)):
        return ["raw timing contains a token outside the frozen tokenizer vocabulary"]
    try:
        open_ids = tokenizer.encode("<think>", add_special_tokens=False).ids
        close_ids = tokenizer.encode("</think>", add_special_tokens=False).ids
        framed_ids = list(token_ids)
        if open_ids and framed_ids[:len(open_ids)] == open_ids:
            del framed_ids[:len(open_ids)]
        close_positions = [
            index for index in range(len(framed_ids) - len(close_ids) + 1)
            if close_ids and framed_ids[index:index + len(close_ids)] == close_ids
        ]
        if close_positions:
            # Match the frozen serving parser: its THINKING state uses strstr()
            # and therefore the first close marker is the sole channel boundary.
            # Later close markers are ordinary content and remain covered by the
            # exact decoded-byte comparison below.
            close_index = close_positions[0]
            reasoning_ids = framed_ids[:close_index]
            content_ids = framed_ids[close_index + len(close_ids):]
        elif generated_reasoning and not generated_content:
            reasoning_ids, content_ids = framed_ids, []
        elif generated_content and not generated_reasoning:
            reasoning_ids, content_ids = [], framed_ids
        else:
            return ["raw timing has no unambiguous reasoning/content token boundary"]
        decoded_reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=False)
        decoded_content = tokenizer.decode(content_ids, skip_special_tokens=False)
    except Exception as error:
        return [f"cannot bind raw timing tokens to client output: {error}"]
    if decoded_reasoning == generated_reasoning and decoded_content == generated_content:
        return []
    return [
        "raw token/client byte mismatch: "
        f"raw_reasoning_bytes={len(decoded_reasoning.encode('utf-8'))}, "
        f"client_reasoning_bytes={len(generated_reasoning.encode('utf-8'))}, "
        f"raw_content_bytes={len(decoded_content.encode('utf-8'))}, "
        f"client_content_bytes={len(generated_content.encode('utf-8'))}"
    ]


def randomness_is_after_freeze(round_number: int, freeze_commit_time: int) -> bool:
    if round_number < 1 or freeze_commit_time < 0:
        return False
    round_time = DRAND_GENESIS_UNIX + (round_number - 1) * DRAND_PERIOD_SECONDS
    return round_time > freeze_commit_time


def validate_randomness_order(
    freeze_path: Path = FREEZE, randomness_path: Path = RANDOMNESS,
) -> None:
    relative = str(freeze_path.relative_to(ROOT))
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", relative],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    freeze_time = int(completed.stdout.strip())
    randomness = SHARED.strict_json(randomness_path)
    if not randomness_is_after_freeze(int(randomness["round"]), freeze_time):
        raise ValueError("public randomness does not postdate the freeze commit")


def configuration_sha256(values: dict[str, str]) -> str:
    canonical = b"".join(
        name.encode("ascii") + b"=" + values.get(name, "<UNSET>").encode() + b"\n"
        for name in ENV_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def trace_environment(
    mode: str, out: Path, *, corpus_smoke: bool = False, quality_corpus: bool = False,
) -> dict[str, str]:
    if mode not in {"off", "on"}:
        raise ValueError("invalid trace arm")
    values = dict(SHARED.COMMON_ENV)
    values.update({
        "DS4_LOCK_FILE": str(out / "runtime.lock"),
        "DS4_GLM_SYNC_TRACE": "1",
    })
    large_corpus = corpus_smoke or quality_corpus
    if large_corpus:
        values["DS4_CUDA_EXPERT_CACHE_GB"] = CORPUS_CUDA_CACHE_GB
    if quality_corpus:
        values["DS4_GLM_STREAMING_TOKEN_PREFILL_MAX"] = "0"
        values["DS4_JSON_REPLACE_INVALID_UTF8"] = "1"
    if mode == "on":
        values.update({
            "DS4_METAL_GRAPH_DUMP_PREFIX": str(out / "trace/request"),
            "DS4_METAL_GRAPH_DUMP_NAME": TRACE_NAMES,
            "DS4_METAL_GRAPH_DUMP_LAYER": "all" if large_corpus else "4",
        })
        if large_corpus:
            values["DS4_GLM_UNION_TRACE_CORPUS"] = "1"
    return values


def matched_configuration_sha256(
    *, corpus_smoke: bool = False, quality_corpus: bool = False,
) -> str:
    values = dict(SHARED.COMMON_ENV)
    values.update({"DS4_LOCK_FILE": "<ARM_LOCAL>", "DS4_GLM_SYNC_TRACE": "1"})
    if corpus_smoke or quality_corpus:
        values["DS4_CUDA_EXPERT_CACHE_GB"] = CORPUS_CUDA_CACHE_GB
    if quality_corpus:
        values["DS4_GLM_STREAMING_TOKEN_PREFILL_MAX"] = "0"
        values["DS4_JSON_REPLACE_INVALID_UTF8"] = "1"
    return configuration_sha256(values)


def full_indexed_chunks(log: Path) -> list[list[int]]:
    return full_indexed_chunks_text(log.read_text(encoding="utf-8", errors="strict"))


def full_indexed_chunks_text(text: str) -> list[list[int]]:
    rows = [[int(match.group(1)), int(match.group(2))]
            for match in SYNC_RE.finditer(text)]
    if not rows or len({tuple(row) for row in rows}) != len(rows):
        raise ValueError("full-indexed chunk evidence is missing or duplicated")
    for index, (pos, count) in enumerate(rows):
        if count <= 0 or (index and pos != rows[index - 1][0] + rows[index - 1][1]):
            raise ValueError("full-indexed chunks overlap or have a gap")
    return rows


def cuda_cache_runtime(text: str) -> dict[str, int | float]:
    candidates = [line for line in text.splitlines()
                  if line.startswith(CUDA_CACHE_PREFIX)]
    if len(candidates) != 1:
        raise ValueError("one resolved CUDA expert-cache arena record is required")
    match = CUDA_CACHE_RE.fullmatch(candidates[0])
    if match is None:
        raise ValueError("resolved CUDA expert-cache arena record is malformed")
    slots_text, expert_mib_text, arena_gib_text = match.groups()
    slots = int(slots_text)
    expert_mib = float(expert_mib_text)
    arena_gib = float(arena_gib_text)
    derived_gib = slots * expert_mib / 1024.0
    if (not all(math.isfinite(value) and value > 0.0
                for value in (expert_mib, arena_gib)) or
            arena_gib > float(CORPUS_CUDA_CACHE_GB) or
            abs(derived_gib - arena_gib) > 0.02):
        raise ValueError("resolved CUDA expert-cache arena is invalid")
    return {"slots": slots, "arena_gib": arena_gib}


def smoke_verdict(
    off: dict[str, Any],
    on: dict[str, Any],
    trace_score: dict[str, Any],
    off_containment: dict[str, Any],
    on_containment: dict[str, Any],
    *,
    min_prompt_tokens: int = MIN_PROMPT_TOKENS,
    require_multichunk: bool = False,
    expected_corpus_seed: int | None = None,
) -> dict[str, Any]:
    common_hashes = (
        "binary_sha256", "model_sha256", "tokenizer_sha256",
        "fixture_sha256", "configuration_sha256",
    )
    prompt_tokens = off.get("prompt_tokens")
    chunks = off.get("full_indexed_chunks")
    chunks_well_formed = (
        isinstance(chunks, list) and bool(chunks) and
        all(
            isinstance(row, list) and len(row) == 2 and
            isinstance(row[0], int) and not isinstance(row[0], bool) and
            isinstance(row[1], int) and not isinstance(row[1], bool) and
            row[0] >= 0 and row[1] > 0
            for row in chunks
        )
    )
    chunks_contiguous = chunks_well_formed and all(
        chunks[index][0] == chunks[index - 1][0] + chunks[index - 1][1]
        for index in range(1, len(chunks))
    )
    exact_coverage = (
        isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
        prompt_tokens >= min_prompt_tokens and on.get("prompt_tokens") == prompt_tokens and
        chunks_well_formed and chunks_contiguous and chunks[0][0] == 0 and
        sum(row[1] for row in chunks) == prompt_tokens and
        chunks[-1][0] + chunks[-1][1] == prompt_tokens and
        (not require_multichunk or (
            prompt_tokens > 2048 and len(chunks) >= 2 and
            any(row[1] == 2048 for row in chunks)
        ))
    )
    off_corpus = off.get("corpus_requests")
    on_corpus = on.get("corpus_requests")
    corpus_mode = off_corpus is not None or on_corpus is not None
    corpus_scope = True
    corpus_event_floor = True
    corpus_cuda_cache = True
    if corpus_mode:
        def valid_requests(value: Any) -> bool:
            if (not isinstance(value, list) or len(value) != 2 or
                    not isinstance(expected_corpus_seed, int) or
                    isinstance(expected_corpus_seed, bool)):
                return False
            for expected_id, item in enumerate(value, 1):
                if (not isinstance(item, dict) or item.get("request_id") != expected_id or
                        item.get("seed") != expected_corpus_seed + expected_id - 1 or
                        not isinstance(item.get("prompt_tokens"), int) or
                        isinstance(item.get("prompt_tokens"), bool) or
                        item["prompt_tokens"] < MIN_PROMPT_TOKENS):
                    return False
                item_chunks = item.get("full_indexed_chunks")
                if (not isinstance(item_chunks, list) or not item_chunks or
                        any(not isinstance(row, list) or len(row) != 2 or
                            any(not isinstance(number, int) or isinstance(number, bool)
                                for number in row) or row[1] <= 0
                            for row in item_chunks)):
                    return False
                if (item_chunks[0][0] != 0 or
                        any(item_chunks[index][0] !=
                            item_chunks[index - 1][0] + item_chunks[index - 1][1]
                            for index in range(1, len(item_chunks))) or
                        sum(row[1] for row in item_chunks) != item["prompt_tokens"]):
                    return False
                signature = item.get("response_signature")
                request_sha256 = (signature.get("request_sha256")
                                  if isinstance(signature, dict) else None)
                if (not isinstance(request_sha256, str) or
                        re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None):
                    return False
            return True

        off_hashes = ({item["response_signature"]["request_sha256"] for item in off_corpus}
                      if valid_requests(off_corpus) else set())
        on_hashes = ({item["response_signature"]["request_sha256"] for item in on_corpus}
                     if valid_requests(on_corpus) else set())
        corpus_scope = (
            valid_requests(off_corpus) and valid_requests(on_corpus) and
            off.get("expert_cache_budget") == CORPUS_CACHE_EXPERTS and
            on.get("expert_cache_budget") == CORPUS_CACHE_EXPERTS and
            len(off_hashes) == 2 and off_hashes == on_hashes and
            all(
                all(left.get(key) == right.get(key) for key in (
                    "request_id", "seed", "prompt_tokens", "full_indexed_chunks",
                    "response_signature",
                ))
                for left, right in zip(off_corpus, on_corpus)
            ) and trace_score.get("requests") == 2
        )
        corpus_event_floor = (
            isinstance(trace_score.get("token_layer_events"), int) and
            trace_score["token_layer_events"] >= 76800
        )
        off_runtime = off.get("cuda_cache_runtime")
        on_runtime = on.get("cuda_cache_runtime")
        corpus_cuda_cache = (
            off.get("cuda_expert_cache_gb") == CORPUS_CUDA_CACHE_GB and
            on.get("cuda_expert_cache_gb") == CORPUS_CUDA_CACHE_GB and
            isinstance(off_runtime, dict) and set(off_runtime) == {"slots", "arena_gib"} and
            off_runtime == on_runtime and
            isinstance(off_runtime.get("slots"), int) and
            not isinstance(off_runtime.get("slots"), bool) and
            off_runtime["slots"] > 0 and
            isinstance(off_runtime.get("arena_gib"), (int, float)) and
            not isinstance(off_runtime.get("arena_gib"), bool) and
            math.isfinite(float(off_runtime["arena_gib"])) and
            0.0 < float(off_runtime["arena_gib"]) <= float(CORPUS_CUDA_CACHE_GB)
        )
    checks = {
        "arm_modes": off.get("mode") == "off" and on.get("mode") == "on",
        "frozen_identity": all(off.get(key) == on.get(key) for key in common_hashes),
        "byte_and_token_identity": off.get("response_signature") == on.get("response_signature"),
        "matched_indexed_chunks": off.get("full_indexed_chunks") == on.get("full_indexed_chunks"),
        "prompt_tokens_and_exact_coverage": exact_coverage,
        "off_emitted_no_trace": off.get("trace_files") == 0,
        "on_emitted_trace": isinstance(on.get("trace_files"), int) and on.get("trace_files", 0) > 0,
        "trace_score_passed": trace_score.get("verdict") == "PASS",
        "containment_clean": off_containment.get("clean") is True and on_containment.get("clean") is True,
        "corpus_request_scope": corpus_scope,
        "corpus_event_floor": corpus_event_floor,
        "corpus_cuda_cache": corpus_cuda_cache,
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def _run_quality_requests(
    *, port: int, out: Path, server_log: Path, log: Any,
    ledger_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    client = BENCH_CLIENT.Client(
        f"http://127.0.0.1:{port}", None, request_timeout_s=1200,
    )
    client.get_model("glm-5.2")
    tokenizer = Tokenizer.from_file(str(SHARED.TOKENIZER))
    prompts = ledger_bundle["_prompts"]
    records: list[dict[str, Any]] = []
    for expected in ledger_bundle["cases"]:
        prompt = prompts[expected["case_id"]]
        log.flush()
        os.fsync(log.fileno())
        log_start = server_log.stat().st_size
        stream = client.stream_chat(quality_request_payload(prompt, expected["seed"]))
        log.flush()
        os.fsync(log.fileno())
        with server_log.open("rb") as log_reader:
            log_reader.seek(log_start)
            request_log = log_reader.read().decode("utf-8", errors="strict")
        usage = stream.get("usage")
        if (stream.get("done") is not True or not isinstance(usage, dict) or
                stream.get("request_sha256") != expected["request_sha256"] or
                stream.get("finish_reason") != "length" or
                usage.get("prompt_tokens") != expected["expected_prompt_tokens"] or
                usage.get("completion_tokens") != QUALITY_MAX_TOKENS):
            raise ValueError(f"quality request {expected['case_id']} was incomplete")
        generated = stream.get("generated_text")
        reasoning = stream.get("generated_reasoning")
        content = stream.get("generated_content")
        if not all(isinstance(value, str) for value in (generated, reasoning, content)):
            raise ValueError("quality response text schema is invalid")
        raw_timing = BENCH_CLIENT.read_token_timing(
            server_log, log_start, expected_request=stream["response_id"],
            expected_count=QUALITY_MAX_TOKENS,
        )
        sse_content_events = len(stream["token_timestamps_ns"])
        output_errors = quality_raw_visible_output_errors(
            tokenizer, raw_timing["token_ids"], reasoning, content,
        )
        if sse_content_events <= 0:
            output_errors.append("no content-bearing SSE event was observed")
        if output_errors:
            raise ValueError(f"quality output is not independently observable: {output_errors}")
        combined_visible = reasoning + content
        utf8_regression_reproduced = (
            raw_timing["token_ids"][:len(UTF8_REGRESSION_TOKEN_PREFIX)] ==
            list(UTF8_REGRESSION_TOKEN_PREFIX) and
            combined_visible.startswith(UTF8_REGRESSION_VISIBLE_PREFIX)
        )
        if (expected.get("utf8_regression_expected") is True and
                not utf8_regression_reproduced):
            raise ValueError("known invalid-UTF-8 regression was not reproduced")
        records.append({
            "case_id": expected["case_id"],
            "group_id": expected["group_id"],
            "split": expected["split"],
            "request_id": expected["request_id"],
            "request_sha256": expected["request_sha256"],
            "prompt_tokens": usage["prompt_tokens"],
            "full_indexed_chunks": full_indexed_chunks_text(request_log),
            "completion_tokens": usage["completion_tokens"],
            "finish_reason": stream["finish_reason"],
            "generated_reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            "generated_reasoning_bytes": len(reasoning.encode()),
            "generated_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "generated_content_bytes": len(content.encode()),
            "token_ids": raw_timing["token_ids"],
            "sse_content_events": sse_content_events,
            "utf8_regression_reproduced": utf8_regression_reproduced,
        })
    response_path = out / "responses.json"
    response_path.write_text(
        json.dumps(records, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return records


def _arm(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    binary = args.binary.resolve()
    expected = trace_environment(
        args.mode, out, corpus_smoke=args.corpus_smoke,
        quality_corpus=args.quality_corpus,
    )
    observed = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    if observed != expected:
        raise ValueError("trace arm environment differs from frozen configuration")
    if (SHARED.sha256(binary) != args.binary_sha256 or
            args.model_sha256 != SHARED.MODEL_SHA256 or
            SHARED.sha256(SHARED.TOKENIZER) != SHARED.TOKENIZER_SHA256):
        raise ValueError("candidate binary changed")
    if out.exists() or not str(out).startswith("/home/bmarti44/.local/state/glm52-"):
        raise ValueError("unsafe or existing output directory")
    out.mkdir(mode=0o700, parents=True)
    trace = out / "trace"
    trace.mkdir(mode=0o700)
    result_path = out / "result.json"
    server_log = out / "server.log"
    large_corpus = args.corpus_smoke or args.quality_corpus
    command = SHARED.server_command(
        binary, args.port,
        cache_experts=(CORPUS_CACHE_EXPERTS if large_corpus else "40GB"),
    )
    ledger_bundle = (
        build_quality_case_ledger(Path(args.candidate), seed=args.seed)
        if args.quality_corpus else None
    )
    if ledger_bundle is not None and args.quality_probe:
        ledger_bundle = quality_probe_ledger(
            ledger_bundle, case_id=args.quality_probe_case,
        )
    if ledger_bundle is not None:
        ledger_path = out / "ledger.json"
        ledger_path.write_text(
            json.dumps(
                {key: value for key, value in ledger_bundle.items() if key != "_prompts"},
                sort_keys=True, indent=2, allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
    server = None
    with server_log.open("xb") as log:
        try:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=False)
            SHARED.wait_ready(server, args.port)
            request_records: list[dict[str, Any]] = []
            if ledger_bundle is not None:
                request_records = _run_quality_requests(
                    port=args.port, out=out, server_log=server_log, log=log,
                    ledger_bundle=ledger_bundle,
                )
            for request_index in range(0 if args.quality_corpus else (2 if args.corpus_smoke else 1)):
                current_result = (out / f"result-{request_index + 1}.json"
                                  if args.corpus_smoke else result_path)
                log.flush()
                os.fsync(log.fileno())
                log_start = server_log.stat().st_size
                completed = subprocess.run([
                    sys.executable, str(SHARED.BENCH),
                    "--base-url", f"http://127.0.0.1:{args.port}",
                    "--out", str(current_result),
                    "--stack-label", f"union-trace-{args.mode}-r{request_index + 1}",
                    "--model-id", "glm-5.2", "--tokenizer-path", str(SHARED.TOKENIZER),
                    "--tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                    "--output-tokenizer-path", str(SHARED.TOKENIZER),
                    "--output-tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                    "--token-timing-log", str(server_log), "--reps", "1", "--warmup", "0",
                    "--context-levels", str(args.context_level), "--max-tokens", "128",
                    "--min-completion-tokens", "128", "--request-timeout", "1200",
                    "--seed", str(args.seed + request_index),
                ], stdin=subprocess.DEVNULL, capture_output=True, timeout=1350, check=False)
                (out / f"bench-{request_index + 1}.stdout.log").write_bytes(completed.stdout)
                (out / f"bench-{request_index + 1}.stderr.log").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark request {request_index + 1} failed rc={completed.returncode}"
                    )
                log.flush()
                os.fsync(log.fileno())
                with server_log.open("rb") as log_reader:
                    log_reader.seek(log_start)
                    request_log = log_reader.read().decode("utf-8", errors="strict")
                signature = SHARED.response_signature(current_result)
                payload = SHARED.strict_json(current_result)
                reps = payload["cells"][0]["reps"]
                prompt_tokens = reps[0].get("prompt_tokens") if len(reps) == 1 else None
                if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or
                        prompt_tokens < max(MIN_PROMPT_TOKENS, args.context_level)):
                    raise ValueError("benchmark prompt-token coverage is insufficient")
                request_records.append({
                    "request_id": request_index + 1,
                    "seed": args.seed + request_index,
                    "prompt_tokens": prompt_tokens,
                    "full_indexed_chunks": full_indexed_chunks_text(request_log),
                    "response_signature": signature,
                    "result_sha256": SHARED.sha256(current_result),
                })
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
            log.flush()
            os.fsync(log.fileno())
    if server is None or server.returncode != 0:
        raise RuntimeError(f"server did not exit cleanly rc={getattr(server, 'returncode', None)}")
    log_text = server_log.read_text(encoding="utf-8", errors="strict")
    SHARED.require_no_gpu_fault(log_text, "server log")
    if args.quality_corpus and log_text.count(UTF8_NORMALIZATION_LOG) != 1:
        raise RuntimeError("UTF-8 normalization activation was not logged exactly once")
    resolved_cuda_cache = cuda_cache_runtime(log_text) if large_corpus else None
    if not request_records:
        raise RuntimeError("arm produced no requests")
    chunks = request_records[0]["full_indexed_chunks"]
    files = [path for path in trace.iterdir()]
    total_trace_bytes = sum(path.stat().st_size for path in files if path.is_file())
    if total_trace_bytes > args.max_trace_bytes:
        raise RuntimeError("trace exceeded its context-derived byte ceiling")
    if args.mode == "off" and files:
        raise RuntimeError("off arm emitted trace files")
    if args.mode == "on" and not files:
        raise RuntimeError("on arm emitted no trace files")
    fixture_digest = (str(ledger_bundle["fixture_content_sha256"])
                      if ledger_bundle is not None else hashlib.sha256(
        b"".join(bytes.fromhex(str(item["response_signature"]["request_sha256"]))
                 for item in request_records)
    ).hexdigest() if args.corpus_smoke else
        str(request_records[0]["response_signature"]["request_sha256"]))
    result_digest = (SHARED.sha256(out / "responses.json")
                     if ledger_bundle is not None else hashlib.sha256(
        b"".join(bytes.fromhex(str(item["result_sha256"])) for item in request_records)
    ).hexdigest() if args.corpus_smoke else str(request_records[0]["result_sha256"]))
    record = {
        "mode": args.mode,
        "binary_sha256": args.binary_sha256,
        "model_sha256": args.model_sha256,
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "fixture_sha256": fixture_digest,
        "configuration_sha256": matched_configuration_sha256(
            corpus_smoke=args.corpus_smoke, quality_corpus=args.quality_corpus,
        ),
        "environment_sha256": configuration_sha256(expected),
        "response_signature": (request_records
                               if args.quality_corpus else
                               [item["response_signature"] for item in request_records]
                               if args.corpus_smoke else request_records[0]["response_signature"]),
        "prompt_tokens": request_records[0]["prompt_tokens"],
        "full_indexed_chunks": chunks,
        "trace_files": len(files),
        "trace_bytes": total_trace_bytes,
        "result_sha256": result_digest,
        "server_log_sha256": SHARED.sha256(server_log),
    }
    if large_corpus:
        record["corpus_requests"] = request_records
        record["expert_cache_budget"] = CORPUS_CACHE_EXPERTS
        record["cuda_expert_cache_gb"] = CORPUS_CUDA_CACHE_GB
        record["cuda_cache_runtime"] = resolved_cuda_cache
    if ledger_bundle is not None:
        record["quality_ledger_sha256"] = SHARED.sha256(out / "ledger.json")
        record["split_plan_sha256"] = ledger_bundle["split_plan_sha256"]
        record["expected_token_layer_events"] = ledger_bundle["expected_token_layer_events"]
    (out / "arm.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.tag) is None:
        raise ValueError("invalid tag")
    if not MIN_PROMPT_TOKENS <= args.context_level <= MAX_CONTEXT_LEVEL:
        raise ValueError("context level is outside the bounded trace range")
    if args.require_multichunk and args.context_level <= 2048:
        raise ValueError("multi-chunk qualification requires context level above 2048")
    if args.require_multichunk and (args.corpus_smoke or args.quality_corpus):
        raise ValueError("corpus and single-request multichunk modes are exclusive")
    if args.corpus_smoke and args.quality_corpus:
        raise ValueError("corpus modes are mutually exclusive")
    if args.quality_probe and not args.quality_corpus:
        raise ValueError("quality probe requires quality corpus mode")
    if args.quality_probe_case is not None and not args.quality_probe:
        raise ValueError("quality probe case requires quality probe mode")
    if (args.quality_probe_case is not None and
            re.fullmatch(r"case_[0-9]{3}", args.quality_probe_case) is None):
        raise ValueError("quality probe case id is invalid")
    large_corpus = args.corpus_smoke or args.quality_corpus
    layer_count = 75 if large_corpus else 1
    request_count = 1 if args.quality_probe else QUALITY_REQUEST_COUNT if args.quality_corpus else 2 if args.corpus_smoke else 1
    per_request_tokens = (
        QUALITY_DISK_MAX_TOKENS if args.quality_corpus else args.context_level + 1024
    )
    max_trace_bytes = per_request_tokens * TRACE_BYTES_PER_TOKEN_LAYER * layer_count * request_count
    freeze_path = QUALITY_FREEZE if args.quality_corpus else CORPUS_FREEZE if args.corpus_smoke else FREEZE
    randomness_path = (
        QUALITY_RANDOMNESS if args.quality_corpus else
        CORPUS_RANDOMNESS if args.corpus_smoke else RANDOMNESS
    )
    freeze = SHARED.frozen_inputs(freeze_path, randomness_path)
    validate_randomness_order(freeze_path, randomness_path)
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    quality_ledger = (
        build_quality_case_ledger(candidate, seed=int(freeze["seed"]))
        if args.quality_corpus else None
    )
    if quality_ledger is not None and args.quality_probe:
        quality_ledger = quality_probe_ledger(
            quality_ledger, case_id=args.quality_probe_case,
        )
    SHARED.no_other_inference()
    available_gib = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))) / 1048576
    if available_gib < 110:
        raise RuntimeError(f"only {available_gib:.3f} GiB available")
    root = Path(f"/home/bmarti44/.local/state/glm52-{args.tag}")
    if root.exists():
        raise FileExistsError(root)
    usage = shutil.disk_usage(root.parent)
    if usage.free < max_trace_bytes + TRACE_DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient trace disk space plus preservation reserve")
    root.mkdir(mode=0o700, parents=True)

    containment: dict[str, dict[str, Any]] = {}
    final_artifacts_by_mode: dict[str, list[Path]] = {}
    for index, mode in enumerate(("off", "on")):
        out = root / mode
        values = trace_environment(
            mode, out, corpus_smoke=args.corpus_smoke,
            quality_corpus=args.quality_corpus,
        )
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        final_artifacts = [str(out / "arm.json")]
        if args.corpus_smoke:
            final_artifacts.extend(str(out / f"result-{request_id}.json")
                                   for request_id in (1, 2))
        elif args.quality_corpus:
            final_artifacts.extend((str(out / "ledger.json"), str(out / "responses.json")))
        else:
            final_artifacts.append(str(out / "result.json"))
        final_artifacts.append(str(out / "server.log"))
        final_artifacts_by_mode[mode] = [Path(path) for path in final_artifacts]
        environment.update({
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": str(freeze["binary_sha256"]),
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(ENV_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": configuration_sha256(values),
            "GLM_SAFE_MEMORY_HIGH_GIB": (
                CORPUS_MEMORY_HIGH_GIB if large_corpus else "69"
            ),
            "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "7200" if args.quality_corpus else "3600",
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join(final_artifacts),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-{mode}", "--", sys.executable,
            str(Path(__file__).resolve()), "_arm", "--mode", mode,
            "--out", str(out), "--binary", str(binary),
            "--binary-sha256", str(freeze["binary_sha256"]),
            "--model-sha256", str(freeze["model_sha256"]),
            "--seed", str(freeze["seed"]), "--port", str(args.port + index),
            "--candidate", str(candidate),
            "--context-level", str(args.context_level),
            "--max-trace-bytes", str(max_trace_bytes),
            *(["--require-multichunk"] if args.require_multichunk else []),
            *(["--corpus-smoke"] if args.corpus_smoke else []),
            *(["--quality-corpus"] if args.quality_corpus else []),
            *(["--quality-probe"] if args.quality_probe else []),
            *(["--quality-probe-case", args.quality_probe_case]
              if args.quality_probe_case is not None else []),
        ], env=environment, stdin=subprocess.DEVNULL, capture_output=True,
           timeout=7300 if args.quality_corpus else 3700, check=False)
        (root / f"{mode}.containment.stdout.log").write_bytes(completed.stdout)
        (root / f"{mode}.containment.stderr.log").write_bytes(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"contained {mode} arm failed rc={completed.returncode}")
        record = SHARED.containment_record(root / f"{mode}.containment.stdout.log")
        containment[mode] = {"clean": True, **record}
        verify_final_artifact_receipts(containment[mode], final_artifacts_by_mode[mode])
        (root / f"{mode}.containment.json").write_text(
            json.dumps(containment[mode], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        SHARED.no_other_inference()

    off = SHARED.strict_json(root / "off/arm.json")
    on = SHARED.strict_json(root / "on/arm.json")
    if args.quality_corpus:
        assert quality_ledger is not None
        expected_ledger_sha256 = SHARED.sha256(root / "off/ledger.json")
        if SHARED.sha256(root / "on/ledger.json") != expected_ledger_sha256:
            raise ValueError("quality arm ledgers differ")
        for mode, arm in (("off", off), ("on", on)):
            if arm.get("quality_ledger_sha256") != expected_ledger_sha256:
                raise ValueError(f"{mode} arm does not bind its quality ledger")
            if arm.get("expected_token_layer_events") != quality_ledger["expected_token_layer_events"]:
                raise ValueError(f"{mode} arm expected event count differs from frozen ledger")
            if arm.get("fixture_sha256") != quality_ledger["fixture_content_sha256"]:
                raise ValueError(f"{mode} arm fixture differs from frozen ledger")
    for mode in ("off", "on"):
        SHARED.require_no_gpu_fault(
            (root / mode / "server.log").read_text(encoding="utf-8", errors="strict"),
            f"{mode} server log",
        )
    if args.quality_corpus:
        assert quality_ledger is not None
        expected_requests = {
            int(item["request_id"]): [tuple(row) for row in item["full_indexed_chunks"]]
            for item in on["corpus_requests"]
        }
        trace_score = TRACE_SCORER.score_trace(
            root / "on/trace", root / "on/server.log", max_bytes=max_trace_bytes,
            expected_layers=set(range(3, 78)), expected_chunks=[],
            expected_requests=expected_requests,
        )
        expected_identity = {
            "binary_sha256": str(freeze["binary_sha256"]),
            "model_sha256": str(freeze["model_sha256"]),
            "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
            "fixture_sha256": quality_ledger["fixture_content_sha256"],
            "split_plan_sha256": quality_ledger["split_plan_sha256"],
            "configuration_sha256": matched_configuration_sha256(quality_corpus=True),
            "off_environment_sha256": configuration_sha256(trace_environment(
                "off", root / "off", quality_corpus=True,
            )),
            "on_environment_sha256": configuration_sha256(trace_environment(
                "on", root / "on", quality_corpus=True,
            )),
        }
        arm_identity_checks = quality_arm_identity_checks(
            off, on, trace_score, expected_identity,
        )
    elif args.corpus_smoke:
        expected_requests = {
            int(item["request_id"]): [tuple(row) for row in item["full_indexed_chunks"]]
            for item in on["corpus_requests"]
        }
        trace_score = TRACE_SCORER.score_trace(
            root / "on/trace", root / "on/server.log", max_bytes=max_trace_bytes,
            expected_layers=set(range(3, 78)), expected_chunks=[],
            expected_requests=expected_requests,
        )
    else:
        expected_chunks = [tuple(row) for row in off["full_indexed_chunks"]]
        trace_score = TRACE_SCORER.score_trace(
            root / "on/trace", root / "on/server.log", max_bytes=max_trace_bytes,
            expected_layers={TRACE_LAYER}, expected_chunks=expected_chunks,
        )
    verdict = (
        quality_capture_verdict(
            quality_ledger["cases"], off["corpus_requests"], on["corpus_requests"],
            trace_score, containment["off"], containment["on"],
        ) if args.quality_corpus and quality_ledger is not None else
        smoke_verdict(
            off, on, trace_score, containment["off"], containment["on"],
            min_prompt_tokens=(2049 if args.require_multichunk else MIN_PROMPT_TOKENS),
            require_multichunk=args.require_multichunk,
            expected_corpus_seed=(int(freeze["seed"]) if args.corpus_smoke else None),
        )
    )
    if args.quality_corpus:
        verdict["checks"].update(arm_identity_checks)
        verdict["verdict"] = "PASS" if all(verdict["checks"].values()) else "FAIL"
        for mode in ("off", "on"):
            verify_final_artifact_receipts(
                containment[mode], final_artifacts_by_mode[mode],
            )
    SHARED.frozen_inputs(freeze_path, randomness_path)
    if args.quality_probe:
        qualification = {
            "scope": "quality_one_case_safety_probe",
            "quality_cases": 1,
            "quality_case_id": quality_ledger["cases"][0]["case_id"],
            "expected_prompt_tokens": quality_ledger["total_expected_prompt_tokens"],
            "expected_token_layer_events": quality_ledger["expected_token_layer_events"],
            "fixture_content_sha256": quality_ledger["fixture_content_sha256"],
            "split_plan_sha256": quality_ledger["split_plan_sha256"],
        }
    elif args.quality_corpus:
        qualification = {
            "scope": "quality_100_case_all_routed_layer_corpus",
            "quality_cases": QUALITY_REQUEST_COUNT,
            "expected_prompt_tokens": quality_ledger["total_expected_prompt_tokens"],
            "expected_token_layer_events": quality_ledger["expected_token_layer_events"],
            "fixture_content_sha256": quality_ledger["fixture_content_sha256"],
            "split_plan_sha256": quality_ledger["split_plan_sha256"],
        }
    elif args.corpus_smoke:
        qualification = {
            "scope": "multi_request_all_routed_layer_corpus_smoke",
            "high_row_2048_status": "OPEN",
            "minimum_token_layer_events": 76800,
        }
    elif args.require_multichunk:
        qualification = {
            "scope": "high_row_multichunk",
            "high_row_2048_status": "PASS" if verdict["verdict"] == "PASS" else "FAIL",
        }
    else:
        qualification = {
            "scope": "short_single_indexed_batch_only",
            "high_row_2048_status": "OPEN",
        }
    summary = {
        "schema_version": 1,
        **qualification,
        "candidate_hash": freeze["candidate_hash"],
        "engine_commit": freeze["engine_commit"],
        "binary_sha256": freeze["binary_sha256"],
        "model_sha256": freeze["model_sha256"],
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "seed": freeze["seed"],
        "context_level": args.context_level,
        "max_trace_bytes": max_trace_bytes,
        "off_arm_sha256": SHARED.sha256(root / "off/arm.json"),
        "on_arm_sha256": SHARED.sha256(root / "on/arm.json"),
        "off_containment_sha256": SHARED.sha256(root / "off.containment.json"),
        "on_containment_sha256": SHARED.sha256(root / "on.containment.json"),
        "trace_score": trace_score,
        **verdict,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if summary["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    public = sub.add_parser("run")
    public.add_argument("--tag", required=True)
    public.add_argument("--port", type=int, default=18090)
    public.add_argument("--context-level", type=int, default=512)
    public.add_argument("--require-multichunk", action="store_true")
    public.add_argument("--corpus-smoke", action="store_true")
    public.add_argument("--quality-corpus", action="store_true")
    public.add_argument("--quality-probe", action="store_true")
    public.add_argument("--quality-probe-case")
    public.set_defaults(func=run)
    internal = sub.add_parser("_arm")
    internal.add_argument("--mode", choices=("off", "on"), required=True)
    internal.add_argument("--out", type=Path, required=True)
    internal.add_argument("--binary", type=Path, required=True)
    internal.add_argument("--candidate", type=Path, required=True)
    internal.add_argument("--binary-sha256", required=True)
    internal.add_argument("--model-sha256", required=True)
    internal.add_argument("--seed", type=int, required=True)
    internal.add_argument("--port", type=int, required=True)
    internal.add_argument("--context-level", type=int, required=True)
    internal.add_argument("--max-trace-bytes", type=int, required=True)
    internal.add_argument("--require-multichunk", action="store_true")
    internal.add_argument("--corpus-smoke", action="store_true")
    internal.add_argument("--quality-corpus", action="store_true")
    internal.add_argument("--quality-probe", action="store_true")
    internal.add_argument("--quality-probe-case")
    internal.set_defaults(func=_arm)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
