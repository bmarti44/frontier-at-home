#!/usr/bin/env python3
"""Compact validated GLM union traces into training-ready P0 records."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import types
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = ROOT / "scripts/75_glm_union_trace_score.py"


def _read_regular_snapshot(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        blocks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"input ended before its observed size: {path}")
            blocks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"input grew while being snapshotted: {path}")
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size) !=
                (after.st_dev, after.st_ino, after.st_size)):
            raise ValueError(f"input identity changed while being snapshotted: {path}")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _load_bytes(name: str, payload: bytes, origin: Path):
    module = types.ModuleType(name)
    module.__file__ = str(origin)
    exec(compile(payload, str(origin), "exec"), module.__dict__)
    return module


SCORER_SOURCE = _read_regular_snapshot(SCORER_PATH)
SCORER_SHA256 = hashlib.sha256(SCORER_SOURCE).hexdigest()
TRACE_SCORER = _load_bytes("union_trace_scorer_for_compact", SCORER_SOURCE, SCORER_PATH)
N_EMBD = 6144
N_EXPERT = 256
N_EXPERT_USED = 8
TOP_K = 32
HIDDEN_GROUP_SIZE = 32
CORPUS_MIN_TOKEN_LAYER_EVENTS = 76800
FILE_RE = re.compile(
    r"^(?P<prefix>.+)_glm_indexed_(?P<kind>ffn_norm|router_logits|router_probs|router_selected|router_bias)-"
    r"(?P<layer>\d+)_pos(?P<pos>\d+)\.(?P<ext>f32|i32)$"
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SCORER_CHECKS = {
    "inputs", "no_trace_errors", "unique_nonempty_log_events",
    "exact_indexed_chunk_coverage", "regular_files_only", "recognized_files_only",
    "one_prefix", "unique_file_keys", "byte_budget", "event_keys_match",
    "exact_triplet_shapes", "finite_values_and_valid_ids",
    "selected_matches_router_formula", "router_probs_match_logits",
}
CORPUS_SCORER_CHECKS = SCORER_CHECKS | {
    "recognized_log_events_only", "router_bias_constant_per_layer", "utf8_server_log",
}
REQUEST_PREFIX_RE = re.compile(r"^request_r(?P<request>[0-9]{8})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"JSON input is not UTF-8: {label}") from error
    value = json.loads(
        text, object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_read_regular_snapshot(path), str(path))


def unpack_hidden_int4(packed: np.ndarray, scale: np.ndarray, width: int) -> np.ndarray:
    if (packed.ndim != 2 or scale.ndim != 2 or packed.shape[0] != scale.shape[0] or
            width <= 0 or packed.shape[1] != (width + 1) // 2):
        raise ValueError("malformed packed hidden feature")
    unpacked = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    unpacked[:, 0::2] = (packed & np.uint8(0x0f)).astype(np.int8) - 8
    unpacked[:, 1::2] = (packed >> np.uint8(4)).astype(np.int8) - 8
    group_size = (width + scale.shape[1] - 1) // scale.shape[1]
    expanded_scale = np.repeat(scale.astype(np.float32), group_size, axis=1)[:, :width]
    return unpacked[:, :width].astype(np.float32) * expanded_scale


def _pack_hidden_int4(hidden: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    if group_size <= 0 or hidden.shape[1] % group_size:
        raise ValueError("hidden width must be divisible by the quantization group size")
    grouped = hidden.reshape(hidden.shape[0], -1, group_size)
    absmax = np.max(np.abs(grouped), axis=2)
    scale32 = np.where(absmax > 0.0, absmax / 7.0, 1.0).astype(np.float32)
    scale = scale32.astype(np.float16)
    if np.any((absmax > 0.0) & (scale == 0.0)) or not np.isfinite(scale).all():
        raise ValueError("hidden int4 scale is not representable in fp16")
    quantized = np.clip(
        np.rint(grouped / scale.astype(np.float32)[:, :, None]), -7, 7,
    ).astype(np.int8).reshape(hidden.shape)
    if quantized.shape[1] % 2:
        quantized = np.pad(quantized, ((0, 0), (0, 1)))
    low = (quantized[:, 0::2] + 8).astype(np.uint8)
    high = ((quantized[:, 1::2] + 8).astype(np.uint8) << np.uint8(4))
    return low | high, scale


def compact_arrays(
    hidden: np.ndarray,
    logits: np.ndarray,
    bias: np.ndarray,
    selected: np.ndarray,
    *,
    top_k: int = TOP_K,
    router_probs: np.ndarray | None = None,
    hidden_group_size: int = HIDDEN_GROUP_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Return compact arrays and measured representation error."""
    if router_probs is None:
        raise ValueError("captured router probabilities are mandatory")
    arrays = (hidden, logits, bias, selected, router_probs)
    if any(not isinstance(value, np.ndarray) for value in arrays):
        raise ValueError("all inputs must be numpy arrays")
    if (hidden.ndim != 2 or logits.ndim != 2 or bias.ndim != 1 or selected.ndim != 2 or
            hidden.shape[0] == 0 or logits.shape[0] != hidden.shape[0] or
            bias.shape[0] != logits.shape[1] or selected.shape[0] != hidden.shape[0] or
            not 1 <= top_k <= logits.shape[1] or selected.shape[1] > top_k):
        raise ValueError("trace array shapes are inconsistent")
    if router_probs.shape != logits.shape:
        raise ValueError("captured router probability shape is inconsistent")
    if (not np.issubdtype(hidden.dtype, np.floating) or
            not np.issubdtype(logits.dtype, np.floating) or
            not np.issubdtype(bias.dtype, np.floating) or
            not np.issubdtype(selected.dtype, np.integer) or
            not np.issubdtype(router_probs.dtype, np.floating)):
        raise ValueError("trace array dtypes are invalid")
    finite_arrays = (hidden, logits, bias, router_probs)
    if not all(np.isfinite(value).all() for value in finite_arrays):
        raise ValueError("trace contains non-finite values")
    n_expert = logits.shape[1]
    if n_expert > 256 or np.any(selected < 0) or np.any(selected >= n_expert):
        raise ValueError("selected expert id is outside uint8 range")
    if any(np.unique(row).size != row.size for row in selected):
        raise ValueError("selected expert ids are duplicated")

    logits32 = logits.astype(np.float32, copy=False)
    bias32 = bias.astype(np.float32, copy=False)
    probabilities = np.empty_like(logits32)
    positive = logits32 >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits32[positive]))
    exponent = np.exp(logits32[~positive])
    probabilities[~positive] = exponent / (1.0 + exponent)
    captured_probs = router_probs.astype(np.float32, copy=False)
    probability_delta = float(np.max(np.abs(probabilities - captured_probs)))
    if probability_delta > 1e-4 or np.any(captured_probs < 0.0) or np.any(captured_probs > 1.0):
        raise ValueError("captured router probabilities do not match logits")
    probabilities = captured_probs
    effective_scores = np.add(probabilities, bias32[None, :], dtype=np.float32)
    top_ids = np.argsort(-effective_scores, axis=1, kind="stable")[:, :top_k]
    expected_selected = top_ids[:, :selected.shape[1]]
    if not np.array_equal(expected_selected, selected):
        raise ValueError("selected ids do not match the effective router ordering")
    top_logits32 = np.take_along_axis(logits32, top_ids, axis=1)
    top_logits = top_logits32.astype(np.float16)
    if not np.isfinite(top_logits).all():
        raise ValueError("top logits overflow fp16")

    hidden32 = hidden.astype(np.float32, copy=False)
    effective_group_size = min(hidden_group_size, hidden.shape[1])
    packed, scale = _pack_hidden_int4(hidden32, effective_group_size)
    restored = unpack_hidden_int4(packed, scale, hidden.shape[1])
    error = restored - hidden32
    denominator = float(np.sum(hidden32.astype(np.float64) ** 2))
    numerator = float(np.sum(error.astype(np.float64) ** 2))
    metrics: dict[str, int | float] = {
        "rows": int(hidden.shape[0]),
        "hidden_values": int(hidden.size),
        "hidden_group_size": effective_group_size,
        "hidden_max_abs_error": float(np.max(np.abs(error))),
        "hidden_rmse": math.sqrt(numerator / hidden.size),
        "hidden_nrmse": math.sqrt(numerator / denominator) if denominator else 0.0,
        "top_logit_max_abs_error": float(np.max(np.abs(top_logits.astype(np.float32) - top_logits32))),
        "router_probability_max_abs_error": probability_delta,
    }
    return {
        "selected_ids": selected.astype(np.uint8),
        "top_ids": top_ids.astype(np.uint8),
        "top_logits": top_logits,
        "hidden_q4": packed,
        "hidden_scale": scale,
    }, metrics


def _valid_chunks(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("qualified source has no chunks")
    result: list[tuple[int, int]] = []
    previous_end = 0
    for item in value:
        if (not isinstance(item, list) or len(item) != 2 or
                any(not isinstance(number, int) or isinstance(number, bool) for number in item)):
            raise ValueError("qualified source chunk schema is invalid")
        pos, rows = item
        if pos != previous_end or rows <= 0:
            raise ValueError("qualified source chunks are not ordered and contiguous")
        result.append((pos, rows))
        previous_end = pos + rows
    return result


def _require_tracked_snapshot(
    path: Path, repository_root: Path, *, required_prefix: str | None = None,
) -> bytes:
    repository_root = repository_root.resolve(strict=True)
    snapshot = _read_regular_snapshot(path)
    try:
        relative = path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("trusted input is outside the repository") from error
    if required_prefix is not None and not str(relative).startswith(required_prefix):
        raise ValueError("trusted input is outside its required repository directory")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=repository_root, stdin=subprocess.DEVNULL, capture_output=True,
    )
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repository_root,
        stdin=subprocess.DEVNULL, capture_output=True,
    )
    if (tracked.returncode != 0 or committed.returncode != 0 or
            committed.stdout != snapshot):
        raise ValueError("trusted input is not tracked and clean at HEAD")
    return snapshot


def _require_tracked_receipt(receipt_path: Path, repository_root: Path) -> bytes:
    return _require_tracked_snapshot(
        receipt_path, repository_root, required_prefix="results/glm52-gates/",
    )


def _repository_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
    )
    value = completed.stdout.strip()
    if not COMMIT_RE.fullmatch(value):
        raise ValueError("repository HEAD is malformed")
    return value


def _review_is_accepted(value: Any) -> bool:
    return (
        isinstance(value, dict) and
        set(value) == {
            "round", "gap_reviewer_score", "adversarial_reviewer_score", "critical", "high",
        } and
        isinstance(value["round"], int) and not isinstance(value["round"], bool) and
        value["round"] >= 1 and
        all(
            isinstance(value[key], int) and not isinstance(value[key], bool) and
            90 <= value[key] <= 100
            for key in ("gap_reviewer_score", "adversarial_reviewer_score")
        ) and value["critical"] == [] and value["high"] == []
    )


def _validate_corpus_source_bundle(
    source_root: Path,
    receipt_path: Path,
    receipt_bytes: bytes,
    receipt: dict[str, Any],
    repository_head: str,
    request_index: int | None,
    minimum_prompt_tokens: int,
) -> dict[str, Any]:
    """Validate the complete qualified corpus, then expose exactly one request shard."""
    if not isinstance(request_index, int) or isinstance(request_index, bool) or request_index <= 0:
        raise ValueError("a positive request index is required for a corpus source")
    trace = source_root / "on" / "trace"
    control_paths = {
        "summary": source_root / "summary.json",
        "on_arm": source_root / "on" / "arm.json",
        "off_arm": source_root / "off" / "arm.json",
        "on_server_log": source_root / "on" / "server.log",
        "off_server_log": source_root / "off" / "server.log",
        "off_containment": source_root / "off.containment.json",
        "on_containment": source_root / "on.containment.json",
    }
    for mode in ("off", "on"):
        for index in (1, 2):
            control_paths[f"{mode}_result_{index}"] = source_root / mode / f"result-{index}.json"
    if not trace.is_dir() or trace.is_symlink():
        raise ValueError("qualified corpus trace layout is invalid")
    controls = {name: _read_regular_snapshot(path) for name, path in control_paths.items()}
    summary = _strict_json_bytes(controls["summary"], str(control_paths["summary"]))
    arms = {
        mode: _strict_json_bytes(controls[f"{mode}_arm"], str(control_paths[f"{mode}_arm"]))
        for mode in ("off", "on")
    }
    containments = {
        mode: _strict_json_bytes(
            controls[f"{mode}_containment"], str(control_paths[f"{mode}_containment"]),
        ) for mode in ("off", "on")
    }
    receipt_keys = {
        "schema_version", "candidate_hash", "engine_commit", "classification", "scope",
        "high_row_2048_status", "summary_sha256", "off_arm_sha256", "on_arm_sha256",
        "off_result_1_sha256", "off_result_2_sha256", "on_result_1_sha256",
        "on_result_2_sha256", "off_server_log_sha256", "on_server_log_sha256",
        "off_containment_sha256", "on_containment_sha256", "observed",
        "pre_runtime_authorization_review", "post_runtime_review", "retained_directory",
        "conclusion",
    }
    summary_keys = {
        "schema_version", "scope", "high_row_2048_status", "candidate_hash",
        "engine_commit", "binary_sha256", "model_sha256", "tokenizer_sha256", "seed",
        "context_level", "max_trace_bytes", "minimum_token_layer_events",
        "off_arm_sha256", "on_arm_sha256", "off_containment_sha256",
        "on_containment_sha256", "trace_score", "checks", "verdict",
    }
    if set(receipt) != receipt_keys or set(summary) != summary_keys:
        raise ValueError("corpus receipt or summary schema differs from the qualified runner")
    if (
        receipt.get("classification") != "PASS" or summary.get("verdict") != "PASS" or
        receipt.get("scope") != "multi_request_all_routed_layer_corpus_smoke" or
        summary.get("scope") != receipt.get("scope") or
        receipt.get("high_row_2048_status") != summary.get("high_row_2048_status") or
        minimum_prompt_tokens <= 0 or
        not _review_is_accepted(receipt.get("pre_runtime_authorization_review")) or
        not _review_is_accepted(receipt.get("post_runtime_review"))
    ):
        raise ValueError("corpus bundle is not qualified")

    bindings = {
        "summary_sha256": _sha256_bytes(controls["summary"]),
        "off_arm_sha256": _sha256_bytes(controls["off_arm"]),
        "on_arm_sha256": _sha256_bytes(controls["on_arm"]),
        "off_server_log_sha256": _sha256_bytes(controls["off_server_log"]),
        "on_server_log_sha256": _sha256_bytes(controls["on_server_log"]),
        "off_containment_sha256": _sha256_bytes(controls["off_containment"]),
        "on_containment_sha256": _sha256_bytes(controls["on_containment"]),
    }
    for mode in ("off", "on"):
        for index in (1, 2):
            bindings[f"{mode}_result_{index}_sha256"] = _sha256_bytes(
                controls[f"{mode}_result_{index}"]
            )
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise ValueError("corpus receipt does not bind every control artifact")
    if any(summary.get(key) != bindings[key] for key in (
            "off_arm_sha256", "on_arm_sha256", "off_containment_sha256",
            "on_containment_sha256")):
        raise ValueError("corpus summary does not bind its arms and containment evidence")
    if (
        receipt.get("candidate_hash") != summary.get("candidate_hash") or
        receipt.get("engine_commit") != summary.get("engine_commit") or
        not COMMIT_RE.fullmatch(str(summary.get("candidate_hash", ""))) or
        not COMMIT_RE.fullmatch(str(summary.get("engine_commit", "")))
    ):
        raise ValueError("corpus candidate lineage differs")
    for key in ("binary_sha256", "model_sha256", "tokenizer_sha256"):
        if (
            not HEX64_RE.fullmatch(str(summary.get(key, ""))) or
            any(arm.get(key) != summary.get(key) for arm in arms.values())
        ):
            raise ValueError(f"corpus {key} lineage differs")

    corpus_requests: dict[str, dict[int, dict[str, Any]]] = {}
    for mode, arm in arms.items():
        rows = arm.get("corpus_requests")
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError("corpus arm does not contain exactly two requests")
        indexed: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                    "request_id", "seed", "prompt_tokens", "full_indexed_chunks",
                    "response_signature", "result_sha256"}:
                raise ValueError("corpus request schema is malformed")
            index = row.get("request_id")
            if not isinstance(index, int) or isinstance(index, bool) or index in indexed:
                raise ValueError("corpus request id is malformed or duplicated")
            chunks = _valid_chunks(row.get("full_indexed_chunks"))
            response = row.get("response_signature")
            if (
                row.get("prompt_tokens") != sum(count for _, count in chunks) or
                row.get("prompt_tokens", 0) < minimum_prompt_tokens or
                not isinstance(response, dict) or
                not HEX64_RE.fullmatch(str(response.get("request_sha256", ""))) or
                row.get("result_sha256") != bindings.get(f"{mode}_result_{index}_sha256")
            ):
                raise ValueError("corpus request coverage or binding differs")
            indexed[index] = row
        corpus_requests[mode] = indexed
    request_ids = set(corpus_requests["off"])
    if request_ids != {1, 2} or set(corpus_requests["on"]) != request_ids:
        raise ValueError("corpus request set differs")
    if request_index not in request_ids:
        raise ValueError("requested corpus shard is not qualified")
    for index in sorted(request_ids):
        left, right = corpus_requests["off"][index], corpus_requests["on"][index]
        if (
            left["seed"] != right["seed"] or
            left["prompt_tokens"] != right["prompt_tokens"] or
            left["full_indexed_chunks"] != right["full_indexed_chunks"] or
            left["response_signature"] != right["response_signature"]
        ):
            raise ValueError("corpus OFF/ON request fixtures or outputs differ")
    request_hashes = {
        corpus_requests["on"][index]["response_signature"]["request_sha256"]
        for index in request_ids
    }
    if len(request_hashes) != len(request_ids):
        raise ValueError("corpus request fixtures are not distinct")
    on_arm, off_arm = arms["on"], arms["off"]
    common_hashes = (
        "binary_sha256", "model_sha256", "tokenizer_sha256", "fixture_sha256",
        "configuration_sha256",
    )
    expected_response_list = [
        corpus_requests["on"][index]["response_signature"] for index in sorted(request_ids)
    ]
    expected_chunks_list = [
        corpus_requests["on"][index]["full_indexed_chunks"] for index in sorted(request_ids)
    ]
    expected_prompt_tokens = [
        corpus_requests["on"][index]["prompt_tokens"] for index in sorted(request_ids)
    ]
    cache = on_arm.get("cuda_cache_runtime")
    cache_arena = cache.get("arena_gib") if isinstance(cache, dict) else None
    cache_ok = (
        off_arm.get("expert_cache_budget") == on_arm.get("expert_cache_budget") == "32GB" and
        off_arm.get("cuda_expert_cache_gb") == on_arm.get("cuda_expert_cache_gb") == "56" and
        isinstance(cache, dict) and cache == off_arm.get("cuda_cache_runtime") and
        cache.get("slots") == 5754 and isinstance(cache.get("arena_gib"), (int, float)) and
        not isinstance(cache_arena, bool) and math.isfinite(float(cache_arena)) and
        0.0 < float(cache_arena) <= 56.0
    )

    trace_score = summary.get("trace_score")
    if (
        not isinstance(trace_score, dict) or trace_score.get("verdict") != "PASS" or
        not isinstance(trace_score.get("checks"), dict) or
        set(trace_score["checks"]) != CORPUS_SCORER_CHECKS or
        not all(value is True for value in trace_score["checks"].values())
    ):
        raise ValueError("corpus fixed-scorer verdict is not PASS")
    artifact_rows = trace_score.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise ValueError("corpus scorer has no artifacts")
    artifacts: dict[str, dict[str, Any]] = {}
    keys: set[tuple[int, int, int, str]] = set()
    for item in artifact_rows:
        if (
            not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"} or
            not isinstance(item["name"], str) or item["name"] in artifacts or
            not isinstance(item["bytes"], int) or item["bytes"] < 0 or
            not HEX64_RE.fullmatch(str(item["sha256"]))
        ):
            raise ValueError("corpus artifact receipt is malformed")
        match = FILE_RE.fullmatch(item["name"])
        prefix = REQUEST_PREFIX_RE.fullmatch(match.group("prefix")) if match else None
        if match is None or prefix is None:
            raise ValueError("corpus artifact name is unrecognized")
        expected_extension = "i32" if match.group("kind") == "router_selected" else "f32"
        if match.group("ext") != expected_extension:
            raise ValueError("corpus artifact extension is invalid for its kind")
        key = (
            int(prefix.group("request")), int(match.group("layer")),
            int(match.group("pos")), match.group("kind"),
        )
        if key in keys:
            raise ValueError("corpus artifact key is duplicated")
        keys.add(key)
        artifacts[item["name"]] = item
    if {key[0] for key in keys} != request_ids:
        raise ValueError("corpus trace request prefixes differ")
    observed_paths = list(trace.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in observed_paths) or
        {path.name for path in observed_paths} != set(artifacts)
    ):
        raise ValueError("corpus trace file set is not exact and regular")
    observed = receipt.get("observed")
    first_layer = observed.get("routed_layer_first") if isinstance(observed, dict) else None
    last_layer = observed.get("routed_layer_last") if isinstance(observed, dict) else None
    if first_layer != 3 or last_layer != 77:
        raise ValueError("corpus routed layer range is malformed")
    layers = list(range(3, 78))
    wanted_kinds = {"ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias"}
    request_chunks = {
        index: _valid_chunks(corpus_requests["on"][index]["full_indexed_chunks"])
        for index in request_ids
    }
    expected_keys = {
        (index, layer, pos, kind)
        for index in request_ids for layer in layers
        for pos, _ in request_chunks[index] for kind in wanted_kinds
    }
    if keys != expected_keys:
        raise ValueError("corpus request/layer/chunk tensor coverage is incomplete")
    expected_events = sum(len(layers) * len(request_chunks[index]) for index in request_ids)
    expected_rows = sum(
        len(layers) * sum(rows for _, rows in request_chunks[index]) for index in request_ids
    )
    total_bytes = sum(item["bytes"] for item in artifacts.values())
    qualified_floor = summary.get("minimum_token_layer_events")
    if (
        trace_score.get("events") != expected_events or
        trace_score.get("total_rows") != expected_rows or
        trace_score.get("token_layer_events") != expected_rows or
        trace_score.get("total_bytes") != total_bytes or
        on_arm.get("trace_files") != len(artifacts) or on_arm.get("trace_bytes") != total_bytes or
        not isinstance(qualified_floor, int) or isinstance(qualified_floor, bool) or
        qualified_floor < CORPUS_MIN_TOKEN_LAYER_EVENTS or expected_rows < qualified_floor
    ):
        raise ValueError("corpus scorer totals are inconsistent")
    with tempfile.TemporaryDirectory(prefix="glm52-corpus-score.") as snapshot_directory:
        score_root = Path(snapshot_directory)
        score_trace = score_root / "trace"
        score_trace.mkdir()
        for path in observed_paths:
            item = artifacts[path.name]
            payload = _read_regular_snapshot(path)
            if len(payload) != item["bytes"] or _sha256_bytes(payload) != item["sha256"]:
                raise ValueError("corpus trace artifact differs from scorer receipt")
            (score_trace / path.name).write_bytes(payload)
        score_log = score_root / "server.log"
        score_log.write_bytes(controls["on_server_log"])
        rescored = TRACE_SCORER.score_trace(
            score_trace, score_log, max_bytes=summary.get("max_trace_bytes", 0),
            expected_layers=set(layers), expected_chunks=[],
            expected_requests={index: request_chunks[index] for index in request_ids},
        )
    if rescored != trace_score:
        raise ValueError("fixed scorer does not reproduce the qualified corpus")

    recomputed_checks = {
        "arm_modes": off_arm.get("mode") == "off" and on_arm.get("mode") == "on",
        "frozen_identity": all(off_arm.get(key) == on_arm.get(key) for key in common_hashes),
        "byte_and_token_identity": (
            off_arm.get("response_signature") == on_arm.get("response_signature") ==
            expected_response_list
        ),
        "matched_indexed_chunks": (
            off_arm.get("full_indexed_chunks") == on_arm.get("full_indexed_chunks") ==
            expected_chunks_list[0] and len({repr(value) for value in expected_chunks_list}) == 1
        ),
        "prompt_tokens_and_exact_coverage": (
            off_arm.get("prompt_tokens") == on_arm.get("prompt_tokens") == expected_prompt_tokens[0] and
            len(set(expected_prompt_tokens)) == 1
        ),
        "off_emitted_no_trace": off_arm.get("trace_files") == 0 and off_arm.get("trace_bytes") == 0,
        "on_emitted_trace": on_arm.get("trace_files", 0) > 0 and on_arm.get("trace_bytes", 0) > 0,
        "trace_score_passed": rescored.get("verdict") == "PASS",
        "containment_clean": all(value.get("clean") is True for value in containments.values()),
        "corpus_request_scope": len(request_ids) == 2 and len(request_hashes) == 2,
        "corpus_event_floor": expected_rows >= qualified_floor,
        "corpus_cuda_cache": cache_ok,
    }
    if summary.get("checks") != recomputed_checks or not all(recomputed_checks.values()):
        raise ValueError("corpus top-level OFF/ON qualification does not reproduce")
    observed_keys = {
        "context_level", "requests", "prompt_tokens_per_request",
        "completion_tokens_per_request", "full_indexed_chunks_per_request",
        "distinct_request_fixtures", "byte_and_token_identity", "containment_clean",
        "streaming_cache_budget", "cuda_cache_environment_gb", "cuda_cache_slots",
        "cuda_cache_arena_gib", "off_trace_files", "on_trace_files", "on_trace_bytes",
        "trace_events", "token_layer_events", "routed_layer_first", "routed_layer_last",
        "minimum_available_memory_gib", "maximum_cgroup_memory_bytes",
        "maximum_cgroup_swap_bytes", "kernel_oom_or_xid", "trace_score_verdict",
    }
    available_memory = observed.get("minimum_available_memory_gib")
    cgroup_memory = observed.get("maximum_cgroup_memory_bytes")
    safe_available_memory = (
        isinstance(available_memory, dict) and set(available_memory) == {"off", "on"} and
        all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)) and float(value) >= 18.0
            for value in available_memory.values()
        )
    )
    safe_cgroup_memory = (
        isinstance(cgroup_memory, dict) and set(cgroup_memory) == {"off", "on"} and
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in cgroup_memory.values())
    )
    if (
        set(observed) != observed_keys or
        observed.get("context_level") != summary.get("context_level") or
        observed.get("requests") != len(request_ids) or
        observed.get("prompt_tokens_per_request") != expected_prompt_tokens or
        observed.get("completion_tokens_per_request") != [
            row["completion_tokens"] for row in expected_response_list
        ] or observed.get("full_indexed_chunks_per_request") != expected_chunks_list or
        observed.get("distinct_request_fixtures") != len(request_hashes) or
        observed.get("byte_and_token_identity") is not True or
        observed.get("containment_clean") is not True or
        observed.get("streaming_cache_budget") != "32GB" or
        observed.get("cuda_cache_environment_gb") != 56 or
        observed.get("cuda_cache_slots") != cache.get("slots") or
        observed.get("cuda_cache_arena_gib") != cache_arena or
        observed.get("off_trace_files") != 0 or
        observed.get("on_trace_files") != len(artifacts) or
        observed.get("on_trace_bytes") != total_bytes or
        observed.get("trace_events") != expected_events or
        observed.get("token_layer_events") != expected_rows or
        observed.get("routed_layer_first") != first_layer or
        observed.get("routed_layer_last") != last_layer or
        not safe_available_memory or not safe_cgroup_memory or
        observed.get("maximum_cgroup_swap_bytes") != 0 or
        observed.get("kernel_oom_or_xid") is not False or
        observed.get("trace_score_verdict") != "PASS"
    ):
        raise ValueError("corpus receipt observations do not reproduce")

    selected_files: dict[tuple[int, int, str], Path] = {}
    for name in artifacts:
        match = FILE_RE.fullmatch(name)
        prefix = REQUEST_PREFIX_RE.fullmatch(match.group("prefix")) if match else None
        if prefix and int(prefix.group("request")) == request_index:
            selected_files[(int(match.group("layer")), int(match.group("pos")), match.group("kind"))] = trace / name
    for layer in layers:
        bias_hashes = {
            artifacts[selected_files[(layer, pos, "router_bias")].name]["sha256"]
            for pos, _ in request_chunks[request_index]
        }
        if len(bias_hashes) != 1:
            raise ValueError("router bias differs across chunks of selected request layer")
    selected_request = corpus_requests["on"][request_index]
    lineage = {
        "source_receipt_sha256": _sha256_bytes(receipt_bytes),
        "source_summary_sha256": bindings["summary_sha256"],
        "source_arm_sha256": bindings["on_arm_sha256"],
        "source_server_log_sha256": bindings["on_server_log_sha256"],
        "candidate_hash": summary.get("candidate_hash"),
        "engine_commit": summary.get("engine_commit"),
        "binary_sha256": summary.get("binary_sha256"),
        "model_sha256": summary.get("model_sha256"),
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "fixture_sha256": selected_request["response_signature"]["request_sha256"],
        "configuration_sha256": on_arm.get("configuration_sha256"),
        "request_index": request_index,
        "request_id": selected_request["response_signature"]["request_sha256"],
        "seed": selected_request["seed"],
        "scorer_sha256": SCORER_SHA256,
        "repository_head": repository_head,
    }
    if any(value is None for value in lineage.values()):
        raise ValueError("corpus source lineage is incomplete")
    return {
        "trace": trace, "layers": layers, "chunks": request_chunks[request_index],
        "files": selected_files, "artifacts": artifacts, "lineage": lineage,
    }


def _validate_quality_source_bundle(
    source_root: Path,
    receipt_path: Path,
    receipt_bytes: bytes,
    receipt: dict[str, Any],
    repository_head: str,
) -> dict[str, Any]:
    """Validate the complete 100-case source once and expose all request shards."""
    trace = source_root / "on" / "trace"
    control_paths = {
        "summary": source_root / "summary.json",
        "off_arm": source_root / "off" / "arm.json",
        "on_arm": source_root / "on" / "arm.json",
        "off_ledger": source_root / "off" / "ledger.json",
        "on_ledger": source_root / "on" / "ledger.json",
        "off_responses": source_root / "off" / "responses.json",
        "on_responses": source_root / "on" / "responses.json",
        "off_server_log": source_root / "off" / "server.log",
        "on_server_log": source_root / "on" / "server.log",
        "off_containment": source_root / "off.containment.json",
        "on_containment": source_root / "on.containment.json",
    }
    if not trace.is_dir() or trace.is_symlink():
        raise ValueError("qualified quality trace layout is invalid")
    controls = {name: _read_regular_snapshot(path) for name, path in control_paths.items()}
    documents = {
        name: _strict_json_bytes(payload, str(control_paths[name]))
        for name, payload in controls.items()
        if name not in {"off_server_log", "on_server_log", "off_responses", "on_responses"}
    }
    response_lists = {}
    for mode in ("off", "on"):
        payload = controls[f"{mode}_responses"]
        try:
            value = json.loads(payload.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("quality response input is malformed") from error
        if not isinstance(value, list):
            raise ValueError("quality response input is not a list")
        response_lists[mode] = value
    summary = documents["summary"]
    arms = {mode: documents[f"{mode}_arm"] for mode in ("off", "on")}
    ledgers = {mode: documents[f"{mode}_ledger"] for mode in ("off", "on")}
    containments = {mode: documents[f"{mode}_containment"] for mode in ("off", "on")}
    receipt_keys = {
        "schema_version", "candidate_hash", "engine_commit", "classification", "scope",
        "summary_sha256", "off_arm_sha256", "on_arm_sha256", "off_ledger_sha256",
        "on_ledger_sha256", "off_responses_sha256", "on_responses_sha256",
        "off_server_log_sha256", "on_server_log_sha256", "off_containment_sha256",
        "on_containment_sha256", "crashlog_bindings", "observed",
        "pre_runtime_authorization_review", "post_runtime_review", "retained_directory",
        "conclusion",
    }
    summary_keys = {
        "schema_version", "scope", "candidate_hash", "engine_commit", "binary_sha256",
        "model_sha256", "tokenizer_sha256", "seed", "context_level", "max_trace_bytes",
        "quality_cases", "expected_prompt_tokens", "expected_token_layer_events",
        "fixture_content_sha256", "split_plan_sha256", "off_arm_sha256", "on_arm_sha256",
        "off_containment_sha256", "on_containment_sha256", "trace_score", "checks", "verdict",
    }
    if set(receipt) != receipt_keys or set(summary) != summary_keys:
        raise ValueError("quality receipt or summary schema differs from the qualified runner")
    if (
        receipt.get("classification") != "PASS" or summary.get("verdict") != "PASS" or
        receipt.get("scope") != summary.get("scope") or
        summary.get("scope") != "quality_100_case_all_routed_layer_corpus" or
        summary.get("quality_cases") != 100 or
        not _review_is_accepted(receipt.get("pre_runtime_authorization_review")) or
        not _review_is_accepted(receipt.get("post_runtime_review"))
    ):
        raise ValueError("quality corpus bundle is not qualified")
    bindings = {
        f"{name}_sha256": _sha256_bytes(payload)
        for name, payload in controls.items()
    }
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise ValueError("quality receipt does not bind every control artifact")
    if any(summary.get(key) != bindings[key] for key in (
            "off_arm_sha256", "on_arm_sha256", "off_containment_sha256",
            "on_containment_sha256")):
        raise ValueError("quality summary does not bind its arms and containment evidence")
    if (
        receipt.get("candidate_hash") != summary.get("candidate_hash") or
        receipt.get("engine_commit") != summary.get("engine_commit") or
        not COMMIT_RE.fullmatch(str(summary.get("candidate_hash", ""))) or
        not COMMIT_RE.fullmatch(str(summary.get("engine_commit", "")))
    ):
        raise ValueError("quality candidate lineage differs")
    common = (
        "binary_sha256", "model_sha256", "tokenizer_sha256", "fixture_sha256",
        "split_plan_sha256", "configuration_sha256", "quality_ledger_sha256",
        "expected_token_layer_events",
    )
    if any(
        arms["off"].get(key) != arms["on"].get(key)
        for key in common
    ) or any(not HEX64_RE.fullmatch(str(summary.get(key, ""))) for key in (
            "binary_sha256", "model_sha256", "tokenizer_sha256",
            "fixture_content_sha256", "split_plan_sha256")):
        raise ValueError("quality arm identity differs")
    if any(arms[mode].get(key) != summary.get(key) for mode in ("off", "on") for key in (
            "binary_sha256", "model_sha256", "tokenizer_sha256", "split_plan_sha256")):
        raise ValueError("quality summary lineage differs from its arms")
    if (
        arms["off"].get("mode") != "off" or arms["on"].get("mode") != "on" or
        controls["off_ledger"] != controls["on_ledger"] or
        controls["off_responses"] != controls["on_responses"] or
        arms["off"].get("corpus_requests") != response_lists["off"] or
        arms["on"].get("corpus_requests") != response_lists["on"] or
        arms["off"].get("response_signature") != response_lists["off"] or
        arms["on"].get("response_signature") != response_lists["on"]
    ):
        raise ValueError("quality OFF/ON ledgers, responses, or modes differ")
    ledger = ledgers["on"]
    cases = ledger.get("cases")
    responses = response_lists["on"]
    if (
        not isinstance(cases, list) or len(cases) != 100 or len(responses) != 100 or
        ledger.get("expected_token_layer_events") != summary.get("expected_token_layer_events") or
        ledger.get("total_expected_prompt_tokens") != summary.get("expected_prompt_tokens") or
        ledger.get("fixture_content_sha256") != summary.get("fixture_content_sha256") or
        ledger.get("split_plan_sha256") != summary.get("split_plan_sha256") or
        ledger.get("tokenizer_sha256") != summary.get("tokenizer_sha256") or
        ledger.get("seed") != summary.get("seed")
    ):
        raise ValueError("quality ledger coverage or lineage differs")
    request_chunks: dict[int, list[tuple[int, int]]] = {}
    request_metadata: dict[int, dict[str, Any]] = {}
    for expected, observed in zip(cases, responses):
        request_id = expected.get("request_id") if isinstance(expected, dict) else None
        if (
            not isinstance(request_id, int) or isinstance(request_id, bool) or
            request_id != len(request_chunks) + 1 or not isinstance(observed, dict) or
            observed.get("request_id") != request_id or
            any(observed.get(key) != expected.get(key) for key in (
                "case_id", "group_id", "split", "request_sha256")) or
            observed.get("prompt_tokens") != expected.get("expected_prompt_tokens") or
            observed.get("completion_tokens") != 8 or observed.get("finish_reason") != "length"
        ):
            raise ValueError("quality request identity or completion differs")
        chunks = _valid_chunks(observed.get("full_indexed_chunks"))
        if sum(rows for _, rows in chunks) != observed.get("prompt_tokens"):
            raise ValueError("quality request chunk coverage differs")
        request_chunks[request_id] = chunks
        request_metadata[request_id] = {
            "request_index": request_id,
            "request_id": observed["request_sha256"],
            "case_id": observed["case_id"],
            "group_id": observed["group_id"],
            "split": observed["split"],
            "seed": expected.get("seed"),
            "prompt_tokens": observed["prompt_tokens"],
        }
    if len({item["request_id"] for item in request_metadata.values()}) != 100:
        raise ValueError("quality request fixtures are not distinct")

    trace_score = summary.get("trace_score")
    if (
        not isinstance(trace_score, dict) or trace_score.get("verdict") != "PASS" or
        not isinstance(trace_score.get("checks"), dict) or
        set(trace_score["checks"]) != CORPUS_SCORER_CHECKS or
        not all(value is True for value in trace_score["checks"].values())
    ):
        raise ValueError("quality fixed-scorer verdict is not PASS")
    artifact_rows = trace_score.get("artifacts")
    if not isinstance(artifact_rows, list) or len(artifact_rows) != 37500:
        raise ValueError("quality scorer artifact count differs")
    artifacts: dict[str, dict[str, Any]] = {}
    keys: set[tuple[int, int, int, str]] = set()
    for item in artifact_rows:
        if (
            not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"} or
            not isinstance(item["name"], str) or item["name"] in artifacts or
            not isinstance(item["bytes"], int) or item["bytes"] < 0 or
            not HEX64_RE.fullmatch(str(item["sha256"]))
        ):
            raise ValueError("quality artifact receipt is malformed")
        match = FILE_RE.fullmatch(item["name"])
        prefix = REQUEST_PREFIX_RE.fullmatch(match.group("prefix")) if match else None
        if match is None or prefix is None:
            raise ValueError("quality artifact name is unrecognized")
        key = (int(prefix.group("request")), int(match.group("layer")),
               int(match.group("pos")), match.group("kind"))
        if key in keys:
            raise ValueError("quality artifact key is duplicated")
        keys.add(key)
        artifacts[item["name"]] = item
    observed_paths = list(trace.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in observed_paths) or
        {path.name for path in observed_paths} != set(artifacts)
    ):
        raise ValueError("quality trace file set is not exact and regular")
    layers = list(range(3, 78))
    wanted = {"ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias"}
    expected_keys = {
        (request, layer, pos, kind)
        for request, chunks in request_chunks.items() for layer in layers
        for pos, _ in chunks for kind in wanted
    }
    if keys != expected_keys:
        raise ValueError("quality request/layer/chunk tensor coverage is incomplete")
    expected_rows = sum(
        len(layers) * sum(rows for _, rows in chunks) for chunks in request_chunks.values()
    )
    total_bytes = sum(item["bytes"] for item in artifacts.values())
    if (
        trace_score.get("requests") != 100 or trace_score.get("events") != 7500 or
        trace_score.get("total_rows") != expected_rows or
        trace_score.get("token_layer_events") != expected_rows or
        trace_score.get("total_bytes") != total_bytes or
        expected_rows != summary.get("expected_token_layer_events") or
        arms["off"].get("trace_files") != 0 or arms["off"].get("trace_bytes") != 0 or
        arms["on"].get("trace_files") != len(artifacts) or
        arms["on"].get("trace_bytes") != total_bytes
    ):
        raise ValueError("quality scorer totals are inconsistent")
    with tempfile.TemporaryDirectory(prefix="glm52-quality-score.") as snapshot_directory:
        score_root = Path(snapshot_directory)
        score_trace = score_root / "trace"
        score_trace.mkdir()
        for path in observed_paths:
            item = artifacts[path.name]
            payload = _read_regular_snapshot(path)
            if len(payload) != item["bytes"] or _sha256_bytes(payload) != item["sha256"]:
                raise ValueError("quality trace artifact differs from scorer receipt")
            (score_trace / path.name).write_bytes(payload)
        score_log = score_root / "server.log"
        score_log.write_bytes(controls["on_server_log"])
        rescored = TRACE_SCORER.score_trace(
            score_trace, score_log, max_bytes=summary.get("max_trace_bytes", 0),
            expected_layers=set(layers), expected_chunks=[], expected_requests=request_chunks,
        )
    if rescored != trace_score:
        raise ValueError("fixed scorer does not reproduce the quality corpus")
    if not isinstance(summary.get("checks"), dict) or not all(summary["checks"].values()):
        raise ValueError("quality top-level checks are not PASS")
    if any(containments[mode].get("clean") is not True for mode in ("off", "on")):
        raise ValueError("quality containment is not clean")
    crashlog_bindings = receipt.get("crashlog_bindings")
    if not isinstance(crashlog_bindings, dict) or set(crashlog_bindings) != {"off", "on"}:
        raise ValueError("quality crashlog bindings are malformed")
    safety: dict[str, dict[str, int | float]] = {}
    for mode in ("off", "on"):
        directory = Path(str(containments[mode].get("crash_directory", "")))
        expected = crashlog_bindings[mode]
        if not isinstance(expected, dict) or set(expected) != {
                "cmd_sha256", "kernel_sha256", "main_sha256", "samples_sha256"}:
            raise ValueError("quality crashlog binding schema is malformed")
        crash_payloads = {
            name: _read_regular_snapshot(directory / f"{name}.log")
            for name in ("cmd", "kernel", "main", "samples")
        }
        actual = {f"{name}_sha256": _sha256_bytes(payload)
                  for name, payload in crash_payloads.items()}
        if actual != expected or any(
            containments[mode].get(key) != expected[key]
            for key in ("kernel_sha256", "main_sha256", "samples_sha256")
        ):
            raise ValueError("quality crashlog bytes differ from their receipt")
        minimum_available_kib: int | None = None
        maximum_cgroup_bytes = 0
        maximum_swap_bytes = 0
        for line in crash_payloads["samples"].decode("utf-8", errors="strict").splitlines():
            values = {}
            for field in line.split()[1:]:
                if "=" not in field:
                    continue
                key, value = field.split("=", 1)
                if value.isdigit():
                    values[key] = int(value)
            available = values.get("mem_avail_kb")
            if available is not None:
                minimum_available_kib = (
                    available if minimum_available_kib is None
                    else min(minimum_available_kib, available)
                )
            maximum_cgroup_bytes = max(
                maximum_cgroup_bytes, values.get("cgroup_peak_bytes", 0),
            )
            maximum_swap_bytes = max(
                maximum_swap_bytes, values.get("cgroup_swap_current_bytes", 0),
            )
        if minimum_available_kib is None or maximum_cgroup_bytes <= 0:
            raise ValueError("quality safety samples are incomplete")
        fault_text = b"\n".join((
            crash_payloads["kernel"], controls[f"{mode}_server_log"],
        )).decode("utf-8", errors="strict")
        safety[mode] = {
            "minimum_available_gib": minimum_available_kib / 1048576,
            "maximum_cgroup_memory_bytes": maximum_cgroup_bytes,
            "maximum_cgroup_swap_bytes": maximum_swap_bytes,
            "kernel_oom_or_xid": bool(re.search(
                r"(?:\bXid\b|out of memory|oom-kill)", fault_text, re.IGNORECASE,
            )),
        }
    observed = receipt.get("observed")
    expected_observed = {
        "quality_cases_per_arm": 100,
        "prompt_tokens": summary["expected_prompt_tokens"],
        "completion_tokens_per_request": 8,
        "byte_and_token_identity": controls["off_responses"] == controls["on_responses"],
        "containment_clean": all(containments[mode]["clean"] is True for mode in ("off", "on")),
        "off_trace_files": arms["off"]["trace_files"],
        "on_trace_files": arms["on"]["trace_files"],
        "on_trace_bytes": arms["on"]["trace_bytes"],
        "trace_events": rescored["events"],
        "token_layer_events": rescored["token_layer_events"],
        "routed_layer_first": layers[0],
        "routed_layer_last": layers[-1],
        "minimum_available_memory_gib": {
            mode: safety[mode]["minimum_available_gib"] for mode in ("off", "on")
        },
        "maximum_cgroup_memory_bytes": {
            mode: safety[mode]["maximum_cgroup_memory_bytes"] for mode in ("off", "on")
        },
        "maximum_cgroup_swap_bytes": max(
            int(safety[mode]["maximum_cgroup_swap_bytes"]) for mode in ("off", "on")
        ),
        "kernel_oom_or_xid": any(
            bool(safety[mode]["kernel_oom_or_xid"]) for mode in ("off", "on")
        ),
        "trace_score_verdict": rescored["verdict"],
        "natural_malformed_utf8_requests": sum(
            item.get("utf8_regression_reproduced") is True for item in responses
        ),
    }
    if (
        not isinstance(observed, dict) or set(observed) != set(expected_observed) or
        any(
            abs(float(observed["minimum_available_memory_gib"][mode]) -
                float(expected_observed["minimum_available_memory_gib"][mode])) > 1e-9
            for mode in ("off", "on")
        ) or
        {key: value for key, value in observed.items()
         if key != "minimum_available_memory_gib"} !=
        {key: value for key, value in expected_observed.items()
         if key != "minimum_available_memory_gib"} or
        Path(str(receipt.get("retained_directory", ""))).resolve(strict=True) != source_root
    ):
        raise ValueError("quality receipt observations do not reproduce")

    file_maps: dict[int, dict[tuple[int, int, str], Path]] = {
        request: {} for request in request_chunks
    }
    for name in artifacts:
        match = FILE_RE.fullmatch(name)
        assert match is not None
        prefix = REQUEST_PREFIX_RE.fullmatch(match.group("prefix"))
        assert prefix is not None
        request = int(prefix.group("request"))
        file_maps[request][(
            int(match.group("layer")), int(match.group("pos")), match.group("kind"),
        )] = trace / name
    lineage = {
        "source_receipt_sha256": _sha256_bytes(receipt_bytes),
        "source_summary_sha256": bindings["summary_sha256"],
        "source_arm_sha256": bindings["on_arm_sha256"],
        "source_server_log_sha256": bindings["on_server_log_sha256"],
        "candidate_hash": summary["candidate_hash"],
        "engine_commit": summary["engine_commit"],
        "binary_sha256": summary["binary_sha256"],
        "model_sha256": summary["model_sha256"],
        "tokenizer_sha256": summary["tokenizer_sha256"],
        "fixture_sha256": summary["fixture_content_sha256"],
        "configuration_sha256": arms["on"]["configuration_sha256"],
        "seed": summary["seed"],
        "scorer_sha256": SCORER_SHA256,
        "repository_head": repository_head,
    }
    return {
        "trace": trace, "layers": layers, "artifacts": artifacts, "lineage": lineage,
        "request_chunks": request_chunks, "request_files": file_maps,
        "request_metadata": request_metadata, "total_rows": expected_rows,
    }


def validate_source_bundle(
    source_root: Path,
    receipt_path: Path,
    *,
    repository_root: Path = ROOT,
    require_tracked_receipt: bool = True,
    require_tracked_scorer: bool | None = None,
    minimum_prompt_tokens: int = 512,
    request_index: int | None = None,
) -> dict[str, Any]:
    """Validate and bind one already-qualified capture bundle."""
    if source_root.is_symlink() or receipt_path.is_symlink():
        raise ValueError("qualified source paths may not be symlinks")
    source_root = source_root.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    if require_tracked_scorer is None:
        require_tracked_scorer = require_tracked_receipt
    receipt_bytes = (_require_tracked_receipt(receipt_path, repository_root)
                     if require_tracked_receipt else _read_regular_snapshot(receipt_path))
    if require_tracked_scorer:
        scorer_bytes = _require_tracked_snapshot(SCORER_PATH, repository_root)
        if scorer_bytes != SCORER_SOURCE:
            raise ValueError("loaded scorer differs from the tracked scorer snapshot")
    repository_head = _repository_head(ROOT)
    receipt = _strict_json_bytes(receipt_bytes, str(receipt_path))
    if receipt.get("scope") == "quality_100_case_all_routed_layer_corpus":
        if request_index is not None:
            raise ValueError("quality corpus compaction always validates and emits all requests")
        return _validate_quality_source_bundle(
            source_root, receipt_path, receipt_bytes, receipt, repository_head,
        )
    if receipt.get("scope") == "multi_request_all_routed_layer_corpus_smoke":
        return _validate_corpus_source_bundle(
            source_root, receipt_path, receipt_bytes, receipt, repository_head,
            request_index, minimum_prompt_tokens,
        )
    if request_index is not None:
        raise ValueError("request index is only valid for a corpus source")
    summary_path = source_root / "summary.json"
    arm_path = source_root / "on" / "arm.json"
    server_log = source_root / "on" / "server.log"
    trace = source_root / "on" / "trace"
    if not trace.is_dir() or trace.is_symlink() or server_log.is_symlink() or not server_log.is_file():
        raise ValueError("qualified source layout is invalid")
    control_paths = {
        "summary": summary_path,
        "arm": arm_path,
        "server_log": server_log,
        "off_arm": source_root / "off" / "arm.json",
        "off_result": source_root / "off" / "result.json",
        "on_result": source_root / "on" / "result.json",
        "off_server_log": source_root / "off" / "server.log",
        "off_containment": source_root / "off.containment.json",
        "on_containment": source_root / "on.containment.json",
    }
    controls = {name: _read_regular_snapshot(path) for name, path in control_paths.items()}
    summary = _strict_json_bytes(controls["summary"], str(summary_path))
    arm = _strict_json_bytes(controls["arm"], str(arm_path))
    receipt_keys = {
        "schema_version", "candidate_hash", "engine_commit", "classification", "scope",
        "high_row_2048_status", "summary_sha256", "off_arm_sha256", "on_arm_sha256",
        "off_result_sha256", "on_result_sha256", "off_server_log_sha256",
        "on_server_log_sha256", "observed", "pre_runtime_authorization_review",
        "post_runtime_review", "conclusion",
    }
    summary_keys = {
        "schema_version", "scope", "high_row_2048_status", "candidate_hash",
        "engine_commit", "binary_sha256", "model_sha256", "tokenizer_sha256", "seed",
        "context_level", "max_trace_bytes", "off_arm_sha256", "on_arm_sha256",
        "off_containment_sha256", "on_containment_sha256", "trace_score", "checks", "verdict",
    }
    if set(receipt) != receipt_keys or set(summary) != summary_keys:
        raise ValueError("source receipt or summary schema differs from the qualified runner")
    valid_scope = (
        (summary.get("scope") == "high_row_multichunk" and
         summary.get("high_row_2048_status") == "PASS") or
        (summary.get("scope") == "short_single_indexed_batch_only" and
         summary.get("high_row_2048_status") == "OPEN")
    )
    if (receipt.get("classification") != "PASS" or summary.get("verdict") != "PASS" or
            arm.get("mode") != "on" or receipt.get("scope") != summary.get("scope") or
            receipt.get("high_row_2048_status") != summary.get("high_row_2048_status") or
            not valid_scope or minimum_prompt_tokens <= 0):
        raise ValueError("source bundle is not qualified")
    post_review = receipt.get("post_runtime_review")
    if (not isinstance(post_review, dict) or set(post_review) != {
            "round", "gap_reviewer_score", "adversarial_reviewer_score", "critical", "high"} or
            not isinstance(post_review["round"], int) or
            not all(isinstance(post_review[key], int) and post_review[key] >= 90
                    for key in ("gap_reviewer_score", "adversarial_reviewer_score")) or
            post_review["critical"] != [] or post_review["high"] != []):
        raise ValueError("source post-runtime review is incomplete")
    bound_hashes = {
        "summary_sha256": _sha256_bytes(controls["summary"]),
        "on_arm_sha256": _sha256_bytes(controls["arm"]),
        "on_server_log_sha256": _sha256_bytes(controls["server_log"]),
    }
    if any(receipt.get(key) != value for key, value in bound_hashes.items()):
        raise ValueError("source receipt hashes do not match the capture")
    if summary.get("on_arm_sha256") != bound_hashes["on_arm_sha256"]:
        raise ValueError("source summary does not bind the ON arm")
    if (receipt.get("candidate_hash") != summary.get("candidate_hash") or
            receipt.get("engine_commit") != summary.get("engine_commit") or
            not COMMIT_RE.fullmatch(str(summary.get("candidate_hash", ""))) or
            not COMMIT_RE.fullmatch(str(summary.get("engine_commit", "")))):
        raise ValueError("source candidate lineage differs")
    for key in ("binary_sha256", "model_sha256", "tokenizer_sha256"):
        if summary.get(key) != arm.get(key) or not HEX64_RE.fullmatch(str(summary.get(key, ""))):
            raise ValueError(f"source {key} lineage differs")

    chunks = _valid_chunks(arm.get("full_indexed_chunks"))
    response = arm.get("response_signature")
    if (not isinstance(response, dict) or
            arm.get("fixture_sha256") != response.get("request_sha256") or
            not HEX64_RE.fullmatch(str(arm.get("fixture_sha256", ""))) or
            not HEX64_RE.fullmatch(str(arm.get("configuration_sha256", ""))) or
            arm.get("prompt_tokens") != sum(rows for _, rows in chunks) or
            receipt.get("observed", {}).get("full_indexed_chunks") != arm.get("full_indexed_chunks")):
        raise ValueError("source chunk coverage differs from the qualified receipt")
    trace_score = summary.get("trace_score")
    if (not isinstance(trace_score, dict) or trace_score.get("verdict") != "PASS" or
            not isinstance(trace_score.get("checks"), dict) or
            set(trace_score["checks"]) != SCORER_CHECKS or
            not all(value is True for value in trace_score["checks"].values())):
        raise ValueError("source fixed-scorer verdict is not PASS")
    artifact_rows = trace_score.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise ValueError("source scorer has no artifacts")
    artifacts: dict[str, dict[str, Any]] = {}
    keys: set[tuple[int, int, str]] = set()
    prefixes: set[str] = set()
    for item in artifact_rows:
        if (not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"} or
                not isinstance(item["name"], str) or item["name"] in artifacts or
                not isinstance(item["bytes"], int) or item["bytes"] < 0 or
                not isinstance(item["sha256"], str)):
            raise ValueError("source artifact receipt is malformed")
        match = FILE_RE.fullmatch(item["name"])
        if not match:
            raise ValueError("source artifact name is unrecognized")
        expected_extension = "i32" if match.group("kind") == "router_selected" else "f32"
        if match.group("ext") != expected_extension:
            raise ValueError("source artifact extension is invalid for its kind")
        key = (int(match.group("layer")), int(match.group("pos")), match.group("kind"))
        if key in keys:
            raise ValueError("source artifact key is duplicated")
        keys.add(key)
        prefixes.add(match.group("prefix"))
        artifacts[item["name"]] = item
    if len(prefixes) != 1:
        raise ValueError("source artifacts have mixed prefixes")
    observed_paths = list(trace.iterdir())
    if (any(path.is_symlink() or not path.is_file() for path in observed_paths) or
            {path.name for path in observed_paths} != set(artifacts)):
        raise ValueError("source trace file set is not exact and regular")
    layers = sorted({layer for layer, _, _ in keys})
    wanted_kinds = {"ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias"}
    expected_keys = {
        (layer, pos, kind) for layer in layers for pos, _ in chunks for kind in wanted_kinds
    }
    if keys != expected_keys or any(not 4 <= layer < 79 for layer in layers):
        raise ValueError("source layer/chunk tensor coverage is incomplete")
    expected_events = len(layers) * len(chunks)
    expected_rows = len(layers) * sum(rows for _, rows in chunks)
    total_bytes = sum(item["bytes"] for item in artifacts.values())
    if (trace_score.get("events") != expected_events or trace_score.get("total_rows") != expected_rows or
            trace_score.get("total_bytes") != total_bytes or arm.get("trace_files") != len(artifacts) or
            arm.get("trace_bytes") != total_bytes):
        raise ValueError("source scorer totals are inconsistent")
    with tempfile.TemporaryDirectory(prefix="glm52-trace-score.") as snapshot_directory:
        score_root = Path(snapshot_directory)
        score_trace = score_root / "trace"
        score_trace.mkdir()
        for path in observed_paths:
            item = artifacts[path.name]
            payload = _read_regular_snapshot(path)
            if len(payload) != item["bytes"] or _sha256_bytes(payload) != item["sha256"]:
                raise ValueError("source trace artifact differs from scorer receipt")
            (score_trace / path.name).write_bytes(payload)
        score_log = score_root / "server.log"
        score_log.write_bytes(controls["server_log"])
        rescored = TRACE_SCORER.score_trace(
            score_trace, score_log, max_bytes=summary.get("max_trace_bytes", 0),
            expected_layers=set(layers), expected_chunks=chunks,
        )
        if rescored != trace_score:
            raise ValueError("fixed scorer does not reproduce the qualified trace result")

    off_arm = _strict_json_bytes(controls["off_arm"], str(control_paths["off_arm"]))
    off_containment = _strict_json_bytes(
        controls["off_containment"], str(control_paths["off_containment"]),
    )
    on_containment = _strict_json_bytes(
        controls["on_containment"], str(control_paths["on_containment"]),
    )
    path_bindings = {
        "off_arm_sha256": _sha256_bytes(controls["off_arm"]),
        "on_arm_sha256": bound_hashes["on_arm_sha256"],
        "off_result_sha256": _sha256_bytes(controls["off_result"]),
        "on_result_sha256": _sha256_bytes(controls["on_result"]),
        "off_server_log_sha256": _sha256_bytes(controls["off_server_log"]),
        "on_server_log_sha256": bound_hashes["on_server_log_sha256"],
    }
    if any(receipt.get(key) != value for key, value in path_bindings.items()):
        raise ValueError("source receipt does not bind all OFF/ON runtime artifacts")
    if (summary.get("off_arm_sha256") != path_bindings["off_arm_sha256"] or
            summary.get("off_containment_sha256") != _sha256_bytes(controls["off_containment"]) or
            summary.get("on_containment_sha256") != _sha256_bytes(controls["on_containment"]) or
            off_arm.get("result_sha256") != path_bindings["off_result_sha256"] or
            arm.get("result_sha256") != path_bindings["on_result_sha256"] or
            off_arm.get("server_log_sha256") != path_bindings["off_server_log_sha256"]):
        raise ValueError("source summary or arm runtime hashes differ")
    common_hashes = (
        "binary_sha256", "model_sha256", "tokenizer_sha256",
        "fixture_sha256", "configuration_sha256",
    )
    prompt_tokens = off_arm.get("prompt_tokens")
    off_chunks = off_arm.get("full_indexed_chunks")
    require_multichunk = summary["scope"] == "high_row_multichunk"
    exact_coverage = (
        isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
        prompt_tokens >= minimum_prompt_tokens and arm.get("prompt_tokens") == prompt_tokens and
        off_chunks == arm.get("full_indexed_chunks") and off_chunks == [list(row) for row in chunks] and
        (not require_multichunk or (
            prompt_tokens > 2048 and len(chunks) >= 2 and any(rows == 2048 for _, rows in chunks)
        ))
    )
    recomputed_checks = {
        "arm_modes": off_arm.get("mode") == "off" and arm.get("mode") == "on",
        "frozen_identity": all(off_arm.get(key) == arm.get(key) for key in common_hashes),
        "byte_and_token_identity": off_arm.get("response_signature") == arm.get("response_signature"),
        "matched_indexed_chunks": off_chunks == arm.get("full_indexed_chunks"),
        "prompt_tokens_and_exact_coverage": exact_coverage,
        "off_emitted_no_trace": off_arm.get("trace_files") == 0,
        "on_emitted_trace": isinstance(arm.get("trace_files"), int) and arm.get("trace_files", 0) > 0,
        "trace_score_passed": rescored.get("verdict") == "PASS",
        "containment_clean": off_containment.get("clean") is True and on_containment.get("clean") is True,
    }
    if summary.get("checks") != recomputed_checks or not all(recomputed_checks.values()):
        raise ValueError("source top-level OFF/ON qualification does not reproduce")
    observed = receipt.get("observed")
    expected_observed = {
        "context_level": summary.get("context_level"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens_per_arm": arm.get("response_signature", {}).get("completion_tokens"),
        "full_indexed_chunks": arm.get("full_indexed_chunks"),
        "byte_and_token_identity": recomputed_checks["byte_and_token_identity"],
        "containment_clean": recomputed_checks["containment_clean"],
        "off_trace_files": off_arm.get("trace_files"),
        "on_trace_files": arm.get("trace_files"),
        "on_trace_bytes": arm.get("trace_bytes"),
        "trace_events": rescored.get("events"),
        "trace_score_verdict": rescored.get("verdict"),
    }
    if observed != expected_observed:
        raise ValueError("source receipt observations do not reproduce")
    file_map = {
        (int(match.group("layer")), int(match.group("pos")), match.group("kind")): trace / name
        for name in artifacts
        for match in [FILE_RE.fullmatch(name)]
        if match is not None
    }
    for layer in layers:
        bias_hashes = {
            artifacts[file_map[(layer, pos, "router_bias")].name]["sha256"]
            for pos, _ in chunks
        }
        if len(bias_hashes) != 1:
            raise ValueError("router bias differs across chunks of one layer")

    lineage = {
        "source_receipt_sha256": _sha256_bytes(receipt_bytes),
        "source_summary_sha256": bound_hashes["summary_sha256"],
        "source_arm_sha256": bound_hashes["on_arm_sha256"],
        "source_server_log_sha256": bound_hashes["on_server_log_sha256"],
        "candidate_hash": summary.get("candidate_hash"),
        "engine_commit": summary.get("engine_commit"),
        "binary_sha256": summary.get("binary_sha256"),
        "model_sha256": summary.get("model_sha256"),
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "fixture_sha256": arm.get("fixture_sha256"),
        "configuration_sha256": arm.get("configuration_sha256"),
        "request_id": response["request_sha256"],
        "seed": summary.get("seed"),
        "scorer_sha256": SCORER_SHA256,
        "repository_head": repository_head,
    }
    if any(value is None for value in lineage.values()):
        raise ValueError("source lineage is incomplete")
    return {
        "trace": trace, "layers": layers, "chunks": chunks,
        "files": file_map, "artifacts": artifacts, "lineage": lineage,
    }


def publish_bundle(
    destination: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any],
) -> dict[str, Any]:
    requested = destination.absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(requested)
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if not parent.is_dir():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=parent))
    try:
        records = temporary / "records.npz"
        with records.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        final_manifest = {
            **manifest,
            "output_file": "records.npz",
            "output_sha256": _sha256(records),
            "output_bytes": records.stat().st_size,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(final_manifest, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace rename is unavailable")
        renameat2.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        if renameat2(
            at_fdcwd, os.fsencode(temporary), at_fdcwd, os.fsencode(destination),
            rename_noreplace,
        ) != 0:
            error_number = ctypes.get_errno()
            if error_number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(error_number, os.strerror(error_number), destination)
            raise OSError(error_number, os.strerror(error_number), destination)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final_manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_bound_array(
    path: Path, dtype: str, shape: tuple[int, ...], expected: dict[str, Any],
) -> np.ndarray:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"]:
            raise ValueError("source tensor descriptor is not the qualified regular file")
        blocks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("source tensor ended before its qualified size")
            blocks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("source tensor exceeds its qualified size")
        after = os.fstat(descriptor)
        payload = b"".join(blocks)
        if ((before.st_dev, before.st_ino, before.st_size) !=
                (after.st_dev, after.st_ino, after.st_size) or
                hashlib.sha256(payload).hexdigest() != expected["sha256"]):
            raise ValueError("source tensor bytes differ from the qualified scorer artifact")
    finally:
        os.close(descriptor)
    expected_values = math.prod(shape)
    array_value = np.frombuffer(payload, dtype=dtype)
    if array_value.size != expected_values:
        raise ValueError("source tensor shape differs from the qualified schema")
    return array_value.reshape(shape)


def run(args: argparse.Namespace) -> int:
    source = validate_source_bundle(
        args.source_root, args.source_receipt, request_index=args.request_index,
    )
    trace = source["trace"]
    quality_corpus = "request_chunks" in source
    if quality_corpus:
        request_ids = sorted(source["request_chunks"])
        work = [
            (request, layer, pos, rows)
            for request in request_ids for layer in source["layers"]
            for pos, rows in source["request_chunks"][request]
        ]
        holdout_count = min(4096, int(source["total_rows"]))
        holdout_targets = set(np.linspace(
            0, int(source["total_rows"]) - 1, holdout_count, dtype=np.int64,
        ).tolist())
    else:
        request_ids = []
        work = [
            (None, layer, pos, rows)
            for layer in source["layers"] for pos, rows in source["chunks"]
        ]
        holdout_targets = set()

    compact_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("selected_ids", "top_ids", "top_logits", "hidden_q4", "hidden_scale")
    }
    layers: list[np.ndarray] = []
    token_positions: list[np.ndarray] = []
    request_indices: list[np.ndarray] = []
    holdout_rows: list[np.ndarray] = []
    holdout_hidden: list[np.ndarray] = []
    metric_parts: list[dict[str, int | float]] = []
    sources: dict[str, str] = {}
    global_row = 0
    for request, layer, pos, rows in work:
        file_map = source["request_files"][request] if quality_corpus else source["files"]
        files = {
            kind: file_map[(layer, pos, kind)]
            for kind in ("ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias")
        }
        hidden = _read_bound_array(
            files["ffn_norm"], "<f4", (rows, N_EMBD),
            source["artifacts"][files["ffn_norm"].name],
        )
        logits = _read_bound_array(
            files["router_logits"], "<f4", (rows, N_EXPERT),
            source["artifacts"][files["router_logits"].name],
        )
        probabilities = _read_bound_array(
            files["router_probs"], "<f4", (rows, N_EXPERT),
            source["artifacts"][files["router_probs"].name],
        )
        bias = _read_bound_array(
            files["router_bias"], "<f4", (N_EXPERT,),
            source["artifacts"][files["router_bias"].name],
        )
        selected = _read_bound_array(
            files["router_selected"], "<i4", (rows, N_EXPERT_USED),
            source["artifacts"][files["router_selected"].name],
        )
        compact, metrics = compact_arrays(
            hidden, logits, bias, selected, router_probs=probabilities,
        )
        for name, value in compact.items():
            compact_parts[name].append(value)
        layers.append(np.full(rows, layer, dtype=np.uint16))
        token_positions.append(np.arange(pos, pos + rows, dtype=np.uint32))
        if quality_corpus:
            assert request is not None
            request_indices.append(np.full(rows, request, dtype=np.uint16))
            selected_holdout = [
                row - global_row
                for row in sorted(holdout_targets)
                if global_row <= row < global_row + rows
            ]
            if selected_holdout:
                holdout_rows.append(np.asarray(
                    [global_row + row for row in selected_holdout], dtype=np.uint32,
                ))
                holdout_hidden.append(hidden[selected_holdout].astype(np.float16))
        global_row += rows
        metric_parts.append(metrics)
        for path in files.values():
            sources[path.name] = source["artifacts"][path.name]["sha256"]

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in compact_parts.items()}
    arrays["layer"] = np.concatenate(layers)
    arrays["token_position"] = np.concatenate(token_positions)
    if quality_corpus:
        arrays["request_index"] = np.concatenate(request_indices)
        arrays["hidden_fp16_holdout_row"] = np.concatenate(holdout_rows)
        arrays["hidden_fp16_holdout"] = np.concatenate(holdout_hidden, axis=0)
    total_values = sum(int(part["hidden_values"]) for part in metric_parts)
    weighted_squared = sum(
        float(part["hidden_rmse"]) ** 2 * int(part["hidden_values"])
        for part in metric_parts
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "format": "glm52-union-p0-npz-v2" if quality_corpus else "glm52-union-p0-npz-v1",
        "rows": int(arrays["layer"].size),
        "layers": source["layers"],
        "chunks": (
            {str(request): source["request_chunks"][request] for request in request_ids}
            if quality_corpus else source["chunks"]
        ),
        "top_k": TOP_K,
        "hidden_quantization": "symmetric-groupwise-int4-range-minus7-plus7-fp16-scale",
        "hidden_group_size": HIDDEN_GROUP_SIZE,
        "hidden_rmse": math.sqrt(weighted_squared / total_values),
        "hidden_nrmse_max_by_chunk": max(float(part["hidden_nrmse"]) for part in metric_parts),
        "hidden_max_abs_error": max(float(part["hidden_max_abs_error"]) for part in metric_parts),
        "top_logit_max_abs_error": max(float(part["top_logit_max_abs_error"]) for part in metric_parts),
        "router_probability_max_abs_error": max(
            float(part["router_probability_max_abs_error"]) for part in metric_parts
        ),
        "source_sha256": dict(sorted(sources.items())),
        "compactor_sha256": _sha256(Path(__file__).resolve()),
        "lineage": source["lineage"],
    }
    if quality_corpus:
        record.update({
            "requests": len(request_ids),
            "request_metadata": [source["request_metadata"][request] for request in request_ids],
            "hidden_fp16_holdout_rows": int(arrays["hidden_fp16_holdout_row"].size),
            "hidden_fp16_holdout_selection": "4096-evenly-spaced-global-rows",
            "raw_source_retained": str(args.source_root.resolve()),
        })
    final_manifest = publish_bundle(args.out_dir, arrays, record)
    print(json.dumps(final_manifest, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True, type=Path)
    result.add_argument("--source-receipt", required=True, type=Path)
    result.add_argument("--request-index", type=int)
    result.add_argument("--out-dir", required=True, type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
