#!/usr/bin/env python3
"""Compact validated GLM union traces into training-ready P0 records."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = ROOT / "scripts/75_glm_union_trace_score.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE_SCORER = _load("union_trace_scorer_for_compact", SCORER_PATH)
N_EMBD = 6144
N_EXPERT = 256
N_EXPERT_USED = 8
TOP_K = 32
HIDDEN_GROUP_SIZE = 32
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input is not a regular file: {path}")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


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


def _require_tracked_receipt(receipt_path: Path, repository_root: Path) -> None:
    repository_root = repository_root.resolve(strict=True)
    try:
        relative = receipt_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("source receipt is outside the trusted repository") from error
    if not str(relative).startswith("results/glm52-gates/"):
        raise ValueError("source receipt is outside the gate evidence directory")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=repository_root, stdin=subprocess.DEVNULL, capture_output=True,
    )
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repository_root,
        stdin=subprocess.DEVNULL, capture_output=True,
    )
    if (tracked.returncode != 0 or committed.returncode != 0 or
            committed.stdout != receipt_path.read_bytes()):
        raise ValueError("source receipt is not tracked and clean at HEAD")


def validate_source_bundle(
    source_root: Path,
    receipt_path: Path,
    *,
    repository_root: Path = ROOT,
    require_tracked_receipt: bool = True,
    minimum_prompt_tokens: int = 512,
) -> dict[str, Any]:
    """Validate and bind one already-qualified capture bundle."""
    if source_root.is_symlink() or receipt_path.is_symlink():
        raise ValueError("qualified source paths may not be symlinks")
    source_root = source_root.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    if require_tracked_receipt:
        _require_tracked_receipt(receipt_path, repository_root)
    summary_path = source_root / "summary.json"
    arm_path = source_root / "on" / "arm.json"
    server_log = source_root / "on" / "server.log"
    trace = source_root / "on" / "trace"
    if not trace.is_dir() or trace.is_symlink() or server_log.is_symlink() or not server_log.is_file():
        raise ValueError("qualified source layout is invalid")
    receipt = _strict_json(receipt_path)
    summary = _strict_json(summary_path)
    arm = _strict_json(arm_path)
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
        "summary_sha256": _sha256(summary_path),
        "on_arm_sha256": _sha256(arm_path),
        "on_server_log_sha256": _sha256(server_log),
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
    for path in observed_paths:
        item = artifacts[path.name]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError("source trace artifact differs from scorer receipt")

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
    rescored = TRACE_SCORER.score_trace(
        trace, server_log, max_bytes=summary.get("max_trace_bytes", 0),
        expected_layers=set(layers), expected_chunks=chunks,
    )
    if rescored != trace_score:
        raise ValueError("fixed scorer does not reproduce the qualified trace result")

    off_arm_path = source_root / "off" / "arm.json"
    off_result_path = source_root / "off" / "result.json"
    on_result_path = source_root / "on" / "result.json"
    off_server_log = source_root / "off" / "server.log"
    off_containment_path = source_root / "off.containment.json"
    on_containment_path = source_root / "on.containment.json"
    off_arm = _strict_json(off_arm_path)
    off_containment = _strict_json(off_containment_path)
    on_containment = _strict_json(on_containment_path)
    path_bindings = {
        "off_arm_sha256": _sha256(off_arm_path),
        "on_arm_sha256": bound_hashes["on_arm_sha256"],
        "off_result_sha256": _sha256(off_result_path),
        "on_result_sha256": _sha256(on_result_path),
        "off_server_log_sha256": _sha256(off_server_log),
        "on_server_log_sha256": bound_hashes["on_server_log_sha256"],
    }
    if any(receipt.get(key) != value for key, value in path_bindings.items()):
        raise ValueError("source receipt does not bind all OFF/ON runtime artifacts")
    if (summary.get("off_arm_sha256") != path_bindings["off_arm_sha256"] or
            summary.get("off_containment_sha256") != _sha256(off_containment_path) or
            summary.get("on_containment_sha256") != _sha256(on_containment_path) or
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
        bias_hashes = {_sha256(file_map[(layer, pos, "router_bias")]) for pos, _ in chunks}
        if len(bias_hashes) != 1:
            raise ValueError("router bias differs across chunks of one layer")

    lineage = {
        "source_receipt_sha256": _sha256(receipt_path),
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
    source = validate_source_bundle(args.source_root, args.source_receipt)
    trace = source["trace"]
    chunks = source["chunks"]

    compact_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("selected_ids", "top_ids", "top_logits", "hidden_q4", "hidden_scale")
    }
    layers: list[np.ndarray] = []
    token_positions: list[np.ndarray] = []
    metric_parts: list[dict[str, int | float]] = []
    sources: dict[str, str] = {}
    for layer in source["layers"]:
        for pos, rows in chunks:
            files = {
                kind: source["files"][(layer, pos, kind)]
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
            metric_parts.append(metrics)
            for path in files.values():
                sources[path.name] = source["artifacts"][path.name]["sha256"]

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in compact_parts.items()}
    arrays["layer"] = np.concatenate(layers)
    arrays["token_position"] = np.concatenate(token_positions)
    total_values = sum(int(part["hidden_values"]) for part in metric_parts)
    weighted_squared = sum(
        float(part["hidden_rmse"]) ** 2 * int(part["hidden_values"])
        for part in metric_parts
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "format": "glm52-union-p0-npz-v1",
        "rows": int(arrays["layer"].size),
        "layers": source["layers"],
        "chunks": chunks,
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
    final_manifest = publish_bundle(args.out_dir, arrays, record)
    print(json.dumps(final_manifest, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True, type=Path)
    result.add_argument("--source-receipt", required=True, type=Path)
    result.add_argument("--out-dir", required=True, type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
