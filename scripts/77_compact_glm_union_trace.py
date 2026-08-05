#!/usr/bin/env python3
"""Compact validated GLM union traces into training-ready P0 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np


N_EMBD = 6144
N_EXPERT = 256
N_EXPERT_USED = 8
TOP_K = 32
HIDDEN_GROUP_SIZE = 32
FILE_RE = re.compile(
    r"^(?P<prefix>.+)_glm_indexed_(?P<kind>ffn_norm|router_logits|router_probs|router_selected|router_bias)-"
    r"(?P<layer>\d+)_pos(?P<pos>\d+)\.(?P<ext>f32|i32)$"
)


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


def validate_source_bundle(source_root: Path, receipt_path: Path) -> dict[str, Any]:
    """Validate and bind one already-qualified capture bundle."""
    if source_root.is_symlink() or receipt_path.is_symlink():
        raise ValueError("qualified source paths may not be symlinks")
    source_root = source_root.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    summary_path = source_root / "summary.json"
    arm_path = source_root / "on" / "arm.json"
    server_log = source_root / "on" / "server.log"
    trace = source_root / "on" / "trace"
    if not trace.is_dir() or trace.is_symlink() or server_log.is_symlink() or not server_log.is_file():
        raise ValueError("qualified source layout is invalid")
    receipt = _strict_json(receipt_path)
    summary = _strict_json(summary_path)
    arm = _strict_json(arm_path)
    if (receipt.get("classification") != "PASS" or summary.get("verdict") != "PASS" or
            arm.get("mode") != "on"):
        raise ValueError("source bundle is not qualified")
    bound_hashes = {
        "summary_sha256": _sha256(summary_path),
        "on_arm_sha256": _sha256(arm_path),
        "on_server_log_sha256": _sha256(server_log),
    }
    if any(receipt.get(key) != value for key, value in bound_hashes.items()):
        raise ValueError("source receipt hashes do not match the capture")
    if summary.get("on_arm_sha256") != bound_hashes["on_arm_sha256"]:
        raise ValueError("source summary does not bind the ON arm")
    if receipt.get("candidate_hash") != summary.get("candidate_hash"):
        raise ValueError("source candidate lineage differs")
    for key in ("binary_sha256", "model_sha256", "tokenizer_sha256"):
        if summary.get(key) != arm.get(key):
            raise ValueError(f"source {key} lineage differs")

    chunks = _valid_chunks(arm.get("full_indexed_chunks"))
    if (arm.get("prompt_tokens") != sum(rows for _, rows in chunks) or
            receipt.get("observed", {}).get("full_indexed_chunks") != arm.get("full_indexed_chunks")):
        raise ValueError("source chunk coverage differs from the qualified receipt")
    trace_score = summary.get("trace_score")
    if (not isinstance(trace_score, dict) or trace_score.get("verdict") != "PASS" or
            not isinstance(trace_score.get("checks"), dict) or
            not trace_score["checks"] or not all(value is True for value in trace_score["checks"].values())):
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

    response = arm.get("response_signature")
    if not isinstance(response, dict) or not isinstance(response.get("request_sha256"), str):
        raise ValueError("source request identity is missing")
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
        os.rename(temporary, destination)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final_manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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
            hidden = np.fromfile(files["ffn_norm"], dtype="<f4").reshape(rows, N_EMBD)
            logits = np.fromfile(files["router_logits"], dtype="<f4").reshape(rows, N_EXPERT)
            probabilities = np.fromfile(files["router_probs"], dtype="<f4").reshape(rows, N_EXPERT)
            bias = np.fromfile(files["router_bias"], dtype="<f4").reshape(N_EXPERT)
            selected = np.fromfile(files["router_selected"], dtype="<i4").reshape(rows, N_EXPERT_USED)
            compact, metrics = compact_arrays(
                hidden, logits, bias, selected, router_probs=probabilities,
            )
            for name, value in compact.items():
                compact_parts[name].append(value)
            layers.append(np.full(rows, layer, dtype=np.uint16))
            token_positions.append(np.arange(pos, pos + rows, dtype=np.uint32))
            metric_parts.append(metrics)
            for path in files.values():
                sources[path.name] = _sha256(path)

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
