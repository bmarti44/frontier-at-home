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
    arrays = (hidden, logits, bias, selected) + (() if router_probs is None else (router_probs,))
    if any(not isinstance(value, np.ndarray) for value in arrays):
        raise ValueError("all inputs must be numpy arrays")
    if (hidden.ndim != 2 or logits.ndim != 2 or bias.ndim != 1 or selected.ndim != 2 or
            hidden.shape[0] == 0 or logits.shape[0] != hidden.shape[0] or
            bias.shape[0] != logits.shape[1] or selected.shape[0] != hidden.shape[0] or
            not 1 <= top_k <= logits.shape[1] or selected.shape[1] > top_k):
        raise ValueError("trace array shapes are inconsistent")
    if router_probs is not None and router_probs.shape != logits.shape:
        raise ValueError("captured router probability shape is inconsistent")
    if (not np.issubdtype(hidden.dtype, np.floating) or
            not np.issubdtype(logits.dtype, np.floating) or
            not np.issubdtype(bias.dtype, np.floating) or
            not np.issubdtype(selected.dtype, np.integer) or
            (router_probs is not None and not np.issubdtype(router_probs.dtype, np.floating))):
        raise ValueError("trace array dtypes are invalid")
    finite_arrays = (hidden, logits, bias) + (() if router_probs is None else (router_probs,))
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
    probability_delta = 0.0
    if router_probs is not None:
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


def _parse_chunk(value: str) -> tuple[int, int]:
    try:
        pos, rows = (int(item) for item in value.split(":", 1))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("chunk must be POS:ROWS") from error
    if pos < 0 or rows <= 0:
        raise argparse.ArgumentTypeError("chunk values are out of range")
    return pos, rows


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    trace = args.trace.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if not trace.is_dir() or output.parent != manifest.parent or not output.parent.is_dir():
        raise ValueError("trace/output paths are invalid")
    if output.exists() or manifest.exists():
        raise FileExistsError("compact output already exists")
    chunks = args.chunk
    if len(set(chunks)) != len(chunks):
        raise ValueError("duplicate expected chunks")

    compact_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("selected_ids", "top_ids", "top_logits", "hidden_q4", "hidden_scale")
    }
    layers: list[np.ndarray] = []
    token_positions: list[np.ndarray] = []
    metric_parts: list[dict[str, int | float]] = []
    sources: dict[str, str] = {}
    for layer in args.layer:
        for pos, rows in chunks:
            files: dict[str, Path] = {}
            for path in trace.iterdir():
                match = FILE_RE.fullmatch(path.name)
                if match and int(match.group("layer")) == layer and int(match.group("pos")) == pos:
                    files[match.group("kind")] = path
            required = {"ffn_norm", "router_logits", "router_probs", "router_selected", "router_bias"}
            if set(files) < required:
                raise ValueError(f"missing trace tensors for layer={layer} pos={pos}")
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
    _atomic_npz(output, arrays)
    total_values = sum(int(part["hidden_values"]) for part in metric_parts)
    weighted_squared = sum(
        float(part["hidden_rmse"]) ** 2 * int(part["hidden_values"])
        for part in metric_parts
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "format": "glm52-union-p0-npz-v1",
        "rows": int(arrays["layer"].size),
        "layers": args.layer,
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
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
    }
    manifest.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--layer", required=True, type=int, action="append")
    result.add_argument("--chunk", required=True, type=_parse_chunk, action="append")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
