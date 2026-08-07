#!/usr/bin/env python3
"""Score optimistic FP4 candidates on frozen real GLM-5.2 KV/query tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE = "scripts/93_score_w9_fp4_falsifier.py"
TEST_RELATIVE = "scripts/tests/test_w9_fp4_falsifier.py"
PLAN_RELATIVE = "results/glm52-gates/W9-fp4-falsifier-plan-v1.json"
DRAND_VERIFIER = REPO / "scripts/89_verify_drand_receipt.mjs"
NODE = pathlib.Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")

LAYERS = (0, 2, 10, 26, 42, 58, 74, 77)
KV_ROWS = 8192
QUERY_ROWS = 128
QUERY_HEADS = 64
WIDTH = 512
SELECTED_CAPACITY = 2048
BLOCK_WIDTH = 32
MAXIMUM_RELATIVE_RMSE = 0.05
CANDIDATES = (
    "plain_e2m1_f32_scale",
    "hadamard_e2m1_f32_scale",
    "hadamard_e2m1_f32_scale_channel_correction",
)
CAPTURE_HASHES = {
    "kv.f32": "805b30d0c4ac89bd5cd361c455c3c6eb49d69d32b49d8afb935b5a75a7de33ed",
    "query.f32": "a9346a4d3a8fc8fd6915905cc8c7f2a595816b957713a1fbfa8a1cfe182d0a9b",
    "selected.u32": "fe9edd824633783800ff16f428c1b42bcebb15276a386b951fb0727897b0eba0",
    "selected-count.u32": "99b56dbff1dd5899c41016ba76257216a809143ccf13fa0bb119343b07c42cdc",
    "metadata.json": "ddf1d406b4c1c3164d292f2aed94e740d29c538e884139ed00eaecb517adef27",
    "W9_CAPTURE_COMPLETE": "228a01a07809971faed980c97cc0d0ddee97c421905e7f82da5855250db81b52",
}
CAPTURE_SIZES = {
    "kv.f32": 134217728,
    "query.f32": 134217728,
    "selected.u32": 8388608,
    "selected-count.u32": 4096,
}
E2M1_LEVELS = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32,
)
E2M1_MIDPOINTS = (E2M1_LEVELS[:-1] + E2M1_LEVELS[1:]) / 2.0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: pathlib.Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {path}: {token}")),
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def e2m1_quantize(rows: np.ndarray, block_width: int = BLOCK_WIDTH) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] % block_width:
        raise ValueError("row width must be divisible by block width")
    if block_width < 1:
        raise ValueError("block width must be positive")
    if not np.isfinite(array).all():
        raise ValueError("quantizer input must be finite")
    blocks = array.reshape(array.shape[0], -1, block_width)
    amax = np.max(np.abs(blocks), axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / np.float32(6.0), np.float32(1.0))
    normalized = blocks / scale
    codes = np.searchsorted(E2M1_MIDPOINTS, normalized, side="left")
    quantized = E2M1_LEVELS[codes] * scale
    return np.asarray(quantized.reshape(array.shape), dtype=np.float32)


def hadamard_rotate(rows: np.ndarray, signs: np.ndarray) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    sign_array = np.asarray(signs, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Hadamard input must be two-dimensional")
    width = array.shape[1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    if sign_array.shape != (width,) or not np.isin(sign_array, (-1.0, 1.0)).all():
        raise ValueError("Hadamard signs must be exactly +/-1")
    if not np.isfinite(array).all():
        raise ValueError("Hadamard input must be finite")
    output = np.ascontiguousarray(array * sign_array)
    step = 1
    while step < width:
        for start in range(0, width, step * 2):
            left = output[:, start:start + step].copy()
            right = output[:, start + step:start + 2 * step].copy()
            output[:, start:start + step] = left + right
            output[:, start + step:start + 2 * step] = left - right
        step *= 2
    output *= np.float32(1.0 / math.sqrt(width))
    return output


def _rank(seed: bytes, domain: bytes, index: int) -> bytes:
    return hashlib.sha256(
        b"GLM52-W9-FP4-RANK-V1\0" + seed + b"\0" + domain + b"\0"
        + index.to_bytes(8, "big")
    ).digest()


def split_indices(count: int, seed: bytes, domain: bytes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if count < 2 or count % 2 or len(seed) != 32 or not domain:
        raise ValueError("split requires an even count, 32-byte seed, and domain")
    ordered = sorted(range(count), key=lambda index: (_rank(seed, domain, index), index))
    midpoint = count // 2
    return tuple(sorted(ordered[:midpoint])), tuple(sorted(ordered[midpoint:]))


def layer_signs(seed: bytes, layer: int, width: int = WIDTH) -> np.ndarray:
    if len(seed) != 32 or layer < 0 or width < 1:
        raise ValueError("invalid sign seed domain")
    values = []
    domain = b"signs/" + str(layer).encode("ascii")
    for index in range(width):
        values.append(1.0 if _rank(seed, domain, index)[0] & 1 else -1.0)
    return np.asarray(values, dtype=np.float32)


def fit_channel_correction(reference: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(quantized, dtype=np.float32)
    if ref.ndim != 2 or ref.shape != candidate.shape or ref.shape[0] < 1:
        raise ValueError("correction arrays must have equal nonempty 2D shapes")
    if not np.isfinite(ref).all() or not np.isfinite(candidate).all():
        raise ValueError("correction arrays must be finite")
    numerator = np.sum(ref.astype(np.float64) * candidate.astype(np.float64), axis=0)
    denominator = np.sum(candidate.astype(np.float64) ** 2, axis=0)
    alpha = np.ones(ref.shape[1], dtype=np.float64)
    np.divide(numerator, denominator, out=alpha, where=denominator > 0)
    if not np.isfinite(alpha).all():
        raise ValueError("non-finite channel correction")
    return alpha.astype(np.float32)


def query_weighted_error(
    queries: np.ndarray,
    reference_keys: np.ndarray,
    candidate_keys: np.ndarray,
    selected: np.ndarray,
    selected_sentinel: int,
    heldout_queries: np.ndarray,
    heldout_keys: np.ndarray,
) -> dict[str, float | int]:
    query_array = np.asarray(queries, dtype=np.float32)
    reference = np.asarray(reference_keys, dtype=np.float32)
    candidate = np.asarray(candidate_keys, dtype=np.float32)
    selection = np.asarray(selected, dtype=np.uint32)
    query_mask = np.asarray(heldout_queries, dtype=np.bool_)
    key_mask = np.asarray(heldout_keys, dtype=np.bool_)
    if (query_array.ndim != 3 or reference.ndim != 2 or
            candidate.shape != reference.shape or query_array.shape[2] != reference.shape[1] or
            selection.ndim != 2 or selection.shape[0] != query_array.shape[0] or
            query_mask.shape != (query_array.shape[0],) or
            key_mask.shape != (reference.shape[0],)):
        raise ValueError("metric input shape mismatch")
    if not (np.isfinite(query_array).all() and np.isfinite(reference).all()
            and np.isfinite(candidate).all()):
        raise ValueError("metric inputs must be finite")
    numerator = 0.0
    denominator = 0.0
    pairs = 0
    query_rows_used = 0
    key_references = 0
    for query_index in np.flatnonzero(query_mask):
        raw_ids = selection[query_index]
        invalid = raw_ids[(raw_ids != selected_sentinel) & (raw_ids >= reference.shape[0])]
        if invalid.size:
            raise ValueError("selected key ID is out of range")
        ids = raw_ids[raw_ids != selected_sentinel].astype(np.int64, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("duplicate selected key ID")
        ids = ids[key_mask[ids]]
        if not ids.size:
            continue
        query = query_array[query_index]
        base_logits = query @ reference[ids].T
        error_logits = query @ (candidate[ids] - reference[ids]).T
        numerator += float(np.sum(np.square(error_logits, dtype=np.float64), dtype=np.float64))
        denominator += float(np.sum(np.square(base_logits, dtype=np.float64), dtype=np.float64))
        pairs += int(ids.size * query.shape[0])
        key_references += int(ids.size)
        query_rows_used += 1
    if pairs < 1 or denominator <= 0 or not math.isfinite(numerator + denominator):
        raise ValueError("held-out metric has no finite positive-denominator pairs")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "relative_rmse": math.sqrt(numerator / denominator),
        "pairs": pairs,
        "query_rows": query_rows_used,
        "key_references": key_references,
    }


def _verify_capture(capture: pathlib.Path) -> dict[str, Any]:
    if not capture.is_dir() or capture.is_symlink():
        raise ValueError("capture root must be a real directory")
    for name, expected in CAPTURE_HASHES.items():
        path = capture / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"capture hash mismatch: {name}")
        if name in CAPTURE_SIZES and path.stat().st_size != CAPTURE_SIZES[name]:
            raise ValueError(f"capture size mismatch: {name}")
    if (capture / "W9_CAPTURE_COMPLETE").read_bytes() != b"W9_CAPTURE_COMPLETE\n":
        raise ValueError("capture completion marker mismatch")
    metadata = strict_json(capture / "metadata.json")
    expected = {
        "schema": "glm52-w9-real-capture-v1",
        "layers": list(LAYERS),
        "kv_rows_per_layer": KV_ROWS,
        "kv_width": WIDTH,
        "query_rows_per_layer": QUERY_ROWS,
        "query_heads": QUERY_HEADS,
        "query_width": WIDTH,
        "selected_capacity": SELECTED_CAPACITY,
        "sample_position_start": 0,
        "sample_position_stride": 64,
        "selected_padding_sentinel": 8193,
        "storage_padding_sentinel": 4294967295,
        "artifacts": CAPTURE_SIZES,
        "dtype": {"kv": "f32", "query": "f32", "selected": "u32"},
    }
    if metadata != expected:
        raise ValueError("capture metadata mismatch")
    counts = np.memmap(
        capture / "selected-count.u32", dtype="<u4", mode="r",
        shape=(len(LAYERS), QUERY_ROWS),
    )
    if not np.all(counts == SELECTED_CAPACITY):
        raise ValueError("selected counts are incomplete")
    return metadata


def _verify_randomness(receipt_path: pathlib.Path, minimum_round: int) -> dict[str, Any]:
    receipt = strict_json(receipt_path)
    if set(receipt) != {"round", "randomness", "signature", "previous_signature"}:
        raise ValueError("unexpected drand receipt schema")
    round_number = receipt.get("round")
    randomness = receipt.get("randomness")
    signature = receipt.get("signature")
    previous = receipt.get("previous_signature")
    if (type(round_number) is not int or round_number <= minimum_round or
            not isinstance(randomness, str) or not re.fullmatch(r"[0-9a-f]{64}", randomness) or
            not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{192}", signature) or
            not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{192}", previous)):
        raise ValueError("invalid or stale drand receipt")
    subprocess.run(
        [str(NODE), str(DRAND_VERIFIER), str(round_number), randomness, signature, previous],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    return receipt


def _verify_frozen_source(candidate_commit: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_commit):
        raise ValueError("candidate commit must be 40 lowercase hex characters")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout:
        raise ValueError("repository must be clean")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", candidate_commit, "HEAD"],
        cwd=REPO, check=True,
    )
    bindings: dict[str, str] = {}
    for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, PLAN_RELATIVE):
        current = (REPO / relative).read_bytes()
        frozen = subprocess.run(
            ["git", "show", f"{candidate_commit}:{relative}"], cwd=REPO,
            check=True, capture_output=True,
        ).stdout
        if current != frozen:
            raise ValueError(f"working source differs from frozen candidate: {relative}")
        bindings[relative] = hashlib.sha256(current).hexdigest()
    return bindings


def _require_idle_host() -> None:
    for process_name in ("ds4-server", "ds4", "fio"):
        result = subprocess.run(
            ["/usr/bin/pgrep", "-x", process_name], check=False,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            raise ValueError(f"competing process is active: {process_name}")
        if result.returncode != 1:
            raise ValueError(f"could not inspect process state: {process_name}")


def _validate_selected(selection: np.ndarray, sentinel: int) -> None:
    for row_index, row in enumerate(selection):
        invalid = row[(row != sentinel) & (row >= KV_ROWS)]
        if invalid.size:
            raise ValueError("selected key ID is out of range")
        valid = row[row != sentinel]
        if np.unique(valid).size != valid.size:
            raise ValueError("duplicate selected key ID")
        position = row_index * 64
        if valid.size and int(valid.max()) > position:
            raise ValueError("selected key ID violates causal boundary")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    capture = pathlib.Path(args.capture_root).resolve()
    receipt_path = pathlib.Path(args.randomness_receipt).resolve()
    _require_idle_host()
    source_bindings = _verify_frozen_source(args.candidate_commit)
    metadata = _verify_capture(capture)
    receipt = _verify_randomness(receipt_path, args.minimum_drand_round)
    master_seed = hashlib.sha256(
        b"GLM52-W9-FP4-SPLIT-V1\0" + bytes.fromhex(receipt["randomness"])
        + bytes.fromhex(CAPTURE_HASHES["kv.f32"])
        + bytes.fromhex(CAPTURE_HASHES["query.f32"])
        + bytes.fromhex(args.candidate_commit)
    ).digest()

    kv = np.memmap(
        capture / "kv.f32", dtype="<f4", mode="r",
        shape=(len(LAYERS), KV_ROWS, WIDTH),
    )
    queries = np.memmap(
        capture / "query.f32", dtype="<f4", mode="r",
        shape=(len(LAYERS), QUERY_ROWS, QUERY_HEADS, WIDTH),
    )
    selected = np.memmap(
        capture / "selected.u32", dtype="<u4", mode="r",
        shape=(len(LAYERS), QUERY_ROWS, SELECTED_CAPACITY),
    )
    raw_rows: list[dict[str, Any]] = []
    totals = {
        candidate: {"numerator": 0.0, "denominator": 0.0, "pairs": 0,
                    "query_rows": 0, "key_references": 0}
        for candidate in CANDIDATES
    }
    split_hashes: dict[str, dict[str, str]] = {}

    for layer_index, layer in enumerate(LAYERS):
        reference = np.array(kv[layer_index], dtype=np.float32, copy=True)
        query = np.array(queries[layer_index], dtype=np.float32, copy=True)
        selection = np.array(selected[layer_index], dtype=np.uint32, copy=True)
        if not np.isfinite(reference).all() or not np.isfinite(query).all():
            raise ValueError(f"non-finite capture values at layer {layer}")
        _validate_selected(selection, int(metadata["selected_padding_sentinel"]))
        calibration_keys, heldout_keys_tuple = split_indices(
            KV_ROWS, master_seed, f"keys/{layer}".encode("ascii"),
        )
        _, heldout_queries_tuple = split_indices(
            QUERY_ROWS, master_seed, f"queries/{layer}".encode("ascii"),
        )
        key_mask = np.zeros(KV_ROWS, dtype=np.bool_)
        key_mask[list(heldout_keys_tuple)] = True
        query_mask = np.zeros(QUERY_ROWS, dtype=np.bool_)
        query_mask[list(heldout_queries_tuple)] = True
        split_hashes[str(layer)] = {
            "calibration_keys_sha256": hashlib.sha256(
                np.asarray(calibration_keys, dtype="<u4").tobytes()).hexdigest(),
            "heldout_keys_sha256": hashlib.sha256(
                np.asarray(heldout_keys_tuple, dtype="<u4").tobytes()).hexdigest(),
            "heldout_queries_sha256": hashlib.sha256(
                np.asarray(heldout_queries_tuple, dtype="<u4").tobytes()).hexdigest(),
        }

        plain = e2m1_quantize(reference)
        metrics = query_weighted_error(
            query, reference, plain, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[0], **metrics})
        for key in totals[CANDIDATES[0]]:
            totals[CANDIDATES[0]][key] += metrics[key]
        del plain

        signs = layer_signs(master_seed, layer)
        rotated_reference = hadamard_rotate(reference, signs)
        rotated_queries = hadamard_rotate(query.reshape(-1, WIDTH), signs).reshape(query.shape)
        rotated_quantized = e2m1_quantize(rotated_reference)
        metrics = query_weighted_error(
            rotated_queries, rotated_reference, rotated_quantized, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[1], **metrics})
        for key in totals[CANDIDATES[1]]:
            totals[CANDIDATES[1]][key] += metrics[key]

        alpha = fit_channel_correction(
            rotated_reference[list(calibration_keys)],
            rotated_quantized[list(calibration_keys)],
        )
        corrected = rotated_quantized * alpha
        metrics = query_weighted_error(
            rotated_queries, rotated_reference, corrected, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[2],
                         "alpha_min": float(alpha.min()),
                         "alpha_max": float(alpha.max()), **metrics})
        for key in totals[CANDIDATES[2]]:
            totals[CANDIDATES[2]][key] += metrics[key]

    aggregate: dict[str, dict[str, float | int]] = {}
    for candidate, values in totals.items():
        numerator = float(values["numerator"])
        denominator = float(values["denominator"])
        if denominator <= 0:
            raise ValueError(f"candidate has nonpositive denominator: {candidate}")
        aggregate[candidate] = {
            **values,
            "relative_rmse": math.sqrt(numerator / denominator),
        }
    winner = min(CANDIDATES, key=lambda name: aggregate[name]["relative_rmse"])
    best_error = float(aggregate[winner]["relative_rmse"])
    verdict = "PASS" if best_error <= MAXIMUM_RELATIVE_RMSE else "FAIL"
    manifest = {
        "schema": "glm52-w9-fp4-falsifier-manifest-v1",
        "candidate_commit": args.candidate_commit,
        "source_bindings": source_bindings,
        "capture_root": str(capture),
        "capture_hashes": CAPTURE_HASHES,
        "randomness_receipt_sha256": sha256_file(receipt_path),
        "drand_round": receipt["round"],
        "minimum_drand_round": args.minimum_drand_round,
        "master_seed_sha256": hashlib.sha256(master_seed).hexdigest(),
        "split_hashes": split_hashes,
        "layers": list(LAYERS),
        "candidates": list(CANDIDATES),
        "block_width": BLOCK_WIDTH,
        "scale": "continuous_f32_amax_div_6",
        "numpy_version": np.__version__,
    }
    summary = {
        "schema": "glm52-w9-fp4-falsifier-summary-v1",
        "gate": "W9",
        "checks": {
            "real_capture": True,
            "capture_width_512": True,
            "source_hashes_and_shapes": True,
            "finite_data": True,
            "calibration_heldout_disjoint": True,
            "all_layers_contributed": all(
                aggregate[name]["query_rows"] > 0 and aggregate[name]["pairs"] > 0
                for name in CANDIDATES),
        },
        "formula": "PASS iff every check is true and the minimum preregistered held-out query-weighted relative RMSE is <= 0.05",
        "maximum_allowed_error": MAXIMUM_RELATIVE_RMSE,
        "candidates": aggregate,
        "winner": winner,
        "query_weighted_error": best_error,
        "verdict": verdict,
        "interpretation": "PASS only authorizes an exact packed-format plus fixed 100-case NLL/top-1 gate; FAIL stops W9 before kernel work.",
    }
    if not all(summary["checks"].values()):
        summary["verdict"] = "FAIL"
    return manifest, raw_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--randomness-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--minimum-drand-round", required=True, type=int)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit("output path must not exist")
    output.mkdir(mode=0o700, parents=False)
    try:
        manifest, raw_rows, summary = run(args)
        write_json(output / "manifest.json", manifest)
        with (output / "raw.jsonl").open("x", encoding="utf-8") as handle:
            for row in raw_rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_json(output / "summary.json", summary)
        terminal = {
            "schema": "glm52-w9-fp4-falsifier-terminal-v1",
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "raw_sha256": sha256_file(output / "raw.jsonl"),
            "summary_sha256": sha256_file(output / "summary.json"),
            "verdict": summary["verdict"],
        }
        write_json(output / "terminal-receipt.json", terminal)
        print(f"W9_FP4_FALSIFIER_{summary['verdict']} {output}")
        return 0 if summary["verdict"] == "PASS" else 1
    except BaseException as error:
        failure = {
            "schema": "glm52-w9-fp4-falsifier-failure-v1",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    sys.exit(main())
