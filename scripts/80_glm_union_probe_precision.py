#!/usr/bin/env python3
"""Fit the selected GLM union probe and run the frozen feature-precision diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import time
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CV_PATH = ROOT / "scripts/79_glm_union_probe_cv.py"
FREEZE = ROOT / "results/glm52-gates/R0c-union-probe-p1-precision-freeze.json"
SPLIT_RECEIPT = ROOT / "results/glm52-gates/R0c-union-probe-splits-pass-76faed9.json"
DIAGNOSTIC = Path("/home/bmarti44/.local/state/glm52-p1-splits-r127-76faed9/train-precision-diagnostic")
RANK = 32
EVENT_FIELDS = (
    "global_row", "local_row", "prediction_row", "request", "target_size",
    "q4_hits", "fp16_hits", "top32_overlap_count",
)


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CV = _load_module("glm_union_probe_cv_for_precision", CV_PATH)
PROBE = CV.PROBE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"diagnostic input is not regular: {path}")
        payload = bytearray()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"diagnostic input ended early: {path}")
            payload.extend(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"diagnostic input grew while being snapshotted: {path}")
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size) !=
                (after.st_dev, after.st_ino, after.st_size)):
            raise ValueError(f"diagnostic input identity changed: {path}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def diagnostic_pair_metrics(
    requests: np.ndarray,
    targets: np.ndarray,
    q4_logits: np.ndarray,
    fp16_logits: np.ndarray,
) -> dict[str, object]:
    """Score paired logits and their top-32 overlap without changing row order."""
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(targets, np.ndarray) or targets.shape != (requests.size, 256) or
        targets.dtype != np.bool_ or
        not isinstance(q4_logits, np.ndarray) or q4_logits.shape != targets.shape or
        not isinstance(fp16_logits, np.ndarray) or fp16_logits.shape != targets.shape or
        not np.issubdtype(q4_logits.dtype, np.floating) or
        not np.issubdtype(fp16_logits.dtype, np.floating) or
        not np.isfinite(q4_logits).all() or not np.isfinite(fp16_logits).all()
    ):
        raise ValueError("paired diagnostic inputs are malformed")
    q4_rank = np.argsort(-q4_logits, axis=1, kind="stable").astype(np.uint16)
    fp16_rank = np.argsort(-fp16_logits, axis=1, kind="stable").astype(np.uint16)
    q4_raw = CV.event_evidence(requests, targets, q4_rank)
    fp16_raw = CV.event_evidence(requests, targets, fp16_rank)
    overlaps = np.asarray([
        len(set(q4_rank[row, :32]).intersection(fp16_rank[row, :32]))
        for row in range(requests.size)
    ], dtype=np.uint8)
    request_overlap = {
        str(int(request)): {
            "overlap_sum": int(overlaps[requests == request].sum()),
            "events": int(np.sum(requests == request)),
        }
        for request in np.unique(requests)
    }
    return {
        "q4": CV.score_event_evidence(
            q4_raw["request"], q4_raw["target_size"], q4_raw["hits"], CV.BUDGETS,
        ),
        "fp16": CV.score_event_evidence(
            fp16_raw["request"], fp16_raw["target_size"], fp16_raw["hits"], CV.BUDGETS,
        ),
        "top32_overlap": request_overlap,
        "evidence": {
            "request": requests.astype(np.uint16, copy=True),
            "target_size": q4_raw["target_size"],
            "q4_hits": q4_raw["hits"],
            "fp16_hits": fp16_raw["hits"],
            "top32_overlap_count": overlaps,
        },
    }


def aggregate_overlap(
    layers: list[dict[str, dict[str, int]]],
    expected_requests: set[str] | None = None,
) -> dict[str, float | int]:
    if not layers:
        raise ValueError("top-32 overlap evidence is missing")
    totals: dict[str, dict[str, int]] = {}
    for layer in layers:
        for request, record in layer.items():
            if set(record) != {"overlap_sum", "events"} or record["events"] <= 0:
                raise ValueError("top-32 overlap record is malformed")
            target = totals.setdefault(request, {"overlap_sum": 0, "events": 0})
            target["overlap_sum"] += record["overlap_sum"]
            target["events"] += record["events"]
    if expected_requests is not None and set(totals) != expected_requests:
        raise ValueError("precision overlap request coverage differs")
    return {
        "requests": len(totals),
        "events": sum(value["events"] for value in totals.values()),
        "event_weighted_overlap": (
            sum(value["overlap_sum"] for value in totals.values()) /
            (32 * sum(value["events"] for value in totals.values()))
        ),
        "macro_request_overlap": float(np.mean([
            value["overlap_sum"] / (32 * value["events"]) for value in totals.values()
        ])),
    }


def aggregate_sparse_request_metrics(
    layers: list[dict[str, dict[str, dict[str, float | int]]]],
    expected_requests: set[str],
) -> dict[str, dict[str, float | int]]:
    """Aggregate sparse layer events within request, then macro-average requests."""
    if (
        not isinstance(layers, list) or not layers or
        any(not isinstance(layer, dict) for layer in layers) or
        not isinstance(expected_requests, set) or not expected_requests or
        any(not isinstance(request, str) or not request.isdigit() for request in expected_requests)
    ):
        raise ValueError("precision sparse metrics are missing")
    budgets = set(layers[0])
    if not budgets or any(set(layer) != budgets for layer in layers):
        raise ValueError("precision sparse metric budget sets differ")
    required = {"recall_sum", "precision_sum", "wasted_sum", "coverage_sum", "events"}
    output: dict[str, dict[str, float | int]] = {}
    for budget in sorted(budgets, key=int):
        observed_requests = set().union(*(set(layer[budget]) for layer in layers))
        if observed_requests != expected_requests:
            raise ValueError("precision sparse request coverage differs")
        totals = {
            request: {name: 0.0 for name in required - {"events"}} | {"events": 0}
            for request in sorted(expected_requests, key=int)
        }
        for layer in layers:
            for request, record in layer[budget].items():
                if (
                    request not in expected_requests or set(record) != required or
                    not isinstance(record["events"], int) or record["events"] <= 0
                ):
                    raise ValueError("precision sparse request metric record is malformed")
                for name in required - {"events"}:
                    value = record[name]
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("precision sparse request metric is non-finite")
                    totals[request][name] += float(value)
                totals[request]["events"] += record["events"]
        if any(record["events"] <= 0 for record in totals.values()):
            raise ValueError("precision sparse request has no events")
        request_means = {
            name: [totals[request][name] / totals[request]["events"] for request in totals]
            for name in required - {"events"}
        }
        event_count = sum(record["events"] for record in totals.values())
        output[budget] = {
            "requests": len(totals),
            "events": event_count,
            "macro_request_recall": float(np.mean(request_means["recall_sum"])),
            "macro_request_precision": float(np.mean(request_means["precision_sum"])),
            "macro_request_wasted_experts": float(np.mean(request_means["wasted_sum"])),
            "macro_request_full_set_coverage": float(np.mean(request_means["coverage_sum"])),
            "event_weighted_recall": sum(value["recall_sum"] for value in totals.values()) / event_count,
            "event_weighted_precision": sum(value["precision_sum"] for value in totals.values()) / event_count,
            "event_weighted_wasted_experts": sum(value["wasted_sum"] for value in totals.values()) / event_count,
            "event_weighted_full_set_coverage": sum(value["coverage_sum"] for value in totals.values()) / event_count,
        }
    return output


def _load_diagnostic() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    receipt = json.loads(PROBE._tracked_bytes(SPLIT_RECEIPT).decode("utf-8"))
    binding = receipt["observed"]["splits"]["train-precision-diagnostic"]
    manifest_path = DIAGNOSTIC / "manifest.json"
    records_path = DIAGNOSTIC / "records.npz"
    manifest_bytes = _snapshot_regular(manifest_path)
    records_bytes = _snapshot_regular(records_path)
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != binding["manifest_sha256"] or
        hashlib.sha256(records_bytes).hexdigest() != binding["output_sha256"] or
        len(records_bytes) != binding["output_bytes"]
    ):
        raise ValueError("precision-diagnostic split differs from its binding")
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    if (
        manifest.get("split") != "train-precision-diagnostic" or
        manifest.get("requests") != 5 or manifest.get("rows") != 12225
    ):
        raise ValueError("precision-diagnostic manifest differs")
    expected_names = {
        "hidden_q4", "hidden_scale", "layer", "request_index", "selected_ids",
        "token_position", "top_ids", "top_logits", "hidden_fp16_holdout_row",
        "hidden_fp16_holdout",
    }
    with zipfile.ZipFile(io.BytesIO(records_bytes), "r") as raw_archive:
        members = raw_archive.namelist()
        if (
            len(members) != len(expected_names) or
            set(members) != {f"{name}.npy" for name in expected_names} or
            raw_archive.testzip() is not None
        ):
            raise ValueError("precision-diagnostic archive members differ")
    with np.load(io.BytesIO(records_bytes), allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("precision-diagnostic array set differs")
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_diagnostic_arrays(arrays)
    return arrays, binding


def validate_diagnostic_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected = {
        "request_index": ((12225,), np.dtype(np.uint16)),
        "layer": ((12225,), np.dtype(np.uint16)),
        "token_position": ((12225,), np.dtype(np.uint32)),
        "selected_ids": ((12225, 8), np.dtype(np.uint8)),
        "hidden_q4": ((12225, 3072), np.dtype(np.uint8)),
        "hidden_scale": ((12225, 192), np.dtype(np.float16)),
        "top_ids": ((12225, 32), np.dtype(np.uint8)),
        "top_logits": ((12225, 32), np.dtype(np.float16)),
        "hidden_fp16_holdout_row": ((203,), np.dtype(np.uint32)),
        "hidden_fp16_holdout": ((203, 6144), np.dtype(np.float16)),
    }
    if (
        not isinstance(arrays, dict) or set(arrays) != set(expected) or
        any(not isinstance(arrays[name], np.ndarray) or arrays[name].shape != shape or
            arrays[name].dtype != dtype for name, (shape, dtype) in expected.items())
    ):
        raise ValueError("precision-diagnostic tensor schema differs")
    holdout = arrays["hidden_fp16_holdout_row"].astype(np.int64)
    if (
        np.any(arrays["request_index"] <= 0) or
        np.any((arrays["layer"] < 3) | (arrays["layer"] > 77)) or
        np.any(np.diff(holdout) <= 0) or holdout[0] < 0 or holdout[-1] >= 12225 or
        any(np.unique(row).size != 8 for row in arrays["selected_ids"]) or
        any(np.unique(row).size != 32 for row in arrays["top_ids"]) or
        not np.isfinite(arrays["hidden_scale"]).all() or
        not np.isfinite(arrays["top_logits"]).all() or
        not np.isfinite(arrays["hidden_fp16_holdout"]).all()
    ):
        raise ValueError("precision-diagnostic tensor values violate the frozen contract")


def _state_from_file(path: Path, schema: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    state = CV._read_bound_npz(path, schema)
    if set(state) != {"down.weight", "up.weight", "up.bias"}:
        raise ValueError("final probe state tensor set differs")
    return state


def fit_rank32_layer(
    sources: list[dict[str, np.ndarray]], layer_id: int, *, device: str = "cuda",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Deterministically fit one final head from authorized training sources."""
    data = CV._layer_arrays(sources, layer_id)
    request = data["request_index"]
    rows, targets, valid = PROBE.multi_k_targets(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    hidden = PROBE.unpack_probe_hidden(data["hidden_q4"], data["hidden_scale"]).astype(np.float16)
    history = PROBE.causal_expert_history(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    features = np.concatenate((hidden, history.astype(np.float16)), axis=1)[rows]
    weights = PROBE.request_balanced_weights(request, rows, valid)
    fit_rows = np.arange(rows.size, dtype=np.int64)
    return PROBE.train_probe_head(
        features, targets, valid, weights, fit_rows, RANK,
        epochs=8, batch_rows=512, seed=20260805, device=device,
    )


def validate_model_training(
    requested: Path,
    model_layers: dict[str, object],
    source_binding: dict[str, object],
    *,
    device: str = "cuda",
) -> None:
    """Retrain independently and exact-compare every persisted model state."""
    sources, groups = CV._load_authorized_sources(source_binding)
    try:
        for layer_id in CV.LAYERS:
            record = model_layers[str(layer_id)]
            observed = _state_from_file(requested / record["file"], record["schema"])
            expected, report = fit_rank32_layer(sources, layer_id, device=device)
            if report != record["training"] or set(observed) != set(expected):
                raise ValueError(f"precision trained model differs: layer {layer_id}")
            for name, value in expected.items():
                if (
                    observed[name].dtype != value.dtype or
                    observed[name].shape != value.shape or
                    not np.array_equal(observed[name], value)
                ):
                    raise ValueError(f"precision trained model differs: layer {layer_id} {name}")
            print(json.dumps({"retrained_layer": layer_id}, sort_keys=True), flush=True)
            del observed, expected
            gc.collect()
    finally:
        del sources, groups
        gc.collect()


def diagnostic_layer_contract(
    diagnostic: dict[str, np.ndarray], layer_id: int,
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray,
]:
    global_rows = np.flatnonzero(diagnostic["layer"] == layer_id)
    holdout_global = diagnostic["hidden_fp16_holdout_row"].astype(np.int64)
    selected_mask = np.isin(holdout_global, global_rows)
    selected_global = holdout_global[selected_mask]
    local_rows = np.searchsorted(global_rows, selected_global).astype(np.int64)
    if selected_global.size and not np.array_equal(global_rows[local_rows], selected_global):
        raise ValueError("diagnostic holdout row does not map to its layer")
    data = {
        name: value[global_rows] for name, value in diagnostic.items()
        if name not in {"hidden_fp16_holdout_row", "hidden_fp16_holdout"}
    }
    rows, targets, valid = PROBE.multi_k_targets(
        data["request_index"], data["layer"], data["token_position"], data["selected_ids"],
    )
    prediction_index = np.full(data["layer"].size, -1, dtype=np.int64)
    prediction_index[rows] = np.arange(rows.size)
    selected_prediction = prediction_index[local_rows]
    scorable = selected_prediction >= 0
    contracts = {}
    for k_index, k in enumerate(CV.K_VALUES):
        active = np.zeros(selected_prediction.size, dtype=np.bool_)
        active[scorable] = valid[selected_prediction[scorable], k_index]
        active_prediction = selected_prediction[active]
        contracts[str(k)] = {
            "global_row": selected_global[active].astype(np.uint32),
            "local_row": local_rows[active].astype(np.uint32),
            "prediction_row": active_prediction.astype(np.uint32),
            "request": data["request_index"][local_rows[active]].astype(np.uint16),
            "target_size": targets[active_prediction, k_index].sum(axis=1).astype(np.uint8),
            "targets": targets[active_prediction, k_index],
        }
    return data, contracts, selected_mask, scorable


def regenerate_diagnostic_events(
    requested: Path,
    model_layers: dict[str, object],
    diagnostic: dict[str, np.ndarray],
    *,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Regenerate integer evidence from bound states and exact paired features."""
    event_arrays: dict[str, np.ndarray] = {}
    for layer_id in CV.LAYERS:
        data, contracts, selected_mask, scorable = diagnostic_layer_contract(diagnostic, layer_id)
        selected_global = diagnostic["hidden_fp16_holdout_row"].astype(np.int64)[selected_mask]
        scorable_global = selected_global[scorable]
        if scorable_global.size:
            global_rows = np.flatnonzero(diagnostic["layer"] == layer_id)
            local_rows = np.searchsorted(global_rows, scorable_global)
            history = PROBE.causal_expert_history(
                data["request_index"], data["layer"], data["token_position"], data["selected_ids"],
            )
            q4_hidden = PROBE.unpack_probe_hidden(data["hidden_q4"], data["hidden_scale"])
            fp16_hidden = diagnostic["hidden_fp16_holdout"][selected_mask][scorable].astype(np.float32)
            q4_features = np.concatenate(
                (q4_hidden[local_rows], history[local_rows]), axis=1,
            ).astype(np.float16)
            fp16_features = np.concatenate(
                (fp16_hidden, history[local_rows]), axis=1,
            ).astype(np.float16)
            state_record = model_layers[str(layer_id)]
            state = _state_from_file(requested / state_record["file"], state_record["schema"])
            q4_logits = PROBE.predict_probe_head(q4_features, state, RANK, device=device)
            fp16_logits = PROBE.predict_probe_head(fp16_features, state, RANK, device=device)
        for k_index, k in enumerate(CV.K_VALUES):
            contract = contracts[str(k)]
            count = contract["request"].size
            prefix = f"layer{layer_id}_k{k}_"
            if count:
                active = np.isin(scorable_global, contract["global_row"])
                paired = diagnostic_pair_metrics(
                    contract["request"], contract["targets"],
                    q4_logits[active, k_index], fp16_logits[active, k_index],
                )
                values = paired["evidence"]
            else:
                values = {
                    "request": np.empty(0, dtype=np.uint16),
                    "target_size": np.empty(0, dtype=np.uint8),
                    "q4_hits": np.empty((0, len(CV.BUDGETS)), dtype=np.uint8),
                    "fp16_hits": np.empty((0, len(CV.BUDGETS)), dtype=np.uint8),
                    "top32_overlap_count": np.empty(0, dtype=np.uint8),
                }
            for field in ("global_row", "local_row", "prediction_row"):
                event_arrays[prefix + field] = contract[field]
            for field in ("request", "target_size", "q4_hits", "fp16_hits", "top32_overlap_count"):
                event_arrays[prefix + field] = values[field]
        print(json.dumps({
            "diagnosed_layer": layer_id, "paired_rows": int(scorable_global.size),
            "retained_rows": int(selected_global.size),
        }, sort_keys=True), flush=True)
    return event_arrays


def validate_semantic_events(
    event_path: Path,
    binding: dict[str, object],
    requested: Path,
    model_layers: dict[str, object],
    diagnostic: dict[str, np.ndarray],
    *,
    device: str = "cuda",
) -> None:
    """Require persisted event evidence to equal fresh model inference exactly."""
    if not isinstance(binding, dict) or not isinstance(binding.get("schema"), dict):
        raise ValueError("precision semantic event binding differs")
    persisted = CV._read_bound_npz(event_path, binding["schema"])
    regenerated = regenerate_diagnostic_events(
        requested, model_layers, diagnostic, device=device,
    )
    if set(persisted) != set(regenerated):
        raise ValueError("precision semantic event members differ")
    for name, expected in regenerated.items():
        observed = persisted[name]
        if (
            observed.dtype != expected.dtype or
            observed.shape != expected.shape or
            not np.array_equal(observed, expected)
        ):
            raise ValueError(f"precision semantic event differs: {name}")


def replay_diagnostic_events(
    event_path: Path,
    binding: dict[str, object],
    diagnostic: dict[str, np.ndarray],
) -> dict[str, object]:
    expected_names = {
        f"layer{layer}_k{k}_{field}"
        for layer in CV.LAYERS for k in CV.K_VALUES for field in EVENT_FIELDS
    }
    if (
        not isinstance(binding, dict) or set(binding) != {"sha256", "bytes", "schema"} or
        binding.get("sha256") != _sha256(event_path) or
        binding.get("bytes") != event_path.stat().st_size or
        not isinstance(binding.get("schema"), dict) or
        set(binding["schema"]) != expected_names
    ):
        raise ValueError("precision event archive binding differs")
    evidence = CV._read_bound_npz(event_path, binding["schema"])
    q4_layers = {str(k): [] for k in CV.K_VALUES}
    fp16_layers = {str(k): [] for k in CV.K_VALUES}
    overlap_layers = {str(k): [] for k in CV.K_VALUES}
    expected_requests = {str(k): set() for k in CV.K_VALUES}
    zero_cells = []
    for layer_id in CV.LAYERS:
        _data, contracts, _selected, _scorable = diagnostic_layer_contract(diagnostic, layer_id)
        for k in CV.K_VALUES:
            prefix = f"layer{layer_id}_k{k}_"
            contract = contracts[str(k)]
            for field in ("global_row", "local_row", "prediction_row", "request", "target_size"):
                if not np.array_equal(evidence[prefix + field], contract[field]):
                    raise ValueError(f"precision event row contract differs: layer {layer_id} K{k}")
            count = contract["request"].size
            expected_shapes = {
                "q4_hits": (count, len(CV.BUDGETS)),
                "fp16_hits": (count, len(CV.BUDGETS)),
                "top32_overlap_count": (count,),
            }
            for field, shape in expected_shapes.items():
                value = evidence[prefix + field]
                if value.dtype != np.uint8 or value.shape != shape:
                    raise ValueError("precision event metric schema differs")
            if count == 0:
                zero_cells.append({"layer": layer_id, "K": k})
                continue
            expected_requests[str(k)].update(map(str, np.unique(contract["request"])))
            q4 = CV.score_event_evidence(
                contract["request"], contract["target_size"], evidence[prefix + "q4_hits"],
                CV.BUDGETS,
            )
            fp16 = CV.score_event_evidence(
                contract["request"], contract["target_size"], evidence[prefix + "fp16_hits"],
                CV.BUDGETS,
            )
            overlap = evidence[prefix + "top32_overlap_count"]
            if np.any(overlap > 32):
                raise ValueError("precision top-32 overlap is impossible")
            overlap_record = {
                str(int(request)): {
                    "overlap_sum": int(overlap[contract["request"] == request].sum()),
                    "events": int(np.sum(contract["request"] == request)),
                }
                for request in np.unique(contract["request"])
            }
            q4_layers[str(k)].append(q4)
            fp16_layers[str(k)].append(fp16)
            overlap_layers[str(k)].append(overlap_record)
    if any(not values for values in [*q4_layers.values(), *fp16_layers.values(), *overlap_layers.values()]):
        raise ValueError("precision evidence has no nonempty K coverage")
    diagnostic_requests = set(map(str, np.unique(diagnostic["request_index"])))
    if any(requests != diagnostic_requests for requests in expected_requests.values()):
        raise ValueError("precision diagnostic request coverage differs by K")
    scorable_rows = int(sum(
        diagnostic_layer_contract(diagnostic, layer)[3].sum() for layer in CV.LAYERS
    ))
    return {
        "q4": {
            k: aggregate_sparse_request_metrics(values, expected_requests[k])
            for k, values in q4_layers.items()
        },
        "fp16": {
            k: aggregate_sparse_request_metrics(values, expected_requests[k])
            for k, values in fp16_layers.items()
        },
        "top32_overlap": {
            k: aggregate_overlap(values, expected_requests[k])
            for k, values in overlap_layers.items()
        },
        "coverage": {
            "holdout_rows": int(diagnostic["hidden_fp16_holdout_row"].size),
            "scorable_rows": scorable_rows,
            "unscorable_rows": int(diagnostic["hidden_fp16_holdout_row"].size) - scorable_rows,
            "events_by_K": {
                k: int(sum(
                    diagnostic_layer_contract(diagnostic, layer)[1][k]["request"].size
                    for layer in CV.LAYERS
                )) for k in map(str, CV.K_VALUES)
            },
            "zero_cells": zero_cells,
        },
    }


def build_precision_summary(
    repository_head: str,
    manifest_sha256: str,
    model_manifest_sha256: str,
    diagnostic_binding: dict[str, object],
    event_binding: dict[str, object],
    replayed: dict[str, object],
    runtime_final_sha256: str,
) -> dict[str, object]:
    q4_headline = replayed["q4"]["4"]["32"]
    fp16_headline = replayed["fp16"]["4"]["32"]
    macro_deficit = 100 * (
        fp16_headline["macro_request_recall"] - q4_headline["macro_request_recall"]
    )
    event_deficit = 100 * (
        fp16_headline["event_weighted_recall"] - q4_headline["event_weighted_recall"]
    )
    overlap = replayed["top32_overlap"]["4"]["event_weighted_overlap"]
    verdict = "PASS" if (
        macro_deficit <= 2.0 and event_deficit <= 2.0 and overlap >= 0.9
    ) else "FAIL"
    return {
        "schema_version": 1,
        "classification": "P1_FEATURE_PRECISION_COMPLETE",
        "verdict": verdict,
        "repository_head": repository_head,
        "manifest_sha256": manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "diagnostic_binding": diagnostic_binding,
        "diagnostic_event_binding": event_binding,
        "q4": replayed["q4"],
        "fp16": replayed["fp16"],
        "top32_overlap": replayed["top32_overlap"],
        "coverage": replayed["coverage"],
        "headline": {
            "K": 4,
            "budget": 32,
            "macro_request_recall_deficit_pp": macro_deficit,
            "event_weighted_recall_deficit_pp": event_deficit,
            "event_weighted_top32_overlap": overlap,
            "macro_request_recall_deficit_pp_max": 2.0,
            "event_weighted_recall_deficit_pp_max": 2.0,
            "mean_top32_overlap_min": 0.9,
        },
        "runtime_final_sha256": runtime_final_sha256,
        "claim_limit": (
            "Five-case feature-precision diagnostic only. Calibration/test values were not scored "
            "or used, but their NPZ members were previously materialized for schema inspection; "
            "see R0c-split-isolation-incident-2026-08-05.json."
        ),
    }


def validate_completed_output(requested: Path) -> dict[str, object]:
    expected_files = {
        "manifest.json", "runtime-start.json", "model-manifest.json",
        "diagnostic-events.npz", "runtime-final.json", "summary.json",
        *(f"layer-{layer:03d}-rank32.npz" for layer in CV.LAYERS),
    }
    if {path.name for path in requested.iterdir()} != expected_files:
        raise ValueError("precision completed artifact set differs")
    manifest_path = requested / "manifest.json"
    manifest = CV._read_json_snapshot(manifest_path)
    source_binding = PROBE.validate_training_sources(CV.QUALITY, CV.LONGS)
    if (
        set(manifest) != {
            "schema_version", "classification", "repository_head", "driver_sha256",
            "cv_driver_sha256", "probe_sha256", "freeze_sha256", "runtime_start_sha256",
            "training_source_binding", "selected_rank", "layers",
        } or manifest.get("schema_version") != 1 or
        manifest.get("classification") != "P1_PRECISION_IN_PROGRESS" or
        manifest.get("repository_head") != CV._repository_head() or
        manifest.get("driver_sha256") != _sha256(Path(__file__).resolve()) or
        manifest.get("cv_driver_sha256") != _sha256(CV_PATH) or
        manifest.get("probe_sha256") != _sha256(CV.PROBE_PATH) or
        manifest.get("freeze_sha256") != _sha256(FREEZE) or
        manifest.get("training_source_binding") != source_binding or
        manifest.get("selected_rank") != RANK or manifest.get("layers") != list(CV.LAYERS)
    ):
        raise ValueError("precision completed manifest differs")
    runtime_start_path = requested / "runtime-start.json"
    runtime_start = CV._read_json_snapshot(runtime_start_path)
    if (
        manifest["runtime_start_sha256"] != _sha256(runtime_start_path) or
        runtime_start.get("classification") != "P1_PRECISION_RUNTIME_START" or
        runtime_start.get("kernel_faults_at_start") != []
    ):
        raise ValueError("precision runtime-start binding differs")
    model_manifest_path = requested / "model-manifest.json"
    model_manifest = CV._read_json_snapshot(model_manifest_path)
    if (
        set(model_manifest) != {
            "schema_version", "classification", "parent_manifest_sha256",
            "diagnostic_opened", "layers",
        } or model_manifest.get("schema_version") != 1 or
        model_manifest.get("classification") != "P1_RANK32_FINAL_MODELS" or
        model_manifest.get("parent_manifest_sha256") != _sha256(manifest_path) or
        model_manifest.get("diagnostic_opened") is not False or
        not isinstance(model_manifest.get("layers"), dict) or
        set(model_manifest["layers"]) != {str(layer) for layer in CV.LAYERS}
    ):
        raise ValueError("precision model manifest differs")
    for layer in CV.LAYERS:
        record = model_manifest["layers"][str(layer)]
        if (
            not isinstance(record, dict) or set(record) != {
                "file", "sha256", "bytes", "schema", "training",
            } or record.get("file") != f"layer-{layer:03d}-rank32.npz" or
            record.get("sha256") != _sha256(requested / record["file"]) or
            record.get("bytes") != (requested / record["file"]).stat().st_size or
            not isinstance(record.get("schema"), dict) or
            not isinstance(record.get("training"), dict) or
            record["training"].get("rank") != RANK or
            record["training"].get("epochs") != 8 or
            record["training"].get("batch_rows") != 512 or
            record["training"].get("seed") != 20260805 or
            record["training"].get("deterministic_algorithms") is not True
        ):
            raise ValueError("precision model layer binding differs")
        _state_from_file(requested / record["file"], record["schema"])
    validate_model_training(
        requested, model_manifest["layers"], source_binding, device="cuda",
    )
    diagnostic, diagnostic_binding = _load_diagnostic()
    summary = CV._read_json_snapshot(requested / "summary.json")
    event_binding = summary.get("diagnostic_event_binding")
    validate_semantic_events(
        requested / "diagnostic-events.npz", event_binding, requested,
        model_manifest["layers"], diagnostic,
    )
    replayed = replay_diagnostic_events(
        requested / "diagnostic-events.npz", event_binding, diagnostic,
    )
    runtime_final_path = requested / "runtime-final.json"
    runtime_final = CV._read_json_snapshot(runtime_final_path)
    if (
        set(runtime_final) != {
            "schema_version", "classification", "start_epoch", "end_epoch", "post_gpu",
            "post_mem_available_kib", "kernel_log", "kernel_log_sha256", "kernel_faults",
        } or runtime_final.get("schema_version") != 1 or
        runtime_final.get("classification") != "P1_PRECISION_RUNTIME_PASS" or
        runtime_final.get("start_epoch") != runtime_start.get("start_epoch") or
        not isinstance(runtime_final.get("end_epoch"), int) or
        runtime_final["end_epoch"] < runtime_start["start_epoch"] or
        not isinstance(runtime_final.get("kernel_log"), str) or
        runtime_final.get("kernel_log_sha256") != hashlib.sha256(
            runtime_final["kernel_log"].encode()
        ).hexdigest() or runtime_final.get("kernel_faults") != [] or
        CV.runtime_fault_lines(runtime_final["kernel_log"]) or
        not isinstance(runtime_final.get("post_mem_available_kib"), int) or
        runtime_final["post_mem_available_kib"] <= 0
    ):
        raise ValueError("precision runtime-final evidence differs")
    expected = build_precision_summary(
        manifest["repository_head"], _sha256(manifest_path), _sha256(model_manifest_path),
        diagnostic_binding, event_binding, replayed, _sha256(runtime_final_path),
    )
    if summary != expected:
        raise ValueError("precision summary does not replay persisted evidence")
    return summary


def execute(out_dir: Path) -> int:
    PROBE._tracked_bytes(Path(__file__).resolve())
    PROBE._tracked_bytes(CV_PATH)
    PROBE._tracked_bytes(FREEZE)
    source_binding = PROBE.validate_training_sources(CV.QUALITY, CV.LONGS)
    requested = out_dir.absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("precision diagnostic requires a fresh output path")
    requested.mkdir(mode=0o700, parents=False)
    start_epoch = int(time.time())
    runtime_start = {
        "schema_version": 1,
        "classification": "P1_PRECISION_RUNTIME_START",
        "start_epoch": start_epoch,
        "mem_available_kib": CV._mem_available_kib(),
        "gpu": CV._gpu_snapshot(),
        "kernel_faults_at_start": CV.runtime_fault_lines(CV._kernel_log_since(start_epoch)),
    }
    if runtime_start["kernel_faults_at_start"]:
        raise RuntimeError("runtime fault exists at precision start")
    CV._write_json_exclusive(requested / "runtime-start.json", runtime_start)
    manifest = {
        "schema_version": 1,
        "classification": "P1_PRECISION_IN_PROGRESS",
        "repository_head": CV._repository_head(),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "cv_driver_sha256": _sha256(CV_PATH),
        "probe_sha256": _sha256(CV.PROBE_PATH),
        "freeze_sha256": _sha256(FREEZE),
        "runtime_start_sha256": _sha256(requested / "runtime-start.json"),
        "training_source_binding": source_binding,
        "selected_rank": RANK,
        "layers": list(CV.LAYERS),
    }
    CV._write_json_exclusive(requested / "manifest.json", manifest)

    sources, groups = CV._load_authorized_sources(source_binding)
    model_layers: dict[str, object] = {}
    for layer_id in CV.LAYERS:
        state, report = fit_rank32_layer(sources, layer_id, device="cuda")
        state_path = requested / f"layer-{layer_id:03d}-rank32.npz"
        binding = CV._write_npz_exclusive(state_path, state)
        model_layers[str(layer_id)] = {
            "file": state_path.name,
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "schema": binding["schema"],
            "training": report,
        }
        if CV.runtime_fault_lines(CV._kernel_log_since(start_epoch)):
            raise RuntimeError("runtime fault appeared during final probe fitting")
        print(json.dumps({"trained_layer": layer_id}, sort_keys=True), flush=True)
        del state
        gc.collect()
    model_manifest = {
        "schema_version": 1,
        "classification": "P1_RANK32_FINAL_MODELS",
        "parent_manifest_sha256": _sha256(requested / "manifest.json"),
        "diagnostic_opened": False,
        "layers": model_layers,
    }
    CV._write_json_exclusive(requested / "model-manifest.json", model_manifest)
    model_manifest_sha256 = _sha256(requested / "model-manifest.json")
    del sources, groups
    gc.collect()

    diagnostic, diagnostic_binding = _load_diagnostic()
    event_arrays = regenerate_diagnostic_events(
        requested, model_layers, diagnostic, device="cuda",
    )
    event_binding = CV._write_npz_exclusive(requested / "diagnostic-events.npz", event_arrays)
    validate_semantic_events(
        requested / "diagnostic-events.npz", event_binding, requested,
        model_layers, diagnostic, device="cuda",
    )
    replayed = replay_diagnostic_events(
        requested / "diagnostic-events.npz", event_binding, diagnostic,
    )
    q4_summary = replayed["q4"]
    fp16_summary = replayed["fp16"]
    overlap_summary = replayed["top32_overlap"]
    post_gpu = CV._gpu_snapshot()
    post_mem = CV._mem_available_kib()
    kernel_log = CV._kernel_log_since(start_epoch)
    faults = CV.runtime_fault_lines(kernel_log)
    if faults:
        raise RuntimeError("runtime fault appeared before precision completion")
    runtime_final = {
        "schema_version": 1,
        "classification": "P1_PRECISION_RUNTIME_PASS",
        "start_epoch": start_epoch,
        "end_epoch": int(time.time()),
        "post_gpu": post_gpu,
        "post_mem_available_kib": post_mem,
        "kernel_log": kernel_log,
        "kernel_log_sha256": hashlib.sha256(kernel_log.encode()).hexdigest(),
        "kernel_faults": faults,
    }
    CV._write_json_exclusive(requested / "runtime-final.json", runtime_final)
    summary = build_precision_summary(
        manifest["repository_head"], _sha256(requested / "manifest.json"),
        model_manifest_sha256, diagnostic_binding, event_binding, replayed,
        _sha256(requested / "runtime-final.json"),
    )
    CV._write_json_exclusive(requested / "summary.json", summary)
    validated = validate_completed_output(requested)
    post_validation_faults = CV.runtime_fault_lines(CV._kernel_log_since(start_epoch))
    if post_validation_faults:
        raise RuntimeError("runtime fault appeared during completed precision validation")
    print(json.dumps(validated, sort_keys=True, indent=2, allow_nan=False))
    return 0 if validated["verdict"] == "PASS" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("run",))
    result.add_argument("--out-dir", required=True, type=Path)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(execute(arguments.out_dir))
