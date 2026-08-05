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


def aggregate_overlap(layers: list[dict[str, dict[str, int]]]) -> dict[str, float | int]:
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
    if (
        arrays["hidden_fp16_holdout_row"].shape != (203,) or
        arrays["hidden_fp16_holdout"].shape != (203, 6144) or
        arrays["hidden_fp16_holdout"].dtype != np.float16
    ):
        raise ValueError("precision-diagnostic FP16 holdout differs")
    return arrays, binding


def _state_from_file(path: Path, schema: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    state = CV._read_bound_npz(path, schema)
    if set(state) != {"down.weight", "up.weight", "up.bias"}:
        raise ValueError("final probe state tensor set differs")
    return state


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
        state, report = PROBE.train_probe_head(
            features, targets, valid, weights, fit_rows, RANK,
            epochs=8, batch_rows=512, seed=20260805, device="cuda",
        )
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
        del data, hidden, history, features, weights, state
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
    q4_layers = {str(k): [] for k in CV.K_VALUES}
    fp16_layers = {str(k): [] for k in CV.K_VALUES}
    overlap_layers = {str(k): [] for k in CV.K_VALUES}
    event_arrays: dict[str, np.ndarray] = {}
    holdout_global = diagnostic["hidden_fp16_holdout_row"].astype(np.int64)
    for layer_id in CV.LAYERS:
        global_rows = np.flatnonzero(diagnostic["layer"] == layer_id)
        selected_mask = np.isin(holdout_global, global_rows)
        selected_global = holdout_global[selected_mask]
        if selected_global.size == 0:
            print(json.dumps({"diagnosed_layer": layer_id, "paired_rows": 0}, sort_keys=True), flush=True)
            continue
        local_rows = np.searchsorted(global_rows, selected_global)
        if not np.array_equal(global_rows[local_rows], selected_global):
            raise ValueError("diagnostic holdout row does not map to its layer")
        data = {name: value[global_rows] for name, value in diagnostic.items()
                if name not in {"hidden_fp16_holdout_row", "hidden_fp16_holdout"}}
        rows, targets, valid = PROBE.multi_k_targets(
            data["request_index"], data["layer"], data["token_position"], data["selected_ids"],
        )
        prediction_index = np.full(data["layer"].size, -1, dtype=np.int64)
        prediction_index[rows] = np.arange(rows.size)
        selected_prediction = prediction_index[local_rows]
        if np.any(selected_prediction < 0):
            raise ValueError("diagnostic holdout row has no prediction target")
        history = PROBE.causal_expert_history(
            data["request_index"], data["layer"], data["token_position"], data["selected_ids"],
        )
        q4_hidden = PROBE.unpack_probe_hidden(data["hidden_q4"], data["hidden_scale"])
        fp16_hidden = diagnostic["hidden_fp16_holdout"][selected_mask].astype(np.float32)
        q4_features = np.concatenate((q4_hidden[local_rows], history[local_rows]), axis=1).astype(np.float16)
        fp16_features = np.concatenate((fp16_hidden, history[local_rows]), axis=1).astype(np.float16)
        state_record = model_layers[str(layer_id)]
        state = _state_from_file(requested / state_record["file"], state_record["schema"])
        q4_logits = PROBE.predict_probe_head(q4_features, state, RANK, device="cuda")
        fp16_logits = PROBE.predict_probe_head(fp16_features, state, RANK, device="cuda")
        for k_index, k in enumerate(CV.K_VALUES):
            active = valid[selected_prediction, k_index]
            if not active.any():
                continue
            paired = diagnostic_pair_metrics(
                data["request_index"][local_rows[active]],
                targets[selected_prediction[active], k_index],
                q4_logits[active, k_index], fp16_logits[active, k_index],
            )
            q4_layers[str(k)].append(paired["q4"])
            fp16_layers[str(k)].append(paired["fp16"])
            overlap_layers[str(k)].append(paired["top32_overlap"])
            for name, value in paired["evidence"].items():
                event_arrays[f"layer{layer_id}_k{k}_{name}"] = value
        print(json.dumps({"diagnosed_layer": layer_id}, sort_keys=True), flush=True)
    event_binding = CV._write_npz_exclusive(requested / "diagnostic-events.npz", event_arrays)
    q4_summary = {k: CV.aggregate_request_metrics(values) for k, values in q4_layers.items()}
    fp16_summary = {k: CV.aggregate_request_metrics(values) for k, values in fp16_layers.items()}
    overlap_summary = {k: aggregate_overlap(values) for k, values in overlap_layers.items()}
    q4_headline = q4_summary["4"]["32"]
    fp16_headline = fp16_summary["4"]["32"]
    macro_deficit = 100 * (
        fp16_headline["macro_request_recall"] - q4_headline["macro_request_recall"]
    )
    event_deficit = 100 * (
        fp16_headline["event_weighted_recall"] - q4_headline["event_weighted_recall"]
    )
    verdict = "PASS" if (
        macro_deficit <= 2.0 and event_deficit <= 2.0 and
        overlap_summary["4"]["event_weighted_overlap"] >= 0.9
    ) else "FAIL"
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
    summary = {
        "schema_version": 1,
        "classification": "P1_FEATURE_PRECISION_COMPLETE",
        "verdict": verdict,
        "repository_head": manifest["repository_head"],
        "manifest_sha256": _sha256(requested / "manifest.json"),
        "model_manifest_sha256": model_manifest_sha256,
        "diagnostic_binding": diagnostic_binding,
        "diagnostic_event_binding": event_binding,
        "q4": q4_summary,
        "fp16": fp16_summary,
        "top32_overlap": overlap_summary,
        "headline": {
            "K": 4,
            "budget": 32,
            "macro_request_recall_deficit_pp": macro_deficit,
            "event_weighted_recall_deficit_pp": event_deficit,
            "event_weighted_top32_overlap": overlap_summary["4"]["event_weighted_overlap"],
            "macro_request_recall_deficit_pp_max": 2.0,
            "event_weighted_recall_deficit_pp_max": 2.0,
            "mean_top32_overlap_min": 0.9,
        },
        "runtime_final_sha256": _sha256(requested / "runtime-final.json"),
        "claim_limit": (
            "Five-case feature-precision diagnostic only. Calibration/test values were not scored "
            "or used, but their NPZ members were previously materialized for schema inspection; "
            "see R0c-split-isolation-incident-2026-08-05.json."
        ),
    }
    CV._write_json_exclusive(requested / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if verdict == "PASS" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("run",))
    result.add_argument("--out-dir", required=True, type=Path)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(execute(arguments.out_dir))
