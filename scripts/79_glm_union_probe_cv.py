#!/usr/bin/env python3
"""Run frozen train-only grouped CV for the GLM direct expert-union probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts/78_glm_union_probe.py"
TRAIN_FREEZE = ROOT / "results/glm52-gates/R0c-union-probe-p1-training-freeze.json"
QUALITY = Path("/home/bmarti44/.local/state/glm52-p1-splits-r127-76faed9/train-fit")
LONGS = [
    Path("/home/bmarti44/.local/state/glm52-p0-shards/2ff949c-request-00000001"),
    Path("/home/bmarti44/.local/state/glm52-p0-shards/2ff949c-request-00000002"),
]
LAYERS = tuple(range(3, 78))
RANKS = (8, 16, 32)
K_VALUES = (2, 4, 8)
BUDGETS = (16, 32, 64)


def _load_probe_module():
    specification = importlib.util.spec_from_file_location("glm_union_probe_for_cv", PROBE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load frozen probe implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PROBE = _load_probe_module()


def event_evidence(
    requests: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, np.ndarray]:
    """Return scorer-replayable integer hit evidence for one K/method/layer."""
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(targets, np.ndarray) or targets.shape != (requests.size, 256) or
        targets.dtype != np.bool_ or
        not isinstance(rankings, np.ndarray) or rankings.shape != targets.shape or
        not np.issubdtype(rankings.dtype, np.integer) or
        np.any(rankings < 0) or np.any(rankings >= 256) or
        any(np.unique(row).size != 256 for row in rankings) or
        any(not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 256
            for budget in budgets) or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("event evidence input schema is invalid")
    target_size = targets.sum(axis=1)
    if np.any(target_size <= 0) or np.any(target_size > 255):
        raise ValueError("event target cardinality is invalid")
    event_rows = np.arange(requests.size)[:, None]
    hits = np.stack([
        targets[event_rows, rankings[:, :budget]].sum(axis=1)
        for budget in budgets
    ], axis=1)
    return {
        "request": requests.astype(np.uint16, copy=True),
        "target_size": target_size.astype(np.uint8),
        "hits": hits.astype(np.uint8),
    }


def score_event_evidence(
    requests: np.ndarray,
    target_size: np.ndarray,
    hits: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Derive exact request metrics only from integer event evidence."""
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(target_size, np.ndarray) or target_size.shape != requests.shape or
        not np.issubdtype(target_size.dtype, np.integer) or
        np.any(target_size <= 0) or np.any(target_size > 255) or
        not isinstance(hits, np.ndarray) or hits.shape != (requests.size, len(budgets)) or
        not np.issubdtype(hits.dtype, np.integer) or np.any(hits < 0) or
        any(not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 256
            for budget in budgets) or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("event evidence schema is invalid")
    if hits.shape[1] > 1 and np.any(np.diff(hits.astype(np.int16), axis=1) < 0):
        raise ValueError("event evidence hit counts are not monotonic by budget")
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for budget_index, budget in enumerate(budgets):
        observed_hits = hits[:, budget_index].astype(np.int64)
        if np.any(observed_hits > target_size) or np.any(observed_hits > budget):
            raise ValueError("event evidence contains impossible hit counts")
        values = {
            "recall_sum": observed_hits / target_size,
            "precision_sum": observed_hits / budget,
            "wasted_sum": budget - observed_hits,
            "coverage_sum": (observed_hits == target_size).astype(np.float64),
        }
        records = {}
        for request in np.unique(requests):
            mask = requests == request
            records[str(int(request))] = {
                name: float(value[mask].sum()) for name, value in values.items()
            } | {"events": int(mask.sum())}
        output[str(budget)] = records
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_snapshot(path: Path) -> dict[str, object]:
    """Parse and hash-sensitive JSON from one stable regular-file snapshot."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"JSON artifact is not a regular file: {path}")
        payload = bytearray()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"JSON artifact ended early: {path}")
            payload.extend(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"JSON artifact grew during validation: {path}")
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size) !=
                (after.st_dev, after.st_ino, after.st_size)):
            raise ValueError(f"JSON artifact identity changed: {path}")
    finally:
        os.close(descriptor)

    def pairs(values):
        output = {}
        for key, value in values:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    value = json.loads(
        bytes(payload).decode("utf-8", errors="strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError("repository HEAD is malformed")
    return result


def runtime_fault_lines(kernel_log: str) -> list[str]:
    """Extract CUDA/Xid/OOM fault lines that invalidate a training run."""
    if not isinstance(kernel_log, str):
        raise ValueError("kernel log must be text")
    pattern = re.compile(
        r"(?:NVRM:\s*Xid\b|\boom-kill\b|Out of memory:\s*Killed process|CUDA[^\n]*(?:fault|error))",
        re.IGNORECASE,
    )
    return [line for line in kernel_log.splitlines() if pattern.search(line)]


def _kernel_log_since(start_epoch: int) -> str:
    completed = subprocess.run(
        ["journalctl", "-k", "--since", f"@{start_epoch}", "--no-pager", "-o", "cat"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"kernel fault log is unavailable: {completed.stderr.strip()}")
    return completed.stdout


def _gpu_snapshot() -> dict[str, str]:
    gpu = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=uuid,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ], stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    applications = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if gpu.returncode != 0 or applications.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError("GPU runtime snapshot is unavailable")
    return {"gpu": gpu.stdout.strip(), "compute_applications": applications.stdout.strip()}


def _mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is unavailable")


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(path_fd)
        finally:
            os.close(path_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _array_schema(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _read_bound_npz(
    path: Path, expected_schema: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("event evidence archive is not a regular file")
        payload = bytearray()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("event evidence archive ended early")
            payload.extend(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("event evidence archive grew during validation")
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size) !=
                (after.st_dev, after.st_ino, after.st_size)):
            raise ValueError("event evidence archive identity changed")
    finally:
        os.close(descriptor)
    import io
    snapshot = bytes(payload)
    expected_members = {f"{name}.npy" for name in expected_schema}
    with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
        names = archive.namelist()
        if len(names) != len(expected_members) or set(names) != expected_members:
            raise ValueError("event evidence archive member set differs")
        if archive.testzip() is not None:
            raise ValueError("event evidence archive checksum failed")
    output: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
        if set(archive.files) != set(expected_schema):
            raise ValueError("event evidence array set differs")
        for name, schema in expected_schema.items():
            value = np.ascontiguousarray(archive[name])
            if value.dtype.hasobject or _array_schema(value) != schema:
                raise ValueError(f"event evidence array binding differs: {name}")
            output[name] = value.copy()
    return output


def _write_npz_exclusive(
    path: Path, arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    if not arrays or any(not isinstance(value, np.ndarray) or value.dtype.hasobject for value in arrays.values()):
        raise ValueError("event evidence arrays are malformed")
    frozen = {name: np.ascontiguousarray(value).copy() for name, value in arrays.items()}
    schema = {name: _array_schema(value) for name, value in frozen.items()}
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **frozen)
            handle.flush()
            os.fsync(handle.fileno())
        _read_bound_npz(temporary, schema)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _read_bound_npz(path, schema)
        return {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "schema": schema,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _load_authorized_sources(
    binding: dict[str, object],
) -> tuple[list[dict[str, np.ndarray]], dict[int, str]]:
    sources: list[dict[str, np.ndarray]] = []
    with np.load(QUALITY / "records.npz", allow_pickle=False) as archive:
        sources.append({
            name: archive[name].copy() for name in (
                "request_index", "layer", "token_position", "selected_ids",
                "hidden_q4", "hidden_scale",
            )
        })
    metadata = json.loads((QUALITY / "manifest.json").read_text(encoding="utf-8"))[
        "request_metadata"
    ]
    groups = {int(row["request_index"]): str(row["group_id"]) for row in metadata}
    for request_id, directory in enumerate(LONGS, 101):
        with np.load(directory / "records.npz", allow_pickle=False) as archive:
            source = {
                name: archive[name].copy() for name in (
                    "layer", "token_position", "selected_ids", "hidden_q4", "hidden_scale",
                )
            }
        source["request_index"] = np.full(source["layer"].size, request_id, dtype=np.uint16)
        sources.append(source)
        groups[request_id] = "long-fixture-lineage-2ff949c"
    if set(groups) != set(int(value) for source in sources for value in np.unique(source["request_index"])):
        raise ValueError("authorized training group map differs from source requests")
    return sources, groups


def _layer_arrays(
    sources: list[dict[str, np.ndarray]], layer_id: int,
) -> dict[str, np.ndarray]:
    names = ("request_index", "layer", "token_position", "selected_ids", "hidden_q4", "hidden_scale")
    pieces = {name: [] for name in names}
    for source in sources:
        mask = source["layer"] == layer_id
        if not mask.any():
            raise ValueError(f"training source has no routed layer {layer_id}")
        for name in names:
            pieces[name].append(source[name][mask])
    return {name: np.concatenate(values) for name, values in pieces.items()}


def _merge_disjoint(
    destination: dict[str, dict[str, dict[str, float | int]]],
    addition: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    if set(destination) != set(addition):
        raise ValueError("CV metric budget sets differ")
    for budget in destination:
        overlap = set(destination[budget]).intersection(addition[budget])
        if overlap:
            raise ValueError("CV request was scored in more than one fold")
        destination[budget].update(addition[budget])


def _empty_metric() -> dict[str, dict[str, dict[str, float | int]]]:
    return {str(budget): {} for budget in BUDGETS}


def expected_layer_contract(
    data: dict[str, np.ndarray], groups: dict[int, str], layer_id: int,
) -> dict[str, object]:
    request = data["request_index"]
    rows, _targets, valid = PROBE.multi_k_targets(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    request_events = {}
    for k_index, k in enumerate(K_VALUES):
        active_requests = request[rows[valid[:, k_index]]]
        identities, counts = np.unique(active_requests, return_counts=True)
        request_events[str(k)] = {
            str(int(identity)): int(count) for identity, count in zip(identities, counts)
        }
    prediction_folds = np.asarray(
        [PROBE.grouped_fold(groups[int(value)]) for value in request[rows]], dtype=np.uint8,
    )
    return {
        "layer": layer_id,
        "source_rows": int(request.size),
        "prediction_rows": int(rows.size),
        "requests": int(np.unique(request).size),
        "request_events": request_events,
        "fit_rows_by_fold": {
            str(fold): int(np.sum(prediction_folds != fold)) for fold in range(3)
        },
    }


def fold_training_weights(
    request: np.ndarray,
    rows: np.ndarray,
    valid: np.ndarray,
    prediction_folds: np.ndarray,
    validation_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive request-balanced weights from fitting rows only."""
    if (
        not isinstance(prediction_folds, np.ndarray) or prediction_folds.ndim != 1 or
        prediction_folds.size != rows.size or not np.issubdtype(prediction_folds.dtype, np.integer) or
        not isinstance(validation_fold, int) or isinstance(validation_fold, bool) or
        validation_fold not in (0, 1, 2)
    ):
        raise ValueError("fold weight input is invalid")
    fitting = np.flatnonzero(prediction_folds != validation_fold).astype(np.int64)
    if fitting.size == 0 or not np.any(prediction_folds == validation_fold):
        raise ValueError("grouped CV fold is empty")
    fitting_weights = PROBE.request_balanced_weights(
        request, rows[fitting], valid[fitting],
    )
    weights = np.zeros(valid.shape, dtype=np.float32)
    weights[fitting] = fitting_weights
    return weights, fitting


def validate_layer_checkpoint(
    checkpoint: dict[str, object],
    contract: dict[str, object],
    identity: dict[str, str],
    previous_checkpoint_sha256: str,
) -> None:
    """Fail closed on stale, malformed, or physically impossible CV layer evidence."""
    top_keys = {
        "schema_version", "classification", "repository_head", "driver_sha256",
        "probe_sha256", "training_source_binding_sha256", "previous_checkpoint_sha256",
        "layer", "source_rows", "prediction_rows", "requests", "frequency", "probe",
        "training", "event_evidence_file", "event_evidence_sha256", "event_evidence_bytes",
        "event_evidence_schema",
    }
    if (
        not isinstance(checkpoint, dict) or set(checkpoint) != top_keys or
        checkpoint.get("schema_version") != 1 or
        checkpoint.get("classification") != "TRAIN_ONLY_LAYER_CV" or
        not isinstance(contract, dict) or
        set(contract) != {
            "layer", "source_rows", "prediction_rows", "requests",
            "request_events", "fit_rows_by_fold",
        } or
        not isinstance(identity, dict) or
        set(identity) != {
            "repository_head", "driver_sha256", "probe_sha256",
            "training_source_binding_sha256",
        } or
        any(checkpoint.get(key) != value for key, value in identity.items()) or
        checkpoint.get("previous_checkpoint_sha256") != previous_checkpoint_sha256 or
        any(checkpoint.get(key) != contract[key]
            for key in ("layer", "source_rows", "prediction_rows", "requests"))
    ):
        raise ValueError("CV layer checkpoint identity or schema differs")
    if (
        checkpoint.get("event_evidence_file") != f"layer-{contract['layer']:03d}-events.npz" or
        not isinstance(checkpoint.get("event_evidence_sha256"), str) or
        len(checkpoint["event_evidence_sha256"]) != 64 or
        not isinstance(checkpoint.get("event_evidence_bytes"), int) or
        checkpoint["event_evidence_bytes"] <= 0 or
        not isinstance(checkpoint.get("event_evidence_schema"), dict)
    ):
        raise ValueError("CV layer event evidence binding differs")
    if (
        len(identity["repository_head"]) != 40 or
        any(len(identity[key]) != 64 for key in identity if key != "repository_head") or
        len(previous_checkpoint_sha256) != 64
    ):
        raise ValueError("CV layer checkpoint digest is malformed")

    expected_events = contract["request_events"]
    metric_record_keys = {
        "recall_sum", "precision_sum", "wasted_sum", "coverage_sum", "events",
    }

    def validate_method(method: object) -> None:
        if not isinstance(method, dict) or set(method) != {str(k) for k in K_VALUES}:
            raise ValueError("CV checkpoint K coverage differs")
        for k in K_VALUES:
            by_budget = method[str(k)]
            expected_requests = expected_events[str(k)]
            if not isinstance(by_budget, dict) or set(by_budget) != {str(b) for b in BUDGETS}:
                raise ValueError("CV checkpoint budget coverage differs")
            for budget in BUDGETS:
                records = by_budget[str(budget)]
                if not isinstance(records, dict) or set(records) != set(expected_requests):
                    raise ValueError("CV checkpoint request coverage differs")
                for request, expected_count in expected_requests.items():
                    record = records[request]
                    if (
                        not isinstance(record, dict) or set(record) != metric_record_keys or
                        record.get("events") != expected_count or
                        not isinstance(expected_count, int) or expected_count <= 0
                    ):
                        raise ValueError("CV checkpoint event count differs")
                    values = [record[name] for name in metric_record_keys - {"events"}]
                    if any(
                        not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not np.isfinite(value) for value in values
                    ):
                        raise ValueError("CV checkpoint metric is non-finite")
                    recall = float(record["recall_sum"])
                    precision = float(record["precision_sum"])
                    wasted = float(record["wasted_sum"])
                    coverage = float(record["coverage_sum"])
                    if (
                        not 0 <= recall <= expected_count or
                        not 0 <= precision <= expected_count or
                        not 0 <= coverage <= expected_count or
                        not 0 <= wasted <= budget * expected_count or
                        coverage > recall + 1e-9 or
                        not np.isclose(
                            precision * budget + wasted,
                            budget * expected_count,
                            rtol=0.0, atol=1e-6,
                        ) or
                        not np.isclose(precision * budget, round(precision * budget), atol=1e-6) or
                        not np.isclose(wasted, round(wasted), atol=1e-6)
                    ):
                        raise ValueError("CV checkpoint metric violates physical bounds")

    validate_method(checkpoint["frequency"])
    probe = checkpoint["probe"]
    if not isinstance(probe, dict) or set(probe) != {str(rank) for rank in RANKS}:
        raise ValueError("CV checkpoint rank coverage differs")
    for method in probe.values():
        validate_method(method)

    training = checkpoint["training"]
    report_keys = {
        "fit_rows", "rank", "epochs", "batch_rows", "seed", "positive_weights",
        "epoch_losses", "epoch_k_losses", "deterministic_algorithms",
    }
    if not isinstance(training, dict) or set(training) != {str(rank) for rank in RANKS}:
        raise ValueError("CV checkpoint training rank coverage differs")
    for rank in RANKS:
        folds = training[str(rank)]
        if not isinstance(folds, dict) or set(folds) != {"0", "1", "2"}:
            raise ValueError("CV checkpoint training fold coverage differs")
        for fold in range(3):
            report = folds[str(fold)]
            if (
                not isinstance(report, dict) or set(report) != report_keys or
                report.get("fit_rows") != contract["fit_rows_by_fold"][str(fold)] or
                report.get("rank") != rank or report.get("epochs") != 8 or
                report.get("batch_rows") != 512 or report.get("seed") != 20260805 or
                report.get("deterministic_algorithms") is not True
            ):
                raise ValueError("CV checkpoint training report differs")
            positive = report["positive_weights"]
            losses = report["epoch_losses"]
            k_losses = report["epoch_k_losses"]
            if (
                not isinstance(positive, list) or len(positive) != 3 or
                not isinstance(losses, list) or len(losses) != 8 or
                not isinstance(k_losses, list) or len(k_losses) != 8 or
                any(not isinstance(row, list) or len(row) != 3 for row in k_losses) or
                any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
                    for value in [*positive, *losses, *(v for row in k_losses for v in row)]) or
                any(not np.isclose(losses[index], np.mean(k_losses[index]), rtol=0, atol=1e-12)
                    for index in range(8))
            ):
                raise ValueError("CV checkpoint training losses differ")


def run_layer(
    data: dict[str, np.ndarray], groups: dict[int, str], layer_id: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    request = data["request_index"]
    rows, targets, valid = PROBE.multi_k_targets(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    history = PROBE.causal_expert_history(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    hidden = PROBE.unpack_probe_hidden(data["hidden_q4"], data["hidden_scale"]).astype(np.float16)
    features = np.concatenate((hidden, history.astype(np.float16)), axis=1)[rows]
    del hidden, history
    request_fold = {identity: PROBE.grouped_fold(group) for identity, group in groups.items()}
    source_folds = np.asarray([request_fold[int(value)] for value in request], dtype=np.uint8)
    prediction_folds = source_folds[rows]
    frequency = {str(k): _empty_metric() for k in K_VALUES}
    probe = {str(rank): {str(k): _empty_metric() for k in K_VALUES} for rank in RANKS}
    training: dict[str, dict[str, object]] = {str(rank): {} for rank in RANKS}
    evidence_parts = {
        str(k): {
            "row": [], "request": [], "target_size": [], "frequency_hits": [],
            **{f"probe_{rank}_hits": [] for rank in RANKS},
        }
        for k in K_VALUES
    }
    for fold in range(3):
        validation = prediction_folds == fold
        weights, fitting = fold_training_weights(
            request, rows, valid, prediction_folds, fold,
        )
        fit_source = source_folds != fold
        frequency_order = PROBE.frequency_prior_by_layer(
            data["layer"][fit_source], data["selected_ids"][fit_source],
        )[layer_id]
        for k_index, k in enumerate(K_VALUES):
            active = validation & valid[:, k_index]
            rankings = np.tile(frequency_order, (int(active.sum()), 1))
            raw = event_evidence(
                request[rows[active]], targets[active, k_index], rankings,
            )
            observed = score_event_evidence(
                raw["request"], raw["target_size"], raw["hits"], BUDGETS,
            )
            _merge_disjoint(frequency[str(k)], observed)
            evidence_parts[str(k)]["row"].append(rows[active].astype(np.uint32))
            evidence_parts[str(k)]["request"].append(raw["request"])
            evidence_parts[str(k)]["target_size"].append(raw["target_size"])
            evidence_parts[str(k)]["frequency_hits"].append(raw["hits"])
        for rank in RANKS:
            state, report = PROBE.train_probe_head(
                features, targets, valid, weights, fitting, rank,
                epochs=8, batch_rows=512, seed=20260805, device="cuda",
            )
            logits = PROBE.predict_probe_head(
                features[validation], state, rank, batch_rows=512, device="cuda",
            )
            training[str(rank)][str(fold)] = report
            for k_index, k in enumerate(K_VALUES):
                active_all = np.flatnonzero(validation & valid[:, k_index])
                active_validation = valid[validation, k_index]
                rankings = np.argsort(
                    -logits[active_validation, k_index], axis=1, kind="stable",
                ).astype(np.uint16)
                raw = event_evidence(
                    request[rows[active_all]], targets[active_all, k_index], rankings,
                )
                observed = score_event_evidence(
                    raw["request"], raw["target_size"], raw["hits"], BUDGETS,
                )
                _merge_disjoint(probe[str(rank)][str(k)], observed)
                evidence_parts[str(k)][f"probe_{rank}_hits"].append(raw["hits"])
            del state, logits
    expected_requests = set(str(value) for value in np.unique(request))
    for method in [frequency, *probe.values()]:
        for by_k in method.values():
            if any(set(by_k[str(budget)]) != expected_requests for budget in BUDGETS):
                raise ValueError("CV method does not cover every training request")
    evidence: dict[str, np.ndarray] = {}
    for k_index, k in enumerate(K_VALUES):
        parts = evidence_parts[str(k)]
        combined = {name: np.concatenate(values) for name, values in parts.items()}
        order = np.argsort(combined["row"], kind="stable")
        expected_rows = rows[valid[:, k_index]].astype(np.uint32)
        if (
            not np.array_equal(combined["row"][order], expected_rows) or
            not np.array_equal(combined["request"][order], request[expected_rows]) or
            np.unique(combined["row"]).size != expected_rows.size
        ):
            raise ValueError("CV event evidence row coverage differs")
        for name, value in combined.items():
            evidence[f"k{k}_{name}"] = value[order]
        for method, metrics in [
            ("frequency_hits", frequency[str(k)]),
            *[(f"probe_{rank}_hits", probe[str(rank)][str(k)]) for rank in RANKS],
        ]:
            replay = score_event_evidence(
                evidence[f"k{k}_request"], evidence[f"k{k}_target_size"],
                evidence[f"k{k}_{method}"], BUDGETS,
            )
            if replay != metrics:
                raise ValueError("CV event evidence does not replay its aggregate")
    result = {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_LAYER_CV",
        "layer": layer_id,
        "source_rows": int(request.size),
        "prediction_rows": int(rows.size),
        "requests": len(expected_requests),
        "frequency": frequency,
        "probe": probe,
        "training": training,
    }
    return result, evidence


def replay_layer_event_evidence(
    path: Path,
    binding: dict[str, object],
    data: dict[str, np.ndarray],
    result: dict[str, object],
) -> dict[str, object]:
    """Rebuild authoritative metrics from the final persisted layer archive."""
    if (
        not isinstance(binding, dict) or set(binding) != {"sha256", "bytes", "schema"} or
        binding.get("sha256") != _sha256(path) or
        binding.get("bytes") != path.stat().st_size or
        not isinstance(binding.get("schema"), dict)
    ):
        raise ValueError("persisted event evidence file binding differs")
    expected_names = {
        f"k{k}_{name}"
        for k in K_VALUES
        for name in (
            "row", "request", "target_size", "frequency_hits",
            *(f"probe_{rank}_hits" for rank in RANKS),
        )
    }
    if set(binding["schema"]) != expected_names:
        raise ValueError("persisted event evidence schema names differ")
    evidence = _read_bound_npz(path, binding["schema"])
    request = data["request_index"]
    rows, targets, valid = PROBE.multi_k_targets(
        request, data["layer"], data["token_position"], data["selected_ids"],
    )
    frequency: dict[str, object] = {}
    probe: dict[str, dict[str, object]] = {str(rank): {} for rank in RANKS}
    for k_index, k in enumerate(K_VALUES):
        active = valid[:, k_index]
        expected_rows = rows[active].astype(np.uint32)
        expected_request = request[expected_rows].astype(np.uint16)
        expected_target_size = targets[active, k_index].sum(axis=1).astype(np.uint8)
        exact = {
            "row": expected_rows,
            "request": expected_request,
            "target_size": expected_target_size,
        }
        for name, expected in exact.items():
            observed = evidence[f"k{k}_{name}"]
            if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                raise ValueError(f"persisted event evidence source contract differs: k{k}_{name}")
        frequency[str(k)] = score_event_evidence(
            expected_request, expected_target_size, evidence[f"k{k}_frequency_hits"], BUDGETS,
        )
        for rank in RANKS:
            probe[str(rank)][str(k)] = score_event_evidence(
                expected_request, expected_target_size,
                evidence[f"k{k}_probe_{rank}_hits"], BUDGETS,
            )
    if frequency != result.get("frequency") or probe != result.get("probe"):
        raise ValueError("persisted event evidence does not replay the layer result")
    replayed = dict(result)
    replayed["frequency"] = frequency
    replayed["probe"] = probe
    return replayed


def build_cv_summary(
    layers: list[dict[str, object]],
    repository_head: str,
    manifest_sha256: str,
    source_binding: dict[str, object],
    checkpoint_tail_sha256: str,
) -> dict[str, object]:
    frequency_summary = {
        str(k): aggregate_request_metrics([layer["frequency"][str(k)] for layer in layers])
        for k in K_VALUES
    }
    probe_summary = {
        str(rank): {
            str(k): aggregate_request_metrics([
                layer["probe"][str(rank)][str(k)] for layer in layers
            ])
            for k in K_VALUES
        }
        for rank in RANKS
    }
    selected_rank = max(
        RANKS,
        key=lambda rank: (
            probe_summary[str(rank)]["4"]["32"]["macro_request_recall"], -rank,
        ),
    )
    return {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_CV_COMPLETE",
        "repository_head": repository_head,
        "manifest_sha256": manifest_sha256,
        "training_source_binding": source_binding,
        "checkpoint_chain_tail_sha256": checkpoint_tail_sha256,
        "completed_layers": list(LAYERS),
        "frequency": frequency_summary,
        "probe": probe_summary,
        "selected_rank": selected_rank,
        "selection_formula": "maximum K=4 budget=32 macro-request recall; exact tie chooses smaller rank",
        "claim_limit": "Train-only rank selection; no diagnostic, calibration, or test metric was opened.",
    }


def validate_completed_output(
    requested: Path,
    source_binding: dict[str, object],
    sources: list[dict[str, np.ndarray]],
    groups: dict[int, str],
    identity: dict[str, str],
) -> dict[str, object]:
    """Reopen the complete run and replay every reported metric from persisted evidence."""
    expected_files = {"manifest.json", "runtime-start.json", "runtime-final.json", "summary.json"}
    expected_files.update(f"layer-{layer:03d}.json" for layer in LAYERS)
    expected_files.update(f"layer-{layer:03d}-events.npz" for layer in LAYERS)
    if {path.name for path in requested.iterdir()} != expected_files:
        raise ValueError("completed CV artifact set differs")
    manifest_path = requested / "manifest.json"
    manifest = _read_json_snapshot(manifest_path)
    expected_manifest_keys = {
        "schema_version", "classification", "repository_head", "driver_sha256",
        "probe_sha256", "training_freeze_sha256", "training_source_binding",
        "training_source_binding_sha256", "layers", "ranks", "K", "budgets",
        "runtime_start_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys or manifest.get("schema_version") != 1 or
        manifest.get("classification") != "TRAIN_ONLY_CV_IN_PROGRESS" or
        manifest.get("repository_head") != identity["repository_head"] or
        manifest.get("driver_sha256") != identity["driver_sha256"] or
        manifest.get("probe_sha256") != identity["probe_sha256"] or
        manifest.get("training_source_binding") != source_binding or
        manifest.get("training_source_binding_sha256") != identity["training_source_binding_sha256"] or
        manifest.get("layers") != list(LAYERS) or manifest.get("ranks") != list(RANKS) or
        manifest.get("K") != list(K_VALUES) or manifest.get("budgets") != list(BUDGETS)
    ):
        raise ValueError("completed CV manifest differs")
    runtime_start_path = requested / "runtime-start.json"
    runtime_start = _read_json_snapshot(runtime_start_path)
    if (
        set(runtime_start) != {
            "schema_version", "classification", "start_epoch", "mem_available_kib",
            "gpu", "kernel_faults_at_start",
        } or runtime_start.get("schema_version") != 1 or
        runtime_start.get("classification") != "TRAIN_ONLY_CV_RUNTIME_START" or
        not isinstance(runtime_start.get("start_epoch"), int) or
        not isinstance(runtime_start.get("mem_available_kib"), int) or
        runtime_start["mem_available_kib"] <= 0 or
        not isinstance(runtime_start.get("gpu"), dict) or
        runtime_start.get("kernel_faults_at_start") != [] or
        manifest.get("runtime_start_sha256") != _sha256(runtime_start_path)
    ):
        raise ValueError("completed CV runtime start differs")
    previous = _sha256(manifest_path)
    layers: list[dict[str, object]] = []
    for layer_id in LAYERS:
        data = _layer_arrays(sources, layer_id)
        contract = expected_layer_contract(data, groups, layer_id)
        layer_path = requested / f"layer-{layer_id:03d}.json"
        layer = _read_json_snapshot(layer_path)
        validate_layer_checkpoint(layer, contract, identity, previous)
        event_path = requested / str(layer["event_evidence_file"])
        binding = {
            "sha256": layer["event_evidence_sha256"],
            "bytes": layer["event_evidence_bytes"],
            "schema": layer["event_evidence_schema"],
        }
        layers.append(replay_layer_event_evidence(event_path, binding, data, layer))
        previous = _sha256(layer_path)
    runtime_final_path = requested / "runtime-final.json"
    runtime_final = _read_json_snapshot(runtime_final_path)
    if (
        set(runtime_final) != {
            "schema_version", "classification", "start_epoch", "end_epoch",
            "kernel_log_sha256", "kernel_log", "kernel_faults", "post_gpu",
            "post_mem_available_kib", "deterministic_algorithms",
        } or runtime_final.get("schema_version") != 1 or
        runtime_final.get("classification") != "TRAIN_ONLY_CV_RUNTIME_PASS" or
        runtime_final.get("start_epoch") != runtime_start["start_epoch"] or
        not isinstance(runtime_final.get("end_epoch"), int) or
        runtime_final["end_epoch"] < runtime_start["start_epoch"] or
        not isinstance(runtime_final.get("kernel_log"), str) or
        runtime_final.get("kernel_log_sha256") != hashlib.sha256(
            runtime_final["kernel_log"].encode("utf-8")
        ).hexdigest() or runtime_final.get("kernel_faults") != [] or
        runtime_fault_lines(runtime_final["kernel_log"]) or
        not isinstance(runtime_final.get("post_gpu"), dict) or
        not isinstance(runtime_final.get("post_mem_available_kib"), int) or
        runtime_final["post_mem_available_kib"] <= 0 or
        runtime_final.get("deterministic_algorithms") is not True
    ):
        raise ValueError("completed CV runtime final differs")
    expected_summary = build_cv_summary(
        layers, identity["repository_head"], _sha256(manifest_path), source_binding, previous,
    )
    expected_summary["runtime_final_sha256"] = _sha256(runtime_final_path)
    summary = _read_json_snapshot(requested / "summary.json")
    if summary != expected_summary:
        raise ValueError("completed CV summary does not replay persisted evidence")
    return summary


def execute(command: str, out_dir: Path) -> int:
    PROBE._tracked_bytes(PROBE_PATH)
    PROBE._tracked_bytes(Path(__file__).resolve())
    freeze_bytes = PROBE._tracked_bytes(TRAIN_FREEZE)
    head = _repository_head()
    source_binding = PROBE.validate_training_sources(QUALITY, LONGS)
    source_binding_sha256 = hashlib.sha256(json.dumps(
        source_binding, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_CV_IN_PROGRESS",
        "repository_head": head,
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "probe_sha256": _sha256(PROBE_PATH),
        "training_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "training_source_binding": source_binding,
        "training_source_binding_sha256": source_binding_sha256,
        "layers": list(LAYERS),
        "ranks": list(RANKS),
        "K": list(K_VALUES),
        "budgets": list(BUDGETS),
    }
    requested = out_dir.absolute()
    if command != "run" or requested.exists() or requested.is_symlink():
        raise FileExistsError("qualifying CV requires a fresh output path")
    requested.mkdir(mode=0o700, parents=False)
    start_epoch = int(time.time())
    runtime_start = {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_CV_RUNTIME_START",
        "start_epoch": start_epoch,
        "mem_available_kib": _mem_available_kib(),
        "gpu": _gpu_snapshot(),
        "kernel_faults_at_start": runtime_fault_lines(_kernel_log_since(start_epoch)),
    }
    if runtime_start["kernel_faults_at_start"]:
        raise RuntimeError("GPU/OOM fault appeared at CV start")
    runtime_start_path = requested / "runtime-start.json"
    _write_json_exclusive(runtime_start_path, runtime_start)
    manifest["runtime_start_sha256"] = _sha256(runtime_start_path)
    _write_json_exclusive(requested / "manifest.json", manifest)
    if runtime_fault_lines(_kernel_log_since(start_epoch)):
        raise RuntimeError("GPU/OOM fault exists in CV runtime interval")
    identity = {
        "repository_head": head,
        "driver_sha256": manifest["driver_sha256"],
        "probe_sha256": manifest["probe_sha256"],
        "training_source_binding_sha256": source_binding_sha256,
    }
    sources, groups = _load_authorized_sources(source_binding)
    previous_checkpoint_sha256 = _sha256(requested / "manifest.json")
    layers: list[dict[str, object]] = []
    for layer_id in LAYERS:
        output = requested / f"layer-{layer_id:03d}.json"
        event_output = requested / f"layer-{layer_id:03d}-events.npz"
        data = _layer_arrays(sources, layer_id)
        contract = expected_layer_contract(data, groups, layer_id)
        if output.exists() or event_output.exists():
            raise FileExistsError("fresh CV path contains a layer artifact")
        result, event_arrays = run_layer(data, groups, layer_id)
        event_binding = _write_npz_exclusive(event_output, event_arrays)
        result = replay_layer_event_evidence(
            event_output, event_binding, data, result,
        )
        result.update({
            **identity,
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
            "event_evidence_file": event_output.name,
            "event_evidence_sha256": event_binding["sha256"],
            "event_evidence_bytes": event_binding["bytes"],
            "event_evidence_schema": event_binding["schema"],
        })
        validate_layer_checkpoint(
            result, contract, identity, previous_checkpoint_sha256,
        )
        if runtime_fault_lines(_kernel_log_since(start_epoch)):
            raise RuntimeError("GPU/OOM fault appeared during CV layer")
        _write_json_exclusive(output, result)
        previous_checkpoint_sha256 = _sha256(output)
        layers.append(result)
        print(json.dumps({"completed_layer": layer_id}, sort_keys=True), flush=True)
        del data
        gc.collect()
    summary = build_cv_summary(
        layers, head, _sha256(requested / "manifest.json"), source_binding,
        previous_checkpoint_sha256,
    )
    if _sha256(runtime_start_path) != manifest["runtime_start_sha256"]:
        raise RuntimeError("CV runtime start binding changed before completion")
    post_gpu = _gpu_snapshot()
    post_mem_available_kib = _mem_available_kib()
    kernel_log = _kernel_log_since(start_epoch)
    faults = runtime_fault_lines(kernel_log)
    if faults:
        raise RuntimeError("GPU/OOM fault appeared before CV completion")
    runtime_final = {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_CV_RUNTIME_PASS",
        "start_epoch": start_epoch,
        "end_epoch": int(time.time()),
        "kernel_log_sha256": hashlib.sha256(kernel_log.encode("utf-8")).hexdigest(),
        "kernel_log": kernel_log,
        "kernel_faults": faults,
        "post_gpu": post_gpu,
        "post_mem_available_kib": post_mem_available_kib,
        "deterministic_algorithms": True,
    }
    runtime_final_path = requested / "runtime-final.json"
    _write_json_exclusive(runtime_final_path, runtime_final)
    summary["runtime_final_sha256"] = _sha256(runtime_final_path)
    _write_json_exclusive(requested / "summary.json", summary)
    validated_summary = validate_completed_output(
        requested, source_binding, sources, groups, identity,
    )
    print(json.dumps(validated_summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("run",))
    result.add_argument("--out-dir", required=True, type=Path)
    return result


def accumulate_request_metrics(
    requests: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    budgets: tuple[int, ...] = (16, 32, 64),
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return exact per-request sums/counts for one K and one routed layer."""
    if (
        not isinstance(requests, np.ndarray) or requests.ndim != 1 or requests.size == 0 or
        not np.issubdtype(requests.dtype, np.integer) or np.any(requests <= 0) or
        not isinstance(targets, np.ndarray) or targets.shape != (requests.size, 256) or
        not isinstance(rankings, np.ndarray) or rankings.shape != targets.shape or
        not np.issubdtype(rankings.dtype, np.integer) or
        np.any(rankings < 0) or np.any(rankings >= 256) or
        any(np.unique(row).size != 256 for row in rankings) or
        any(not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 256
            for budget in budgets) or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("CV metric input schema is invalid")
    if targets.dtype != np.bool_:
        if not np.issubdtype(targets.dtype, np.integer) or np.any((targets != 0) & (targets != 1)):
            raise ValueError("CV targets must be Boolean")
        targets = targets.astype(np.bool_)
    observed = event_evidence(requests, targets, rankings, budgets)
    return score_event_evidence(
        observed["request"], observed["target_size"], observed["hits"], budgets,
    )


def aggregate_request_metrics(
    layers: list[dict[str, dict[str, dict[str, float | int]]]],
) -> dict[str, dict[str, float]]:
    """Aggregate event sums within request, then macro-average across requests."""
    if not isinstance(layers, list) or not layers or any(not isinstance(layer, dict) for layer in layers):
        raise ValueError("CV layer metrics are missing")
    budgets = set(layers[0])
    if not budgets or any(set(layer) != budgets for layer in layers):
        raise ValueError("CV layer budget sets differ")
    output: dict[str, dict[str, float]] = {}
    required = {"recall_sum", "precision_sum", "wasted_sum", "coverage_sum", "events"}
    for budget in sorted(budgets, key=int):
        request_set = set(layers[0][budget])
        if not request_set or any(set(layer[budget]) != request_set for layer in layers):
            raise ValueError("CV request coverage differs across layers")
        requests = sorted(request_set, key=int)
        totals = {
            request: {name: 0.0 for name in required - {"events"}} | {"events": 0}
            for request in requests
        }
        for layer in layers:
            for request, record in layer[budget].items():
                if set(record) != required or not isinstance(record["events"], int) or record["events"] <= 0:
                    raise ValueError("CV request metric record is malformed")
                for name in required - {"events"}:
                    value = record[name]
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("CV request metric is non-finite")
                    totals[request][name] += float(value)
                totals[request]["events"] += record["events"]
        request_means = {
            name: [totals[request][name] / totals[request]["events"] for request in requests]
            for name in required - {"events"}
        }
        event_count = sum(totals[request]["events"] for request in requests)
        output[budget] = {
            "requests": len(requests),
            "events": event_count,
            "macro_request_recall": float(np.mean(request_means["recall_sum"])),
            "macro_request_precision": float(np.mean(request_means["precision_sum"])),
            "macro_request_wasted_experts": float(np.mean(request_means["wasted_sum"])),
            "macro_request_full_set_coverage": float(np.mean(request_means["coverage_sum"])),
            "event_weighted_recall": sum(totals[r]["recall_sum"] for r in requests) / event_count,
            "event_weighted_precision": sum(totals[r]["precision_sum"] for r in requests) / event_count,
            "event_weighted_wasted_experts": sum(totals[r]["wasted_sum"] for r in requests) / event_count,
            "event_weighted_full_set_coverage": sum(totals[r]["coverage_sum"] for r in requests) / event_count,
        }
    return output


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(execute(arguments.command, arguments.out_dir))
