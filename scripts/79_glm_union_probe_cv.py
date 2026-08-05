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
import subprocess

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError("repository HEAD is malformed")
    return result


def runtime_fault_lines(kernel_log: str) -> list[str]:
    """Extract CUDA/Xid/OOM fault lines that invalidate a training run."""
    raise NotImplementedError


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
        "training",
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
) -> dict[str, object]:
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
            observed = accumulate_request_metrics(
                request[rows[active]], targets[active, k_index], rankings,
            )
            _merge_disjoint(frequency[str(k)], observed)
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
                observed = accumulate_request_metrics(
                    request[rows[active_all]], targets[active_all, k_index], rankings,
                )
                _merge_disjoint(probe[str(rank)][str(k)], observed)
            del state, logits
    expected_requests = set(str(value) for value in np.unique(request))
    for method in [frequency, *probe.values()]:
        for by_k in method.values():
            if any(set(by_k[str(budget)]) != expected_requests for budget in BUDGETS):
                raise ValueError("CV method does not cover every training request")
    return {
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


def execute(command: str, out_dir: Path) -> int:
    PROBE._tracked_bytes(PROBE_PATH)
    PROBE._tracked_bytes(Path(__file__).resolve())
    freeze_bytes = PROBE._tracked_bytes(TRAIN_FREEZE)
    head = _repository_head()
    source_binding = PROBE.validate_training_sources(QUALITY, LONGS)
    source_binding_sha256 = hashlib.sha256(json.dumps(
        source_binding, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    expected_manifest = {
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
    if command == "run":
        if requested.exists() or requested.is_symlink():
            raise FileExistsError(requested)
        requested.mkdir(mode=0o700, parents=False)
        manifest = expected_manifest
        _write_json_exclusive(requested / "manifest.json", manifest)
    else:
        requested = requested.resolve(strict=True)
        manifest = json.loads((requested / "manifest.json").read_text(encoding="utf-8"))
        if manifest != expected_manifest:
            raise ValueError("CV resume candidate differs")
    if (requested / "summary.json").exists():
        raise FileExistsError("CV summary already exists")
    identity = {
        "repository_head": head,
        "driver_sha256": manifest["driver_sha256"],
        "probe_sha256": manifest["probe_sha256"],
        "training_source_binding_sha256": source_binding_sha256,
    }
    sources, groups = _load_authorized_sources(source_binding)
    previous_checkpoint_sha256 = _sha256(requested / "manifest.json")
    for layer_id in LAYERS:
        output = requested / f"layer-{layer_id:03d}.json"
        data = _layer_arrays(sources, layer_id)
        contract = expected_layer_contract(data, groups, layer_id)
        if output.exists():
            prior = json.loads(output.read_text(encoding="utf-8"))
            validate_layer_checkpoint(
                prior, contract, identity, previous_checkpoint_sha256,
            )
            previous_checkpoint_sha256 = _sha256(output)
            continue
        result = run_layer(data, groups, layer_id)
        result.update({
            **identity,
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
        })
        validate_layer_checkpoint(
            result, contract, identity, previous_checkpoint_sha256,
        )
        _write_json_exclusive(output, result)
        previous_checkpoint_sha256 = _sha256(output)
        print(json.dumps({"completed_layer": layer_id}, sort_keys=True), flush=True)
        del data
        gc.collect()
    layers = [json.loads((requested / f"layer-{layer:03d}.json").read_text()) for layer in LAYERS]
    frequency_summary = {
        str(k): aggregate_request_metrics([layer["frequency"][str(k)] for layer in layers])
        for k in K_VALUES
    }
    probe_summary = {
        str(rank): {
            str(k): aggregate_request_metrics([layer["probe"][str(rank)][str(k)] for layer in layers])
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
    summary = {
        "schema_version": 1,
        "classification": "TRAIN_ONLY_CV_COMPLETE",
        "repository_head": head,
        "manifest_sha256": _sha256(requested / "manifest.json"),
        "training_source_binding": source_binding,
        "checkpoint_chain_tail_sha256": previous_checkpoint_sha256,
        "completed_layers": list(LAYERS),
        "frequency": frequency_summary,
        "probe": probe_summary,
        "selected_rank": selected_rank,
        "selection_formula": "maximum K=4 budget=32 macro-request recall; exact tie chooses smaller rank",
        "claim_limit": "Train-only rank selection; no diagnostic, calibration, or test metric was opened.",
    }
    _write_json_exclusive(requested / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("run", "resume"))
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
    sizes = targets.sum(axis=1)
    if np.any(sizes <= 0):
        raise ValueError("CV target union is empty")
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    event_rows = np.arange(requests.size)[:, None]
    for budget in budgets:
        hits = targets[event_rows, rankings[:, :budget]].sum(axis=1).astype(np.float64)
        values = {
            "recall_sum": hits / sizes,
            "precision_sum": hits / budget,
            "wasted_sum": budget - hits,
            "coverage_sum": (hits == sizes).astype(np.float64),
        }
        per_request = {}
        for request in np.unique(requests):
            mask = requests == request
            per_request[str(int(request))] = {
                name: float(value[mask].sum()) for name, value in values.items()
            } | {"events": int(mask.sum())}
        output[str(budget)] = per_request
    return output


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
        requests = set(layers[0][budget])
        if not requests or any(set(layer[budget]) != requests for layer in layers):
            raise ValueError("CV request coverage differs across layers")
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
