#!/usr/bin/env python3
"""Train and score frozen GLM multi-token expert-union probes."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


K_VALUES = (2, 4, 8)
BUDGETS = (16, 32, 64)
N_EXPERT = 256
ROOT = Path(__file__).resolve().parents[1]
QUALITY_COMPACTION_RECEIPT = (
    ROOT / "results/glm52-gates/R0b-union-quality-corpus-compaction-pass-440d15d.json"
)
SPLIT_PLAN = ROOT / "results/glm52-gates/R0b-union-p0-split-plan.json"
P1_SPLIT_RECEIPT = ROOT / "results/glm52-gates/R0c-union-probe-splits-pass-76faed9.json"
LONG_COMPACTION_RECEIPT = (
    ROOT / "results/glm52-gates/R0b-union-corpus-compaction-pass-2ff949c.json"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SPLIT_COUNTS = {
    "train-fit": 55,
    "train-precision-diagnostic": 5,
    "calibration": 20,
    "test": 20,
}


def partition_request_rows(
    request_index: np.ndarray,
    request_metadata: list[dict[str, object]],
    expected_counts: dict[str, int] = SPLIT_COUNTS,
    expected_case_splits: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Return exact row indices for the preregistered request-grouped splits."""
    if (
        not isinstance(request_index, np.ndarray) or request_index.ndim != 1 or
        request_index.size == 0 or not np.issubdtype(request_index.dtype, np.integer) or
        np.any(request_index <= 0) or np.any(np.diff(request_index.astype(np.int64)) < 0) or
        not isinstance(request_metadata, list) or not request_metadata or
        not isinstance(expected_counts, dict) or set(expected_counts) != set(SPLIT_COUNTS) or
        any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in expected_counts.values())
    ):
        raise ValueError("split input schema is invalid")
    metadata_by_request: dict[int, dict[str, object]] = {}
    case_ids: set[str] = set()
    group_ids: set[str] = set()
    split_counts = {name: 0 for name in expected_counts}
    for row in request_metadata:
        if not isinstance(row, dict):
            raise ValueError("request metadata row is malformed")
        request = row.get("request_index")
        case = row.get("case_id")
        group = row.get("group_id")
        split = row.get("split")
        if (
            not isinstance(request, int) or isinstance(request, bool) or request <= 0 or
            request in metadata_by_request or not isinstance(case, str) or not case or
            case in case_ids or split not in expected_counts
        ):
            raise ValueError("request metadata identity or split is invalid")
        if expected_case_splits is not None and (
            not isinstance(group, str) or not group or group != case or group in group_ids or
            expected_case_splits.get(case) != split
        ):
            raise ValueError("request metadata differs from the preregistered case split")
        metadata_by_request[request] = row
        case_ids.add(case)
        if isinstance(group, str):
            group_ids.add(group)
        split_counts[str(split)] += 1
    if expected_case_splits is not None and (
        not isinstance(expected_case_splits, dict) or
        set(expected_case_splits) != case_ids or
        any(split not in expected_counts for split in expected_case_splits.values())
    ):
        raise ValueError("preregistered case split coverage differs")
    if split_counts != expected_counts:
        raise ValueError("request split counts differ from the frozen plan")
    observed_requests = set(int(value) for value in np.unique(request_index))
    if observed_requests != set(metadata_by_request):
        raise ValueError("request rows and metadata are not a bijection")
    split_by_request = {
        request: str(row["split"]) for request, row in metadata_by_request.items()
    }
    result = {
        split: np.flatnonzero(np.asarray([
            split_by_request[int(request)] == split for request in request_index
        ], dtype=np.bool_)).astype(np.int64)
        for split in expected_counts
    }
    combined = np.concatenate(list(result.values()))
    if (
        combined.size != request_index.size or
        not np.array_equal(np.sort(combined), np.arange(request_index.size))
    ):
        raise ValueError("split rows are incomplete or overlap")
    return result


def split_compact_arrays(
    arrays: dict[str, np.ndarray], split_rows: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Project a combined compact corpus into isolated split archives."""
    if (
        not isinstance(arrays, dict) or not arrays or
        not isinstance(split_rows, dict) or set(split_rows) != set(SPLIT_COUNTS) or
        any(not isinstance(value, np.ndarray) for value in arrays.values()) or
        "request_index" not in arrays or "layer" not in arrays or
        "hidden_fp16_holdout_row" not in arrays or "hidden_fp16_holdout" not in arrays
    ):
        raise ValueError("compact split input schema is invalid")
    row_count = arrays["request_index"].shape[0]
    holdout_names = {"hidden_fp16_holdout_row", "hidden_fp16_holdout"}
    main_names = set(arrays) - holdout_names
    if (
        row_count <= 0 or any(value.ndim == 0 or value.shape[0] != row_count
                              for name, value in arrays.items() if name in main_names) or
        any(np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
            for value in arrays.values())
    ):
        raise ValueError("compact split row coverage or finiteness differs")
    holdout_row = arrays["hidden_fp16_holdout_row"]
    holdout_hidden = arrays["hidden_fp16_holdout"]
    if (
        holdout_row.ndim != 1 or not np.issubdtype(holdout_row.dtype, np.integer) or
        holdout_hidden.ndim != 2 or holdout_hidden.shape[0] != holdout_row.size or
        np.any(holdout_row < 0) or np.any(holdout_row >= row_count) or
        (holdout_row.size > 1 and np.any(np.diff(holdout_row.astype(np.int64)) <= 0))
    ):
        raise ValueError("compact split holdout coverage is invalid")
    normalized_rows: dict[str, np.ndarray] = {}
    for split, rows in split_rows.items():
        if (
            not isinstance(rows, np.ndarray) or rows.ndim != 1 or
            not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or
            np.any(rows >= row_count) or
            (rows.size > 1 and np.any(np.diff(rows.astype(np.int64)) <= 0))
        ):
            raise ValueError(f"split row indices are invalid: {split}")
        normalized_rows[split] = rows.astype(np.int64, copy=False)
    combined = np.concatenate(list(normalized_rows.values()))
    if combined.size != row_count or not np.array_equal(np.sort(combined), np.arange(row_count)):
        raise ValueError("compact split rows are incomplete or overlap")

    result: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in normalized_rows.items():
        projected = {name: arrays[name][rows] for name in sorted(main_names)}
        old_to_new = np.full(row_count, -1, dtype=np.int64)
        old_to_new[rows] = np.arange(rows.size)
        selected_holdout = old_to_new[holdout_row.astype(np.int64)] >= 0
        projected["hidden_fp16_holdout_row"] = old_to_new[
            holdout_row[selected_holdout].astype(np.int64)
        ].astype(np.uint32)
        projected["hidden_fp16_holdout"] = holdout_hidden[selected_holdout]
        if (
            projected["hidden_fp16_holdout"].shape[0] !=
            projected["hidden_fp16_holdout_row"].size or
            any(value.shape[0] != rows.size for name, value in projected.items()
                if name not in holdout_names)
        ):
            raise ValueError("projected compact split is inconsistent")
        result[split] = projected
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_sources(
    train_fit_dir: Path, long_train_dirs: list[Path],
) -> dict[str, object]:
    """Bind the only archives authorized to contribute P1 fitting rows."""
    if not isinstance(long_train_dirs, list) or len(long_train_dirs) != 2:
        raise ValueError("exactly two grouped long-training shards are required")
    train_fit_dir = train_fit_dir.resolve(strict=True)
    long_train_dirs = [path.resolve(strict=True) for path in long_train_dirs]
    if len(set(long_train_dirs)) != 2:
        raise ValueError("long-training shard paths are duplicated")

    split_receipt = json.loads(_tracked_bytes(P1_SPLIT_RECEIPT).decode("utf-8"))
    try:
        train_binding = split_receipt["observed"]["splits"]["train-fit"]
    except (KeyError, TypeError) as error:
        raise ValueError("P1 split receipt has no train-fit binding") from error
    train_manifest_path = train_fit_dir / "manifest.json"
    train_records_path = train_fit_dir / "records.npz"
    train_manifest_bytes = train_manifest_path.read_bytes()
    train_manifest = json.loads(train_manifest_bytes.decode("utf-8", errors="strict"))
    if (
        train_manifest.get("format") != "glm52-union-p1-split-npz-v1" or
        train_manifest.get("split") != "train-fit" or
        hashlib.sha256(train_manifest_bytes).hexdigest() != train_binding.get("manifest_sha256") or
        _sha256(train_records_path) != train_binding.get("output_sha256") or
        train_records_path.stat().st_size != train_binding.get("output_bytes")
    ):
        raise ValueError("train-fit archive differs from its authoritative binding")

    long_receipt = json.loads(_tracked_bytes(LONG_COMPACTION_RECEIPT).decode("utf-8"))
    shard_bindings = long_receipt.get("shards")
    if not isinstance(shard_bindings, list) or len(shard_bindings) != 2:
        raise ValueError("long-training receipt has the wrong shard count")
    by_directory = {}
    for binding in shard_bindings:
        if not isinstance(binding, dict) or not isinstance(binding.get("directory"), str):
            raise ValueError("long-training shard binding is malformed")
        directory = Path(binding["directory"]).resolve(strict=True)
        if directory in by_directory:
            raise ValueError("long-training receipt duplicates a shard")
        by_directory[directory] = binding
    if set(long_train_dirs) != set(by_directory):
        raise ValueError("training input includes an unauthorized long shard")

    accepted_long = []
    for directory in long_train_dirs:
        binding = by_directory[directory]
        manifest_path = directory / "manifest.json"
        records_path = directory / "records.npz"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
        if (
            manifest.get("format") != "glm52-union-p0-npz-v1" or
            hashlib.sha256(manifest_bytes).hexdigest() != binding.get("manifest_sha256") or
            _sha256(records_path) != binding.get("output_sha256") or
            records_path.stat().st_size != binding.get("records_bytes")
        ):
            raise ValueError("long-training archive differs from its authoritative binding")
        accepted_long.append({
            "directory": str(directory),
            "manifest_sha256": binding["manifest_sha256"],
            "output_sha256": binding["output_sha256"],
            "output_bytes": binding["records_bytes"],
        })
    return {
        "train_fit": {
            "directory": str(train_fit_dir),
            "manifest_sha256": train_binding["manifest_sha256"],
            "output_sha256": train_binding["output_sha256"],
            "output_bytes": train_binding["output_bytes"],
        },
        "long_train": accepted_long,
    }


def _tracked_bytes(path: Path) -> bytes:
    """Read a repository input once and require byte equality with HEAD."""
    path = path.resolve(strict=True)
    relative = path.relative_to(ROOT)
    observed = path.read_bytes()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True,
    )
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT,
        stdin=subprocess.DEVNULL, capture_output=True,
    )
    if tracked.returncode != 0 or committed.returncode != 0 or committed.stdout != observed:
        raise ValueError(f"trusted input is not tracked and clean at HEAD: {relative}")
    return observed


def _load_compactor():
    path = ROOT / "scripts/77_compact_glm_union_trace.py"
    _tracked_bytes(path)
    specification = importlib.util.spec_from_file_location("glm_union_compactor_for_split", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the frozen corpus publisher")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _repository_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=True,
    )
    value = completed.stdout.strip()
    if not COMMIT_RE.fullmatch(value):
        raise ValueError("repository HEAD is malformed")
    return value


def _case_ids_from_manifest(path: Path, expected_sha256: str) -> list[str]:
    content = _tracked_bytes(path)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("split block manifest hash differs")
    lines = content.decode("utf-8", errors="strict").splitlines()
    if not lines or lines[0] != "# id\tprompt_file\tcontinuation_file\tresponse_file":
        raise ValueError("split block manifest header differs")
    result: list[str] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4 or not re.fullmatch(r"case_[0-9]{3}", fields[0]):
            raise ValueError("split block manifest row is malformed")
        result.append(fields[0])
    if not result or len(result) != len(set(result)):
        raise ValueError("split block manifest case IDs are empty or duplicated")
    return result


def expected_case_splits(split_plan: dict[str, object]) -> dict[str, str]:
    """Derive the exact frozen case-to-split mapping from the bound block manifests."""
    if not isinstance(split_plan, dict):
        raise ValueError("split plan is malformed")
    splits = split_plan.get("splits")
    holdout = split_plan.get("full_precision_hidden_holdout")
    if not isinstance(splits, dict) or not isinstance(holdout, dict):
        raise ValueError("split plan schema differs")
    result: dict[str, str] = {}
    for plan_name, output_name in (
        ("train", "train-fit"), ("calibration", "calibration"), ("test", "test"),
    ):
        specification = splits.get(plan_name)
        if not isinstance(specification, dict):
            raise ValueError("split plan block differs")
        block_manifests = specification.get("block_manifests")
        if not isinstance(block_manifests, list) or not block_manifests:
            raise ValueError("split plan has no block manifests")
        for block in block_manifests:
            if not isinstance(block, dict) or set(block) != {"path", "sha256"}:
                raise ValueError("split plan manifest binding differs")
            path_value, digest = block["path"], block["sha256"]
            if not isinstance(path_value, str) or not isinstance(digest, str):
                raise ValueError("split plan manifest binding is malformed")
            for case_id in _case_ids_from_manifest(ROOT / path_value, digest):
                if case_id in result:
                    raise ValueError("case ID crosses frozen split blocks")
                result[case_id] = output_name
    diagnostic = holdout.get("case_ids")
    if (
        not isinstance(diagnostic, list) or len(diagnostic) != 5 or
        len(set(diagnostic)) != len(diagnostic) or
        any(result.get(case_id) != "train-fit" for case_id in diagnostic)
    ):
        raise ValueError("precision diagnostic cases differ")
    for case_id in diagnostic:
        result[case_id] = "train-precision-diagnostic"
    observed_counts = {name: sum(value == name for value in result.values()) for name in SPLIT_COUNTS}
    if observed_counts != SPLIT_COUNTS:
        raise ValueError("frozen case split counts differ")
    return result


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def split_archive(source_dir: Path, out_root: Path) -> dict[str, object]:
    """Validate and publish isolated quality splits without analyzing held-out labels."""
    source_dir = source_dir.resolve(strict=True)
    manifest_path = source_dir / "manifest.json"
    records_path = source_dir / "records.npz"
    splitter_bytes = _tracked_bytes(Path(__file__))
    splitter_sha256 = hashlib.sha256(splitter_bytes).hexdigest()
    repository_head = _repository_head()
    receipt = json.loads(_tracked_bytes(QUALITY_COMPACTION_RECEIPT).decode("utf-8"))
    split_plan_bytes = _tracked_bytes(SPLIT_PLAN)
    split_plan_sha256 = hashlib.sha256(split_plan_bytes).hexdigest()
    split_plan = json.loads(split_plan_bytes.decode("utf-8", errors="strict"))
    case_splits = expected_case_splits(split_plan)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != receipt.get("manifest_sha256") or
        _sha256(records_path) != receipt.get("output_sha256") or
        records_path.stat().st_size != receipt.get("output_bytes") or
        manifest.get("format") != "glm52-union-p0-npz-v2" or
        manifest.get("requests") != receipt.get("observed", {}).get("requests") or
        manifest.get("rows") != receipt.get("observed", {}).get("rows") or
        manifest.get("output_sha256") != receipt.get("output_sha256") or
        manifest.get("raw_source_retained") != receipt.get("retained_raw_directory")
    ):
        raise ValueError("quality compact source differs from its reviewed receipt")
    schema = manifest.get("array_schema")
    if not isinstance(schema, dict) or len(schema) != 10:
        raise ValueError("quality compact array schema is malformed")
    with np.load(records_path, allow_pickle=False) as archive:
        if len(archive.files) != len(schema) or set(archive.files) != set(schema):
            raise ValueError("quality compact array set differs")
        arrays = {}
        for name in archive.files:
            value = archive[name]
            expected = schema[name]
            if (
                value.dtype.str != expected.get("dtype") or
                list(value.shape) != expected.get("shape") or
                hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() !=
                expected.get("sha256") or
                (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all())
            ):
                raise ValueError(f"quality compact array differs: {name}")
            arrays[name] = value.copy()
    metadata = manifest.get("request_metadata")
    split_rows = partition_request_rows(
        arrays["request_index"], metadata, SPLIT_COUNTS, case_splits,
    )
    projected = split_compact_arrays(arrays, split_rows)

    publisher = _load_compactor()
    requested = out_root.absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(requested)
    parent = requested.parent.resolve(strict=True)
    out_root = parent / requested.name
    staging = Path(tempfile.mkdtemp(prefix=f".{out_root.name}.tmp.", dir=parent))
    published = {}
    try:
        for split in SPLIT_COUNTS:
            request_ids = sorted(set(int(value) for value in projected[split]["request_index"]))
            split_metadata = [
                row for row in metadata if int(row["request_index"]) in request_ids
            ]
            record = {
                "schema_version": 1,
                "format": "glm52-union-p1-split-npz-v1",
                "split": split,
                "requests": len(request_ids),
                "rows": int(projected[split]["request_index"].size),
                "request_metadata": split_metadata,
                "source_manifest_sha256": receipt["manifest_sha256"],
                "source_output_sha256": receipt["output_sha256"],
                "split_plan_sha256": split_plan_sha256,
                "splitter_sha256": splitter_sha256,
                "repository_head": repository_head,
            }
            split_manifest = publisher.publish_bundle(
                staging / split, projected[split], record,
            )
            published[split] = {
                "requests": record["requests"],
                "rows": record["rows"],
                "manifest_sha256": _sha256(staging / split / "manifest.json"),
                "output_sha256": split_manifest["output_sha256"],
                "output_bytes": split_manifest["output_bytes"],
            }
        result = {
            "schema_version": 1,
            "classification": "DERIVED_SPLITS",
            "source_manifest_sha256": receipt["manifest_sha256"],
            "source_output_sha256": receipt["output_sha256"],
            "split_plan_sha256": split_plan_sha256,
            "splitter_sha256": splitter_sha256,
            "repository_head": repository_head,
            "splits": published,
        }
        top = staging / "manifest.json"
        with top.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(staging, out_root)
        staging = None
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return result
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def run(args: argparse.Namespace) -> int:
    result = split_archive(args.source_dir, args.out_root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    split = subcommands.add_parser("split", help="publish the frozen grouped corpus splits")
    split.add_argument("--source-dir", required=True, type=Path)
    split.add_argument("--out-root", required=True, type=Path)
    return result


def future_union_targets(
    request_index: np.ndarray,
    layer: np.ndarray,
    token_position: np.ndarray,
    selected_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid source-row indices and exact future-K expert-union labels."""
    arrays = (request_index, layer, token_position, selected_ids)
    if any(not isinstance(value, np.ndarray) for value in arrays):
        raise ValueError("future-union inputs must be numpy arrays")
    if (
        request_index.ndim != 1 or layer.ndim != 1 or token_position.ndim != 1 or
        selected_ids.ndim != 2 or selected_ids.shape[0] != request_index.size or
        layer.size != request_index.size or token_position.size != request_index.size or
        request_index.size == 0 or k not in K_VALUES or
        not all(np.issubdtype(value.dtype, np.integer) for value in arrays) or
        np.any(request_index <= 0) or np.any(layer < 0) or np.any(token_position < 0) or
        np.any(selected_ids < 0) or np.any(selected_ids >= N_EXPERT) or
        any(np.unique(row).size != row.size for row in selected_ids)
    ):
        raise ValueError("future-union input schema is invalid")

    groups: list[tuple[int, int]] = []
    starts: list[int] = []
    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for index, key in enumerate(zip(request_index.tolist(), layer.tolist())):
        current = (int(key[0]), int(key[1]))
        if current != previous:
            if current in seen:
                raise ValueError("request/layer group is repeated or reordered")
            if previous is not None and current <= previous:
                raise ValueError("request/layer groups are not in canonical order")
            seen.add(current)
            groups.append(current)
            starts.append(index)
            previous = current
    starts.append(request_index.size)

    output_rows: list[int] = []
    output_targets: list[np.ndarray] = []
    for group_index in range(len(groups)):
        start, stop = starts[group_index], starts[group_index + 1]
        positions = token_position[start:stop].astype(np.int64, copy=False)
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
            raise ValueError("token positions are gapped, duplicated, or reordered")
        for source in range(start, stop - k):
            target = np.zeros(N_EXPERT, dtype=np.bool_)
            target[selected_ids[source + 1:source + k + 1].reshape(-1)] = True
            if not target.any():
                raise ValueError("future expert union is empty")
            output_rows.append(source)
            output_targets.append(target)
    if not output_rows:
        raise ValueError("no complete future-K window exists")
    return np.asarray(output_rows, dtype=np.int64), np.stack(output_targets)


def score_rankings(
    row_indices: np.ndarray,
    targets: np.ndarray,
    rankings: np.ndarray,
    request_index: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, object]:
    """Score macro-request and event-weighted recall/precision without rounding."""
    if (
        not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1 or
        not isinstance(targets, np.ndarray) or targets.shape != (row_indices.size, N_EXPERT) or
        not isinstance(rankings, np.ndarray) or rankings.shape != targets.shape or
        not isinstance(request_index, np.ndarray) or request_index.ndim != 1 or
        row_indices.size == 0 or not np.issubdtype(row_indices.dtype, np.integer) or
        not np.issubdtype(rankings.dtype, np.integer) or
        np.any(row_indices < 0) or np.any(row_indices >= request_index.size) or
        np.any(rankings < 0) or np.any(rankings >= N_EXPERT) or
        any(np.unique(row).size != N_EXPERT for row in rankings) or
        any(not isinstance(value, int) or isinstance(value, bool) or
            not 1 <= value <= N_EXPERT for value in budgets) or
        len(set(budgets)) != len(budgets)
    ):
        raise ValueError("ranking scorer input schema is invalid")
    if not np.issubdtype(targets.dtype, np.bool_):
        if not np.issubdtype(targets.dtype, np.integer) or np.any((targets != 0) & (targets != 1)):
            raise ValueError("targets must be Boolean")
        targets = targets.astype(np.bool_)
    target_sizes = targets.sum(axis=1)
    if np.any(target_sizes <= 0):
        raise ValueError("target union is empty")
    requests = request_index[row_indices]
    unique_requests = np.unique(requests)
    if np.any(unique_requests <= 0):
        raise ValueError("request identity is invalid")

    by_budget: dict[str, dict[str, float]] = {}
    event_rows = np.arange(row_indices.size)[:, None]
    for budget in budgets:
        predicted = rankings[:, :budget]
        hits = targets[event_rows, predicted].sum(axis=1).astype(np.float64)
        recall = hits / target_sizes
        precision = hits / budget
        wasted = budget - hits
        coverage = (hits == target_sizes).astype(np.float64)

        def macro(values: np.ndarray) -> float:
            return float(np.mean([
                np.mean(values[requests == request]) for request in unique_requests
            ]))

        by_budget[str(budget)] = {
            "macro_request_recall": macro(recall),
            "macro_request_precision": macro(precision),
            "macro_request_wasted_experts": macro(wasted),
            "macro_request_full_set_coverage": macro(coverage),
            "event_weighted_recall": float(np.mean(recall)),
            "event_weighted_precision": float(np.mean(precision)),
            "event_weighted_wasted_experts": float(np.mean(wasted)),
            "event_weighted_full_set_coverage": float(np.mean(coverage)),
        }
    return {
        "requests": int(unique_requests.size),
        "events": int(row_indices.size),
        "budgets": list(budgets),
        "by_budget": by_budget,
    }


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "split":
        raise SystemExit(run(arguments))
    raise SystemExit("unsupported command")
