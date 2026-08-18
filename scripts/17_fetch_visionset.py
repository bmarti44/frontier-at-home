#!/usr/bin/env python3
"""Fetch a deterministic, revision-pinned 100-case MMMU vision evalset.

The source revision is deliberately required rather than resolved from mutable
``main``.  The script prefers ``datasets`` when it is installed and otherwise
reads the validation parquet files through ``huggingface_hub`` and pyarrow.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "evalsets" / "mmmu-val-100"
DATASET_REPO = "MMMU/MMMU"
REVISION_RE = re.compile(r"[0-9a-f]{40}")
IMAGE_COLUMN_RE = re.compile(r"image(?:_\d+)?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="immutable 40-character MMMU dataset commit",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not REVISION_RE.fullmatch(args.revision):
        parser.error("--revision must be a lowercase 40-character git commit")
    if args.count < 1:
        parser.error("--count must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_options(value: Any) -> list[str] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 26:
        return None
    options = [str(option).strip() for option in value]
    return options if all(options) else None


def image_bytes(value: Any) -> tuple[bytes, str | None] | None:
    """Return encoded source bytes and its optional source filename."""
    if value is None:
        return None
    if isinstance(value, dict):
        data = value.get("bytes")
        path = value.get("path")
        if data is None and isinstance(path, str):
            source = Path(path)
            if source.is_file():
                data = source.read_bytes()
        if isinstance(data, memoryview):
            data = data.tobytes()
        if isinstance(data, bytearray):
            data = bytes(data)
        return (data, path) if isinstance(data, bytes) and data else None
    # A datasets Image feature can decode to a PIL object. Preserve the
    # original encoded file when its filename remains available.
    filename = getattr(value, "filename", None)
    if isinstance(filename, str) and Path(filename).is_file():
        return Path(filename).read_bytes(), filename
    return None


def image_extension(data: bytes, source_name: str | None) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    suffix = Path(source_name).suffix.lower() if source_name else ""
    if suffix in {".png", ".jpg", ".jpeg"}:
        return suffix
    raise RuntimeError("MMMU image is neither encoded PNG nor JPEG")


def iter_rows_with_datasets(revision: str) -> Iterable[tuple[str, dict[str, Any]]]:
    import datasets  # type: ignore[import-not-found]

    configs = datasets.get_dataset_config_names(DATASET_REPO, revision=revision)
    for subject in sorted(configs):
        dataset = datasets.load_dataset(
            DATASET_REPO,
            subject,
            split="validation",
            revision=revision,
        )
        # Keep the original encoded bytes. Decoding to pixels and re-encoding
        # would change the shipped artifact and its digest.
        for column, feature in dataset.features.items():
            if IMAGE_COLUMN_RE.fullmatch(column) and isinstance(feature, datasets.Image):
                dataset = dataset.cast_column(column, datasets.Image(decode=False))
        for row in dataset:
            yield subject, dict(row)


def iter_rows_with_parquet(revision: str) -> Iterable[tuple[str, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "fetching MMMU requires datasets, or huggingface_hub plus pyarrow "
            "(use .venv-harness)"
        ) from error

    api = HfApi()
    files = api.list_repo_files(
        DATASET_REPO, repo_type="dataset", revision=revision
    )
    parquet_files = sorted(
        path
        for path in files
        if path.lower().endswith(".parquet")
        and "validation" in Path(path).name.lower()
    )
    if not parquet_files:
        raise RuntimeError("pinned MMMU revision contains no validation parquet files")
    for repo_path in parquet_files:
        local = hf_hub_download(
            DATASET_REPO,
            repo_path,
            repo_type="dataset",
            revision=revision,
        )
        parts = Path(repo_path).parts
        subject = parts[-2] if len(parts) > 1 else "default"
        for row in pq.read_table(local).to_pylist():
            row_subject = row.get("subject")
            yield row_subject if isinstance(row_subject, str) else subject, row


def source_rows(revision: str) -> Iterable[tuple[str, dict[str, Any]]]:
    try:
        import datasets  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return iter_rows_with_parquet(revision)
    return iter_rows_with_datasets(revision)


def collect_candidates(
    rows: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for subject, row in rows:
        case_id = row.get("id")
        question = row.get("question")
        options = normalize_options(row.get("options"))
        answer = row.get("answer")
        if not isinstance(case_id, (str, int)) or isinstance(case_id, bool):
            continue
        case_id = str(case_id)
        if not isinstance(question, str) or not question.strip() or options is None:
            continue
        if not isinstance(answer, str):
            continue
        answer = answer.strip().upper()
        if not re.fullmatch(r"[A-Z]", answer):
            continue
        if ord(answer) - ord("A") >= len(options):
            continue
        images = []
        for key, value in row.items():
            if IMAGE_COLUMN_RE.fullmatch(str(key)):
                encoded = image_bytes(value)
                if encoded is not None:
                    images.append(encoded)
        if len(images) != 1:
            continue
        if case_id in seen_ids:
            raise RuntimeError(f"duplicate MMMU id among eligible rows: {case_id}")
        seen_ids.add(case_id)
        data, source_name = images[0]
        by_subject.setdefault(subject, []).append(
            {
                "id": case_id,
                "question": question.strip(),
                "options": options,
                "answer": answer,
                "image_bytes": data,
                "image_extension": image_extension(data, source_name),
            }
        )
    for candidates in by_subject.values():
        candidates.sort(key=lambda case: case["id"])
    return by_subject


def balanced_selection(
    by_subject: dict[str, list[dict[str, Any]]], count: int
) -> list[dict[str, Any]]:
    """Round-robin sorted subjects, consuming each subject's sorted IDs."""
    positions = {subject: 0 for subject in by_subject}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for subject in sorted(by_subject):
            position = positions[subject]
            if position >= len(by_subject[subject]):
                continue
            case = dict(by_subject[subject][position])
            case["subject"] = subject
            selected.append(case)
            positions[subject] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} eligible single-image multiple-choice cases; "
            f"requested {count}"
        )
    return sorted(selected, key=lambda case: (case["id"], case["subject"]))


def safe_image_stem(case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip(".-") or "case"
    suffix = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:80]}-{suffix}"


def write_package(
    destination: Path, revision: str, selected: list[dict[str, Any]]
) -> dict[str, Any]:
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"output already exists; refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mmmu-val-", dir=destination.parent))
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        cases_path = temporary / "cases.jsonl"
        with cases_path.open("w", encoding="utf-8") as stream:
            for case in selected:
                filename = safe_image_stem(case["id"]) + case["image_extension"]
                image_path = images_dir / filename
                if image_path.exists():
                    raise RuntimeError(f"image filename collision: {filename}")
                image_path.write_bytes(case["image_bytes"])
                record = {
                    "id": case["id"],
                    "question": case["question"],
                    "options": case["options"],
                    "answer": case["answer"],
                    "image": f"images/{filename}",
                }
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")

        files: dict[str, dict[str, Any]] = {}
        stored = [cases_path, *sorted(images_dir.iterdir())]
        for path in stored:
            relative = path.relative_to(temporary).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        pins = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET_REPO,
            "revision": revision,
            "split": "validation",
            "selection": "balanced subjects; sorted subjects and ids; no seed",
            "rows": len(selected),
            "files": files,
        }
        (temporary / "pins.json").write_text(
            json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "ok": True,
        "output": str(destination),
        "rows": len(selected),
        "subjects": len({case["subject"] for case in selected}),
        "files_written": len(selected) + 2,
    }


def main() -> int:
    args = parse_args()
    try:
        candidates = collect_candidates(source_rows(args.revision))
        selected = balanced_selection(candidates, args.count)
        summary = write_package(args.out_dir, args.revision, selected)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"17_fetch_visionset.py: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
