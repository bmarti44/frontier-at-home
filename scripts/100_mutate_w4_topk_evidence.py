#!/usr/bin/env python3
"""Deterministically prove the frozen W4 scorer rejects named corruptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    os.chmod(path, 0o600)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=True) + "\n")


def rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "raw.jsonl").read_text().splitlines()]


def rewrite_raw(run: Path, value: list[dict]) -> None:
    raw = run / "raw.jsonl"
    os.chmod(raw, 0o600)
    raw.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n"
                           for row in value))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["raw_sha256"] = digest(raw)
    write_json(run / "manifest.json", manifest)


def rewrite_artifact_binding(run: Path, name: str) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    path = run / manifest["artifacts"][name]["path"]
    manifest["artifacts"][name]["sha256"] = digest(path)
    manifest["artifacts"][name]["bytes"] = path.stat().st_size
    write_json(run / "manifest.json", manifest)


def invoke(run: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/python3", str(run / "scorer.py"), str(run)],
        text=True, capture_output=True, timeout=30, check=False,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})


def normalize_stderr(stderr: str, run: Path) -> str:
    """Remove only the per-execution mutation directory from tracebacks."""
    return stderr.replace(str(run), "<MUTATED_RUN>")


def mutate_missing(run: Path) -> None:
    rewrite_raw(run, rows(run)[:-1])


def mutate_duplicate(run: Path) -> None:
    value = rows(run); rewrite_raw(run, value + [value[-1]])


def mutate_reordered(run: Path) -> None:
    value = rows(run); rewrite_raw(run, [value[1], value[0], *value[2:]])


def mutate_timing(run: Path, value: float) -> None:
    value_rows = rows(run); value_rows[0]["elapsed_ms"] = value
    rewrite_raw(run, value_rows)


def mutate_all_positive_timings(run: Path) -> None:
    value = rows(run)
    for row in value: row["elapsed_ms"] = 10.0 if row["arm"] == "A" else 1.0
    rewrite_raw(run, value)


def mutate_all_ids(run: Path) -> None:
    value = rows(run)
    for row in value: row["ids_sha256"] = "2" * 64
    rewrite_raw(run, value)


def mutate_marker(run: Path) -> None:
    value = rows(run); next(row for row in value if row["arm"] == "B")[
        "effective_marker_present"] = False
    rewrite_raw(run, value)


def mutate_disabled(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["configuration"]["flag_value"] = "0"
    write_json(run / "manifest.json", manifest)


def mutate_stale_binary(run: Path) -> None:
    path = run / "binary"; os.chmod(path, 0o700); path.write_bytes(path.read_bytes() + b"x")


def mutate_transcript_timing(run: Path) -> None:
    path = run / "microgate.stderr"; os.chmod(path, 0o600)
    text = path.read_text(); path.write_text(text.replace("elapsed_ms=", "elapsed_ms=123", 1))
    rewrite_artifact_binding(run, "microgate.stderr")


def mutate_coherent_closure(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["candidate_hash"] = "a" * 40
    for name in ("binary", "ds4.c", "engine.cu", "test.cu", "Makefile", "runner.py"):
        path = run / manifest["artifacts"][name]["path"]
        os.chmod(path, 0o700); path.write_bytes(path.read_bytes() + b"alternate")
        manifest["artifacts"][name]["sha256"] = digest(path)
        manifest["artifacts"][name]["bytes"] = path.stat().st_size
    manifest["binary_sha256"] = manifest["artifacts"]["binary"]["sha256"]
    write_json(run / "manifest.json", manifest)


def mutate_scorer_error(run: Path) -> None:
    path = run / "scorer.py"
    os.chmod(path, 0o700)
    path.write_text("raise SystemExit(3)\n")


MUTATIONS = (
    ("missing_row", mutate_missing), ("duplicate_row", mutate_duplicate),
    ("reordered_rows", mutate_reordered),
    ("nonfinite_timing", lambda run: mutate_timing(run, math.nan)),
    ("zero_timing", lambda run: mutate_timing(run, 0.0)),
    ("positive_finite_timings", mutate_all_positive_timings),
    ("consistent_wrong_ids", mutate_all_ids), ("missing_marker", mutate_marker),
    ("disabled_flag", mutate_disabled), ("stale_binary", mutate_stale_binary),
    ("transcript_timing_mismatch", mutate_transcript_timing),
    ("coherent_alternate_closure", mutate_coherent_closure),
    ("scorer_error", mutate_scorer_error),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    source = args.run_dir.resolve(strict=True)
    baseline = invoke(source)
    if baseline.returncode != 0 or json.loads(baseline.stdout).get("verdict") != "PASS":
        raise SystemExit("authoritative source run does not replay PASS")
    results = []
    for name, mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix=f"w4-mutation-{name}.") as tmp:
            run = Path(tmp) / "run"; shutil.copytree(source, run)
            mutation(run); completed = invoke(run)
            results.append({
                "mutation": name, "exit_code": completed.returncode,
                "rejected": completed.returncode != 0,
                "stdout_sha256": digest_bytes(completed.stdout.encode()),
                "stderr_sha256": digest_bytes(
                    normalize_stderr(completed.stderr, run).encode()),
                "diagnostic_tail": (completed.stderr.strip().splitlines()[-1]
                                    if completed.stderr.strip() else ""),
            })
    all_rejected = all(result["rejected"] for result in results)
    print(json.dumps({
        "schema": "glm52-w4-topk-mutations-v3", "candidate": 5,
        "stderr_normalization": "exact mutation-run path replaced by <MUTATED_RUN>",
        "source_run": str(args.run_dir),
        "artifacts": {"driver": digest(Path(__file__))},
        "mutations": results, "all_rejected": all_rejected,
        "verdict": "PASS" if all_rejected else "FAIL",
    }, indent=2, sort_keys=True))
    return 0 if all_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
