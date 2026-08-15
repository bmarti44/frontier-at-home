#!/usr/bin/env python3
"""Fixed fail-closed scorer for the W6 query-tile CUDA microgate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


T95_DF4 = 2.131846786326649
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30
NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
WIDTHS = (1, 2, 4)
EXPECTED_LOGICAL_BYTES = {1: 2147483648, 2: 1073741824, 4: 536870912}


class ScoreError(ValueError):
    pass


def _pairs(values: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in values:
        if key in result:
            raise ScoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ScoreError(f"non-finite JSON constant: {value}")


def _json(data: bytes, label: str) -> object:
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"malformed {label}: {exc}") from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
            c not in "0123456789abcdef" for c in value):
        raise ScoreError(f"invalid {label}")
    return value


def _read(path: Path) -> bytes:
    if path.is_symlink():
        raise ScoreError(f"symlink rejected: {path.name}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after):
        raise ScoreError(f"artifact changed while read: {path.name}")
    return b"".join(chunks)


def schedule_from_randomness(randomness: str) -> list[int]:
    seed = bytes.fromhex(_hex(randomness, 64, "randomness"))
    tagged = []
    for width in WIDTHS:
        for occurrence in range(5):
            key = hashlib.sha256(seed + f":{width}:{occurrence}".encode()).digest()
            tagged.append((key, width))
    return [width for _, width in sorted(tagged)]


def expected_stderr(schedule: list[int]) -> bytes:
    lines = ["ds4: CUDA backend initialized on NVIDIA GB10 (sm_121) dev=0"]
    for _ in range(12):
        lines.extend(f"ds4: GLM indexer query tiles={width}" for width in (1, 2, 4))
    lines.extend(
        ["ds4: DS4_CUDA_GLM_INDEXER_QUERY_TILES must be 1, 2, or 4"] * 6)
    lines.extend([
        "ds4: GLM indexer query tiles=2",
        "ds4: GLM indexer query-tile reuse is unavailable in quality mode",
    ])
    previous = 2
    for width in schedule:
        if width != previous:
            lines.append(f"ds4: GLM indexer query tiles={width}")
            previous = width
    return ("\n".join(lines) + "\n").encode()


def validate_and_score_rows(schedule: list[int], raw: bytes) -> dict:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ScoreError("raw evidence is not UTF-8") from exc
    if len(lines) != 16 or any(not line for line in lines):
        raise ScoreError("raw evidence must contain 15 observations and one result")
    rows = [_json(line.encode(), f"raw row {i}") for i, line in enumerate(lines)]
    samples = {width: [] for width in WIDTHS}
    timing_keys = {
        "kind", "sequence", "width", "elapsed_ms", "logical_k_bytes",
        "complete_write", "exact_scores", "exact_ids", "canaries_intact",
    }
    for sequence, (row, expected_width) in enumerate(zip(rows[:15], schedule, strict=True)):
        if not isinstance(row, dict) or set(row) != timing_keys:
            raise ScoreError(f"timing row {sequence} schema differs")
        if row["kind"] != "timing" or row["sequence"] != sequence or \
                row["width"] != expected_width:
            raise ScoreError(f"timing row {sequence} order differs")
        elapsed = row["elapsed_ms"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or \
                not math.isfinite(elapsed) or elapsed <= 0:
            raise ScoreError(f"timing row {sequence} elapsed time invalid")
        if row["logical_k_bytes"] != EXPECTED_LOGICAL_BYTES[expected_width]:
            raise ScoreError(f"timing row {sequence} logical byte count differs")
        if any(row[key] is not True for key in (
                "complete_write", "exact_scores", "exact_ids", "canaries_intact")):
            raise ScoreError(f"timing row {sequence} correctness evidence failed")
        samples[expected_width].append(float(elapsed))
    result = rows[15]
    expected_result = {
        "kind": "result", "verdict": "PASS", "correctness_cases": 12,
        "causal_cases": 5, "ragged_row_cases": 1,
        "invalid_values_rejected": 6, "quality_rejected": True,
    }
    if result != expected_result:
        raise ScoreError("terminal correctness result differs")
    if any(len(samples[width]) != 5 for width in WIDTHS):
        raise ScoreError("each width must have exactly five observations")
    comparisons = {}
    for width in (2, 4):
        logs = [math.log(a / b) for a, b in zip(samples[1], samples[width], strict=True)]
        mean_log = statistics.fmean(logs)
        sample_sd = statistics.stdev(logs)
        lower = math.exp(mean_log - T95_DF4 * sample_sd / math.sqrt(5))
        comparisons[str(width)] = {
            "baseline_ms": samples[1], "candidate_ms": samples[width],
            "paired_log_speedups": logs, "geometric_speedup": math.exp(mean_log),
            "speedup_lower_95": lower, "required_speedup_lower_95": 1.05,
            "passes": lower >= 1.05,
        }
    passing = [width for width in (2, 4) if comparisons[str(width)]["passes"]]
    return {
        "formula": "exp(mean(log(width1_ms/widthN_ms)) - t_0.95_df4*sample_sd/sqrt(5))",
        "t_0.95_df4": T95_DF4,
        "samples_ms": {str(width): samples[width] for width in WIDTHS},
        "comparisons": comparisons,
        "selected_width": max(passing, key=lambda width: comparisons[str(width)]["speedup_lower_95"])
        if passing else None,
        "checks": {
            "randomized_schedule_exact": True,
            "complete_outputs_and_canaries": True,
            "scores_and_ids_exact": True,
            "causal_and_ragged_cases": True,
            "logical_bytes_are_derived_not_physical": True,
        },
        "verdict": "PASS" if passing else "NO_RESULT",
    }


def _artifact(run_dir: Path, record: object, name: str) -> bytes:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
        raise ScoreError(f"artifact record differs: {name}")
    rel = record["path"]
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or \
            ".." in Path(rel).parts:
        raise ScoreError(f"unsafe artifact path: {name}")
    data = _read(run_dir / rel)
    if record["bytes"] != len(data) or record["sha256"] != _sha(data):
        raise ScoreError(f"artifact binding mismatch: {name}")
    return data


def score_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve(strict=True)
    manifest_data = _read(run_dir / "manifest.json")
    manifest = _json(manifest_data, "manifest")
    manifest_keys = {
        "schema", "gate", "candidate", "freeze_time_unix", "source_commit",
        "binary_sha256", "scorer_sha256", "runner_sha256", "raw_sha256",
        "configuration", "randomness", "schedule", "invocation", "device",
        "safety", "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys or \
            manifest["schema"] != "glm52-w6-indexer-tile-reuse-manifest-v1" or \
            manifest["gate"] != "W6-indexer-tile-reuse" or manifest["candidate"] != 2:
        raise ScoreError("manifest identity or schema differs")
    _hex(manifest["source_commit"], 40, "source commit")
    _hex(manifest["binary_sha256"], 64, "binary digest")
    if manifest["scorer_sha256"] != _sha(Path(__file__).read_bytes()):
        raise ScoreError("executing scorer differs from manifest")
    expected_configuration = {
        "rows": 1048576, "ragged_rows": 1048575, "timed_tokens": 64,
        "heads": 32, "head_dim": 128, "top_k": 128, "samples_per_width": 5,
        "supported_widths": [1, 2, 4], "required_speedup_lower_95": 1.05,
        "traffic_semantics": "derived_logical_k_bytes_not_physical_traffic",
    }
    if manifest["configuration"] != expected_configuration:
        raise ScoreError("configuration differs")
    artifacts = manifest["artifacts"]
    required = {
        "binary", "ds4.c", "engine.cu", "test.cu", "Makefile", "scorer.py",
        "runner.py", "drand-verifier.mjs", "randomness-receipt.json", "freeze.json",
        "microgate.stdout", "microgate.stderr",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ScoreError("artifact set differs")
    bound = {name: _artifact(run_dir, artifacts[name], name) for name in sorted(required)}
    if _sha(bound["binary"]) != manifest["binary_sha256"] or \
            _sha(bound["scorer.py"]) != manifest["scorer_sha256"] or \
            _sha(bound["runner.py"]) != manifest["runner_sha256"]:
        raise ScoreError("executable closure identity differs")
    freeze = _json(bound["freeze.json"], "freeze")
    if not isinstance(freeze, dict) or freeze.get("candidate") != 2 or \
            freeze.get("source_commit") != manifest["source_commit"] or \
            freeze.get("binary_sha256") != manifest["binary_sha256"] or \
            freeze.get("freeze_time_unix") != manifest["freeze_time_unix"]:
        raise ScoreError("freeze binding differs")
    frozen = freeze.get("artifact_sha256")
    freeze_names = {
        "ds4.c": "ds4.c", "ds4_cuda.cu": "engine.cu",
        "tests/cuda_indexer_tile_reuse_w6.cu": "test.cu", "Makefile": "Makefile",
        "scripts/106_score_w6_indexer_tile_reuse.py": "scorer.py",
        "scripts/107_run_w6_indexer_tile_reuse.py": "runner.py",
        "scripts/89_verify_drand_receipt.mjs": "drand-verifier.mjs",
    }
    if not isinstance(frozen, dict) or any(
            frozen.get(source) != _sha(bound[target]) for source, target in freeze_names.items()):
        raise ScoreError("frozen artifact map differs")
    receipt = _json(bound["randomness-receipt.json"], "randomness receipt")
    if not isinstance(receipt, dict) or set(receipt) != {
            "round", "randomness", "signature", "previous_signature"}:
        raise ScoreError("randomness receipt schema differs")
    randomness = manifest["randomness"]
    if not isinstance(randomness, dict) or randomness != {
            **receipt, "verification": "DRAND_BLS_RECEIPT_OK",
            "receipt_path": "randomness-receipt.json"}:
        raise ScoreError("randomness binding differs")
    round_value = receipt["round"]
    if not isinstance(round_value, int) or isinstance(round_value, bool) or round_value < 1:
        raise ScoreError("randomness round invalid")
    for key, length in (("randomness", 64), ("signature", 192),
                        ("previous_signature", 192)):
        _hex(receipt[key], length, key)
    round_time = DRAND_GENESIS_UNIX + (round_value - 1) * DRAND_PERIOD_SECONDS
    if round_time <= manifest["freeze_time_unix"]:
        raise ScoreError("randomness was not published after freeze")
    verified = subprocess.run(
        [str(NODE), str(run_dir / artifacts["drand-verifier.mjs"]["path"]),
         str(round_value), receipt["randomness"], receipt["signature"],
         receipt["previous_signature"]], text=True, capture_output=True,
        timeout=30, check=False, env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})
    if verified.returncode != 0 or verified.stdout != "DRAND_BLS_RECEIPT_OK\n":
        raise ScoreError("drand receipt verification failed")
    schedule = schedule_from_randomness(receipt["randomness"])
    if manifest["schedule"] != schedule:
        raise ScoreError("manifest schedule differs from public randomness")
    raw = _read(run_dir / "raw.jsonl")
    if _sha(raw) != manifest["raw_sha256"] or raw != bound["microgate.stdout"]:
        raise ScoreError("raw evidence binding differs")
    if bound["microgate.stderr"] != expected_stderr(schedule):
        raise ScoreError("microgate stderr state-machine transcript differs")
    invocation = manifest["invocation"]
    if not isinstance(invocation, dict) or invocation.get("exit_code") != 0 or \
            invocation.get("schedule_env") != ",".join(map(str, schedule)):
        raise ScoreError("invocation differs")
    safety = manifest["safety"]
    if not isinstance(safety, dict) or safety.get("failures") != [] or \
            safety.get("engine_processes_present") is not False or \
            safety.get("fio_present") is not False or \
            safety.get("executed_binary_sha256") != manifest["binary_sha256"]:
        raise ScoreError("safety evidence differs")
    result = validate_and_score_rows(schedule, raw)
    return {
        "schema": "glm52-w6-indexer-tile-reuse-summary-v1",
        **result,
        "raw_sha256": _sha(raw), "manifest_sha256": _sha(manifest_data),
        "binary_sha256": manifest["binary_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        result = score_run(args.run_dir)
    except (OSError, ScoreError, subprocess.SubprocessError) as exc:
        print(f"W6 scorer: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
