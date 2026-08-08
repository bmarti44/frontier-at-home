#!/usr/bin/env python3
"""Fail-closed scorer for the W4 exact top-k CUDA confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


class ScoreError(ValueError):
    pass


ROOT = Path(__file__).resolve().parents[1]
NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
T95_DF4 = 2.131846786
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30
MANIFEST_KEYS = {
    "schema", "gate", "candidate", "candidate_hash", "freeze_time_unix",
    "binary_sha256", "scorer_sha256", "raw_sha256", "configuration",
    "randomness", "invocation", "device", "artifacts",
}
CONFIGURATION = {
    "n_components": 1048576,
    "n_tokens": 8,
    "top_k": 2048,
    "blocks": 5,
    "observations_per_block": 4,
    "flag_name": "DS4_CUDA_TOPK2048_CUB",
    "flag_value": "1",
    "required_speedup_lower_95": 2.0,
}
ROW_KEYS = {
    "schema", "block", "sequence", "arm", "elapsed_ms", "ids_sha256",
    "ids_identical_to_expected", "effective_marker_present",
    "n_components", "n_tokens", "top_k",
}


def _reject_constant(value: str) -> None:
    raise ScoreError(f"non-finite JSON constant: {value}")


def _read_bound(path: Path) -> tuple[bytes, os.stat_result]:
    if path.is_symlink():
        raise ScoreError(f"symlink artifact rejected: {path.name}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ScoreError(f"cannot open artifact {path.name}: {exc}") from exc
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after) or not os.path.isfile(path):
        raise ScoreError(f"artifact changed while read: {path.name}")
    return b"".join(chunks), after


def _json_bytes(data: bytes, label: str) -> object:
    try:
        return json.loads(data, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"malformed {label}: {exc}") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
            c not in "0123456789abcdef" for c in value):
        raise ScoreError(f"invalid {label}")
    return value


def _artifact(run_dir: Path, name: str, record: object) -> bytes:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
        raise ScoreError(f"invalid artifact record: {name}")
    rel = record["path"]
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ScoreError(f"unsafe artifact path: {name}")
    path = run_dir / rel
    data, stat = _read_bound(path)
    if stat.st_size != record["bytes"] or _digest(data) != record["sha256"]:
        raise ScoreError(f"artifact binding mismatch: {name}")
    return data


def _schedule(randomness: str) -> list[str]:
    first_baab = bool(int(randomness[:2], 16) & 1)
    return ["BAAB" if first_baab ^ bool(block & 1) else "ABBA"
            for block in range(5)]


def score_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve(strict=True)
    manifest_data, _ = _read_bound(run_dir / "manifest.json")
    manifest = _json_bytes(manifest_data, "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ScoreError("manifest schema keys differ")
    if manifest["schema"] != "glm52-w4-topk-manifest-v1" or \
            manifest["gate"] != "W4" or manifest["candidate"] != 3:
        raise ScoreError("wrong manifest identity")
    _require_hex(manifest["candidate_hash"], 40, "candidate hash")
    _require_hex(manifest["binary_sha256"], 64, "binary SHA-256")
    if manifest["scorer_sha256"] != _digest(Path(__file__).read_bytes()):
        raise ScoreError("stale or modified scorer")
    if manifest["configuration"] != CONFIGURATION:
        raise ScoreError("configuration differs from fixed gate")
    if not isinstance(manifest["invocation"], dict) or \
            manifest["invocation"].get("exit_code") != 0 or \
            manifest["invocation"].get("environment") != {
                "DS4_CUDA_TOPK2048_CUB": "scheduled-per-arm"}:
        raise ScoreError("invalid invocation or disabled candidate flag")
    if not isinstance(manifest["device"], dict) or not all(
            isinstance(manifest["device"].get(k), str) and manifest["device"][k]
            for k in ("name", "uuid", "driver")):
        raise ScoreError("missing device identity")

    artifacts = manifest["artifacts"]
    required_artifacts = {
        "binary", "engine.cu", "test.cu", "runner.py",
        "scorer.py", "randomness-receipt.json", "drand-verifier.mjs",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        raise ScoreError("artifact set differs")
    bound = {name: _artifact(run_dir, name, artifacts[name])
             for name in sorted(artifacts)}
    if _digest(bound["binary"]) != manifest["binary_sha256"]:
        raise ScoreError("binary identity mismatch")

    receipt = _json_bytes(bound["randomness-receipt.json"], "randomness receipt")
    randomness = manifest["randomness"]
    if not isinstance(receipt, dict) or set(receipt) != {
            "round", "randomness", "signature", "previous_signature"}:
        raise ScoreError("randomness receipt schema differs")
    if not isinstance(randomness, dict) or set(randomness) != set(receipt) | {
            "verification", "receipt_path"}:
        raise ScoreError("manifest randomness schema differs")
    if any(randomness[k] != receipt[k] for k in receipt) or \
            randomness["verification"] != "DRAND_BLS_RECEIPT_OK" or \
            randomness["receipt_path"] != "randomness-receipt.json":
        raise ScoreError("randomness receipt binding mismatch")
    round_value = receipt["round"]
    if not isinstance(round_value, int) or isinstance(round_value, bool) or round_value < 1:
        raise ScoreError("invalid randomness round")
    for key, length in (("randomness", 64), ("signature", 192),
                        ("previous_signature", 192)):
        _require_hex(receipt[key], length, key)
    freeze_time = manifest["freeze_time_unix"]
    round_time = DRAND_GENESIS_UNIX + (round_value - 1) * DRAND_PERIOD_SECONDS
    if not isinstance(freeze_time, int) or isinstance(freeze_time, bool) or \
            round_time <= freeze_time:
        raise ScoreError("randomness was not published strictly after freeze")
    verifier_path = run_dir / artifacts["drand-verifier.mjs"]["path"]
    verified = subprocess.run(
        [str(NODE), str(verifier_path), str(round_value), receipt["randomness"],
         receipt["signature"], receipt["previous_signature"]],
        cwd=run_dir, text=True, capture_output=True, timeout=30,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
        check=False,
    )
    if verified.returncode != 0 or verified.stdout != "DRAND_BLS_RECEIPT_OK\n":
        raise ScoreError("drand BLS verification failed")

    raw_data, _ = _read_bound(run_dir / "raw.jsonl")
    if _digest(raw_data) != manifest["raw_sha256"]:
        raise ScoreError("raw evidence hash mismatch")
    try:
        raw_lines = raw_data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ScoreError("raw evidence is not UTF-8") from exc
    if len(raw_lines) != 20 or not raw_lines or any(not line for line in raw_lines):
        raise ScoreError("raw evidence must contain exactly 20 nonempty rows")
    rows = [_json_bytes(line.encode(), f"raw row {i}")
            for i, line in enumerate(raw_lines)]
    schedules = _schedule(receipt["randomness"])
    expected_hash = None
    block_a = []
    block_b = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ScoreError(f"raw row {index} schema differs")
        block, sequence = divmod(index, 4)
        if row["schema"] != "glm52-w4-topk-observation-v1" or \
                row["block"] != block or row["sequence"] != sequence or \
                row["arm"] != schedules[block][sequence]:
            raise ScoreError(f"raw row {index} order or arm differs")
        elapsed = row["elapsed_ms"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or \
                not math.isfinite(elapsed) or elapsed <= 0:
            raise ScoreError(f"raw row {index} timing invalid")
        if (row["n_components"], row["n_tokens"], row["top_k"]) != \
                (1048576, 8, 2048):
            raise ScoreError(f"raw row {index} fixture differs")
        digest = _require_hex(row["ids_sha256"], 64, "selected IDs SHA-256")
        if expected_hash is None:
            expected_hash = digest
        if digest != expected_hash or row["ids_identical_to_expected"] is not True:
            raise ScoreError(f"raw row {index} selected IDs differ")
        if row["effective_marker_present"] is not (row["arm"] == "B"):
            raise ScoreError(f"raw row {index} effective marker differs")
        (block_a if row["arm"] == "A" else block_b).append((block, float(elapsed)))
    a_means = [statistics.fmean(v for b, v in block_a if b == block)
               for block in range(5)]
    b_means = [statistics.fmean(v for b, v in block_b if b == block)
               for block in range(5)]
    if any(sum(1 for b, _ in block_a if b == block) != 2 or
           sum(1 for b, _ in block_b if b == block) != 2 for block in range(5)):
        raise ScoreError("each block must contain exactly two observations per arm")
    log_ratios = [math.log(a / b) for a, b in zip(a_means, b_means)]
    mean_log = statistics.fmean(log_ratios)
    sample_sd = statistics.stdev(log_ratios)
    lower95 = math.exp(mean_log - T95_DF4 * sample_sd / math.sqrt(5))
    verdict = "PASS" if lower95 >= 2.0 else "FAIL"
    return {
        "schema": "glm52-w4-topk-summary-v1",
        "formula": "exp(mean(log(block_mean_A_ms/block_mean_B_ms)) - t_0.95_df4 * sample_sd/sqrt(5))",
        "t_0.95_df4": T95_DF4,
        "block_a_ms": a_means,
        "block_b_ms": b_means,
        "log_ratios": log_ratios,
        "speedup_lower_95": lower95,
        "required_speedup_lower_95": 2.0,
        "selected_ids_sha256": expected_hash,
        "raw_sha256": _digest(raw_data),
        "manifest_sha256": _digest(manifest_data),
        "checks": {
            "twenty_ordered_observations": True,
            "randomness_bls_verified": True,
            "randomness_strictly_post_freeze": True,
            "artifacts_hash_bound": True,
            "ids_and_order_exact": True,
            "candidate_marker_effective": True,
        },
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = score_run(args.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
