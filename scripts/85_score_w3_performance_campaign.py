#!/usr/bin/env python3
"""Fixed scorer for the five-block W3 completed-time campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any


EXPECTED_ORDERS = (
    "off-on", "on-off",  # ABBA
    "on-off", "off-on",  # BAAB
    "off-on", "on-off",  # ABBA
    "on-off", "off-on",  # BAAB
    "off-on", "on-off",  # ABBA
)
TOKEN_RE = re.compile(
    r"DS4_TOKEN_TIMING request=(\S+) index=(\d+) "
    r"monotonic_ns=(\d+) token=(-?\d+)"
)
T95_DF4 = 2.1318


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=strict_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return raw, value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an exact integer")
    return value


def require_digest(value: Any, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(fr"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{label} is not a lowercase hexadecimal digest")
    return value


def verified_manifest(pair: Path, summary_raw: bytes) -> dict[str, Any]:
    _, manifest = strict_json(pair / "manifest.json")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or artifacts.get("summary.json") != sha256_bytes(summary_raw):
        raise ValueError(f"{pair}: manifest does not bind summary.json")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"{pair}: unsafe manifest artifact path")
        require_digest(expected, f"{pair}:{relative} digest")
        artifact = pair / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"{pair}: artifact digest mismatch: {relative}")
    return manifest


def measured_timing(cmd_path: Path, expected_sha: str) -> dict[str, Any]:
    require_digest(expected_sha, f"{cmd_path} expected digest")
    raw = cmd_path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"{cmd_path}: cmd.log digest mismatch")
    groups: list[tuple[str, list[tuple[int, int, int]]]] = []
    for raw_line in raw.decode("utf-8", errors="strict").splitlines():
        match = TOKEN_RE.fullmatch(raw_line)
        if match is None:
            continue
        request = match.group(1)
        row = tuple(int(match.group(i)) for i in (2, 3, 4))
        if not groups or groups[-1][0] != request:
            if any(previous == request for previous, _ in groups):
                raise ValueError(f"{cmd_path}: interleaved token timing request")
            groups.append((request, []))
        groups[-1][1].append(row)
    if len(groups) != 2:
        raise ValueError(f"{cmd_path}: expected exactly warm and measured timing groups")
    for phase, (_, rows) in zip(("warm", "measured"), groups, strict=True):
        if len(rows) != 129 or [row[0] for row in rows] != list(range(1, 130)):
            raise ValueError(f"{cmd_path}: {phase} group is not exactly 129 ordered tokens")
        stamps = [row[1] for row in rows]
        if any(right <= left for left, right in zip(stamps, stamps[1:])):
            raise ValueError(f"{cmd_path}: {phase} timestamps are not strictly increasing")
    request, rows = groups[1]
    timestamps = [row[1] for row in rows]
    seconds = (timestamps[-1] - timestamps[0]) / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{cmd_path}: invalid completed decode time")
    return {
        "request_id": request,
        "token_timestamps_ns": timestamps,
        "token_ids": [row[2] for row in rows],
        "completed_seconds": seconds,
        "decode_tokens_per_second": 128.0 / seconds,
        "cmd_log_sha256": expected_sha,
    }


def score_pair(pair: Path, expected_order: str) -> dict[str, Any]:
    summary_raw, summary = strict_json(pair / "summary.json")
    manifest = verified_manifest(pair, summary_raw)
    if summary.get("status") != "PASS" or summary.get("arm_order") != expected_order:
        raise ValueError(f"{pair}: pair status or randomized arm order is invalid")
    if exact_int(summary.get("required_completion_tokens"), f"{pair}: token count") != 129:
        raise ValueError(f"{pair}: pair does not require 129 completion tokens")
    checks = summary.get("checks")
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        raise ValueError(f"{pair}: runtime eligibility checks did not all pass")
    bindings = ("binary_sha256", "model_sha256", "tokenizer_sha256", "freeze_sha256")
    for key in bindings:
        require_digest(summary.get(key), f"{pair}:{key}")
        if manifest.get(key) != summary[key]:
            raise ValueError(f"{pair}: manifest {key} mismatch")
    repository_head = require_digest(summary.get("repository_head"), f"{pair}:repository_head", 40)
    if manifest.get("repository_head") != repository_head:
        raise ValueError(f"{pair}: manifest repository head mismatch")
    request_sha = require_digest(summary.get("request_sha256"), f"{pair}:request_sha256")
    if manifest.get("request_sha256") != request_sha:
        raise ValueError(f"{pair}: manifest request digest mismatch")
    randomness = summary.get("public_randomness")
    if not isinstance(randomness, dict):
        raise ValueError(f"{pair}: public randomness is missing")
    round_number = exact_int(randomness.get("round"), f"{pair}: randomness round")
    floor = exact_int(randomness.get("freeze_floor_round"), f"{pair}: freeze floor")
    require_digest(randomness.get("randomness"), f"{pair}: randomness")
    require_digest(randomness.get("signature"), f"{pair}: randomness signature", 192)
    if round_number <= floor:
        raise ValueError(f"{pair}: public randomness predates freeze")
    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"off", "on"}:
        raise ValueError(f"{pair}: expected exact off/on arms")
    measured: dict[str, Any] = {}
    for name in ("off", "on"):
        arm = arms[name]
        if not isinstance(arm, dict) or arm.get("arm") != name:
            raise ValueError(f"{pair}: malformed {name} arm")
        if (arm.get("safe_returncode") != 0 or
                arm.get("independent_completion_tokens") != 129 or
                arm.get("independent_warm_completion_tokens") != 129):
            raise ValueError(f"{pair}: {name} arm is short or unsafe")
        crash = Path(arm.get("crash_evidence", ""))
        crash_hashes = arm.get("crash_artifact_sha256")
        if not isinstance(crash_hashes, dict) or not crash.is_dir():
            raise ValueError(f"{pair}: {name} crash evidence is missing")
        measured[name] = measured_timing(crash / "cmd.log", crash_hashes.get("cmd.log", ""))
    if arms["off"].get("generated_sha256") != arms["on"].get("generated_sha256"):
        raise ValueError(f"{pair}: measured generated bytes differ across arms")
    return {
        "path": str(pair),
        "arm_order": expected_order,
        "request_sha256": request_sha,
        "randomness_round": round_number,
        "randomness": randomness["randomness"],
        "bindings": {key: summary[key] for key in (*bindings, "repository_head")},
        "arms": measured,
        "summary_sha256": sha256_bytes(summary_raw),
        "manifest_sha256": sha256(pair / "manifest.json"),
    }


def upper_ratio(candidate: list[float], baseline: list[float]) -> float:
    logs = [math.log(c / b) for c, b in zip(candidate, baseline, strict=True)]
    mean = statistics.fmean(logs)
    sem = statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean + T95_DF4 * sem)


def write_result(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 10:
        raise ValueError("W3 requires exactly ten fresh-server pairs")
    request_hashes = [row["request_sha256"] for row in rows]
    rounds = [row["randomness_round"] for row in rows]
    random_values = [row["randomness"] for row in rows]
    if len(set(request_hashes)) != 10 or len(set(rounds)) != 10 or len(set(random_values)) != 10:
        raise ValueError("fixtures and public-randomness records must be fresh and unique")
    frozen = rows[0]["bindings"]
    if any(row["bindings"] != frozen for row in rows[1:]):
        raise ValueError("campaign pairs do not share one frozen candidate")
    baseline: list[float] = []
    candidate: list[float] = []
    blocks: list[dict[str, Any]] = []
    for block_index in range(5):
        left, right = rows[2 * block_index:2 * block_index + 2]
        off = [left["arms"]["off"]["completed_seconds"],
               right["arms"]["off"]["completed_seconds"]]
        on = [left["arms"]["on"]["completed_seconds"],
              right["arms"]["on"]["completed_seconds"]]
        baseline.append(statistics.fmean(off))
        candidate.append(statistics.fmean(on))
        blocks.append({
            "block": block_index + 1,
            "schedule": "ABBA" if block_index % 2 == 0 else "BAAB",
            "baseline_seconds": baseline[-1],
            "candidate_seconds": candidate[-1],
            "pair_request_sha256": [left["request_sha256"], right["request_sha256"]],
        })
    bound = upper_ratio(candidate, baseline)
    passed = bound <= 0.95
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    raw_path = output / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "gate": "W3-completed-time-campaign",
        "status": "PASS" if passed else "FAIL",
        "acceptance_formula": (
            "five ABBA/BAAB block means; one-sided 95% upper bound of the "
            "geometric paired candidate/baseline completed-time ratio <= 0.95"
        ),
        "timing_formula": "seconds=(t129-t1)/1e9; decode_tps=128/seconds",
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "completed_time_ratio_upper_95": bound,
        "blocks": blocks,
        "frozen_bindings": frozen,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scorer = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "scorer_sha256": sha256(scorer),
        "raw_sha256": sha256(raw_path),
        "summary_sha256": sha256(summary_path),
        "input_manifest_sha256": [row["manifest_sha256"] for row in rows],
        "input_summary_sha256": [row["summary_sha256"] for row in rows],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pairs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.pairs) != 10:
        raise ValueError("expected exactly ten chronological pair directories")
    rows = [score_pair(path.resolve(), order)
            for path, order in zip(args.pairs, EXPECTED_ORDERS, strict=True)]
    summary = write_result(args.output.resolve(), rows)
    print(json.dumps({"output": str(args.output.resolve()), "status": summary["status"],
                      "completed_time_ratio_upper_95": summary["completed_time_ratio_upper_95"]},
                     sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"W3 campaign scoring failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
