#!/usr/bin/env python3
"""Compare a DeepSeek-V4-Flash-0731 qualification run against the published baselines.

Every comparison first checks that the two runs were generated under the same
contract and refuses to report a delta when they were not. A prior 0731 run
scored GSM8K dev 94/100 against a baseline's 98/100 purely because the baseline
disabled thinking and the candidate did not; a bare number would have read as a
four-point regression. Configuration mismatch is reported as a mismatch, never
as a result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

# Candidate and baseline filenames do not follow one rule, so they are stated
# rather than derived: the MMLU-Pro baseline is filed under "mmlu", and
# HumanEval has no dev/holdout partition at all (split=all, 164 items).
SUITES = (
    {
        "suite": "gsm8k",
        "candidate": "acc-gsm8k-dev-0731.json",
        "baseline": "acc-gsm8k-dev-ds4.json",
    },
    {
        "suite": "mmlu-pro",
        "candidate": "acc-mmlu-pro-dev-0731.json",
        "baseline": "acc-mmlu-dev-ds4.json",
    },
    {
        "suite": "humaneval",
        "candidate": "acc-humaneval-all-0731.json",
        "baseline": "acc-humaneval-ds4.json",
    },
)


def load(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def wilson95(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval; the same estimator the accuracy harness records."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = correct / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
        / denominator
    )
    return (center - margin, center + margin)


def generation_contract(document: dict[str, Any]) -> dict[str, Any]:
    """The fields that must agree before two accuracy runs are comparable."""
    generation = document.get("generation") or {}
    return {
        "endpoint": generation.get("endpoint"),
        "temperature": generation.get("temperature"),
        "seed": generation.get("seed"),
        "max_tokens": generation.get("max_tokens"),
        "stop": generation.get("stop"),
        "extra_body": generation.get("extra_body"),
    }


def compare_accuracy(spec: dict[str, str], candidate_dir: Path) -> dict[str, Any]:
    suite = spec["suite"]
    candidate = load(candidate_dir / spec["candidate"])
    baseline = load(REPO_ROOT / "results" / spec["baseline"])
    if candidate is None or baseline is None:
        return {
            "suite": suite,
            "status": "MISSING",
            "detail": "candidate or baseline artifact absent",
            "candidate_path": spec["candidate"],
            "baseline_path": spec["baseline"],
        }

    candidate_contract = generation_contract(candidate)
    baseline_contract = generation_contract(baseline)
    differing = {
        key: {"baseline": baseline_contract[key], "candidate": candidate_contract[key]}
        for key in baseline_contract
        if baseline_contract[key] != candidate_contract[key]
    }

    candidate_low, candidate_high = wilson95(candidate["correct"], candidate["n"])
    baseline_low, baseline_high = wilson95(baseline["correct"], baseline["n"])
    # Overlapping Wilson intervals mean the suites cannot separate the two
    # models at n=100; a raw point-difference alone would overstate the case.
    overlap = candidate_low <= baseline_high and baseline_low <= candidate_high

    result = {
        "suite": suite,
        "candidate": {
            "correct": candidate["correct"],
            "n": candidate["n"],
            "accuracy": candidate["accuracy"],
            "wilson95": [candidate_low, candidate_high],
            "invalid_count": candidate.get("invalid_count"),
        },
        "baseline": {
            "correct": baseline["correct"],
            "n": baseline["n"],
            "accuracy": baseline["accuracy"],
            "wilson95": [baseline_low, baseline_high],
            "stack_label": baseline.get("stack_label"),
        },
        "delta_accuracy": round(candidate["accuracy"] - baseline["accuracy"], 4),
        "intervals_overlap": overlap,
        "configuration_mismatch": differing,
    }
    if differing:
        result["status"] = "NOT_COMPARABLE"
        result["detail"] = (
            "generation contract differs from the baseline; the delta above is "
            "not attributable to the model"
        )
    elif candidate.get("invalid_count"):
        result["status"] = "INVALID_ITEMS"
    else:
        result["status"] = "COMPARABLE"
    return result


def median_decode(document: dict[str, Any], ctx: int) -> float | None:
    for cell in document.get("cells", []):
        if cell.get("ctx_tokens") == ctx:
            return cell.get("median_decode")
    return None


def compare_speed(candidate_dir: Path) -> dict[str, Any]:
    candidate = load(candidate_dir / "speed-0731.json")
    baseline = load(REPO_ROOT / "results" / "speed-ds4-dspark.json")
    if candidate is None or baseline is None:
        return {"status": "MISSING", "detail": "candidate or baseline artifact absent"}

    candidate_extra = (candidate.get("metadata") or {}).get("extra_body")
    baseline_extra = (baseline.get("metadata") or {}).get("extra_body")
    cells = []
    for cell in candidate.get("cells", []):
        ctx = cell.get("ctx_tokens")
        cells.append(
            {
                "ctx_tokens": ctx,
                "candidate_median_decode": cell.get("median_decode"),
                "baseline_median_decode": median_decode(baseline, ctx),
                "candidate_median_ttft": cell.get("median_ttft"),
                "cell_valid": cell.get("valid"),
                "invalid_reps": cell.get("invalid_reps"),
            }
        )
    return {
        "status": "COMPARABLE" if candidate_extra == baseline_extra else "NOT_COMPARABLE",
        "suite_valid": candidate.get("suite_valid"),
        "candidate_extra_body": candidate_extra,
        "baseline_extra_body": baseline_extra,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dsv4-0731-staging",
        help="directory holding the 0731 qualification artifacts",
    )
    parser.add_argument("--out", type=Path, help="write the comparison JSON here")
    args = parser.parse_args()

    golden = load(args.candidate_dir / "golden-0731.json")
    report = {
        "candidate_dir": str(args.candidate_dir),
        "golden": {
            "pass": None if golden is None else golden.get("pass"),
            "failed_checks": []
            if golden is None
            else [c["name"] for c in golden.get("checks", []) if not c.get("pass")],
        },
        "speed": compare_speed(args.candidate_dir),
        "accuracy": [compare_accuracy(spec, args.candidate_dir) for spec in SUITES],
    }

    print(f"golden: pass={report['golden']['pass']} "
          f"failed={report['golden']['failed_checks'] or 'none'}")
    speed = report["speed"]
    print(f"speed: {speed['status']} suite_valid={speed.get('suite_valid')}")
    for cell in speed.get("cells", []):
        print(
            f"  ctx {cell['ctx_tokens']:>6}: "
            f"candidate={cell['candidate_median_decode']} "
            f"baseline={cell['baseline_median_decode']} "
            f"valid={cell['cell_valid']}"
        )
    for entry in report["accuracy"]:
        if entry["status"] == "MISSING":
            print(f"{entry['suite']}: MISSING")
            continue
        candidate = entry["candidate"]
        baseline = entry["baseline"]
        print(
            f"{entry['suite']}: {entry['status']} "
            f"candidate={candidate['correct']}/{candidate['n']} "
            f"baseline={baseline['correct']}/{baseline['n']} "
            f"delta={entry['delta_accuracy']:+.4f} "
            f"intervals_overlap={entry['intervals_overlap']}"
        )
        if entry["configuration_mismatch"]:
            print(f"    MISMATCH: {json.dumps(entry['configuration_mismatch'])}")

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
