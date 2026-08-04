#!/usr/bin/env python3
"""Score matched baseline/shared-correction router predictions from engine logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINE = re.compile(
    r"^PREDPAIR L(?P<layer>[0-9]+) actual:(?P<actual>(?: [0-9]+){8})"
    r" base:(?P<base>(?: [0-9]+){8})"
    r" shared:(?P<shared>(?: [0-9]+){8})$"
)
MIN_SAMPLES = 1000
MIN_RECALL_GAIN = 0.02


def parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item) for item in value.split())
    if len(ids) != 8 or len(set(ids)) != 8 or any(item < 0 or item >= 256 for item in ids):
        raise ValueError("prediction sets must contain eight unique expert IDs in 0..255")
    return ids


def score(path: Path) -> dict[str, object]:
    samples = baseline_hits = shared_hits = 0
    malformed = 0
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw.startswith("PREDPAIR "):
            continue
        match = LINE.fullmatch(raw)
        if match is None:
            malformed += 1
            continue
        layer = int(match.group("layer"))
        if layer < 4 or layer > 77:
            malformed += 1
            continue
        try:
            actual = set(parse_ids(match.group("actual")))
            baseline = set(parse_ids(match.group("base")))
            shared = set(parse_ids(match.group("shared")))
        except ValueError:
            malformed += 1
            continue
        baseline_hits += len(actual & baseline)
        shared_hits += len(actual & shared)
        samples += 1

    denominator = samples * 8
    baseline_recall = baseline_hits / denominator if denominator else 0.0
    shared_recall = shared_hits / denominator if denominator else 0.0
    gain = shared_recall - baseline_recall
    checks = {
        "minimum_samples": samples >= MIN_SAMPLES,
        "no_malformed_rows": malformed == 0,
        "shared_recall_gain": gain >= MIN_RECALL_GAIN,
    }
    return {
        "schema_version": 1,
        "formula": "recall=sum(|predicted_top8 intersect actual_top8|)/(8*samples)",
        "acceptance": {
            "minimum_samples": MIN_SAMPLES,
            "minimum_absolute_recall_gain": MIN_RECALL_GAIN,
        },
        "samples": samples,
        "malformed_rows": malformed,
        "baseline_recall": baseline_recall,
        "shared_recall": shared_recall,
        "absolute_recall_gain": gain,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = score(args.trace)
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
