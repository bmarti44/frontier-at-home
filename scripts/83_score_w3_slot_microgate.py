#!/usr/bin/env python3
"""Score W3 direct-slot raw samples independently of the CUDA candidate."""

import argparse
import json
import math
import statistics
from pathlib import Path


MUTATIONS = {
    "noop_direct",
    "skip_gate",
    "skip_quant",
    "skip_down",
    "skip_sum",
    "route_slot",
}


def fail(message: str) -> None:
    raise ValueError(message)


def score(path: Path) -> dict:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"line {line_number}: malformed JSON: {exc}")
        if not isinstance(row, dict):
            fail(f"line {line_number}: row is not an object")
        rows.append(row)
    confirmations = [row for row in rows if row.get("kind") == "confirmation"]
    mutations = [row for row in rows if row.get("kind") == "mutation"]
    if len(rows) != 9 or len(confirmations) != 3 or len(mutations) != 6:
        fail("expected exactly three confirmations and six mutations")
    ratios = []
    seen_runs = set()
    for row in confirmations:
        run = row.get("run")
        if run not in {1, 2, 3} or run in seen_runs:
            fail("confirmation run ids must be exactly 1, 2, 3")
        seen_runs.add(run)
        compact = row.get("compact_samples_ms")
        direct = row.get("direct_samples_ms")
        if not isinstance(compact, list) or not isinstance(direct, list):
            fail("confirmation is missing raw timing arrays")
        if len(compact) != 6 or len(direct) != 6:
            fail("each arm must contain exactly six samples")
        values = compact + direct
        if any(not isinstance(value, (int, float)) or
               not math.isfinite(value) or value <= 0 for value in values):
            fail("timing samples must be finite and positive")
        ratio = statistics.median(direct) / statistics.median(compact)
        ratios.append(ratio)
        if row.get("exit_code") != 0 or row.get("output_mismatches") != 0:
            fail("confirmation did not exit cleanly with identical output")
        if row.get("finite_nonzero_reference_values", 0) < 18_432:
            fail("confirmation reference output is trivial or non-finite")
        if row.get("expert_evaluations_per_arm", 0) < 600:
            fail("confirmation is too short")
        if row.get("samples") != 6 or row.get("cuda_event_synchronized") is not True:
            fail("confirmation timing contract is incomplete")
        if ratio > 0.95:
            fail("direct arm misses the preregistered headroom threshold")
    seen_mutations = set()
    for row in mutations:
        name = row.get("name")
        if name not in MUTATIONS or name in seen_mutations:
            fail("mutation names must match the exact fixed set")
        seen_mutations.add(name)
        if row.get("exit_code") == 0 or row.get("output_mismatches", 0) <= 0:
            fail(f"mutation {name} did not fail closed")
    if seen_mutations != MUTATIONS:
        fail("mutation set is incomplete")
    return {
        "schema_version": 1,
        "status": "PASS",
        "confirmation_runs": 3,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "mutations_rejected": 6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    try:
        result = score(args.raw)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL",
                          "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
