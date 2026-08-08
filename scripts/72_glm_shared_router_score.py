#!/usr/bin/env python3
"""Score matched baseline/shared-correction router predictions from engine logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINE = re.compile(
    r"^PREDPAIR E(?P<event>[0-9]+) P(?P<position>[0-9]+)"
    r" L(?P<layer>[0-9]+) actual:(?P<actual>(?: [0-9]+){8})"
    r" base:(?P<base>(?: [0-9]+){8})"
    r" shared:(?P<shared>(?: [0-9]+){8})$"
)
MIN_SAMPLES = 1036
MIN_RECALL_GAIN = 0.02
MIN_POSITIONS = 14
EXPECTED_LAYERS = set(range(4, 78))


def parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item) for item in value.split())
    if len(ids) != 8 or len(set(ids)) != 8 or any(item < 0 or item >= 256 for item in ids):
        raise ValueError("prediction sets must contain eight unique expert IDs in 0..255")
    return ids


def score(path: Path) -> dict[str, object]:
    samples = baseline_hits = shared_hits = 0
    malformed = 0
    events: list[int] = []
    event_keys: set[tuple[int, int, int]] = set()
    positions: set[int] = set()
    layers: set[int] = set()
    ordered_position_layers: list[tuple[int, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw.startswith("PREDPAIR "):
            continue
        match = LINE.fullmatch(raw)
        if match is None:
            malformed += 1
            continue
        event = int(match.group("event"))
        position = int(match.group("position"))
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
        events.append(event)
        event_keys.add((event, position, layer))
        positions.add(position)
        layers.add(layer)
        ordered_position_layers.append((position, layer))
        samples += 1

    denominator = samples * 8
    baseline_recall = baseline_hits / denominator if denominator else 0.0
    shared_recall = shared_hits / denominator if denominator else 0.0
    gain = shared_recall - baseline_recall
    expected_events = list(range(1, samples + 1))
    ordered_positions = sorted(positions)
    contiguous_positions = bool(ordered_positions) and ordered_positions == list(
        range(ordered_positions[0], ordered_positions[-1] + 1)
    )
    expected_sweeps = [
        (position, layer)
        for position in ordered_positions
        for layer in sorted(EXPECTED_LAYERS)
    ]
    complete_sweeps = contiguous_positions and ordered_position_layers == expected_sweeps
    checks = {
        "minimum_samples": samples >= MIN_SAMPLES,
        "no_malformed_rows": malformed == 0,
        "unique_event_keys": len(event_keys) == samples and events == expected_events,
        "position_coverage": len(positions) >= MIN_POSITIONS,
        "layer_coverage": layers == EXPECTED_LAYERS,
        "complete_position_sweeps": complete_sweeps,
        "shared_recall_gain": gain >= MIN_RECALL_GAIN,
    }
    return {
        "schema_version": 1,
        "formula": "recall=sum(|predicted_top8 intersect actual_top8|)/(8*samples)",
        "acceptance": {
            "minimum_samples": MIN_SAMPLES,
            "minimum_unique_positions": MIN_POSITIONS,
            "expected_layers": sorted(EXPECTED_LAYERS),
            "minimum_absolute_recall_gain": MIN_RECALL_GAIN,
        },
        "samples": samples,
        "malformed_rows": malformed,
        "unique_event_keys": len(event_keys),
        "unique_positions": len(positions),
        "observed_layers": sorted(layers),
        "complete_position_sweeps": complete_sweeps,
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
