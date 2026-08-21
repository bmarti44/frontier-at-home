#!/usr/bin/env python3
"""Wait for a stable MemAvailable floor before loading a large model."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


KIB_PER_GIB = 2**20


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def read_available_gib(path: Path) -> float:
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields[:1] == ["MemAvailable:"] and len(fields) == 3 and fields[2] == "kB":
            return int(fields[1]) / KIB_PER_GIB
    raise ValueError(f"invalid or missing MemAvailable in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-gib", type=positive_float, required=True)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=nonnegative_float, default=1)
    parser.add_argument("--timeout-seconds", type=nonnegative_float, default=180)
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    args = parser.parse_args()
    if args.stable_samples < 1:
        parser.error("--stable-samples must be at least 1")

    deadline = time.monotonic() + args.timeout_seconds
    stable = 0
    observed = 0.0
    while True:
        observed = read_available_gib(args.meminfo)
        stable = stable + 1 if observed >= args.required_gib else 0
        if stable >= args.stable_samples:
            print(
                json.dumps(
                    {
                        "pass": True,
                        "required_gib": args.required_gib,
                        "mem_available_gib": round(observed, 3),
                        "stable_samples_observed": stable,
                    },
                    separators=(",", ":"),
                )
            )
            return 0
        # Once the floor is met, finish the requested consecutive reads even
        # when the caller selected a zero-second fail-fast timeout.
        if time.monotonic() >= deadline and stable == 0:
            print(
                json.dumps(
                    {
                        "pass": False,
                        "required_gib": args.required_gib,
                        "mem_available_gib": round(observed, 3),
                        "stable_samples_observed": stable,
                    },
                    separators=(",", ":"),
                )
            )
            return 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
