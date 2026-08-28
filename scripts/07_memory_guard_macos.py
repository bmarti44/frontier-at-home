#!/usr/bin/env python3
"""macOS release gate: wait for a stable available-memory floor.

Port of scripts/03_memory_guard.py (which is byte-frozen by the GLM
lossless-plateau freeze receipt and reads Linux procfs). Availability here
is (free + inactive + purgeable) pages from vm_stat; the compressor makes
this fuzzier than MemAvailable, so callers should require conservative
floors. Same flags and JSON verdict as the Linux guard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time


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


def read_available_gib() -> float:
    page_size = int(
        subprocess.run(
            ["sysctl", "-n", "vm.pagesize"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    )
    output = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=True
    ).stdout
    pages = 0
    matched = False
    for name in ("Pages free", "Pages inactive", "Pages purgeable"):
        match = re.search(rf"{re.escape(name)}:\s+(\d+)", output)
        if match:
            pages += int(match.group(1))
            matched = True
    if not matched:
        raise ValueError("vm_stat output has no recognizable page counters")
    return pages * page_size / 2**30


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-gib", type=positive_float, required=True)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=nonnegative_float, default=1)
    parser.add_argument("--timeout-seconds", type=nonnegative_float, default=180)
    args = parser.parse_args()
    if args.stable_samples < 1:
        parser.error("--stable-samples must be at least 1")

    deadline = time.monotonic() + args.timeout_seconds
    stable = 0
    observed = 0.0
    while True:
        observed = read_available_gib()
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
