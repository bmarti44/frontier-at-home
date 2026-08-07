#!/usr/bin/env python3
"""Fixed W7 restored-frontier equivalence scorer (RED placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path


def score(strict: Path, candidate: Path, cold: Path) -> dict:
    raise RuntimeError("RED: W7 equivalence scoring is not implemented")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.strict, args.candidate, args.cold)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
