#!/usr/bin/env python3
"""Fail-closed scaffold for the contained W7.1 matched campaign runner."""

from __future__ import annotations

import argparse


def derive_schedules(seed_sha256: str) -> list[str]:
    del seed_sha256
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--seed-sha256")
    parser.parse_args()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
