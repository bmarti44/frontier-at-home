#!/usr/bin/env python3
"""Frozen entry point for the W9 real-tensor FP4 offline falsifier."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--randomness-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise SystemExit("W9 FP4 offline falsifier is not implemented")


if __name__ == "__main__":
    main()
