#!/usr/bin/env python3
"""Contained four-arm W9 E2M1 fidelity campaign runner (implementation pending)."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run_w9_e2m1_v1.sh"


def run(_args: argparse.Namespace) -> int:
    raise RuntimeError("W9 E2M1 runner is not implemented")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
