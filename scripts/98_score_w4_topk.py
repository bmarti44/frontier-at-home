#!/usr/bin/env python3
"""Fixed scorer placeholder for the W4 candidate-2 evidence RED."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class ScoreError(ValueError):
    pass


def score_run(run_dir: Path) -> dict:
    del run_dir
    return {"schema": "glm52-w4-topk-summary-v1", "verdict": "NO_RESULT"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = score_run(args.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
