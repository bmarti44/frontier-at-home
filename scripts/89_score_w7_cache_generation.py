#!/usr/bin/env python3
"""Score the W7 stable-model expert-cache generation invariant.

This is deliberately a log scorer, not an engine self-reported verdict.  A
single server process that has begun accepting requests must not announce a
model-generation change while serving a fixed model.  The later paired gate
still owns byte identity and performance acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LISTEN = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
CACHE_ENABLED = "CUDA persistent expert cache enabled:"
INDEXED_RESUME = "GLM sync branch=indexed_resume"
PROMPT_DONE = " prompt done "
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"


def score_text(text: str) -> dict[str, object]:
    lines = text.splitlines()
    listen = next((i for i, line in enumerate(lines) if LISTEN in line), None)
    shutdown = next(
        (i for i, line in enumerate(lines) if SHUTDOWN in line and (listen is None or i > listen)),
        len(lines),
    )
    window = lines[listen + 1 : shutdown] if listen is not None else []
    cache_enabled_count = sum(CACHE_ENABLED in line for line in window)
    indexed_positions = [i for i, line in enumerate(window) if INDEXED_RESUME in line]
    completed_indexed_resume = any(
        any(PROMPT_DONE in later for later in window[i + 1 :])
        for i in indexed_positions
    )
    false_flush_count = sum(FALSE_FLUSH in line for line in window)
    checks = {
        "listener_observed": listen is not None,
        "persistent_cache_enabled_once": cache_enabled_count == 1,
        "indexed_resume_observed": bool(indexed_positions),
        "indexed_resume_completed": completed_indexed_resume,
        "stable_model_generation_flushes_zero": false_flush_count == 0,
    }
    return {
        "formula": "PASS iff all checks are true; generation-change flushes are counted strictly after listener readiness and before shutdown",
        "observed": {
            "cache_enabled_count": cache_enabled_count,
            "indexed_resume_count": len(indexed_positions),
            "false_generation_flush_count": false_flush_count,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_text(args.server_log.read_text(encoding="utf-8", errors="strict"))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
