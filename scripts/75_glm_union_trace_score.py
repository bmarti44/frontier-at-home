#!/usr/bin/env python3
"""Validate one immutable GLM union-trace attempt."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def score_trace(directory: Path, server_log: Path, *, max_bytes: int) -> dict[str, Any]:
    """Return a fail-closed result for one trace attempt (implementation pending)."""
    return {"verdict": "FAIL", "checks": {"implemented": False}}

