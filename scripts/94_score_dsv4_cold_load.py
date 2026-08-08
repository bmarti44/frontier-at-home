#!/usr/bin/env python3
"""RED placeholder for the fixed DSV4 cold-load campaign scorer."""

from __future__ import annotations


def score_campaign(manifest: object, rows: object) -> dict[str, object]:
    del manifest, rows
    return {"verdict": "FAIL", "failure": "RED placeholder"}
