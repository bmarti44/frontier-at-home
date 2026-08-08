#!/usr/bin/env python3
"""Fail-closed scaffold for the preregistered W7.1 matched campaign."""

from __future__ import annotations


def score_campaign_rows(rows: object, schedules: object) -> dict[str, object]:
    del rows, schedules
    return {
        "formula": (
            "five fresh-server ABBA/BAAB blocks; byte-identical outputs/logits; "
            "one-sided 95% TTFT ratio upper bound <=0.95; decode ratio lower bound >=1.00"
        ),
        "checks": {"implementation_present": False},
        "observed": {},
        "verdict": "FAIL",
    }
