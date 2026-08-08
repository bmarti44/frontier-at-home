#!/usr/bin/env python3
"""Fixed scorer shell for the preregistered W7.2 bounded probe."""

from __future__ import annotations


FORMULA = (
    "two fresh-server equal-fixture OFF/ON arms; exact output-token, UTF-8, and "
    "three-logit equality; warm append TTFT_OFF-TTFT_ON >=0.5s; "
    "decode_ON/decode_OFF >=0.99 over >=128 output timestamps"
)


def score_probe_rows(rows: object, order: object) -> dict[str, object]:
    """RED shell: implementation follows the committed behavioral test."""
    return {
        "formula": FORMULA,
        "checks": {"implementation_present": False},
        "observed": {},
        "failure": "scorer implementation not present",
        "verdict": "FAIL",
    }
