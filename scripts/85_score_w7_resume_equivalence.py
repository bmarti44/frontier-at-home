#!/usr/bin/env python3
"""Fixed W7 restored-frontier equivalence scorer."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import sys


N_VOCAB = 154880
LOGIT_BYTES = N_VOCAB * 4
PRIMARY_SHA256 = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
LIVE_SHA256 = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"

EXPECTED_DUMPS = {
    "strict": {
        "logits.sync1.start0.prompt5055.suffix5055",
        "logits.sync2.start0.prompt5066.suffix5066",
    },
    "candidate": {
        "logits.sync1.start0.prompt5055.suffix5055",
        "logits.sync2.start5044.prompt5066.suffix22",
    },
    "cold": {"logits.sync1.start0.prompt5066.suffix5066"},
}


def _pairs(items: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json(path: Path) -> dict:
    value = json.loads(
        path.read_bytes(), object_pairs_hook=_pairs, parse_constant=_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _logits(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) != LOGIT_BYTES:
        raise ValueError(f"{path}: expected {LOGIT_BYTES} bytes, got {len(raw)}")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([0-9a-f]{40}\.kv)", line)
        if match is None or match.group(2) in result:
            raise ValueError(f"{path}: malformed or duplicate inventory row")
        result[match.group(2)] = match.group(1)
    if not result:
        raise ValueError(f"{path}: empty inventory")
    return result


def _trace_pass(path: Path) -> bool:
    try:
        value = _json(path)
        checks = value.get("checks")
        return (
            value.get("verdict") == "PASS"
            and isinstance(checks, dict)
            and bool(checks)
            and all(item is True for item in checks.values())
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _selected_kv_unchanged(arm: Path, log: str) -> bool:
    try:
        matches = re.findall(
            r"kv cache hit text tokens=5044[^\n]* file=([^\s]+\.kv)", log
        )
        if len(matches) != 1:
            return False
        selected = Path(matches[0])
        if selected.parent != arm / "kv":
            return False
        before = _inventory(arm / "kv-before.sha256")
        after = _inventory(arm / "kv-after.sha256")
        return (
            selected.name in before
            and selected.name in after
            and before[selected.name] == after[selected.name]
        )
    except (OSError, ValueError):
        return False


def _response_ok(arm: Path, role: str, expected_tokens: int) -> bool:
    try:
        response = _json(arm / f"{role}-response.json")
        return (
            (arm / f"{role}-http-status").read_text().strip() == "200"
            and response.get("usage", {}).get("prompt_tokens") == expected_tokens
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def score(strict: Path, candidate: Path, cold: Path) -> dict:
    arms = {"strict": strict, "candidate": candidate, "cold": cold}
    checks: dict[str, bool] = {}
    logs: dict[str, str] = {}
    metadata: dict[str, dict] = {}
    for name, arm in arms.items():
        try:
            metadata[name] = _json(arm / "arm.json")
            logs[name] = (arm / "server.log").read_text(
                encoding="utf-8", errors="strict"
            )
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
            metadata[name] = {}
            logs[name] = ""

    checks["arm_identity_exact"] = all(
        metadata[name].get("schema_version") == 1
        and metadata[name].get("arm") == name
        and metadata[name].get("containment_rc") == 0
        for name in arms
    )
    expected_requests = {
        "strict": {"live": LIVE_SHA256, "primary": PRIMARY_SHA256},
        "candidate": {"live": LIVE_SHA256, "primary": PRIMARY_SHA256},
        "cold": {"primary": PRIMARY_SHA256},
    }
    checks["request_hashes_exact"] = all(
        metadata[name].get("request_sha256") == expected_requests[name]
        for name in arms
    )
    checks["http_and_prompt_tokens_exact"] = (
        _response_ok(strict, "live", 5055)
        and _response_ok(strict, "primary", 5066)
        and _response_ok(candidate, "live", 5055)
        and _response_ok(candidate, "primary", 5066)
        and _response_ok(cold, "primary", 5066)
    )
    checks["traces_pass"] = _trace_pass(strict / "trace-result.json") and _trace_pass(
        candidate / "trace-result.json"
    )
    checks["dump_sets_exact"] = all(
        {path.name for path in arm.glob("logits*")} == EXPECTED_DUMPS[name]
        for name, arm in arms.items()
    )
    checks["strict_guard_observed"] = (
        "GLM resume guard: prompt (5066) extends/diverges past evaluated frontier 5055 (checkpoint 5044)"
        in logs["strict"]
        and "GLM sync start=0 prompt=5066 suffix=5066" in logs["strict"]
        and "restored-frontier diagnostic: authoritative" not in logs["strict"]
    )
    checks["candidate_frontier_observed"] = (
        "GLM restored-frontier diagnostic: authoritative checkpoint=5044 compact_rows=5044"
        in logs["candidate"]
        and "GLM sync start=5044 prompt=5066 suffix=22" in logs["candidate"]
        and "GLM resume guard:" not in logs["candidate"]
    )
    checks["cold_control_observed"] = (
        "GLM sync start=0 prompt=5066 suffix=5066" in logs["cold"]
        and "GLM resume guard:" not in logs["cold"]
        and "restored-frontier diagnostic:" not in logs["cold"]
    )
    checks["selected_kv_unchanged"] = _selected_kv_unchanged(
        strict, logs["strict"]
    ) and _selected_kv_unchanged(candidate, logs["candidate"])

    observed = {
        "candidate_argmax": None,
        "cold_argmax": None,
        "strict_argmax": None,
        "max_abs_logit_delta": None,
        "strict_vs_cold_max_abs_logit_delta": None,
    }
    try:
        candidate_logits = _logits(
            candidate / "logits.sync2.start5044.prompt5066.suffix22"
        )
        cold_logits = _logits(cold / "logits.sync1.start0.prompt5066.suffix5066")
        strict_logits = _logits(strict / "logits.sync2.start0.prompt5066.suffix5066")
        checks["candidate_logits_finite"] = all(map(math.isfinite, candidate_logits))
        checks["cold_logits_finite"] = all(map(math.isfinite, cold_logits))
        checks["strict_logits_finite"] = all(map(math.isfinite, strict_logits))
        if all(
            checks[key]
            for key in (
                "candidate_logits_finite",
                "cold_logits_finite",
                "strict_logits_finite",
            )
        ):
            observed["candidate_argmax"] = max(
                range(N_VOCAB), key=candidate_logits.__getitem__
            )
            observed["cold_argmax"] = max(range(N_VOCAB), key=cold_logits.__getitem__)
            observed["strict_argmax"] = max(
                range(N_VOCAB), key=strict_logits.__getitem__
            )
            observed["max_abs_logit_delta"] = max(
                abs(left - right)
                for left, right in zip(candidate_logits, cold_logits, strict=True)
            )
            observed["strict_vs_cold_max_abs_logit_delta"] = max(
                abs(left - right)
                for left, right in zip(strict_logits, cold_logits, strict=True)
            )
        checks["candidate_argmax_matches_cold"] = (
            observed["candidate_argmax"] is not None
            and observed["candidate_argmax"] == observed["cold_argmax"]
        )
        checks["candidate_max_abs_delta_lt_1e_2"] = (
            observed["max_abs_logit_delta"] is not None
            and observed["max_abs_logit_delta"] < 0.01
        )
        checks["strict_control_matches_cold"] = (
            observed["strict_argmax"] is not None
            and observed["strict_argmax"] == observed["cold_argmax"]
            and observed["strict_vs_cold_max_abs_logit_delta"] == 0.0
        )
    except (OSError, ValueError):
        checks["candidate_logits_finite"] = False
        checks["cold_logits_finite"] = False
        checks["strict_logits_finite"] = False
        checks["candidate_argmax_matches_cold"] = False
        checks["candidate_max_abs_delta_lt_1e_2"] = False
        checks["strict_control_matches_cold"] = False

    verdict = "PASS" if checks and all(checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "gate": "W7-resume-bpe-lineage-v1",
        "formula": {
            "max_abs_logit_delta": "max_i(abs(candidate_i-cold_i))",
            "threshold": "strictly less than 0.01",
            "argmax": "first maximum index must be identical",
        },
        "checks": checks,
        "observed": observed,
        "artifact_sha256": {
            name: hashlib.sha256((arm / "server.log").read_bytes()).hexdigest()
            if (arm / "server.log").is_file()
            else None
            for name, arm in arms.items()
        },
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.strict, args.candidate, args.cold)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
