#!/usr/bin/env python3
"""Fail-closed controller and fixed acceptance formulas for the GLM-5.2 goal.

The controller deliberately does not infer PASS from prose or engine logs.  A
gate can advance only through a registered runner which writes a manifest,
raw.jsonl and summary.json under the immutable attempt directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / "results" / "glm52-goal"
STATUSES = frozenset({"PENDING", "RED_CONFIRMED", "PASS", "FAIL", "NO_RESULT"})
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "NO_RESULT"})
GATE_ORDER = (
    "foundation",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8",
    "W9",
    "W10",
    "W11",
    "switch",
    "parity",
    "review",
)

# One-sided 95% Student-t critical values, indexed by degrees of freedom.
_T95 = {
    1: 6.3138,
    2: 2.9200,
    3: 2.3534,
    4: 2.1318,
    5: 2.0150,
    6: 1.9432,
    7: 1.8946,
    8: 1.8595,
    9: 1.8331,
    10: 1.8125,
    11: 1.7959,
    12: 1.7823,
    13: 1.7709,
    14: 1.7613,
    15: 1.7531,
    16: 1.7459,
    17: 1.7396,
    18: 1.7341,
    19: 1.7291,
    20: 1.7247,
    21: 1.7207,
    22: 1.7171,
    23: 1.7139,
    24: 1.7109,
    25: 1.7081,
    26: 1.7056,
    27: 1.7033,
    28: 1.7011,
    29: 1.6991,
    30: 1.6973,
}


class GoalError(RuntimeError):
    """A fail-closed controller or evidence validation error."""


def _finite_positive(values: Iterable[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} is empty")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{label} contains non-positive or non-finite values")
    return result


def decode_tokens_per_second(token_timestamps: Iterable[float]) -> float:
    """Return (N-1)/(tN-t1), requiring at least 128 emitted tokens."""
    timestamps = [float(value) for value in token_timestamps]
    if len(timestamps) < 128:
        raise ValueError("decode requires at least 128 token timestamps")
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("token timestamps must be finite")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("token timestamps must be strictly increasing")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        raise ValueError("decode interval must be positive")
    return (len(timestamps) - 1) / elapsed


def _t95(df: int) -> float:
    if df < 1:
        raise ValueError("at least two paired samples are required")
    return _T95.get(min(df, 30), 1.6449)


def paired_ratio_bound(
    candidate: Iterable[float], reference: Iterable[float], *, side: str
) -> float:
    """One-sided 95% bound of the geometric mean paired ratio.

    Ratios are analyzed in log space. Fixtures must be paired in execution
    order; callers are responsible for ABBA/BAAB block validation.
    """
    left = _finite_positive(candidate, "candidate samples")
    right = _finite_positive(reference, "reference samples")
    if len(left) != len(right):
        raise ValueError("paired sample counts differ")
    if len(left) < 2:
        raise ValueError("at least two paired samples are required")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    logs = [math.log(a / b) for a, b in zip(left, right)]
    mean = statistics.fmean(logs)
    sem = statistics.stdev(logs) / math.sqrt(len(logs))
    delta = _t95(len(logs) - 1) * sem
    return math.exp(mean - delta if side == "lower" else mean + delta)


def performance_verdict(samples: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered matched-performance acceptance formula."""
    required = (
        "decode_glm",
        "decode_dsv4",
        "prefill_glm",
        "prefill_dsv4",
        "prefill_time_glm",
        "prefill_time_dsv4",
        "warm_ttft_glm",
        "warm_ttft_dsv4",
        "cold_ttft_glm",
        "cold_ttft_dsv4",
    )
    missing = [name for name in required if name not in samples]
    if missing:
        raise ValueError(f"missing performance samples: {', '.join(missing)}")
    metrics = {
        "decode_ratio_lower_95": paired_ratio_bound(
            samples["decode_glm"], samples["decode_dsv4"], side="lower"
        ),
        "prefill_ratio_lower_95": paired_ratio_bound(
            samples["prefill_glm"], samples["prefill_dsv4"], side="lower"
        ),
        "prefill_time_ratio_upper_95": paired_ratio_bound(
            samples["prefill_time_glm"],
            samples["prefill_time_dsv4"],
            side="upper",
        ),
        "warm_ttft_ratio_upper_95": paired_ratio_bound(
            samples["warm_ttft_glm"], samples["warm_ttft_dsv4"], side="upper"
        ),
        "cold_ttft_ratio_upper_95": paired_ratio_bound(
            samples["cold_ttft_glm"], samples["cold_ttft_dsv4"], side="upper"
        ),
    }
    checks = {
        "decode": metrics["decode_ratio_lower_95"] >= 0.80,
        "prefill_rate": metrics["prefill_ratio_lower_95"] >= 0.80,
        "prefill_time": metrics["prefill_time_ratio_upper_95"] <= 1.25,
        "warm_ttft": metrics["warm_ttft_ratio_upper_95"] <= 1.20,
        "cold_ttft": metrics["cold_ttft_ratio_upper_95"] <= 1.20,
    }
    return {
        "formula_version": 1,
        "metrics": metrics,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def context_verdict(observation: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed 1M context, retrieval and resource-safety formula."""
    required = {
        "context_cap",
        "processed_tokens",
        "retrieval_pass",
        "negative_control_pass",
        "completed_generation",
        "truncated",
        "oom",
        "xid",
        "available_memory_gib",
    }
    missing = sorted(required - observation.keys())
    if missing:
        raise ValueError(f"missing context fields: {', '.join(missing)}")
    memory = float(observation["available_memory_gib"])
    if not math.isfinite(memory):
        raise ValueError("available memory is non-finite")
    checks = {
        "context_cap": int(observation["context_cap"]) == 1_048_576,
        "processed_tokens": int(observation["processed_tokens"]) >= 1_000_000,
        "retrieval": observation["retrieval_pass"] is True,
        "negative_control": observation["negative_control_pass"] is True,
        "completed_generation": observation["completed_generation"] is True,
        "no_truncation": observation["truncated"] is False,
        "no_oom": observation["oom"] is False,
        "no_xid": observation["xid"] is False,
        "memory_floor": memory >= 10.0,
    }
    return {
        "formula_version": 1,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_raw_record(record: dict[str, Any]) -> None:
    """Reject malformed, failed, short, or unidentifiable measurement arms."""
    if record.get("arm") not in {"A", "B"}:
        raise ValueError("arm must be A or B")
    for field in ("fixture_sha256", "binary_sha256"):
        if not _is_sha256(record.get(field)):
            raise ValueError(f"{field} is not a lowercase SHA-256")
    decode_tokens_per_second(record.get("token_timestamps", ()))
    evaluated = record.get("evaluated_tokens")
    if not isinstance(evaluated, int) or isinstance(evaluated, bool) or evaluated <= 0:
        raise ValueError("evaluated_tokens must be a positive integer")
    seconds = float(record.get("prefill_seconds", math.nan))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("prefill_seconds must be finite and positive")
    if record.get("failures") != []:
        raise ValueError("measurement record contains failures")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "gates": {
            gate: {"status": "PENDING", "attempts": [], "reason": None}
            for gate in GATE_ORDER
        },
    }


def _validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != 1:
        raise GoalError("unsupported state schema")
    gates = state.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATE_ORDER):
        raise GoalError("state gate set is incomplete or unknown")
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") not in STATUSES:
            raise GoalError(f"{name}: invalid status")
        if not isinstance(gate.get("attempts"), list):
            raise GoalError(f"{name}: attempts is not a list")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _load_state(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "state.json"
    if not path.exists():
        state = _initial_state()
        _atomic_json(path, state)
        return state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoalError(f"cannot read state: {exc}") from exc
    _validate_state(state)
    return state


def _selected_gate(state: dict[str, Any]) -> str | None:
    for name in GATE_ORDER:
        if state["gates"][name]["status"] not in TERMINAL_STATUSES:
            return name
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dispatch(state_dir: Path, command: str) -> dict[str, Any]:
    state = _load_state(state_dir)
    selected = _selected_gate(state)
    event = {
        "command": command,
        "selected_gate": selected,
        "time": _utcnow(),
        "action": "awaiting_registered_runner" if selected else "complete",
    }
    events = state_dir / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("resume")
    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = _load_state(args.state_dir)
        if args.command == "status":
            if args.json:
                print(json.dumps(state, sort_keys=True, allow_nan=False))
            else:
                selected = _selected_gate(state)
                print(f"next={selected or 'none'}")
                for name in GATE_ORDER:
                    print(f"{name}: {state['gates'][name]['status']}")
            return 0
        print(
            json.dumps(
                _dispatch(args.state_dir, args.command),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (GoalError, ValueError, OSError) as exc:
        print(f"glm52_goal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
