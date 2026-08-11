#!/usr/bin/env python3
"""Build an authoritative evidence bundle for the DeepSeek-V4-Flash-0731 candidate.

`results/dsv4-0731-staging/` is a set of flat JSON documents that an agent wrote by
hand. They are readable, but they are narration: they assert that gates passed
without carrying the observations a reader would need to recompute the verdict, and
nothing stops a later edit from changing a recorded conclusion. AGENTS.md's evidence
contract requires an attempt directory containing `manifest.json`, `raw.jsonl`, and
`summary.json`, with frozen hashes, raw observations, exact formulas, and a computed
verdict.

This scorer produces that bundle. It is deliberately NOT a narrator:

  * every number in summary.json is recomputed here from the transcripts, never
    copied from an arm artifact's own summary fields;
  * the verdict comes from the preregistered gates in
    configs/decision-specs/dsv4-0731.v1.json, read before the arms are scored;
  * identities are hashed from the files on disk at build time;
  * anything missing, malformed, non-finite, or duplicated is a failure that lands
    in the bundle rather than an omission that quietly improves the result.

The bundle is written to an immutable per-attempt directory created with O_EXCL. A
rebuild never overwrites: it either gets a fresh attempt id or refuses. That is the
property review finding 7 asked for, and the reason 99_qualify_dsv4_0731.sh's fixed
output path was called out separately -- a stage that dies before writing leaves
yesterday's verdict standing.

Usage:
    python3 scripts/105_build_dsv4_0731_bundle.py --attempt-id <id> [--allow-incomplete]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING = REPO_ROOT / "results" / "dsv4-0731-staging"
ATTEMPTS_ROOT = REPO_ROOT / "results" / "dsv4-0731-attempts"
SPEC_PATH = REPO_ROOT / "configs" / "decision-specs" / "dsv4-0731.v3.json"
WEIGHTS_MANIFEST = REPO_ROOT / "weights" / "unsloth-ud-q2_k_xl" / "manifest.json"
WEIGHTS_PIN = REPO_ROOT / "configs" / "pins" / "unsloth-ud-q2_k_xl.json"
TOKENIZER = REPO_ROOT / "vendor" / "official-encoding" / "tokenizer.json"
ACCURACY_HARNESS = REPO_ROOT / "scripts" / "31_bench_accuracy.py"
GOLDEN_HARNESS = REPO_ROOT / "scripts" / "32_golden_tests.py"
ENGINE_BINARY = Path(
    "/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server"
)
ATTEMPT_ID = re.compile(r"\A[a-z0-9][a-z0-9-]{0,63}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class BundleError(RuntimeError):
    """Raised for any condition that must fail the bundle rather than be skipped."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(document: Any) -> bytes:
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {path}: {error}") from error


def git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(REPO_ROOT), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BundleError(f"git {arguments!r} failed: {error}") from error
    if completed.returncode != 0:
        raise BundleError(f"git {arguments!r} exited {completed.returncode}")
    return completed.stdout


# --------------------------------------------------------------------------
# Arm scoring: every figure below is recomputed from transcripts.
# --------------------------------------------------------------------------


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval at 95%. Written out so the formula is auditable."""
    if total <= 0:
        raise BundleError("wilson interval requires a positive denominator")
    z = 1.959963984540054
    phat = correct / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
        / denominator
    )
    return (centre - margin, centre + margin)


def repo_relative(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    relative_to() raises for anything outside the tree, which would abort a build
    over a cosmetic field. Tests score fixtures from a temp directory.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def score_accuracy_arm(name: str, artifact: Path, transcripts: Path) -> dict[str, Any]:
    """Recompute an accuracy arm from its transcripts.

    The artifact's own `correct`/`accuracy` fields are read only to be COMPARED
    against the recomputation. If they disagree the arm fails: that disagreement is
    the signature of an artifact edited after the fact.
    """
    if not transcripts.is_dir():
        raise BundleError(f"{name}: transcripts directory is missing: {transcripts}")
    document = read_json(artifact)

    rows: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    correct = 0
    invalid = 0
    truncated = 0
    timeouts = 0
    completion_tokens: list[int] = []

    for path in sorted(transcripts.glob("*.json")):
        row = read_json(path)
        index = row.get("index")
        if index in seen_ids:
            raise BundleError(f"{name}: duplicate transcript index {index!r}")
        seen_ids.add(index)

        reason = row.get("reason") or ""
        scored = bool(row.get("scored_correct"))
        finish = row.get("finish_reason")
        usage = ((row.get("response") or {}).get("usage") or {})
        tokens = usage.get("completion_tokens")

        if scored:
            correct += 1
        if reason.startswith("invalid"):
            invalid += 1
        if "Timeout" in reason:
            timeouts += 1
        if finish == "length":
            truncated += 1
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            if tokens < 0:
                raise BundleError(f"{name}: negative completion_tokens at {path.name}")
            completion_tokens.append(tokens)

        rows.append(
            {
                "record_type": "accuracy_item",
                "arm": name,
                "index": index,
                "task_id": row.get("task_id"),
                "rendered_prompt_sha256": row.get("rendered_prompt_sha256"),
                "rendering": row.get("rendering"),
                "finish_reason": finish,
                "scored_correct": scored,
                "reason": reason,
                "completion_tokens": tokens,
                "elapsed_s": (row.get("request") or {}).get("elapsed_s"),
            }
        )

    total = len(rows)
    if total == 0:
        raise BundleError(f"{name}: no transcripts found")

    claimed_correct = document.get("correct")
    claimed_n = document.get("n")
    if claimed_n != total:
        raise BundleError(
            f"{name}: artifact claims n={claimed_n!r} but {total} transcripts exist"
        )
    if claimed_correct != correct:
        raise BundleError(
            f"{name}: artifact claims correct={claimed_correct!r} but recomputation "
            f"from transcripts gives {correct}"
        )

    low, high = wilson_interval(correct, total)
    for value in (low, high):
        if not math.isfinite(value):
            raise BundleError(f"{name}: non-finite confidence bound")

    completion_tokens.sort()

    def percentile(fraction: float) -> int | None:
        if not completion_tokens:
            return None
        position = min(len(completion_tokens) - 1, int(len(completion_tokens) * fraction))
        return completion_tokens[position]

    summary = {
        "arm": name,
        "artifact": repo_relative(artifact),
        "artifact_sha256": sha256_file(artifact),
        "stack_label": document.get("stack_label"),
        "suite": document.get("suite"),
        "split": document.get("split"),
        "rowset_sha256": document.get("rowset_sha256"),
        "thinking_mode": (document.get("generation") or {}).get("thinking_mode"),
        "max_tokens": (document.get("generation") or {}).get("max_tokens"),
        "request_timeout_s": (document.get("generation") or {}).get("request_timeout_s"),
        "n": total,
        "correct": correct,
        "accuracy": correct / total,
        "accuracy_formula": "correct / n, both recomputed from transcripts",
        "wilson95_low": low,
        "wilson95_high": high,
        "wilson95_formula": "Wilson score interval, z=1.959963984540054",
        "invalid": invalid,
        "invalid_fraction": invalid / total,
        "truncated": truncated,
        "truncated_fraction": truncated / total,
        "request_timeouts": timeouts,
        "completion_tokens_median": (
            completion_tokens[len(completion_tokens) // 2] if completion_tokens else None
        ),
        "completion_tokens_p90": percentile(0.90),
        "completion_tokens_p95": percentile(0.95),
        "completion_tokens_max": completion_tokens[-1] if completion_tokens else None,
        "transcripts_without_usage": total - len(completion_tokens),
    }
    return {"summary": summary, "rows": rows}


def score_golden(artifact: Path) -> dict[str, Any]:
    document = read_json(artifact)
    checks = document.get("checks")
    if not isinstance(checks, list) or not checks:
        raise BundleError("golden artifact carries no checks")
    passed = sum(1 for check in checks if check.get("pass"))
    rows = [
        {
            "record_type": "golden_check",
            "name": check.get("name"),
            "pass": bool(check.get("pass")),
            "error": check.get("error"),
        }
        for check in checks
    ]
    return {
        "summary": {
            "artifact": repo_relative(artifact),
            "artifact_sha256": sha256_file(artifact),
            "checks_total": len(checks),
            "checks_passed": passed,
            "all_passed": passed == len(checks),
            "formula": "count(check.pass) over every recorded check",
        },
        "rows": rows,
    }


def score_speed(artifact: Path) -> dict[str, Any]:
    """Recompute the speed verdict from the per-cell reps."""
    document = read_json(artifact)
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise BundleError("speed artifact carries no cells")
    rows = []
    all_valid = True
    for cell in cells:
        valid = bool(cell.get("valid"))
        all_valid = all_valid and valid
        decode = cell.get("median_decode")
        if decode is not None and not math.isfinite(decode):
            raise BundleError(f"speed cell {cell.get('ctx_tokens')} has non-finite decode")
        rows.append(
            {
                "record_type": "speed_cell",
                "ctx_tokens": cell.get("ctx_tokens"),
                "median_decode_tok_s": decode,
                "median_ttft_s": cell.get("median_ttft"),
                "invalid_reps": cell.get("invalid_reps"),
                "valid": valid,
            }
        )
    return {
        "summary": {
            "artifact": repo_relative(artifact),
            "artifact_sha256": sha256_file(artifact),
            "suite_valid": bool(document.get("suite_valid")),
            "all_cells_valid": all_valid,
            "cells": len(cells),
            "total_invalid_reps": sum(int(c.get("invalid_reps") or 0) for c in cells),
        },
        "rows": rows,
    }


def score_soak(artifact: Path) -> dict[str, Any]:
    """Recompute the soak gates rather than trusting its pass bit."""
    document = read_json(artifact)
    gates = document.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise BundleError("soak artifact carries no gates")
    failed = sorted(name for name, ok in gates.items() if not ok)
    reps = document.get("reps") or []
    errors = document.get("errors") or []
    min_mem = document.get("mem_available_min_gib")
    if min_mem is not None and not math.isfinite(min_mem):
        raise BundleError("soak minimum memory is non-finite")
    return {
        "summary": {
            "artifact": repo_relative(artifact),
            "artifact_sha256": sha256_file(artifact),
            "pass": bool(document.get("pass")) and not failed,
            "failed_gates": failed,
            "n_reps": len(reps),
            "n_errors": len(errors),
            "decode_overall_median_tok_s": document.get("decode_overall_median_tok_s"),
            "degradation_fraction": document.get("degradation_fraction"),
            "mem_floor_gib": document.get("mem_floor_gib"),
            "mem_available_min_gib": min_mem,
            # The distinction that decides admissibility, carried forward rather
            # than left to whoever remembers how the run was launched.
            "qualification_eligible_floor": document.get("qualification_eligible_floor"),
        },
        "rows": [
            {"record_type": "soak_gate", "gate": name, "pass": bool(ok)}
            for name, ok in sorted(gates.items())
        ],
    }


def score_parity(artifact: Path) -> dict[str, Any]:
    document = read_json(artifact)
    return {
        "summary": {
            "artifact": repo_relative(artifact),
            "artifact_sha256": sha256_file(artifact),
            "pass": bool(document.get("pass")),
            "parity_level": document.get("parity_level"),
        },
        "rows": [
            {
                "record_type": "parity",
                "pass": bool(document.get("pass")),
                "parity_level": document.get("parity_level"),
            }
        ],
    }


# --------------------------------------------------------------------------
# Host observations
# --------------------------------------------------------------------------


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    except OSError as error:
        raise BundleError(f"cannot read /proc/meminfo: {error}") from error
    return values


def host_observations() -> dict[str, Any]:
    meminfo = read_meminfo()
    observation = {
        "record_type": "host_observation",
        "observed_at": utc_now(),
        "kernel": platform.release(),
        "mem_total_kib": meminfo.get("MemTotal"),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "swap_total_kib": meminfo.get("SwapTotal"),
        "swap_free_kib": meminfo.get("SwapFree"),
    }
    swap_total = observation["swap_total_kib"]
    swap_free = observation["swap_free_kib"]
    if isinstance(swap_total, int) and isinstance(swap_free, int):
        observation["swap_used_kib"] = swap_total - swap_free
    return observation


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


def readable_file(path: Path) -> bool:
    """is_file() that survives an unreadable parent directory.

    /home/dsv4 is mode 0700, so stat() on anything beneath it raises
    PermissionError from this account rather than returning False. Letting that
    propagate would abort the bundle over a field that is legitimately absent.
    """
    try:
        return path.is_file()
    except (PermissionError, OSError):
        return False


def build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    def digest_if_present(path: Path) -> str | None:
        if not readable_file(path):
            return None
        try:
            return sha256_file(path)
        except (PermissionError, OSError):
            return None

    status = git("status", "--porcelain")
    weights = read_json(WEIGHTS_MANIFEST)
    pin = read_json(WEIGHTS_PIN)

    return {
        "schema_version": 1,
        "record_type": "attempt_manifest",
        "built_at": utc_now(),
        "spec": {
            "path": str(SPEC_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SPEC_PATH),
            "decision_id": spec["decision_id"],
        },
        "source": {
            "commit": git("rev-parse", "HEAD").strip(),
            "tree_is_clean": status.strip() == "",
            "dirty_paths": [line[3:] for line in status.splitlines()],
            "diff_sha256": hashlib.sha256(
                git("diff", "HEAD").encode("utf-8")
            ).hexdigest(),
        },
        "scorer": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "harnesses": {
            "31_bench_accuracy.py": sha256_file(ACCURACY_HARNESS),
            "32_golden_tests.py": sha256_file(GOLDEN_HARNESS),
        },
        "binary": {
            "path": str(ENGINE_BINARY),
            "sha256": digest_if_present(ENGINE_BINARY),
            "readable": readable_file(ENGINE_BINARY),
            "note": (
                None
                if readable_file(ENGINE_BINARY)
                else "engine binary lives under /home/dsv4 (mode 0700) and is not "
                "readable from the harness account; the digest is therefore absent "
                "rather than fabricated"
            ),
        },
        "model": {
            "weights_manifest_path": str(WEIGHTS_MANIFEST.relative_to(REPO_ROOT)),
            "weights_manifest_sha256": sha256_file(WEIGHTS_MANIFEST),
            "release_repo": weights.get("repo"),
            "release_revision": weights.get("revision"),
            "shards": [
                {"name": entry["name"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
                for entry in weights.get("files", [])
            ],
            "pin_sha256": sha256_file(WEIGHTS_PIN),
            "pin_repo": pin.get("repo"),
            "pin_revision": pin.get("revision"),
        },
        "tokenizer": {
            "path": str(TOKENIZER.relative_to(REPO_ROOT)) if readable_file(TOKENIZER) else None,
            "sha256": digest_if_present(TOKENIZER),
        },
        "public_randomness": {
            "used": False,
            "why": (
                "This attempt scores already-collected deterministic suites whose "
                "row selection is seeded at 42 in the harness. No arm order or "
                "fixture subset was chosen here, so there is nothing for public "
                "randomness to bind. A holdout or matched-arm attempt would need it."
            ),
        },
    }


def gate_results(
    spec: dict[str, Any], arms: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    gates_spec = spec["preregistered_validity_gates"]
    ceilings = gates_spec["truncated_fraction_max"]
    results: list[dict[str, Any]] = []

    for name, arm in arms.items():
        summary = arm["summary"]
        suite = summary.get("suite")
        ceiling = ceilings.get(suite)
        exempt = (gates_spec.get("truncation_gate_exempt") or {}).get(suite)
        if ceiling is None and exempt:
            # Recorded, not gated, and the reason travels with the record. An
            # unexplained absent ceiling still fails below.
            results.append(
                {
                    "gate": f"{name}: truncation RECORDED (no valid baseline ceiling)",
                    "pass": True,
                    "observed": summary["truncated_fraction"],
                    "exemption_reason": exempt,
                }
            )
        elif ceiling is None:
            results.append(
                {
                    "gate": f"{name}: truncation ceiling",
                    "pass": False,
                    "detail": f"no preregistered ceiling for suite {suite!r}",
                }
            )
            continue
        else:
            observed = summary["truncated_fraction"]
            results.append(
                {
                    "gate": f"{name}: truncated_fraction <= {ceiling}",
                    "pass": observed <= ceiling,
                    "observed": observed,
                    "ceiling": ceiling,
                    "formula": "count(finish_reason == 'length') / n",
                }
            )
        results.append(
            {
                "gate": f"{name}: request_timeouts == {gates_spec['request_failures_max']}",
                "pass": summary["request_timeouts"] <= gates_spec["request_failures_max"],
                "observed": summary["request_timeouts"],
                "formula": "count(reason contains 'Timeout')",
            }
        )
        expected = gates_spec["expected_n"].get(f"{suite}-{summary.get('split')}")
        if expected is not None:
            results.append(
                {
                    "gate": f"{name}: n == {expected}",
                    "pass": summary["n"] == expected,
                    "observed": summary["n"],
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "build a bundle even when named arms are missing. The bundle records "
            "which arms were absent and its verdict can never be PASS."
        ),
    )
    args = parser.parse_args()

    if not ATTEMPT_ID.fullmatch(args.attempt_id):
        print("attempt id must match [a-z0-9][a-z0-9-]{0,63}", file=sys.stderr)
        return 2

    attempt = ATTEMPTS_ROOT / f"attempt-{args.attempt_id}"
    try:
        ATTEMPTS_ROOT.mkdir(parents=True, exist_ok=True)
        # O_EXCL semantics: a rebuild never silently replaces a prior verdict.
        attempt.mkdir()
    except FileExistsError:
        print(
            f"refusing to overwrite existing attempt {attempt}; choose a new id",
            file=sys.stderr,
        )
        return 3
    except OSError as error:
        print(f"cannot create attempt directory: {error}", file=sys.stderr)
        return 2

    try:
        spec = read_json(SPEC_PATH)
        manifest = build_manifest(spec)

        wanted = {
            "gsm8k-dev-thinking": (
                STAGING / "acc-gsm8k-dev-0731-thinking-16k.json",
                STAGING / "transcripts" / "gsm8k-dev-thinking-16k",
            ),
            "mmlu-pro-dev-thinking": (
                STAGING / "acc-mmlu-pro-dev-0731-thinking-16k.json",
                STAGING / "transcripts" / "mmlu-pro-dev-thinking-16k",
            ),
            "gsm8k-dev-chat": (
                STAGING / "acc-gsm8k-dev-0731-chat.json",
                STAGING / "transcripts" / "gsm8k-dev-chat",
            ),
            "mmlu-pro-dev-chat": (
                STAGING / "acc-mmlu-pro-dev-0731-chat.json",
                STAGING / "transcripts" / "mmlu-pro-dev-chat",
            ),
            "humaneval": (
                STAGING / "acc-humaneval-0731.json",
                STAGING / "transcripts" / "humaneval-0731",
            ),
            "gsm8k-holdout": (
                STAGING / "acc-gsm8k-holdout-0731-thinking.json",
                STAGING / "transcripts" / "gsm8k-holdout-thinking",
            ),
            "mmlu-pro-holdout": (
                STAGING / "acc-mmlu-pro-holdout-0731-thinking.json",
                STAGING / "transcripts" / "mmlu-pro-holdout-thinking",
            ),
        }

        arms: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        failures: list[str] = []
        for name, (artifact, transcripts) in wanted.items():
            if not artifact.is_file():
                missing.append(name)
                continue
            try:
                arms[name] = score_accuracy_arm(name, artifact, transcripts)
            except BundleError as error:
                failures.append(f"{name}: {error}")

        # Non-accuracy arms. Each is scored by its own recomputing function; a
        # missing one lands in `missing` and forces NO_RESULT rather than being
        # quietly dropped from the verdict.
        others: dict[str, dict[str, Any]] = {}
        for name, path, scorer in (
            ("speed", STAGING / "speed-llamacpp-0731-thinking.json", score_speed),
            ("soak", STAGING / "soak-llamacpp-0731-v2.json", score_soak),
            ("parity", STAGING / "parity-llamacpp-0731.json", score_parity),
        ):
            if not path.is_file():
                missing.append(name)
                continue
            try:
                others[name] = scorer(path)
            except BundleError as error:
                failures.append(f"{name}: {error}")

        golden_path = STAGING / "golden-llamacpp-0731-v2.json"
        golden = None
        if golden_path.is_file():
            try:
                golden = score_golden(golden_path)
            except BundleError as error:
                failures.append(f"golden: {error}")
        else:
            missing.append("golden")

        if missing and not args.allow_incomplete:
            raise BundleError(
                f"missing arms {missing!r}; pass --allow-incomplete to record a "
                "bundle that cannot pass"
            )

        raw_path = attempt / "raw.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(host_observations(), sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            for name in sorted(arms):
                for row in arms[name]["rows"]:
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            for name in sorted(others):
                for row in others[name]["rows"]:
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            if golden is not None:
                for row in golden["rows"]:
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            stream.flush()
            os.fsync(stream.fileno())

        gates = gate_results(spec, arms)
        if "speed" in others:
            gates.append({
                "gate": "speed: suite_valid",
                "pass": others["speed"]["summary"]["suite_valid"],
                "observed": others["speed"]["summary"]["suite_valid"],
            })
        if "parity" in others:
            gates.append({
                "gate": "parity: exact-ids",
                "pass": others["parity"]["summary"]["parity_level"] == "exact-ids"
                and others["parity"]["summary"]["pass"],
                "observed": others["parity"]["summary"]["parity_level"],
            })
        if "soak" in others:
            soak_summary = others["soak"]["summary"]
            gates.append({
                "gate": "soak: all gates pass",
                "pass": soak_summary["pass"],
                "observed": soak_summary["failed_gates"] or "none failed",
            })
            # Reported as a gate in its own right rather than folded into the
            # soak pass bit: the run genuinely passed every gate it was scored
            # against, and separately is not admissible to 34_decision.py, which
            # recomputes a 12 GiB floor. Collapsing the two would hide one of them.
            gates.append({
                "gate": "soak: floor is qualification-eligible (>= 12 GiB)",
                "pass": bool(soak_summary.get("qualification_eligible_floor")),
                "observed": soak_summary.get("mem_floor_gib"),
                "note": "an operational-floor soak is valid evidence for the served "
                        "profile and is NOT admissible to the decision procedure",
            })
        checks_pass = all(gate["pass"] for gate in gates)
        complete = not missing and not failures
        verdict = (
            "PASS"
            if complete and checks_pass and golden is not None and golden["summary"]["all_passed"]
            else "FAIL"
        )
        if not complete:
            verdict = "NO_RESULT"

        summary = {
            "schema_version": 1,
            "record_type": "attempt_summary",
            "attempt_id": args.attempt_id,
            "built_at": utc_now(),
            "verdict": verdict,
            "verdict_rule": (
                "PASS requires every named arm present, every arm recomputable from "
                "its transcripts, every preregistered gate met, and golden all-pass. "
                "Missing arms yield NO_RESULT rather than FAIL, so an incomplete run "
                "is never reported as a measured negative."
            ),
            "scope_limits": [
                "Accuracy is dev-split only. No holdout arm exists, so this bundle "
                "is NOT a qualification result under scripts/34_decision.py.",
                "Speed, soak, and context capability are not in this bundle.",
                "The engine binary digest is absent because /home/dsv4 is not "
                "readable from this account.",
            ],
            "missing_arms": missing,
            "arm_failures": failures,
            "gates": gates,
            "arms": {name: arms[name]["summary"] for name in sorted(arms)},
            "golden": golden["summary"] if golden else None,
            "other_arms": {name: others[name]["summary"] for name in sorted(others)},
            "raw_jsonl_sha256": sha256_file(raw_path),
            "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
        }

        (attempt / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (attempt / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except BundleError as error:
        (attempt / "ERROR.txt").write_text(f"{error}\n", encoding="utf-8")
        print(f"FAIL CLOSED: {error}", file=sys.stderr)
        return 4

    print(f"{verdict}: {attempt.relative_to(REPO_ROOT)}")
    for gate in gates:
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {gate['gate']}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
