#!/usr/bin/env python3
"""Thin Rung 0.1 lifecycle wrapper around the existing speed scorer."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
from glm52_goal import paired_ratio_bound, validate_ab_blocks
from glm52_w1_affine_campaign import (
    _authenticate_drand,
    _drand_record,
    content_complete_fixture_sha256,
)


SLAB_PATH = "/home/bmarti44/.cache/glm52-rung0-artifacts/glm52-experts-v2.slab"
SLAB_SHA256 = (
    "62961905a685e16e3e8f5f98e189511e"
    "b2e65ee6eda7e1a860c1ec58959e5518"
)
MODEL_SHA256 = (
    "a49de64c5020432bdae23de36a423a96"
    "60a5621bc0db8d12b66bd8814b07fea0"
)
MODEL_BYTES = 211_075_856_448
SLAB_BYTES = 190_028_697_600
PROVENANCE_NAMES = tuple(
    sorted(
        {
            "DS4_CUDA_EXPERT_CACHE_GB",
            "DS4_CUDA_EXPERT_CACHE_PIN",
            "DS4_CUDA_EXPERT_CACHE_SLRU",
            "DS4_CUDA_FETCH_THREADS",
            "DS4_CUDA_LOAD_PROFILE",
            "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
            "DS4_GLM_TP_DEBUG",
            "DS4_LOCK_FILE",
            "DS4_TOKEN_TIMING_LOG",
            "DS4_CUDA_EXPERT_SLAB_PATH",
            "DS4_CUDA_EXPERT_SLAB_SHA256",
            "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256",
            "DS4_CUDA_EXPERT_SLAB_TRACE",
            "DS4_CUDA_EXPERT_SLAB_AUTH_TRACE",
            "DS4_GLM_PREFETCH",
            "DS4_CUDA_EXPERT_SLAB_PREFETCH_SHA",
            "DS4_GLM_PREFETCH_THREADS",
        }
    )
)
ROOT = Path(__file__).resolve().parents[1]
CGROUP_RUNNER = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
BENCHMARK = ROOT / "scripts/30_bench_speed.py"
TOKENIZER = Path(
    "/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json"
)
TOKENIZER_SHA256 = (
    "19e773648cb4e65de8660ea6365e10ac"
    "ca112d42a854923df93db4a6f333a82d"
)
FIXTURE = ROOT / "fixtures/ctx-32k.txt"
GLOBAL_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
INSTANCE_LOCK = "/run/user/1000/glm52-rung0-ds4.lock"
CRASH_ROOT = Path("/home/bmarti44/.local/state/glm52-crashlog")
MODEL_PATH = Path(
    "/home/dsv4/ds4-project/gguf-glm/"
    "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
)
INFLIGHT = Path("/sys/class/block/nvme0n1/inflight")
ARENA_BYTES = 68_000_000_000
MEMORY_MARGIN_BYTES = 4 * 1024**3
MEMORY_MAX_EXCURSION_GIB = 2
HOST_KILL_FLOOR_GIB = 18
QUALITY_FIXTURE_RELATIVE = Path(
    "gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv"
)
QUALITY_FIXTURE_CONTENT_SHA256 = (
    "11c5dc7234a21f645141c5431dd80eb5"
    "5ff9b36bc5eb8ca1baff377012bdc0d3"
)
EXPERT_RECORD_PAYLOAD_BYTES = 9_732_096


def arm_schedule(*, flip: bool = False) -> tuple[tuple[int, int, str], ...]:
    """Return the preregistered five-block execution order."""
    rows: list[tuple[int, int, str]] = []
    for block in range(5):
        order = "ABBA" if (block + int(flip)) % 2 == 0 else "BAAB"
        rows.extend((block, sequence, arm) for sequence, arm in enumerate(order))
    return tuple(rows)


def sha_prefetch_schedule(flip: bool = False) -> tuple[tuple[int, int, str], ...]:
    """Five-block, three-arm schedule frozen before the async implementation."""
    orders = ("ABC", "BCA", "CAB", "CBA", "ACB")
    if flip:
        orders = tuple(order[::-1] for order in orders)
    return tuple(
        (block, sequence, arm)
        for block, order in enumerate(orders)
        for sequence, arm in enumerate(order)
    )


def canonical_sha_prefetch_environment(mode: str) -> dict[str, str]:
    """Exact A/B/C environment; absence is represented by an unset key."""
    if mode not in {"off", "demand_sha", "prefetch_sha"}:
        raise ValueError("invalid SHA-prefetch mode")
    result = canonical_engine_environment("off")
    result["DS4_CUDA_EXPERT_SLAB_AUTH_TRACE"] = "1"
    if mode != "off":
        result.update({
            "DS4_CUDA_EXPERT_SLAB_PATH": SLAB_PATH,
            "DS4_CUDA_EXPERT_SLAB_SHA256": SLAB_SHA256,
            "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256": MODEL_SHA256,
        })
    if mode == "prefetch_sha":
        result.update({
            "DS4_GLM_PREFETCH": "1",
            "DS4_CUDA_EXPERT_SLAB_PREFETCH_SHA": "1",
            "DS4_GLM_PREFETCH_THREADS": "8",
        })
    return result


def _canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _validate_sha_prefetch_quality(
    attempts: list[dict[str, Any]], candidate_commit: str,
    binary_sha256: str, fixture_sha256: str,
    configuration_sha256_by_arm: dict[str, str],
) -> dict[str, Any]:
    """Derive the lossless result from three complete raw 100-case attempts."""
    if not isinstance(attempts, list) or len(attempts) != 3:
        raise ValueError("quality evidence requires exact A/B/C attempts")
    expected_mode = {"A": "off", "B": "demand_sha", "C": "prefetch_sha"}
    canonical_rows: list[dict[str, Any]] | None = None
    case_ids: list[str] | None = None
    for attempt, arm in zip(attempts, "ABC", strict=True):
        if not isinstance(attempt, dict) or set(attempt) != {
            "schema_version", "arm", "mode", "candidate_commit",
            "binary_sha256", "configuration_sha256", "fixture_sha256",
            "output_sha256", "rows",
        }:
            raise ValueError("quality attempt schema is invalid")
        rows = attempt["rows"]
        if (
            attempt["schema_version"] != 1
            or attempt["arm"] != arm
            or attempt["mode"] != expected_mode[arm]
            or attempt["candidate_commit"] != candidate_commit
            or attempt["binary_sha256"] != binary_sha256
            or attempt["configuration_sha256"]
            != configuration_sha256_by_arm[arm]
            or attempt["fixture_sha256"] != fixture_sha256
            or re.fullmatch(r"[0-9a-f]{64}", attempt["output_sha256"]) is None
            or not isinstance(rows, list)
            or len(rows) != 100
        ):
            raise ValueError("quality attempt is incomplete or unbound")
        ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "case_id", "target_tokens", "total_nll", "top1_matches",
            }:
                raise ValueError("quality row schema is invalid")
            case_id = row["case_id"]
            target_tokens = row["target_tokens"]
            total_nll = row["total_nll"]
            top1_matches = row["top1_matches"]
            if (
                not isinstance(case_id, str) or not case_id
                or isinstance(target_tokens, bool) or not isinstance(target_tokens, int)
                or target_tokens <= 0
                or isinstance(top1_matches, bool) or not isinstance(top1_matches, int)
                or not 0 <= top1_matches <= target_tokens
                or isinstance(total_nll, bool)
                or not isinstance(total_nll, (int, float))
                or not math.isfinite(float(total_nll)) or total_nll < 0
            ):
                raise ValueError("quality row contains invalid values")
            ids.append(case_id)
        if len(set(ids)) != 100:
            raise ValueError("quality case IDs are duplicated")
        if case_ids is None:
            case_ids = ids
            canonical_rows = rows
        elif ids != case_ids or rows != canonical_rows:
            raise ValueError("lossless candidate changed quality rows")
    assert canonical_rows is not None
    total_tokens = sum(row["target_tokens"] for row in canonical_rows)
    return {
        "case_count": 100,
        "target_tokens": total_tokens,
        "mean_nll": sum(float(row["total_nll"]) for row in canonical_rows)
        / total_tokens,
        "top1_agreement": sum(row["top1_matches"] for row in canonical_rows)
        / total_tokens,
        "token_weighted_delta_nll": 0.0,
        "top1_loss_pp": 0.0,
        "deterministic": True,
    }


def score_sha_prefetch_campaign(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    freeze: dict[str, Any],
    quality_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the frozen A=off/B=demand-SHA/C=prefetch-SHA campaign."""
    expected_manifest_keys = {
        "schema_version", "candidate_commit", "binary_sha256",
        "quality_binary_sha256", "model_generation",
        "configuration_by_arm", "configuration_sha256_by_arm", "fixture_sha256",
        "access_stream_sha256", "campaign_started_monotonic_ns",
        "campaign_finished_monotonic_ns", "freeze_sha256", "randomness",
    }
    if set(manifest) != expected_manifest_keys or manifest["schema_version"] != 2:
        raise ValueError("prefetch manifest schema is invalid")

    def sha256(value: Any, label: str) -> str:
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"{label} is not a lowercase SHA-256")
        return value

    def finite(value: Any, label: str, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} is not numeric")
        result = float(value)
        if not math.isfinite(result) or (positive and result <= 0) or (
            not positive and result < 0
        ):
            raise ValueError(f"{label} is not finite in the required range")
        return result

    def integer(value: Any, label: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} is not an integer in the required range")
        return value

    commit = manifest["candidate_commit"]
    if (not isinstance(commit, str) or len(commit) != 40 or
            any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("candidate commit is invalid")
    binary = sha256(manifest["binary_sha256"], "manifest binary")
    quality_binary = sha256(
        manifest["quality_binary_sha256"], "manifest quality binary"
    )
    model_generation = integer(
        manifest["model_generation"], "manifest model generation", minimum=1
    )
    fixture = sha256(manifest["fixture_sha256"], "manifest fixture")
    access = sha256(manifest["access_stream_sha256"], "manifest access stream")
    environments = manifest["configuration_by_arm"]
    configurations = manifest["configuration_sha256_by_arm"]
    if (
        not isinstance(environments, dict) or set(environments) != {"A", "B", "C"}
        or not isinstance(configurations, dict) or set(configurations) != {"A", "B", "C"}
    ):
        raise ValueError("manifest arm configurations are invalid")
    expected_modes = {"A": "off", "B": "demand_sha", "C": "prefetch_sha"}
    for arm, mode in expected_modes.items():
        expected_environment = canonical_sha_prefetch_environment(mode)
        if environments[arm] != expected_environment:
            raise ValueError(f"arm {arm} environment differs from the fixed mode")
        expected_hash = canonical_environment_sha256(expected_environment)
        if sha256(configurations[arm], f"configuration {arm}") != expected_hash:
            raise ValueError(f"arm {arm} configuration hash is not derived")
    started = integer(
        manifest["campaign_started_monotonic_ns"], "campaign start", minimum=1
    )
    finished = integer(
        manifest["campaign_finished_monotonic_ns"], "campaign finish", minimum=1
    )
    if not started < finished or finished - started > 86_400 * 1_000_000_000:
        raise ValueError("campaign is not a bounded contemporaneous window")
    expected_freeze_keys = {
        "schema_version", "candidate_commit", "binary_sha256",
        "quality_binary_sha256", "source_sha256", "scorer_sha256",
        "tests_sha256", "frozen_epoch_s",
    }
    if (
        not isinstance(freeze, dict) or set(freeze) != expected_freeze_keys
        or freeze["schema_version"] != 1
        or freeze["candidate_commit"] != commit
        or freeze["binary_sha256"] != binary
        or freeze["quality_binary_sha256"] != quality_binary
        or any(
            re.fullmatch(r"[0-9a-f]{64}", freeze[name]) is None
            for name in ("source_sha256", "scorer_sha256", "tests_sha256")
        )
        or not isinstance(freeze["frozen_epoch_s"], (int, float))
        or not math.isfinite(float(freeze["frozen_epoch_s"]))
        or manifest["freeze_sha256"] != _canonical_object_sha256(freeze)
    ):
        raise ValueError("campaign is not bound to the frozen candidate")
    expected_source = _canonical_object_sha256({
        "patch_sha256": sha256_file(
            ROOT / "results/glm52-gates/harness/"
            "ds4-expert-slab-prefetch-sha-pipeline.patch"
        ),
        "state_header_sha256": sha256_file(
            ROOT / "results/glm52-gates/harness/ds4_slab_prefetch_state.h"
        ),
    })
    if (
        freeze["source_sha256"] != expected_source
        or freeze["scorer_sha256"] != sha256_file(Path(__file__).resolve())
        or freeze["tests_sha256"] != sha256_file(
            ROOT / "scripts/tests/test_glm_rung0_sha_prefetch.py"
        )
    ):
        raise ValueError("frozen source, scorer, or acceptance test changed")
    randomness = manifest["randomness"]
    validate_confirmation_record(randomness, commit, binary, quality_binary)
    if randomness["published_epoch_s"] <= freeze["frozen_epoch_s"]:
        raise ValueError("campaign randomness predates the freeze")
    quality = _validate_sha_prefetch_quality(
        quality_attempts, commit, binary, fixture, configurations
    )

    schedule = sha_prefetch_schedule(bool(randomness["flip"]))
    if not isinstance(records, list) or len(records) != len(schedule):
        raise ValueError("campaign requires exactly 15 fresh-server arms")
    exact_record_keys = {
        "schema_version", "block", "sequence", "arm", "mode",
        "server_instance_id", "candidate_commit", "binary_sha256",
        "configuration_sha256", "fixture_sha256", "access_stream_sha256",
        "recorded_monotonic_ns", "reps", "engine", "safety",
    }
    expected_mode = expected_modes
    servers: set[str] = set()
    paired_outputs: dict[tuple[int, int], set[tuple[Any, ...]]] = {}
    metrics: dict[str, dict[str, dict[int, list[float]]]] = {
        arm: {
            "client": {block: [] for block in range(5)},
            "raw": {block: [] for block in range(5)},
            "cold": {block: [] for block in range(5)},
            "warm": {block: [] for block in range(5)},
        }
        for arm in "ABC"
    }
    fetch_ms: dict[str, list[float]] = {"B": [], "C": []}

    for ordinal, (record, scheduled) in enumerate(zip(records, schedule)):
        if set(record) != exact_record_keys or record["schema_version"] != 1:
            raise ValueError(f"arm {ordinal} schema is invalid")
        block, sequence, arm = scheduled
        if (record["block"], record["sequence"], record["arm"]) != scheduled:
            raise ValueError("campaign arms are missing, duplicated, or reordered")
        mode = expected_mode[arm]
        if record["mode"] != mode:
            raise ValueError("arm-to-mode mapping is invalid")
        server = record["server_instance_id"]
        if not isinstance(server, str) or not server or server in servers:
            raise ValueError("campaign did not use unique fresh servers")
        servers.add(server)
        if (record["candidate_commit"] != commit or
                sha256(record["binary_sha256"], "record binary") != binary or
                sha256(record["configuration_sha256"], "record configuration") != configurations[arm] or
                sha256(record["fixture_sha256"], "record fixture") != fixture or
                sha256(record["access_stream_sha256"], "record access stream") != access):
            raise ValueError("arm provenance differs from the frozen manifest")
        recorded = integer(record["recorded_monotonic_ns"], "record time", minimum=1)
        if not started <= recorded <= finished:
            raise ValueError("historical or out-of-window arm is forbidden")

        reps = record["reps"]
        if not isinstance(reps, list) or len(reps) != 2 or [
            rep.get("phase") if isinstance(rep, dict) else None for rep in reps
        ] != ["cold", "warm"]:
            raise ValueError("each fresh server requires cold then warm reps")
        for rep_index, rep in enumerate(reps):
            exact_rep_keys = {
                "phase", "request_sha256", "generated_bytes_sha256",
                "token_ids", "completion_tokens", "raw_token_timestamps_ns",
                "client_token_timestamps_ns", "client_request_started_ns",
                "client_first_token_ns", "client_last_token_ns", "ttft_seconds",
            }
            if not isinstance(rep, dict) or set(rep) != exact_rep_keys:
                raise ValueError("rep schema is invalid")
            count = integer(rep["completion_tokens"], "completion tokens", minimum=128)
            token_ids = rep["token_ids"]
            raw_times = rep["raw_token_timestamps_ns"]
            client_times = rep["client_token_timestamps_ns"]
            if (not isinstance(token_ids, list) or len(token_ids) != count or
                    not isinstance(raw_times, list) or len(raw_times) != count or
                    not isinstance(client_times, list) or len(client_times) < 2 or
                    any(isinstance(value, bool) or not isinstance(value, int)
                        for value in token_ids + raw_times + client_times) or
                    any(right <= left for left, right in zip(raw_times, raw_times[1:])) or
                    any(right <= left for left, right in zip(client_times, client_times[1:]))):
                raise ValueError("generated token/timing evidence is incomplete")
            request_started = integer(rep["client_request_started_ns"], "request start", minimum=1)
            first = integer(rep["client_first_token_ns"], "first token", minimum=1)
            last = integer(rep["client_last_token_ns"], "last token", minimum=1)
            if (client_times[0] != first or client_times[-1] != last or
                    not request_started < first < last or
                    not started <= request_started or last > finished or
                    raw_times[0] < started or raw_times[-1] > finished):
                raise ValueError("client timing endpoints are inconsistent")
            ttft = (first - request_started) / 1e9
            if not math.isclose(
                ttft, finite(rep["ttft_seconds"], "reported TTFT", positive=True),
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError("reported TTFT differs from client endpoints")
            client_elapsed = (client_times[-1] - client_times[0]) / 1e9
            raw_elapsed = (raw_times[-1] - raw_times[0]) / 1e9
            clock_ratio = raw_elapsed / client_elapsed
            if not 0.75 <= clock_ratio <= 1.25:
                raise ValueError("client and raw clocks disagree")
            metrics[arm]["client"][block].append((count - 1) / client_elapsed)
            metrics[arm]["raw"][block].append((count - 1) / raw_elapsed)
            metrics[arm][rep["phase"]][block].append(ttft)
            signature = (
                sha256(rep["request_sha256"], "request"),
                sha256(rep["generated_bytes_sha256"], "generated bytes"),
                tuple(token_ids), count,
            )
            paired_outputs.setdefault((block, rep_index), set()).add(signature)

        engine = record["engine"]
        if not isinstance(engine, dict) or set(engine) != {
            "mode", "model_generation", "slab_reads", "slab_peak_qd",
            "completed_fetch_ms", "telemetry",
        } or engine["mode"] != mode or integer(
            engine["model_generation"], "model generation", minimum=1
        ) != model_generation:
            raise ValueError("engine mode/generation evidence is invalid")
        reads = integer(engine["slab_reads"], "slab reads")
        qd = integer(engine["slab_peak_qd"], "slab queue depth")
        fetch = engine["completed_fetch_ms"]
        if not isinstance(fetch, list) or any(
            not math.isfinite(finite(value, "completed fetch", positive=True))
            for value in fetch
        ):
            raise ValueError("completed-fetch timing is invalid")
        if arm == "A":
            if reads != 0 or qd != 0 or fetch:
                raise ValueError("slab-off arm performed slab work")
        else:
            if reads <= 0 or qd < 2 or len(fetch) < 3:
                raise ValueError("slab arm lacks completed I/O coverage")
            fetch_ms[arm].extend(float(value) for value in fetch)
        telemetry = engine["telemetry"]
        telemetry_keys = {
            "attempts", "sha_successes", "sha_failures", "ready", "late",
            "stale", "fallback", "copies", "validated_bytes", "copied_bytes",
            "publications", "read_ns", "sha_ns", "wait_ns", "copy_ns",
            "current_ready",
        }
        if not isinstance(telemetry, dict) or set(telemetry) != telemetry_keys:
            raise ValueError("prefetch telemetry schema is invalid")
        t = {name: integer(value, f"telemetry {name}") for name, value in telemetry.items()}
        if (t["attempts"] != t["sha_successes"] + t["sha_failures"] or
                t["sha_failures"] != 0 or t["copies"] > t["sha_successes"] or
                t["publications"] > t["copies"] or
                t["validated_bytes"]
                != t["sha_successes"] * EXPERT_RECORD_PAYLOAD_BYTES or
                t["copied_bytes"] != t["copies"] * EXPERT_RECORD_PAYLOAD_BYTES):
            raise ValueError("prefetch telemetry does not reconcile")
        if arm == "A" and any(t.values()):
            raise ValueError("slab-off telemetry is nonzero")
        if arm == "B" and (
            t["attempts"] != reads or t["sha_successes"] != reads
            or t["copies"] != reads or t["publications"] != reads
            or any(t[name] for name in (
                "ready", "late", "stale", "fallback", "current_ready", "wait_ns"
            ))
            or min(t["read_ns"], t["sha_ns"], t["copy_ns"]) <= 0
        ):
            raise ValueError("demand-only arm lacks exact full-SHA coverage")
        if arm == "C" and (
            t["attempts"] <= 0 or t["sha_successes"] <= 0 or t["copies"] <= 0 or
            t["ready"] != t["sha_successes"] or
            t["copies"] + t["stale"] + t["current_ready"] != t["ready"] or
            t["fallback"] != t["late"] or
            min(t["read_ns"], t["sha_ns"], t["copy_ns"]) <= 0
        ):
            raise ValueError("prefetch terminal outcomes do not reconcile")

        safety = record["safety"]
        if not isinstance(safety, dict) or set(safety) != {
            "minimum_available_gib", "swap_bytes", "oom_events", "xid", "survivors"
        } or finite(safety["minimum_available_gib"], "available memory", positive=True) < 10 or \
                safety["swap_bytes"] != 0 or safety["oom_events"] != 0 or \
                safety["xid"] is not False or safety["survivors"] != []:
            raise ValueError("arm safety evidence is invalid")

    if len(paired_outputs) != 10 or any(
        len(values) != 1 for values in paired_outputs.values()
    ):
        raise ValueError("paired requests, bytes, or token IDs differ across arms")

    block_means: dict[str, dict[str, list[float]]] = {
        arm: {metric: [] for metric in ("client", "raw", "cold", "warm")}
        for arm in "ABC"
    }
    for arm in "ABC":
        for metric in block_means[arm]:
            for block in range(5):
                values = metrics[arm][metric][block]
                expected_count = 2 if metric in {"client", "raw"} else 1
                if len(values) != expected_count:
                    raise ValueError("block metric coverage is incomplete")
                block_means[arm][metric].append(statistics.fmean(values))

    decode_bounds: dict[str, float] = {}
    ttft_bounds: dict[str, float] = {}
    for comparator in ("A", "B"):
        for clock in ("client", "raw"):
            decode_bounds[f"C/{comparator}:{'client_wall' if clock == 'client' else 'raw_token'}"] = paired_ratio_bound(
                block_means["C"][clock], block_means[comparator][clock], side="lower"
            )
        for phase in ("cold", "warm"):
            ttft_bounds[f"C/{comparator}:{phase}"] = paired_ratio_bound(
                block_means["C"][phase], block_means[comparator][phase], side="upper"
            )
    fetch_ratio = statistics.median(fetch_ms["C"]) / statistics.median(fetch_ms["B"])
    verdict = "PASS" if (
        fetch_ratio <= 0.90 and all(value > 1.0 for value in decode_bounds.values())
        and all(value <= 1.05 for value in ttft_bounds.values())
    ) else "FAIL"
    return {
        "scorer_id": "glm.rung0.full-sha-prefetch.v1",
        "verdict": verdict,
        "probe_completed_fetch_ratio": fetch_ratio,
        "decode_ratio_lower_95_by_comparator_and_clock": decode_bounds,
        "ttft_ratio_upper_95_by_comparator_and_state": ttft_bounds,
        "block_means": block_means,
        "quality": dict(quality),
    }


def safe_timeout_seconds(mode: str) -> int:
    """Allow the evidence-only 401 GB identity scan to finish unchanged."""
    if mode == "off":
        return 3600
    if mode == "on":
        return 5400
    raise ValueError("unknown slab mode")


def quality_timeout_seconds(mode: str) -> int:
    """Bound one complete 100-case scorer arm, including identity scans."""
    if mode not in {"off", "on"}:
        raise ValueError("unknown slab mode")
    return 9000


def confirmation_seed(
    randomness: str,
    candidate_commit: str,
    binary_sha256: str,
    quality_binary_sha256: str,
) -> str:
    """Bind public randomness to both clean-built candidate executables."""
    if (
        re.fullmatch(r"[0-9a-f]{64}", randomness) is None
        or re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None
        or re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", quality_binary_sha256) is None
    ):
        raise ValueError("confirmation seed input is malformed")
    return hashlib.sha256(
        (
            f"{candidate_commit}:{binary_sha256}:{quality_binary_sha256}:"
            f"{randomness}:glm-rung0-slab"
        ).encode("ascii")
    ).hexdigest()


def authenticate_confirmation(
    path: Path,
    candidate_commit: str,
    binary_sha256: str,
    quality_binary_sha256: str,
    not_before_epoch_s: float,
) -> dict[str, Any]:
    """Require relay-agreed randomness published after the binary freeze."""
    authenticated = _authenticate_drand(_drand_record(path))
    record = {
        name: authenticated[name] for name in ("round", "randomness", "signature")
    }
    chain_identity: tuple[int, int, str] | None = None
    for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
        completed = subprocess.run(
            [
                "/usr/bin/curl", "--disable", "--silent", "--show-error",
                "--fail", "--max-time", "10", "--proto", "=https",
                f"https://{host}/info",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            check=False,
        )
        if completed.returncode:
            raise ValueError(f"latest drand relay unavailable: {host}")
        info = json.loads(
            completed.stdout,
            object_pairs_hook=lambda pairs: _reject_duplicate_json_pairs(
                pairs, f"drand chain metadata from {host}"
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite drand value {value}")
            ),
        )
        identity = (info.get("genesis_time"), info.get("period"), info.get("hash"))
        if (
            not isinstance(identity[0], int)
            or not isinstance(identity[1], int)
            or identity[1] <= 0
            or not isinstance(identity[2], str)
            or re.fullmatch(r"[0-9a-f]{64}", identity[2]) is None
        ):
            raise ValueError(f"drand chain metadata is malformed at {host}")
        if chain_identity is None:
            chain_identity = identity
        elif identity != chain_identity:
            raise ValueError("drand relays disagree on chain metadata")
    assert chain_identity is not None
    published_epoch_s = chain_identity[0] + (record["round"] - 1) * chain_identity[1]
    if published_epoch_s <= not_before_epoch_s:
        raise ValueError("drand round was published before the frozen candidate")
    seed = confirmation_seed(
        record["randomness"], candidate_commit, binary_sha256, quality_binary_sha256
    )
    return {
        **record,
        "chain_hash": chain_identity[2],
        "published_epoch_s": published_epoch_s,
        "seed_sha256": seed,
        "flip": bool(int(seed[:2], 16) & 1),
    }


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, Any]], label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in {label}: {key}")
        result[key] = value
    return result


def validate_confirmation_record(
    record: Any,
    candidate_commit: str,
    binary_sha256: str,
    quality_binary_sha256: str,
) -> None:
    """Validate the complete public-randomness binding consumed by scoring."""
    expected_keys = {
        "round", "randomness", "signature", "chain_hash",
        "published_epoch_s", "seed_sha256", "flip",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or not isinstance(record.get("round"), int)
        or isinstance(record.get("round"), bool)
        or record["round"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", record.get("randomness", ""))
        or not re.fullmatch(r"[0-9a-f]{192}", record.get("signature", ""))
        or hashlib.sha256(bytes.fromhex(record["signature"])).hexdigest()
        != record["randomness"]
        or not re.fullmatch(r"[0-9a-f]{64}", record.get("chain_hash", ""))
        or not isinstance(record.get("published_epoch_s"), int)
        or isinstance(record.get("published_epoch_s"), bool)
        or not isinstance(record.get("flip"), bool)
        or record.get("seed_sha256")
        != confirmation_seed(
            record["randomness"], candidate_commit, binary_sha256,
            quality_binary_sha256,
        )
        or record["flip"] != bool(int(record["seed_sha256"][:2], 16) & 1)
    ):
        raise ValueError("public-randomness binding is incomplete or malformed")


def derive_memory_envelope(
    non_arena_peak_bytes: int, host_total_bytes: int
) -> dict[str, int]:
    """Derive the only accepted full-cache cgroup limit from a real probe."""
    if (
        isinstance(non_arena_peak_bytes, bool)
        or not isinstance(non_arena_peak_bytes, int)
        or isinstance(host_total_bytes, bool)
        or not isinstance(host_total_bytes, int)
        or not 8 * 1024**3 <= non_arena_peak_bytes <= 48 * 1024**3
        or host_total_bytes < 110 * 1024**3
    ):
        raise ValueError("memory probe values are outside the bounded host model")
    required = non_arena_peak_bytes + ARENA_BYTES + MEMORY_MARGIN_BYTES
    memory_high_gib = math.ceil(required / 1024**3)
    memory_max_gib = memory_high_gib + MEMORY_MAX_EXCURSION_GIB
    if (
        memory_high_gib < 32
        or memory_high_gib > 101
        or (memory_max_gib + HOST_KILL_FLOOR_GIB) * 1024**3 > host_total_bytes
    ):
        raise ValueError("measured GLM envelope cannot preserve the host kill floor")
    return {
        "non_arena_peak_bytes": non_arena_peak_bytes,
        "arena_bytes": ARENA_BYTES,
        "margin_bytes": MEMORY_MARGIN_BYTES,
        "memory_high_bytes": memory_high_gib * 1024**3,
        "memory_high_gib": memory_high_gib,
        "memory_max_gib": memory_max_gib,
        "host_total_bytes": host_total_bytes,
        "host_kill_floor_gib": HOST_KILL_FLOOR_GIB,
    }


def parse_quality_tsv(path: Path) -> list[dict[str, Any]]:
    """Read the complete fixed quality suite and reject partial evidence."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 100:
        raise ValueError(f"quality output needs 100 cases, got {len(rows)}")
    cases: list[dict[str, Any]] = []
    for row in rows:
        try:
            case = {
                "case_id": row["id"],
                "tokens": int(row["target_tokens"]),
                "nll_sum": float(row["nll"]),
                "top1_correct": int(row["target_top1_correct"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("quality output contains malformed values") from error
        if (
            not case["case_id"]
            or case["tokens"] <= 0
            or not 0 <= case["top1_correct"] <= case["tokens"]
            or not math.isfinite(case["nll_sum"])
        ):
            raise ValueError("quality output contains invalid values")
        cases.append(case)
    if len({case["case_id"] for case in cases}) != 100:
        raise ValueError("quality output case IDs are duplicated")
    return cases


def fixture_manifest_case_ids(path: Path) -> list[str]:
    """Validate the fixed official manifest's literal on-disk schema."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    identifiers = [row.get("# id") for row in rows]
    if (
        len(identifiers) != 100
        or any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
        or len(set(identifiers)) != 100
    ):
        raise ValueError("quality fixture is not the fixed complete 100-case suite")
    return identifiers


def compare_quality_rows(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Enforce exact teacher-forced identity for byte-preserving transport."""
    if len(baseline) != 100 or len(candidate) != 100:
        raise ValueError("quality comparison requires two complete suites")
    if baseline != candidate:
        raise ValueError("lossless slab transport changed quality evidence")
    tokens = sum(case["tokens"] for case in baseline)
    if tokens <= 0:
        raise ValueError("quality comparison has no target tokens")
    return {
        "case_count": 100,
        "token_weighted_delta_nll": 0.0,
        "top1_loss_pp": 0.0,
        "deterministic": True,
    }


def quality_schedule(*, flip: bool = False) -> tuple[str, ...]:
    """One balanced block with two independent executions of each arm."""
    return ("B", "A", "A", "B") if flip else ("A", "B", "B", "A")


def validate_quality_attempts(
    attempts: list[dict[str, Any]], expected_case_ids: list[str]
) -> dict[str, Any]:
    """Require exact self-replay and exact cross-arm quality identity."""
    if (
        not isinstance(attempts, list)
        or len(attempts) != 4
        or tuple(attempt.get("arm") for attempt in attempts)
        not in {quality_schedule(), quality_schedule(flip=True)}
    ):
        raise ValueError("quality attempts do not match the fixed ABBA schedule")
    if (
        len(expected_case_ids) != 100
        or len(set(expected_case_ids)) != 100
        or any(not isinstance(case_id, str) or not case_id for case_id in expected_case_ids)
    ):
        raise ValueError("official quality case identity is invalid")
    grouped: dict[str, list[list[dict[str, Any]]]] = {"A": [], "B": []}
    for attempt in attempts:
        arm = attempt["arm"]
        mode = "off" if arm == "A" else "on"
        rows = attempt.get("rows")
        engine = attempt.get("engine")
        safety = attempt.get("safety")
        expected_attempt_keys = {
            "arm", "mode", "rows", "output_sha256", "configuration_sha256",
            "engine", "safety",
        }
        expected_safety_keys = {
            "minimum_available_gib", "cgroup_high_events", "cgroup_max_events",
            "cgroup_oom_events", "cgroup_swap_bytes", "xid", "survivors",
            "failures",
        }
        if (
            set(attempt) != expected_attempt_keys
            or attempt.get("mode") != mode
            or not isinstance(rows, list)
            or len(rows) != 100
            or not re.fullmatch(r"[0-9a-f]{64}", attempt.get("output_sha256", ""))
            or attempt.get("configuration_sha256")
            != canonical_environment_sha256(canonical_engine_environment(mode))
            or not isinstance(engine, dict)
            or set(engine) != {"slab_mode", "slab_reads", "slab_peak_qd"}
            or engine.get("slab_mode") != mode
            or not isinstance(safety, dict)
            or set(safety) != expected_safety_keys
            or safety.get("minimum_available_gib", 0) < 10
            or any(
                safety.get(name) != 0
                for name in (
                    "cgroup_high_events", "cgroup_max_events",
                    "cgroup_oom_events", "cgroup_swap_bytes",
                )
            )
            or safety.get("xid") is not False
            or safety.get("survivors") != []
            or safety.get("failures") != []
        ):
            raise ValueError("quality attempt is incomplete or unsafe")
        if mode == "on" and (
            engine["slab_reads"] <= 0 or engine["slab_peak_qd"] < 2
        ):
            raise ValueError("quality slab arm lacks concurrent reads")
        if mode == "off" and (engine["slab_reads"] != 0 or engine["slab_peak_qd"] != 0):
            raise ValueError("quality baseline performed slab reads")
        if [row.get("case_id") for row in rows] != expected_case_ids:
            raise ValueError("quality output IDs differ from the official fixture")
        grouped[arm].append(rows)
    if grouped["A"][0] != grouped["A"][1] or grouped["B"][0] != grouped["B"][1]:
        raise ValueError("quality arm is not deterministic with itself")
    return compare_quality_rows(grouped["A"][0], grouped["B"][0])


def validate_bound_quality_evidence(
    performance_manifest: dict[str, Any],
    quality_manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive quality from raw attempts bound to the performance candidate."""
    for name in (
        "candidate_commit",
        "binary_sha256",
        "model_sha256",
        "memory_envelope_sha256",
    ):
        if quality_manifest.get(name) != performance_manifest.get(name):
            raise ValueError(f"quality evidence differs on {name}")
    performance_randomness = performance_manifest.get("randomness")
    quality_randomness = quality_manifest.get("randomness")
    validate_confirmation_record(
        performance_randomness,
        performance_manifest.get("candidate_commit", ""),
        performance_manifest.get("binary_sha256", ""),
        performance_manifest.get("quality_binary_sha256", ""),
    )
    validate_confirmation_record(
        quality_randomness,
        quality_manifest.get("candidate_commit", ""),
        quality_manifest.get("binary_sha256", ""),
        quality_manifest.get("quality_binary_sha256", ""),
    )
    if (
        not isinstance(performance_randomness, dict)
        or not isinstance(quality_randomness, dict)
        or quality_randomness != performance_randomness
    ):
        raise ValueError("quality and performance public randomness differ")
    expected_ids = quality_manifest.get("ordered_case_ids")
    expected_quality_keys = {
        "schema_version", "candidate_commit", "binary_sha256",
        "quality_binary_sha256", "model_sha256", "model_stat_before",
        "model_stat_after", "fixture_sha256", "fixture_content_sha256",
        "fixture_content_sha256_after", "ordered_case_ids",
        "memory_envelope_sha256", "quality_raw_sha256", "nll_sha256",
        "schedule", "randomness",
    }
    manifest_schedule = tuple(quality_manifest.get("schedule", ()))
    expected_schedule = quality_schedule(flip=bool(quality_randomness.get("flip")))
    if (
        set(quality_manifest) != expected_quality_keys
        or quality_manifest.get("schema_version") != 1
        or quality_manifest.get("fixture_content_sha256")
        != QUALITY_FIXTURE_CONTENT_SHA256
        or quality_manifest.get("fixture_content_sha256_after")
        != QUALITY_FIXTURE_CONTENT_SHA256
        or quality_manifest.get("model_stat_before")
        != quality_manifest.get("model_stat_after")
        or not isinstance(quality_manifest.get("model_stat_before"), dict)
        or quality_manifest.get("quality_binary_sha256")
        != performance_manifest.get("quality_binary_sha256")
        or not isinstance(expected_ids, list)
        or manifest_schedule != expected_schedule
        or tuple(attempt.get("arm") for attempt in attempts) != expected_schedule
    ):
        raise ValueError("quality manifest is not bound to the fixed scorer contract")
    return validate_quality_attempts(attempts, expected_ids)


def validate_performance_binding(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Bind every performance arm to the frozen manifest and exact schedule."""
    raw_schedule = manifest.get("schedule")
    allowed_schedules = {
        tuple(arm_schedule()),
        tuple(arm_schedule(flip=True)),
    }
    try:
        schedule = tuple(tuple(row) for row in raw_schedule)
    except (TypeError, ValueError):
        schedule = ()
    randomness = manifest.get("randomness")
    expected_manifest_keys = {
        "schema_version", "gate", "candidate_source", "candidate_commit",
        "binary_sha256", "quality_binary_sha256", "model_sha256",
        "sidecar_sha256", "tokenizer_sha256", "fixture_sha256", "randomness",
        "seed_sha256", "schedule", "memory_envelope_sha256",
        "memory_high_gib", "memory_max_gib", "kill_floor_gib",
        "artifact_sha256", "sidecar_stat_before",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != 1
        or manifest.get("gate") != "glm-rung0-slab"
        or schedule not in allowed_schedules
        or not isinstance(randomness, dict)
        or randomness.get("flip") not in {False, True}
        or manifest.get("seed_sha256") != randomness.get("seed_sha256")
        or schedule != arm_schedule(flip=randomness["flip"])
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("binary_sha256", ""))
        or not re.fullmatch(
            r"[0-9a-f]{64}", manifest.get("quality_binary_sha256", "")
        )
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("fixture_sha256", ""))
        or len(records) != 20
    ):
        raise ValueError("performance manifest identity or schedule is invalid")
    validate_confirmation_record(
        randomness,
        manifest["candidate_commit"],
        manifest["binary_sha256"],
        manifest["quality_binary_sha256"],
    )
    for record, (block, sequence, arm) in zip(records, schedule, strict=True):
        mode = "off" if arm == "A" else "on"
        if (
            (record.get("block"), record.get("sequence"), record.get("arm"))
            != (block, sequence, arm)
            or record.get("mode") != mode
            or record.get("binary_sha256") != manifest["binary_sha256"]
            or record.get("fixture_sha256") != manifest["fixture_sha256"]
            or record.get("configuration_sha256")
            != canonical_environment_sha256(canonical_engine_environment(mode))
        ):
            raise ValueError("performance raw arm differs from its manifest")


def canonical_engine_environment(mode: str) -> dict[str, str]:
    """Return the exact timed engine environment for one arm."""
    if mode not in {"off", "on"}:
        raise ValueError("mode must be off or on")
    result = {
        "DS4_CUDA_EXPERT_CACHE_GB": "68",
        "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
        "DS4_CUDA_FETCH_THREADS": "8",
        "DS4_CUDA_LOAD_PROFILE": "1",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_GLM_TP_DEBUG": "1",
        "DS4_LOCK_FILE": INSTANCE_LOCK,
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    if mode == "on":
        result.update(
            {
                "DS4_CUDA_EXPERT_SLAB_PATH": SLAB_PATH,
                "DS4_CUDA_EXPERT_SLAB_SHA256": SLAB_SHA256,
                "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256": MODEL_SHA256,
            }
        )
    return result


def memory_probe_environment() -> dict[str, str]:
    """Return the exact cache-off environment used to measure non-arena RSS."""
    return {
        "DS4_CUDA_EXPERT_CACHE_GB": "0",
        "DS4_CUDA_FETCH_THREADS": "8",
        "DS4_CUDA_LOAD_PROFILE": "1",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_GLM_TP_DEBUG": "1",
        "DS4_LOCK_FILE": INSTANCE_LOCK,
        "DS4_TOKEN_TIMING_LOG": "1",
    }


def peak_engine_rss_bytes(samples: str) -> int:
    """Read peak engine RSS from the wrapper's independent /proc sampler."""
    values = [
        int(match.group(1))
        for match in re.finditer(r"\beng_rss_kb=(\d+)\b", samples)
        if int(match.group(1)) > 0
    ]
    if len(values) < 2:
        raise ValueError("memory probe lacks repeated positive engine RSS samples")
    return max(values) * 1024


def measured_non_arena_peak_bytes(samples: str, main: str) -> int:
    """Include RSS, cgroup charges, and unified-memory loss from the host."""
    rss = peak_engine_rss_bytes(samples)
    cgroup_peaks = [
        int(match.group(1))
        for match in re.finditer(r"\bcgroup_peak_bytes=(\d+)\b", samples)
        if int(match.group(1)) > 0
    ]
    baseline_matches = re.findall(r"^MemAvailable:\s+(\d+) kB$", main, re.MULTILINE)
    available_samples = [
        int(value)
        for value in re.findall(r"\bmem_avail_kb=(\d+)\b", samples)
    ]
    if not baseline_matches or len(available_samples) < 2:
        raise ValueError("memory probe lacks whole-system baseline or samples")
    host_loss = (int(baseline_matches[0]) - min(available_samples)) * 1024
    if host_loss <= 0:
        raise ValueError("memory probe did not measure positive host memory use")
    return max([rss, host_loss, *cgroup_peaks])


def quality_command(
    binary: Path, model: Path, manifest: Path, output: Path
) -> list[Any]:
    """Build the existing official scorer invocation for the full fixture."""
    return [
        binary,
        model,
        manifest,
        output,
        "8192",
        "--ssd-streaming",
        "--ssd-streaming-cache-experts",
        "40GB",
    ]


def canonical_environment_sha256(environment: dict[str, str]) -> str:
    """Hash the exact engine environment as glm_safe_run observes it."""
    if not isinstance(environment, dict) or any(
        not isinstance(name, str)
        or name not in PROVENANCE_NAMES
        or not isinstance(value, str)
        for name, value in environment.items()
    ):
        raise ValueError("engine environment is outside the fixed allowlist")
    required = set(canonical_engine_environment("off"))
    if not required.issubset(environment):
        raise ValueError("engine environment lacks a required common setting")
    canonical = b"".join(
        name.encode("ascii")
        + b"="
        + environment.get(name, "<UNSET>").encode("utf-8")
        + b"\n"
        for name in PROVENANCE_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def observed_environment_sha256(environment: dict[str, str]) -> str:
    """Hash any exact allowlisted engine environment, including cache-off."""
    if not isinstance(environment, dict) or any(
        name not in PROVENANCE_NAMES or not isinstance(value, str)
        for name, value in environment.items()
    ):
        raise ValueError("engine environment is outside the fixed allowlist")
    canonical = b"".join(
        name.encode("ascii")
        + b"="
        + environment.get(name, "<UNSET>").encode("utf-8")
        + b"\n"
        for name in PROVENANCE_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def parse_sha_prefetch_engine_log(
    text: str, mode: str, *, model_generation: int
) -> dict[str, Any]:
    """Reduce only authenticated production log records into scorer input."""
    if mode not in {"off", "demand_sha", "prefetch_sha"}:
        raise ValueError("invalid SHA-prefetch log mode")
    if not isinstance(text, str) or model_generation <= 0:
        raise ValueError("invalid SHA-prefetch log input")
    auth_pattern = re.compile(
        r"^SLABAUTH mode=(demand_sha|prefetch_sha) generation=(\d+) "
        r"attempt=(\d+) key=(\d+) submit_ns=(\d+) complete_ns=(\d+) "
        r"payload_bytes=(\d+) ok=([01])$",
        re.MULTILINE,
    )
    auth = [
        {
            "mode": match[0], "generation": int(match[1]),
            "attempt": int(match[2]), "key": int(match[3]),
            "submit_ns": int(match[4]), "complete_ns": int(match[5]),
            "payload_bytes": int(match[6]), "ok": int(match[7]),
        }
        for match in auth_pattern.findall(text)
        if match[0] == mode
    ]
    identifiers = [(row["generation"], row["attempt"]) for row in auth]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate SHA-auth attempt")
    if any(
        row["generation"] != model_generation or row["ok"] != 1
        or row["payload_bytes"] != EXPERT_RECORD_PAYLOAD_BYTES
        or row["submit_ns"] <= 0 or row["complete_ns"] <= row["submit_ns"]
        for row in auth
    ):
        raise ValueError("failed or malformed SHA-auth attempt")

    load_pattern = re.compile(
        r"^LOADPROF L\d+ .*?fetch_ms=([0-9]+(?:\.[0-9]+)?) .*?"
        r"slab_mode=(on|off|error) slab_reads=(\d+) slab_bytes=(\d+) "
        r"slab_actual_bytes=(\d+) slab_peak_qd=(\d+) "
        r"slab_io_ms=([0-9]+(?:\.[0-9]+)?)"
        r"(?: slab_validation_ms=([0-9]+(?:\.[0-9]+)?) "
        r"slab_copy_ms=([0-9]+(?:\.[0-9]+)?))?$",
        re.MULTILINE,
    )
    loads = load_pattern.findall(text)
    if not loads:
        raise ValueError("SHA-prefetch log has no complete LOADPROF records")
    expected_slab_mode = "off" if mode == "off" else "on"
    if any(row[1] != expected_slab_mode for row in loads):
        raise ValueError("LOADPROF mode differs from requested arm")
    reads = sum(int(row[2]) for row in loads)
    logical_bytes = sum(int(row[3]) for row in loads)
    peak_qd = max(int(row[5]) for row in loads)
    if mode == "off":
        if auth or reads or logical_bytes or peak_qd:
            raise ValueError("off arm performed authenticated slab work")
        zero = {
            name: 0 for name in (
                "attempts", "sha_successes", "sha_failures", "ready", "late",
                "stale", "fallback", "copies", "validated_bytes", "copied_bytes",
                "publications", "read_ns", "sha_ns", "wait_ns", "copy_ns",
                "current_ready",
            )
        }
        return {
            "mode": mode, "model_generation": model_generation,
            "slab_reads": 0, "slab_peak_qd": 0,
            "completed_fetch_ms": [], "telemetry": zero,
        }
    completed_ms = [
        (row["complete_ns"] - row["submit_ns"]) / 1e6 for row in auth
    ]
    if len(auth) < 3:
        raise ValueError("arm lacks per-attempt SHA-auth coverage")

    if mode == "demand_sha":
        if any(not row[7] or not row[8] for row in loads):
            raise ValueError("demand SHA timing fields are absent")
        if reads != len(auth) or logical_bytes != len(auth) * EXPERT_RECORD_PAYLOAD_BYTES:
            raise ValueError("demand SHA attempts do not reconcile with LOADPROF")
        sha_ns = round(sum(float(row[7]) for row in loads) * 1e6)
        copy_ns = round(sum(float(row[8]) for row in loads) * 1e6)
        total_ns = sum(row["complete_ns"] - row["submit_ns"] for row in auth)
        telemetry = {
            "attempts": reads, "sha_successes": reads, "sha_failures": 0,
            "ready": 0, "late": 0, "stale": 0, "fallback": 0,
            "copies": reads, "validated_bytes": logical_bytes,
            "copied_bytes": logical_bytes, "publications": reads,
            "read_ns": max(1, total_ns - sha_ns), "sha_ns": sha_ns,
            "wait_ns": 0, "copy_ns": copy_ns, "current_ready": 0,
        }
    else:
        marker = re.compile(
            r"^PREFETCHSHA mode=prefetch_sha generation=(\d+) attempts=(\d+) "
            r"sha_successes=(\d+) sha_failures=(\d+) ready=(\d+) late=(\d+) "
            r"stale=(\d+) fallback=(\d+) copies=(\d+) validated_bytes=(\d+) "
            r"copied_bytes=(\d+) publications=(\d+) read_ns=(\d+) sha_ns=(\d+) "
            r"wait_ns=(\d+) copy_ns=(\d+) current_ready=(\d+) peak_qd=(\d+)$",
            re.MULTILINE,
        )
        markers = marker.findall(text)
        if not markers:
            raise ValueError("prefetch arm has no authoritative telemetry")
        values = [int(value) for value in markers[-1]]
        if values[0] != model_generation or values[1] != len(auth):
            raise ValueError("prefetch attempts do not match raw auth records")
        names = (
            "attempts", "sha_successes", "sha_failures", "ready", "late",
            "stale", "fallback", "copies", "validated_bytes", "copied_bytes",
            "publications", "read_ns", "sha_ns", "wait_ns", "copy_ns",
            "current_ready",
        )
        telemetry = dict(zip(names, values[1:17], strict=True))
        if (
            telemetry["sha_successes"] != len(auth)
            or telemetry["sha_failures"] != 0
            or telemetry["validated_bytes"]
            != len(auth) * EXPERT_RECORD_PAYLOAD_BYTES
        ):
            raise ValueError("prefetch telemetry differs from raw auth records")
        peak_qd = max(peak_qd, values[17])
    return {
        "mode": mode, "model_generation": model_generation,
        "slab_reads": reads, "slab_peak_qd": peak_qd,
        "completed_fetch_ms": completed_ms, "telemetry": telemetry,
    }


def normalize_sha_prefetch_reps(reps: Any) -> list[dict[str, Any]]:
    """Convert the existing independent speed harness output without invention."""
    if not isinstance(reps, list) or len(reps) != 2:
        raise ValueError("SHA-prefetch arm needs exactly two measured reps")
    normalized: list[dict[str, Any]] = []
    for phase, rep in zip(("cold", "warm"), reps, strict=True):
        if not isinstance(rep, dict) or rep.get("valid") is not True:
            raise ValueError("speed harness rep is invalid")
        raw = rep.get("token_timestamps_ns")
        client = rep.get("sse_token_timestamps_ns")
        tokens = rep.get("token_ids")
        count = rep.get("completion_tokens")
        request_started = rep.get("client_request_started_ns")
        first = rep.get("client_first_content_ns")
        last = rep.get("client_last_content_ns")
        if (
            not isinstance(count, int) or isinstance(count, bool) or count < 128
            or rep.get("server_completion_tokens") != count
            or not isinstance(tokens, list) or len(tokens) != count
            or not isinstance(raw, list) or len(raw) != count
            or not isinstance(client, list) or len(client) < 2
            or client[0] != first or client[-1] != last
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       for value in [*tokens, *raw, *client,
                                     request_started, first, last])
            or not request_started < first < last
            or any(right <= left for left, right in zip(raw, raw[1:]))
            or any(right <= left for left, right in zip(client, client[1:]))
        ):
            raise ValueError("speed harness token evidence is incomplete")
        reasoning = rep.get("generated_reasoning_sha256")
        content = rep.get("generated_content_sha256")
        request = rep.get("request_sha256")
        if any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None
               for value in (reasoning, content, request)):
            raise ValueError("speed harness byte identity is malformed")
        normalized.append({
            "phase": phase,
            "request_sha256": request,
            "generated_bytes_sha256": hashlib.sha256(
                f"{reasoning}:{content}".encode("ascii")
            ).hexdigest(),
            "token_ids": tokens,
            "completion_tokens": count,
            "raw_token_timestamps_ns": raw,
            "client_token_timestamps_ns": client,
            "client_request_started_ns": request_started,
            "client_first_token_ns": first,
            "client_last_token_ns": last,
            "ttft_seconds": (first - request_started) / 1e9,
        })
    return normalized


def parse_engine_log(text: str, mode: str) -> dict[str, Any]:
    """Reduce aggregate slab/cache telemetry without trusting its timings."""
    if mode not in {"off", "on"}:
        raise ValueError("mode must be off or on")
    if not isinstance(text, str):
        raise ValueError("engine log is not text")
    trace_lines = sum(line.startswith("SLABIO ") for line in text.splitlines())
    if trace_lines:
        raise ValueError("per-read slab trace contaminated a timed arm")
    if "ds4: expert-cache arena pin: ok" not in text:
        raise ValueError("pinned expert arena was not established")
    if mode == "on":
        model_marker = (
            "ds4: expert slab full-model identity verified via O_DIRECT "
            f"bytes={MODEL_BYTES}"
        )
        sidecar_marker = (
            "ds4: expert slab full-sidecar identity verified via O_DIRECT "
            f"bytes={SLAB_BYTES}"
        )
        enabled_marker = "ds4: CUDA contiguous expert slab enabled records=19456"
        if (
            text.count(model_marker) != 1
            or text.count(sidecar_marker) != 1
            or text.count(enabled_marker) != 1
            or not text.index(model_marker) < text.index(sidecar_marker) < text.index(enabled_marker)
        ):
            raise ValueError("ordered full-identity slab activation is absent")
    elif (
        "ds4: CUDA contiguous expert slab enabled" in text
        or "identity verified via O_DIRECT" in text
    ):
        raise ValueError("default-off arm emitted a slab activation marker")

    load_pattern = re.compile(
        r"^LOADPROF .*\bslab_mode=(on|off|error) "
        r"slab_reads=(\d+) .*\bslab_peak_qd=(\d+)\b",
        re.MULTILINE,
    )
    loads = load_pattern.findall(text)
    if not loads or any(resolved != mode for resolved, _, _ in loads):
        raise ValueError("per-load slab mode is absent or inconsistent")
    reads = sum(int(value) for _, value, _ in loads)
    peak = max(int(value) for _, _, value in loads)
    if mode == "on" and (reads <= 0 or peak < 2):
        raise ValueError("slab arm lacks positive concurrent reads")
    if mode == "off" and (reads != 0 or peak != 0):
        raise ValueError("default-off arm performed slab reads")

    window_pattern = re.compile(
        r"^ds4: expert-cache window tag=models-get lookup_bytes=(\d+) "
        r"hit_bytes=(\d+) stream_sha256=([0-9a-f]{64})$",
        re.MULTILINE,
    )
    windows = [match for match in window_pattern.findall(text) if int(match[0]) > 0]
    if not windows:
        raise ValueError("non-empty expert access-stream digest is absent")
    return {
        "slab_mode": mode,
        "slab_reads": reads,
        "slab_peak_qd": peak,
        "access_stream_sha256": windows[-1][2],
        "arena_pin_ok": True,
        "trace_lines": trace_lines,
    }


def parse_quality_engine_log(text: str, mode: str) -> dict[str, Any]:
    """Prove the official scorer actually exercised the requested slab arm."""
    if "ds4: expert-cache arena pin: ok" not in text:
        raise ValueError("quality arm did not establish the pinned arena")
    marker = "ds4: CUDA contiguous expert slab enabled records=19456"
    if mode == "on":
        model_marker = (
            "ds4: expert slab full-model identity verified via O_DIRECT "
            f"bytes={MODEL_BYTES}"
        )
        sidecar_marker = (
            "ds4: expert slab full-sidecar identity verified via O_DIRECT "
            f"bytes={SLAB_BYTES}"
        )
        if (
            text.count(model_marker) != 1
            or text.count(sidecar_marker) != 1
            or text.count(marker) != 1
            or not text.index(model_marker) < text.index(sidecar_marker) < text.index(marker)
        ):
            raise ValueError("quality arm lacks ordered full-identity activation")
    if mode == "off" and (
        "ds4: CUDA contiguous expert slab enabled" in text
        or "identity verified via O_DIRECT" in text
    ):
        raise ValueError("quality baseline emitted a slab activation marker")
    loads = re.findall(
        r"^LOADPROF .*\bslab_mode=(on|off|error) "
        r"slab_reads=(\d+) .*\bslab_peak_qd=(\d+)\b",
        text,
        re.MULTILINE,
    )
    if not loads or any(resolved != mode for resolved, _, _ in loads):
        raise ValueError("quality arm slab mode is absent or inconsistent")
    reads = sum(int(value) for _, value, _ in loads)
    peak = max(int(value) for _, _, value in loads)
    if mode == "on" and (reads <= 0 or peak < 2):
        raise ValueError("quality slab arm lacks concurrent reads")
    if mode == "off" and (reads or peak):
        raise ValueError("quality baseline performed slab reads")
    return {"slab_mode": mode, "slab_reads": reads, "slab_peak_qd": peak}


def parse_safety_logs(main: str, samples: str, kernel: str) -> dict[str, Any]:
    """Fail closed on containment, memory, process, or kernel evidence."""
    for marker in (
        "executed candidate was verified alive at least once; no identity "
        "contradiction observed by the periodic sampler; actual cadence is "
        "recorded in samples.log; wrapper and descendant checks clean",
        "SAFE_RUN end rc=0 killed=no",
        "cgroup_final ",
    ):
        if marker not in main:
            raise ValueError(f"safety log lacks {marker!r}")
    if "FATAL" in main:
        raise ValueError("safety wrapper reported a fatal condition")
    final = [line for line in main.splitlines() if "cgroup_final " in line]
    if len(final) != 1:
        raise ValueError("safety log lacks one final cgroup record")
    swap_match = re.search(r"swap_current_bytes=(\d+)", final[0])
    events = {
        name: int(value)
        for name, value in re.findall(
            r"\b(high|max|oom|oom_kill) (\d+)(?:,|$)", final[0]
        )
    }
    if swap_match is None or set(events) != {"high", "max", "oom", "oom_kill"}:
        raise ValueError("final cgroup counters are incomplete")
    if int(swap_match.group(1)) != 0 or any(events.values()):
        raise ValueError("cgroup memory or swap event invalidates the arm")
    memory_values: list[int] = []
    sample_swap: list[int] = []
    for line in samples.splitlines():
        memory = re.search(r"\bmem_avail_kb=(\d+)\b", line)
        swap = re.search(r"\bcgroup_swap_current_bytes=(\d+)\b", line)
        if memory is not None and swap is not None:
            memory_values.append(int(memory.group(1)))
            sample_swap.append(int(swap.group(1)))
    if len(memory_values) < 2 or any(sample_swap):
        raise ValueError("external memory samples are incomplete or swapped")
    minimum_available_gib = min(memory_values) / 1_048_576
    if minimum_available_gib < 10:
        raise ValueError("whole-system memory floor was violated")
    if re.search(
        r"NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm",
        kernel,
        re.IGNORECASE,
    ):
        raise ValueError("kernel OOM or Xid evidence invalidates the arm")
    return {
        "minimum_available_gib": minimum_available_gib,
        "cgroup_high_events": events["high"],
        "cgroup_max_events": events["max"],
        "cgroup_oom_events": events["oom"] + events["oom_kill"],
        "cgroup_swap_bytes": int(swap_match.group(1)),
        "xid": False,
        "survivors": [],
        "failures": [],
    }


def parse_slab_staging_telemetry(text: str) -> list[dict[str, int]]:
    """Validate the bounded pinned pool before a slab performance arm."""
    lowered = text.lower()
    if (
        "expert slab pinned staging allocation failed" in lowered
        or "expert slab pinned staging memory query failed" in lowered
        or "NV_ERR_NO_MEMORY" in text
    ):
        raise ValueError("slab pinned staging reported an allocation failure")
    matches = re.findall(
        r"^ds4: expert slab pinned staging ready count=(\d+) "
        r"buffer_bytes=(\d+) total_bytes=(\d+) "
        r"cuda_free_before=(\d+) cuda_free_after=(\d+) cuda_total=(\d+)$",
        text,
        re.MULTILINE,
    )
    pools: list[dict[str, int]] = []
    for fields in matches:
        count, buffer_bytes, total_bytes, free_before, free_after, cuda_total = (
            int(value) for value in fields
        )
        if (
            not 1 <= count <= 8
            or not 8 * 1024**2 <= buffer_bytes <= 16 * 1024**2
            or total_bytes != count * buffer_bytes
            or not 0 < free_before <= cuda_total
            or not 0 < free_after <= cuda_total
        ):
            raise ValueError("slab pinned staging geometry is invalid")
        pools.append(
            {
                "count": count,
                "buffer_bytes": buffer_bytes,
                "total_bytes": total_bytes,
                "cuda_free_before": free_before,
                "cuda_free_after": free_after,
                "cuda_total": cuda_total,
            }
        )
    if not pools:
        raise ValueError("slab pinned staging success telemetry is absent")
    return pools


def summarize_external_io(
    samples: list[tuple[int, int]],
    read_bytes_before: int,
    read_bytes_after: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Summarize externally observed block queue depth and completed reads."""
    if (
        not isinstance(samples, list)
        or len(samples) < 2
        or any(
            not isinstance(sample, tuple)
            or len(sample) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in sample)
            or sample[0] <= 0
            or sample[1] < 0
            for sample in samples
        )
        or any(right[0] <= left[0] for left, right in zip(samples, samples[1:]))
        or isinstance(read_bytes_before, bool)
        or not isinstance(read_bytes_before, int)
        or isinstance(read_bytes_after, bool)
        or not isinstance(read_bytes_after, int)
        or read_bytes_before < 0
        or read_bytes_after <= read_bytes_before
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds <= 0
    ):
        raise ValueError("external I/O samples are incomplete")
    return {
        "read_bytes_delta": read_bytes_after - read_bytes_before,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_read_qd": max(sample[1] for sample in samples),
        "sample_count": len(samples),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def proc_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split(") ", 1)
    if len(fields) != 2:
        raise RuntimeError("process start identity is malformed")
    value = int(fields[1].split()[19])
    if value <= 0:
        raise RuntimeError("process start identity is invalid")
    return value


def proc_read_bytes(pid: int) -> int:
    for line in Path(f"/proc/{pid}/io").read_text(encoding="ascii").splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split(":", 1)[1])
    raise RuntimeError("process completed read-byte counter is absent")


def read_qd() -> int:
    fields = INFLIGHT.read_text(encoding="ascii").split()
    if len(fields) != 2:
        raise RuntimeError("block inflight counter is malformed")
    value = int(fields[0])
    if value < 0:
        raise RuntimeError("block inflight counter is negative")
    return value


def terminate_exact(process: subprocess.Popen[Any], start_ticks: int) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        if proc_start_ticks(process.pid) != start_ticks:
            raise RuntimeError("server PID changed identity before termination")
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        if proc_start_ticks(process.pid) != start_ticks:
            raise RuntimeError("server PID changed identity before SIGKILL")
        process.kill()
        process.wait(timeout=15)


def matching_executable_pids(binary: Path) -> list[int]:
    identity = binary.stat()
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            observed = (entry / "exe").stat()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino):
            matches.append(int(entry.name))
    return matches


def wait_ready(process: subprocess.Popen[Any], port: int) -> None:
    deadline = time.monotonic() + 900
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError("server startup timed out")


def flush_expert_cache_window(port: int) -> None:
    """Request the engine's finalized access-stream counters."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=10
    ) as response:
        if response.status != 200:
            raise RuntimeError("final telemetry flush failed")
        response.read()


def execute_memory_probe_arm(args: argparse.Namespace) -> int:
    """Run one bounded completion for cache-off RSS or slab-on safety."""
    mode = getattr(args, "mode", "off")
    expected_environment = (
        memory_probe_environment()
        if mode == "off"
        else canonical_engine_environment("on")
    )
    observed_environment = {
        name: os.environ[name] for name in PROVENANCE_NAMES if name in os.environ
    }
    if observed_environment != expected_environment:
        raise ValueError("inherited single-request environment differs")
    binary = args.binary.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    if (
        binary.name != "ds4-server"
        or not str(binary.parent).startswith("/home/bmarti44/.cache/glm52-")
        or not binary.is_file()
        or sha256_file(binary) != args.binary_sha256
        or model != MODEL_PATH
        or not model.is_file()
        or out.exists()
        or not str(out).startswith("/home/bmarti44/.local/state/glm52-rung0-")
    ):
        raise ValueError("memory-probe artifact identity is invalid")
    out.mkdir(mode=0o700, parents=True)
    server_environment = {
        "HOME": "/home/bmarti44",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        **expected_environment,
    }
    command = [
        str(binary), "--cuda", "-m", str(model), "-c", "8192",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
    ]
    server: subprocess.Popen[Any] | None = None
    start_ticks: int | None = None
    with (out / "server.log").open("xb") as server_log:
        try:
            server = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=server_environment,
                start_new_session=False,
            )
            start_ticks = proc_start_ticks(server.pid)
            wait_ready(server, args.port)
            probe_result = out / "probe-result.json"
            request_timeout = 2700 if mode == "on" else 300
            completed = subprocess.run(
                [
                    str(ROOT / ".venv-harness/bin/python"),
                    str(BENCHMARK),
                    "--base-url", f"http://127.0.0.1:{args.port}",
                    "--out", str(probe_result),
                    "--stack-label", f"rung0-{mode}-single-request-probe",
                    "--model-id", "glm-5.2",
                    "--output-tokenizer-path", str(TOKENIZER),
                    "--output-tokenizer-sha256", TOKENIZER_SHA256,
                    "--token-timing-log", str(out / "server.log"),
                    "--reps", "1", "--warmup", "0", "--context-levels", "0",
                    "--request-timeout", str(request_timeout),
                    "--max-tokens", "160", "--min-completion-tokens", "128",
                    "--seed", "0",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=server_environment,
                timeout=request_timeout + 300,
                check=False,
            )
            (out / "probe.stdout.log").write_bytes(completed.stdout)
            (out / "probe.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError("cache-off memory probe benchmark failed")
            result = strict_json(probe_result)
            cells = result.get("cells")
            if (
                result.get("suite_valid") is not True
                or not isinstance(cells, list)
                or len(cells) != 1
                or not isinstance(cells[0].get("reps"), list)
                or len(cells[0]["reps"]) != 1
                or cells[0]["reps"][0].get("client_completion_tokens", 0) < 128
            ):
                raise RuntimeError("cache-off memory probe produced fewer than 128 tokens")
            flush_expert_cache_window(args.port)
            time.sleep(1)
            server_log.flush()
            os.fsync(server_log.fileno())
            slab_evidence: dict[str, Any] = {}
            if mode == "on":
                log_text = (out / "server.log").read_text(encoding="utf-8")
                slab_evidence = {
                    "engine": parse_engine_log(log_text, "on"),
                    "staging_pools": parse_slab_staging_telemetry(log_text),
                }
            write_json_exclusive(
                out / "partial.json",
                {
                    "schema_version": 1,
                    "binary_sha256": args.binary_sha256,
                    "mode": mode,
                    "probe_environment_sha256": observed_environment_sha256(
                        expected_environment
                    ),
                    **slab_evidence,
                },
            )
        finally:
            if server is not None and start_ticks is not None:
                terminate_exact(server, start_ticks)
    if matching_executable_pids(binary):
        raise RuntimeError("frozen engine survived memory-probe cleanup")
    return 0


def execute_quality_arm(args: argparse.Namespace) -> int:
    """Validate identities, then replace this process with the frozen scorer."""
    mode = "off" if args.arm == "A" else "on"
    expected_environment = canonical_engine_environment(mode)
    observed_environment = {
        name: os.environ[name] for name in PROVENANCE_NAMES if name in os.environ
    }
    binary = args.binary.resolve()
    fixture_root = args.fixture_root.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if (
        observed_environment != expected_environment
        or binary.name != "ds4-server"
        or not str(binary.parent).startswith("/home/bmarti44/.cache/glm52-")
        or not binary.is_file()
        or sha256_file(binary) != args.binary_sha256
        or not fixture_root.is_dir()
        or not str(fixture_root).startswith("/home/bmarti44/.cache/glm52-")
        or manifest != fixture_root / QUALITY_FIXTURE_RELATIVE
        or not manifest.is_file()
        or sha256_file(manifest) != args.manifest_sha256
        or output.exists()
        or not str(output).startswith("/home/bmarti44/.local/state/glm52-rung0-")
    ):
        raise ValueError("quality arm identity or environment is invalid")
    command = quality_command(binary, MODEL_PATH, manifest, output)
    environment = {
        "HOME": "/home/bmarti44",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        **expected_environment,
    }
    os.chdir(fixture_root)
    os.execve(binary, [os.fspath(part) for part in command], environment)
    raise RuntimeError("quality scorer exec unexpectedly returned")


def execute_arm(args: argparse.Namespace) -> int:
    """Run one fresh server; outer glm_safe_run owns containment and safety."""
    sha_prefetch = args.command == "sha-prefetch-arm"
    if sha_prefetch:
        mode = {"A": "off", "B": "demand_sha", "C": "prefetch_sha"}[args.arm]
        expected_environment = canonical_sha_prefetch_environment(mode)
    else:
        mode = "off" if args.arm == "A" else "on"
        expected_environment = canonical_engine_environment(mode)
    observed_environment = {
        name: os.environ[name] for name in PROVENANCE_NAMES if name in os.environ
    }
    if observed_environment != expected_environment:
        raise ValueError("inherited engine environment differs from fixed arm")
    binary = args.binary.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    if (
        not str(binary.parent).startswith("/home/bmarti44/.cache/glm52-")
        or binary.name != "ds4-server"
        or not binary.is_file()
        or sha256_file(binary) != args.binary_sha256
        or model != MODEL_PATH
        or not model.is_file()
        or out.exists()
        or not str(out).startswith("/home/bmarti44/.local/state/glm52-rung0-")
    ):
        raise ValueError("arm artifact or output identity is invalid")
    if sha256_file(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("GLM tokenizer identity mismatch")
    out.mkdir(mode=0o700, parents=True)
    server_log_path = out / "server.log"
    server_environment = {
        "HOME": "/home/bmarti44",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        **expected_environment,
    }
    command = [
        str(binary),
        "--cuda",
        "-m",
        str(model),
        "-c",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ssd-streaming",
        "--ssd-streaming-cache-experts",
        "40GB",
    ]
    server: subprocess.Popen[Any] | None = None
    server_start_ticks: int | None = None
    with server_log_path.open("xb") as server_log:
        try:
            server = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=server_environment,
                start_new_session=False,
            )
            start_ticks = proc_start_ticks(server.pid)
            server_start_ticks = start_ticks
            started = time.monotonic()
            wait_ready(server, args.port)
            ready_seconds = time.monotonic() - started
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            ).strip()
            server_instance_id = hashlib.sha256(
                f"{boot_id}:{server.pid}:{start_ticks}".encode("ascii")
            ).hexdigest()
            read_before = proc_read_bytes(server.pid)
            io_samples: list[tuple[int, int]] = []
            sampler_error: list[str] = []
            stop_sampler = threading.Event()

            def sample_io() -> None:
                while not stop_sampler.is_set():
                    try:
                        io_samples.append((time.monotonic_ns(), read_qd()))
                    except Exception as error:  # recorded and failed closed below
                        sampler_error.append(f"{type(error).__name__}: {error}")
                        return
                    stop_sampler.wait(0.002)

            sampler = threading.Thread(target=sample_io, daemon=True)
            sampler.start()
            probe_started = time.monotonic()
            completed = subprocess.run(
                [
                    str(ROOT / ".venv-harness/bin/python"),
                    str(ROOT / "scripts/30_bench_speed.py"),
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--out",
                    str(out / "result.json"),
                    "--stack-label",
                    f"rung0-slab-{mode}",
                    "--model-id",
                    "glm-5.2",
                    "--output-tokenizer-path",
                    str(TOKENIZER),
                    "--output-tokenizer-sha256",
                    TOKENIZER_SHA256,
                    "--token-timing-log",
                    str(server_log_path),
                    "--reps",
                    "2",
                    "--warmup", "1",
                    "--request-timeout", "2700",
                    "--context-levels",
                    "0",
                    "--max-tokens",
                    "160",
                    "--min-completion-tokens",
                    "128",
                    "--seed",
                    str(args.seed),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=server_environment,
                timeout=3000,
                check=False,
            )
            (out / "probe.stdout.log").write_bytes(completed.stdout)
            (out / "probe.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"existing speed scorer failed rc={completed.returncode}")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/v1/models", timeout=10
            ) as response:
                if response.status != 200:
                    raise RuntimeError("final telemetry flush failed")
                response.read()
            probe_elapsed = time.monotonic() - probe_started
            read_after = proc_read_bytes(server.pid)
            stop_sampler.set()
            sampler.join(timeout=5)
            if sampler.is_alive() or sampler_error:
                raise RuntimeError(f"external I/O sampler failed: {sampler_error}")
            with (out / "nvme-inflight.log").open("x", encoding="ascii") as stream:
                for timestamp_ns, qd in io_samples:
                    stream.write(f"{timestamp_ns} {qd}\n")
                stream.flush()
                os.fsync(stream.fileno())
            external_io = summarize_external_io(
                io_samples, read_before, read_after, probe_elapsed
            )
            result = strict_json(out / "result.json")
            cells = result.get("cells")
            if (
                result.get("suite_valid") is not True
                or not isinstance(cells, list)
                or len(cells) != 1
                or not isinstance(cells[0], dict)
                or cells[0].get("ctx_tokens") != 0
            ):
                raise ValueError("existing speed scorer result shape is invalid")
            server_log.flush()
            os.fsync(server_log.fileno())
            log_text = server_log_path.read_text(encoding="utf-8")
            if sha_prefetch:
                generic = parse_engine_log(
                    log_text, "off" if mode == "off" else "on"
                )
                engine = parse_sha_prefetch_engine_log(
                    log_text, mode, model_generation=args.model_generation
                )
                record = {
                    "schema_version": 1,
                    "block": args.block,
                    "sequence": args.sequence,
                    "arm": args.arm,
                    "mode": mode,
                    "server_instance_id": server_instance_id,
                    "candidate_commit": args.candidate_commit,
                    "binary_sha256": args.binary_sha256,
                    "configuration_sha256": canonical_environment_sha256(
                        expected_environment
                    ),
                    "fixture_sha256": sha256_file(FIXTURE),
                    "access_stream_sha256": generic["access_stream_sha256"],
                    "recorded_monotonic_ns": time.monotonic_ns(),
                    "reps": normalize_sha_prefetch_reps(cells[0].get("reps")),
                    "engine": engine,
                }
            else:
                engine = parse_engine_log(log_text, mode)
                record = {
                "schema_version": 1,
                "block": args.block,
                "sequence": args.sequence,
                "arm": args.arm,
                "mode": mode,
                "server_instance_id": server_instance_id,
                "binary_sha256": args.binary_sha256,
                "configuration_sha256": canonical_environment_sha256(
                    expected_environment
                ),
                "fixture_sha256": sha256_file(FIXTURE),
                "suite_valid": True,
                "reps": cells[0].get("reps"),
                "engine": engine,
                "external_io": external_io,
                "server_start_to_ready_seconds": ready_seconds,
                }
            write_json_exclusive(out / "partial.json", record)
        finally:
            if server is not None and server_start_ticks is not None:
                terminate_exact(server, server_start_ticks)
    if matching_executable_pids(binary):
        raise RuntimeError("frozen engine executable survived arm cleanup")
    return 0


def no_large_engines() -> None:
    completed = subprocess.run(
        ["/usr/bin/pgrep", "-x", "llama-server|ds4-server"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        raise RuntimeError("another large model engine is active")
    if completed.returncode not in {0, 1}:
        raise RuntimeError("cannot inspect active model engines")


def services_are_stopped() -> None:
    for unit in (
        "dsv4-guard.timer",
        "dsv4-guard.service",
        "deepseek-v4-flash-llamacpp.service",
    ):
        completed = subprocess.run(
            ["/usr/bin/systemctl", "is-active", unit],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.stdout.strip() not in {"inactive", "failed"}:
            raise RuntimeError(f"production unit is not stopped: {unit}")


def stable_start_memory(required_gib: float = 110.0) -> None:
    for _ in range(3):
        available = next(
            int(line.split()[1]) / 1_048_576
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
            if line.startswith("MemAvailable:")
        )
        if available < required_gib:
            raise RuntimeError(
                f"stable start memory is {available:.2f} GiB, below {required_gib:.2f}"
            )
        time.sleep(0.1)


def verify_global_lock_access() -> None:
    descriptor = os.open(GLOBAL_LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        details = os.fstat(descriptor)
        if not GLOBAL_LOCK.is_file() or details.st_nlink != 1:
            raise RuntimeError("global inference lock is not a stable regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def artifact_stat(path: Path) -> dict[str, Any]:
    details = path.stat()
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def verified_memory_envelope(
    path: Path,
    binary: Path,
    binary_sha256: str,
    quality_binary: Path,
    quality_binary_sha256: str,
    candidate_commit: str,
) -> dict[str, Any]:
    """Load a probe-bound envelope and recompute its fixed arithmetic."""
    envelope = strict_json(path)
    expected_keys = {
        "schema_version",
        "binary_sha256",
        "binary_stat",
        "quality_binary_sha256",
        "quality_binary_stat",
        "candidate_commit",
        "probe_environment_sha256",
        "probe_safety",
        "non_arena_peak_bytes",
        "arena_bytes",
        "margin_bytes",
        "memory_high_bytes",
        "memory_high_gib",
        "memory_max_gib",
        "host_total_bytes",
        "host_kill_floor_gib",
    }
    if (
        set(envelope) != expected_keys
        or envelope["schema_version"] != 1
        or envelope["binary_sha256"] != binary_sha256
        or envelope["binary_stat"] != artifact_stat(binary)
        or sha256_file(binary) != binary_sha256
        or envelope["quality_binary_sha256"] != quality_binary_sha256
        or envelope["quality_binary_stat"] != artifact_stat(quality_binary)
        or sha256_file(quality_binary) != quality_binary_sha256
        or envelope["candidate_commit"] != candidate_commit
        or not isinstance(envelope["probe_safety"], dict)
        or envelope["probe_safety"].get("failures") != []
        or not re.fullmatch(r"[0-9a-f]{64}", envelope["probe_environment_sha256"])
    ):
        raise ValueError("memory envelope identity or safety evidence is invalid")
    derived = derive_memory_envelope(
        envelope["non_arena_peak_bytes"], envelope["host_total_bytes"]
    )
    if any(envelope.get(name) != value for name, value in derived.items()):
        raise ValueError("memory envelope arithmetic differs from the fixed formula")
    return envelope


def run_memory_probe(args: argparse.Namespace) -> int:
    """Run one contained cache-off startup and bind its measured RSS."""
    candidate = args.candidate.resolve()
    binary = candidate / "ds4-server"
    quality_candidate = args.quality_candidate.resolve()
    quality_binary = quality_candidate / "ds4-server"
    out = Path(f"/home/bmarti44/.local/state/glm52-rung0-{args.tag}")
    if (
        out.exists()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", args.tag) is None
        or not str(candidate).startswith("/home/bmarti44/.cache/glm52-")
        or binary.name != "ds4-server"
        or not binary.is_file()
        or sha256_file(binary) != args.binary_sha256
        or not str(quality_candidate).startswith("/home/bmarti44/.cache/glm52-")
        or not quality_binary.is_file()
        or sha256_file(quality_binary) != args.quality_binary_sha256
        or re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.quality_binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{40}", args.candidate_commit) is None
        or not 1024 <= args.port <= 65535
    ):
        raise ValueError("memory-probe candidate identity is invalid")
    services_are_stopped()
    no_large_engines()
    stable_start_memory(110.0)
    verify_global_lock_access()
    out.mkdir(mode=0o700, parents=True)
    arm_out = out / "probe"
    crash_before = set(CRASH_ROOT.glob("*")) if CRASH_ROOT.exists() else set()
    probe_environment = memory_probe_environment()
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("DS4_") or name.startswith("GLM_"):
            del environment[name]
    environment.update(probe_environment)
    environment.update(
        {
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": args.binary_sha256,
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(PROVENANCE_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": observed_environment_sha256(
                probe_environment
            ),
            "GLM_SAFE_MEMORY_HIGH_GIB": "48",
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "1200",
        }
    )
    completed = subprocess.run(
        [
            str(CGROUP_RUNNER), "--tag", f"{args.tag}-rss", "--",
            sys.executable, str(Path(__file__).resolve()), "memory-probe-arm",
            "--out", str(arm_out), "--binary", str(binary),
            "--binary-sha256", args.binary_sha256,
            "--model", str(MODEL_PATH), "--port", str(args.port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=1300,
        check=False,
    )
    if arm_out.is_dir():
        (arm_out / "containment.stdout.log").write_bytes(completed.stdout)
        (arm_out / "containment.stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"contained memory probe failed rc={completed.returncode}")
    crash_after = set(CRASH_ROOT.glob("*"))
    matches = [
        path
        for path in crash_after - crash_before
        if path.name.endswith(f"-{args.tag}-rss")
    ]
    if len(matches) != 1:
        raise RuntimeError("memory probe lacks one safety evidence directory")
    for name in ("main.log", "samples.log", "kernel.log", "cmd.log"):
        source = matches[0] / name
        if not source.is_file():
            raise RuntimeError(f"memory probe lacks safety artifact {name}")
        shutil.copy2(source, arm_out / f"safety.{name}")
    main = (arm_out / "safety.main.log").read_text(encoding="utf-8")
    samples = (arm_out / "safety.samples.log").read_text(encoding="utf-8")
    kernel = (arm_out / "safety.kernel.log").read_text(encoding="utf-8")
    safety = parse_safety_logs(main, samples, kernel)
    non_arena_peak_bytes = measured_non_arena_peak_bytes(samples, main)
    host_total_bytes = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        if line.startswith("MemTotal:")
    )
    envelope = {
        "schema_version": 1,
        "binary_sha256": args.binary_sha256,
        "binary_stat": artifact_stat(binary),
        "quality_binary_sha256": args.quality_binary_sha256,
        "quality_binary_stat": artifact_stat(quality_binary),
        "candidate_commit": args.candidate_commit,
        "probe_environment_sha256": observed_environment_sha256(
            probe_environment
        ),
        "probe_safety": safety,
        **derive_memory_envelope(non_arena_peak_bytes, host_total_bytes),
    }
    write_json_exclusive(out / "memory-envelope.json", envelope)
    print(f"RUNG0_MEMORY_ENVELOPE out={out / 'memory-envelope.json'}")
    return 0


def run_quality_campaign(args: argparse.Namespace) -> int:
    """Run ABBA full-suite scorer arms and emit the fixed exact NLL artifact."""
    quality_candidate = args.quality_candidate.resolve()
    binary = quality_candidate / "ds4-server"
    server_candidate = args.server_candidate.resolve()
    server_binary = server_candidate / "ds4-server"
    fixture_root = args.fixture_root.resolve()
    manifest = args.manifest.resolve()
    out = Path(f"/home/bmarti44/.local/state/glm52-rung0-{args.tag}")
    if (
        out.exists()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", args.tag) is None
        or not str(quality_candidate).startswith("/home/bmarti44/.cache/glm52-")
        or not binary.is_file()
        or sha256_file(binary) != args.quality_binary_sha256
        or not str(server_candidate).startswith("/home/bmarti44/.cache/glm52-")
        or not server_binary.is_file()
        or sha256_file(server_binary) != args.server_binary_sha256
        or re.fullmatch(r"[0-9a-f]{64}", args.quality_binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.server_binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{40}", args.candidate_commit) is None
        or not fixture_root.is_dir()
        or not str(fixture_root).startswith("/home/bmarti44/.cache/glm52-")
        or manifest != fixture_root / QUALITY_FIXTURE_RELATIVE
        or not manifest.is_file()
        or not args.drand_json.is_file()
    ):
        raise ValueError("quality campaign identity is invalid")
    fixture_ids = fixture_manifest_case_ids(manifest)
    fixture_content_sha256 = content_complete_fixture_sha256(
        fixture_root, [manifest]
    )
    if fixture_content_sha256 != QUALITY_FIXTURE_CONTENT_SHA256:
        raise ValueError("official quality fixture content hash mismatch")
    model_stat_before = artifact_stat(MODEL_PATH)
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("quality model content hash mismatch before execution")
    envelope = verified_memory_envelope(
        args.memory_envelope.resolve(),
        server_binary,
        args.server_binary_sha256,
        binary,
        args.quality_binary_sha256,
        args.candidate_commit,
    )
    confirmation = authenticate_confirmation(
        args.drand_json.resolve(), args.candidate_commit,
        args.server_binary_sha256, args.quality_binary_sha256,
        max(
            args.memory_envelope.resolve().stat().st_mtime,
            server_binary.stat().st_mtime,
            binary.stat().st_mtime,
        ),
    )
    schedule = quality_schedule(flip=confirmation["flip"])
    memory_high_gib = envelope["memory_high_gib"]
    services_are_stopped()
    no_large_engines()
    stable_start_memory(max(110.0, memory_high_gib + 20.0))
    verify_global_lock_access()
    out.mkdir(mode=0o700, parents=True)
    manifest_sha256 = sha256_file(manifest)
    attempts: list[dict[str, Any]] = []
    raw_stream = (out / "quality-raw.jsonl").open("x", encoding="utf-8")
    try:
        for index, arm in enumerate(schedule):
            services_are_stopped()
            no_large_engines()
            stable_start_memory(max(110.0, memory_high_gib + 20.0))
            mode = "off" if arm == "A" else "on"
            safe_timeout = quality_timeout_seconds(mode)
            label = f"quality-{index:02d}-{arm.lower()}"
            result_path = out / f"{label}.tsv"
            crash_before = set(CRASH_ROOT.glob("*")) if CRASH_ROOT.exists() else set()
            engine_environment = canonical_engine_environment(mode)
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith("DS4_") or name.startswith("GLM_"):
                    del environment[name]
            environment.update(engine_environment)
            environment.update(
                {
                    "GLM_CANDIDATE_SRC": str(quality_candidate),
                    "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
                    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
                    "GLM_SAFE_EXPECTED_BINARY_SHA256": args.quality_binary_sha256,
                    "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(PROVENANCE_NAMES),
                    "GLM_SAFE_EXPECTED_ENV_SHA256": canonical_environment_sha256(
                        engine_environment
                    ),
                    "GLM_SAFE_MEMORY_HIGH_GIB": str(memory_high_gib),
                    "GLM_SAFE_KILL_FLOOR_GIB": str(HOST_KILL_FLOOR_GIB),
                    "GLM_SAFE_MIN_START_GIB": "110",
                    "GLM_SAFE_TIMEOUT_S": str(safe_timeout),
                }
            )
            completed = subprocess.run(
                [
                    str(CGROUP_RUNNER), "--tag", label, "--",
                    sys.executable, str(Path(__file__).resolve()), "quality-arm",
                    "--arm", arm, "--binary", str(binary),
                    "--binary-sha256", args.quality_binary_sha256,
                    "--manifest", str(manifest),
                    "--manifest-sha256", manifest_sha256,
                    "--fixture-root", str(fixture_root),
                    "--output", str(result_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=safe_timeout + 100,
                check=False,
            )
            (out / f"{label}.stdout.log").write_bytes(completed.stdout)
            (out / f"{label}.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"contained quality arm failed rc={completed.returncode}")
            crash_after = set(CRASH_ROOT.glob("*"))
            matches = [
                path for path in crash_after - crash_before if path.name.endswith(f"-{label}")
            ]
            if len(matches) != 1:
                raise RuntimeError(f"quality arm {label} lacks one safety directory")
            safety_files: dict[str, str] = {}
            for name in ("main.log", "samples.log", "kernel.log", "cmd.log"):
                source = matches[0] / name
                if not source.is_file():
                    raise RuntimeError(f"quality arm {label} lacks {name}")
                destination = out / f"{label}.safety.{name}"
                shutil.copy2(source, destination)
                safety_files[name] = destination.read_text(encoding="utf-8")
            safety = parse_safety_logs(
                safety_files["main.log"],
                safety_files["samples.log"],
                safety_files["kernel.log"],
            )
            engine = parse_quality_engine_log(
                safety_files["cmd.log"],
                mode,
            )
            rows = parse_quality_tsv(result_path)
            attempt = {
                "arm": arm,
                "mode": mode,
                "rows": rows,
                "output_sha256": sha256_file(result_path),
                "configuration_sha256": canonical_environment_sha256(
                    engine_environment
                ),
                "engine": engine,
                "safety": safety,
            }
            attempts.append(attempt)
            raw_stream.write(
                json.dumps(attempt, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
            no_large_engines()
    finally:
        raw_stream.close()
    model_stat_after = artifact_stat(MODEL_PATH)
    fixture_content_after = content_complete_fixture_sha256(
        fixture_root, [manifest]
    )
    if (
        model_stat_after != model_stat_before
        or sha256_file(MODEL_PATH) != MODEL_SHA256
        or fixture_content_after != fixture_content_sha256
    ):
        raise RuntimeError("quality model or fixture changed during the campaign")
    result = validate_quality_attempts(attempts, fixture_ids)
    write_json_exclusive(out / "nll.json", result)
    quality_manifest = {
        "schema_version": 1,
        "candidate_commit": args.candidate_commit,
        "binary_sha256": args.server_binary_sha256,
        "quality_binary_sha256": args.quality_binary_sha256,
        "model_sha256": MODEL_SHA256,
        "model_stat_before": model_stat_before,
        "model_stat_after": model_stat_after,
        "fixture_sha256": manifest_sha256,
        "fixture_content_sha256": fixture_content_sha256,
        "fixture_content_sha256_after": fixture_content_after,
        "ordered_case_ids": fixture_ids,
        "memory_envelope_sha256": sha256_file(args.memory_envelope.resolve()),
        "quality_raw_sha256": sha256_file(out / "quality-raw.jsonl"),
        "nll_sha256": sha256_file(out / "nll.json"),
        "schedule": list(schedule),
        "randomness": confirmation,
    }
    write_json_exclusive(out / "quality-manifest.json", quality_manifest)
    print(f"RUNG0_QUALITY_DONE out={out / 'nll.json'}")
    return 0


def run_campaign(args: argparse.Namespace) -> int:
    candidate = args.candidate.resolve()
    binary = candidate / "ds4-server"
    quality_candidate = args.quality_candidate.resolve()
    quality_binary = quality_candidate / "ds4-server"
    out = Path(f"/home/bmarti44/.local/state/glm52-rung0-{args.tag}")
    if (
        out.exists()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", args.tag) is None
        or not candidate.is_dir()
        or not str(candidate).startswith("/home/bmarti44/.cache/glm52-")
        or not binary.is_file()
        or sha256_file(binary) != args.binary_sha256
        or not str(quality_candidate).startswith("/home/bmarti44/.cache/glm52-")
        or not quality_binary.is_file()
        or sha256_file(quality_binary) != args.quality_binary_sha256
        or re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.quality_binary_sha256) is None
        or re.fullmatch(r"[0-9a-f]{40}", args.candidate_commit) is None
        or not args.drand_json.is_file()
        or not 1024 <= args.port <= 65535
    ):
        raise ValueError("campaign identity or bounded configuration is invalid")
    envelope = verified_memory_envelope(
        args.memory_envelope.resolve(), binary, args.binary_sha256,
        quality_binary, args.quality_binary_sha256, args.candidate_commit,
    )
    confirmation = authenticate_confirmation(
        args.drand_json.resolve(), args.candidate_commit,
        args.binary_sha256, args.quality_binary_sha256,
        max(
            args.memory_envelope.resolve().stat().st_mtime,
            binary.stat().st_mtime,
            quality_binary.stat().st_mtime,
        ),
    )
    schedule = arm_schedule(flip=confirmation["flip"])
    memory_high_gib = envelope["memory_high_gib"]
    services_are_stopped()
    no_large_engines()
    stable_start_memory(max(110.0, memory_high_gib + 20.0))
    verify_global_lock_access()
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("full mapped model identity mismatch")
    slab = Path(SLAB_PATH)
    sidecar_before = artifact_stat(slab)
    if sha256_file(slab) != SLAB_SHA256:
        raise ValueError("full expert sidecar identity mismatch")
    out.mkdir(mode=0o700, parents=True)
    arms_root = out / "arms"
    arms_root.mkdir(mode=0o700)
    seed = int(confirmation["seed_sha256"][:8], 16)
    manifest = {
        "schema_version": 1,
        "gate": "glm-rung0-slab",
        "candidate_source": str(candidate),
        "candidate_commit": args.candidate_commit,
        "binary_sha256": args.binary_sha256,
        "quality_binary_sha256": args.quality_binary_sha256,
        "model_sha256": MODEL_SHA256,
        "sidecar_sha256": SLAB_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "fixture_sha256": sha256_file(FIXTURE),
        "randomness": confirmation,
        "seed_sha256": confirmation["seed_sha256"],
        "schedule": [list(row) for row in schedule],
        "memory_envelope_sha256": sha256_file(args.memory_envelope.resolve()),
        "memory_high_gib": memory_high_gib,
        "memory_max_gib": memory_high_gib + MEMORY_MAX_EXCURSION_GIB,
        "kill_floor_gib": 18,
        "artifact_sha256": {
            str(BENCHMARK.relative_to(ROOT)): sha256_file(BENCHMARK),
            str(CGROUP_RUNNER.relative_to(ROOT)): sha256_file(CGROUP_RUNNER),
            "results/glm52-gates/harness/glm_safe_run.sh": sha256_file(
                ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
            ),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(
                Path(__file__).resolve()
            ),
        },
        "sidecar_stat_before": sidecar_before,
    }
    write_json_exclusive(out / "manifest.json", manifest)
    raw_path = out / "raw.jsonl"
    raw_stream = raw_path.open("x", encoding="utf-8")
    try:
        for block, sequence, arm in schedule:
            services_are_stopped()
            no_large_engines()
            stable_start_memory(max(110.0, memory_high_gib + 20.0))
            mode = "off" if arm == "A" else "on"
            safe_timeout = safe_timeout_seconds(mode)
            label = f"r0-b{block}s{sequence}{arm.lower()}"
            arm_out = arms_root / label
            crash_before = set(CRASH_ROOT.glob("*")) if CRASH_ROOT.exists() else set()
            engine_environment = canonical_engine_environment(mode)
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith("DS4_") or name.startswith("GLM_"):
                    del environment[name]
            environment.update(engine_environment)
            environment.update(
                {
                    "GLM_CANDIDATE_SRC": str(candidate),
                    "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
                    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
                    "GLM_SAFE_EXPECTED_BINARY_SHA256": args.binary_sha256,
                    "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(
                        PROVENANCE_NAMES
                    ),
                    "GLM_SAFE_EXPECTED_ENV_SHA256": canonical_environment_sha256(
                        engine_environment
                    ),
                    "GLM_SAFE_MEMORY_HIGH_GIB": str(memory_high_gib),
                    "GLM_SAFE_KILL_FLOOR_GIB": "18",
                    "GLM_SAFE_MIN_START_GIB": "110",
                    "GLM_SAFE_TIMEOUT_S": str(safe_timeout),
                }
            )
            completed = subprocess.run(
                [
                    str(CGROUP_RUNNER),
                    "--tag",
                    label,
                    "--",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "arm",
                    "--out",
                    str(arm_out),
                    "--block",
                    str(block),
                    "--sequence",
                    str(sequence),
                    "--arm",
                    arm,
                    "--binary",
                    str(binary),
                    "--binary-sha256",
                    args.binary_sha256,
                    "--model",
                    str(MODEL_PATH),
                    "--port",
                    str(args.port),
                    "--seed",
                    str(seed),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=safe_timeout + 100,
                check=False,
            )
            if arm_out.is_dir():
                (arm_out / "containment.stdout.log").write_bytes(completed.stdout)
                (arm_out / "containment.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"contained arm {label} failed rc={completed.returncode}"
                )
            crash_after = set(CRASH_ROOT.glob("*"))
            crash_matches = [
                path for path in crash_after - crash_before if path.name.endswith(f"-{label}")
            ]
            if len(crash_matches) != 1:
                raise RuntimeError(f"arm {label} lacks one safety evidence directory")
            crash = crash_matches[0]
            for name in ("main.log", "samples.log", "kernel.log", "cmd.log"):
                source = crash / name
                if not source.is_file():
                    raise RuntimeError(f"arm {label} lacks safety artifact {name}")
                shutil.copy2(source, arm_out / f"safety.{name}")
            partial = strict_json(arm_out / "partial.json")
            lifecycle = partial.pop("server_start_to_ready_seconds", None)
            write_json_exclusive(
                arm_out / "lifecycle.json",
                {"server_start_to_ready_seconds": lifecycle},
            )
            partial["safety"] = parse_safety_logs(
                (arm_out / "safety.main.log").read_text(encoding="utf-8"),
                (arm_out / "safety.samples.log").read_text(encoding="utf-8"),
                (arm_out / "safety.kernel.log").read_text(encoding="utf-8"),
            )
            write_json_exclusive(arm_out / "record.json", partial)
            raw_stream.write(
                json.dumps(
                    partial, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
            no_large_engines()
    finally:
        raw_stream.close()
    sidecar_after = artifact_stat(slab)
    if sidecar_after != sidecar_before or sha256_file(slab) != SLAB_SHA256:
        raise RuntimeError("expert sidecar changed during campaign")
    write_json_exclusive(
        out / "performance-stage.json",
        {
            "status": "COMPLETE_PENDING_NLL",
            "arm_count": 20,
            "sidecar_stat_after": sidecar_after,
            "manifest_sha256": sha256_file(out / "manifest.json"),
            "raw_sha256": sha256_file(raw_path),
        },
    )
    print(f"RUNG0_SLAB_PERF_DONE_PENDING_NLL out={out}")
    return 0


def score_directory(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    performance_manifest = strict_json(campaign / "manifest.json")
    performance_stage = strict_json(campaign / "performance-stage.json")
    performance_raw = campaign / "raw.jsonl"
    if (
        set(performance_stage)
        != {
            "status", "arm_count", "sidecar_stat_after",
            "manifest_sha256", "raw_sha256",
        }
        or performance_stage.get("status") != "COMPLETE_PENDING_NLL"
        or performance_stage.get("arm_count") != 20
        or performance_stage.get("manifest_sha256")
        != sha256_file(campaign / "manifest.json")
        or performance_stage.get("raw_sha256") != sha256_file(performance_raw)
    ):
        raise ValueError("performance raw or manifest differs from its stage receipt")
    records = [
        json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
        for line in performance_raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_performance_binding(performance_manifest, records)
    quality = args.quality_campaign.resolve()
    quality_manifest = strict_json(quality / "quality-manifest.json")
    quality_raw = quality / "quality-raw.jsonl"
    quality_nll = quality / "nll.json"
    if (
        quality_manifest.get("quality_raw_sha256") != sha256_file(quality_raw)
        or quality_manifest.get("nll_sha256") != sha256_file(quality_nll)
    ):
        raise ValueError("quality raw or NLL hash differs from its manifest")
    attempts = [
        json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
        for line in quality_raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bound_quality_schedule = tuple(quality_manifest.get("schedule", ()))
    for index, (attempt, arm) in enumerate(
        zip(attempts, bound_quality_schedule, strict=True)
    ):
        label = f"quality-{index:02d}-{arm.lower()}"
        result_path = quality / f"{label}.tsv"
        safety_paths = {
            name: quality / f"{label}.safety.{name}"
            for name in ("main.log", "samples.log", "kernel.log", "cmd.log")
        }
        if (
            attempt.get("output_sha256") != sha256_file(result_path)
            or attempt.get("rows") != parse_quality_tsv(result_path)
        ):
            raise ValueError("quality raw differs from its official scorer TSV")
        safety = parse_safety_logs(
            safety_paths["main.log"].read_text(encoding="utf-8"),
            safety_paths["samples.log"].read_text(encoding="utf-8"),
            safety_paths["kernel.log"].read_text(encoding="utf-8"),
        )
        engine = parse_quality_engine_log(
            safety_paths["cmd.log"].read_text(encoding="utf-8"),
            "off" if arm == "A" else "on",
        )
        if attempt.get("safety") != safety or attempt.get("engine") != engine:
            raise ValueError("quality raw differs from its external safety logs")
    nll = validate_bound_quality_evidence(
        performance_manifest, quality_manifest, attempts
    )
    if strict_json(quality_nll) != nll:
        raise ValueError("stored NLL differs from raw quality derivation")
    summary = score_campaign(
        records,
        nll,
        quality_bound=True,
        schedule_flip=performance_manifest["randomness"]["flip"],
    )
    write_json_exclusive(campaign / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--out", type=Path, required=True)
    arm.add_argument("--block", type=int, required=True)
    arm.add_argument("--sequence", type=int, required=True)
    arm.add_argument("--arm", choices=("A", "B"), required=True)
    arm.add_argument("--binary", type=Path, required=True)
    arm.add_argument("--binary-sha256", required=True)
    arm.add_argument("--model", type=Path, required=True)
    arm.add_argument("--port", type=int, required=True)
    arm.add_argument("--seed", type=int, required=True)
    sha_arm = subparsers.add_parser("sha-prefetch-arm")
    sha_arm.add_argument("--out", type=Path, required=True)
    sha_arm.add_argument("--block", type=int, required=True)
    sha_arm.add_argument("--sequence", type=int, required=True)
    sha_arm.add_argument("--arm", choices=("A", "B", "C"), required=True)
    sha_arm.add_argument("--binary", type=Path, required=True)
    sha_arm.add_argument("--binary-sha256", required=True)
    sha_arm.add_argument("--candidate-commit", required=True)
    sha_arm.add_argument("--model-generation", type=int, required=True)
    sha_arm.add_argument("--model", type=Path, required=True)
    sha_arm.add_argument("--port", type=int, required=True)
    sha_arm.add_argument("--seed", type=int, required=True)
    probe_arm = subparsers.add_parser("memory-probe-arm")
    probe_arm.add_argument("--out", type=Path, required=True)
    probe_arm.add_argument("--binary", type=Path, required=True)
    probe_arm.add_argument("--binary-sha256", required=True)
    probe_arm.add_argument("--model", type=Path, required=True)
    probe_arm.add_argument("--port", type=int, required=True)
    probe_arm.add_argument("--mode", choices=("off", "on"), default="off")
    probe = subparsers.add_parser("memory-probe")
    probe.add_argument("--tag", required=True)
    probe.add_argument("--candidate", type=Path, required=True)
    probe.add_argument("--quality-candidate", type=Path, required=True)
    probe.add_argument("--candidate-commit", required=True)
    probe.add_argument("--binary-sha256", required=True)
    probe.add_argument("--quality-binary-sha256", required=True)
    probe.add_argument("--port", type=int, default=8032)
    quality_arm = subparsers.add_parser("quality-arm")
    quality_arm.add_argument("--arm", choices=("A", "B"), required=True)
    quality_arm.add_argument("--binary", type=Path, required=True)
    quality_arm.add_argument("--binary-sha256", required=True)
    quality_arm.add_argument("--manifest", type=Path, required=True)
    quality_arm.add_argument("--manifest-sha256", required=True)
    quality_arm.add_argument("--fixture-root", type=Path, required=True)
    quality_arm.add_argument("--output", type=Path, required=True)
    quality = subparsers.add_parser("quality")
    quality.add_argument("--tag", required=True)
    quality.add_argument("--quality-candidate", type=Path, required=True)
    quality.add_argument("--server-candidate", type=Path, required=True)
    quality.add_argument("--quality-binary-sha256", required=True)
    quality.add_argument("--server-binary-sha256", required=True)
    quality.add_argument("--candidate-commit", required=True)
    quality.add_argument("--fixture-root", type=Path, required=True)
    quality.add_argument("--manifest", type=Path, required=True)
    quality.add_argument("--memory-envelope", type=Path, required=True)
    quality.add_argument("--drand-json", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--tag", required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--quality-candidate", type=Path, required=True)
    run.add_argument("--candidate-commit", required=True)
    run.add_argument("--binary-sha256", required=True)
    run.add_argument("--quality-binary-sha256", required=True)
    run.add_argument("--drand-json", type=Path, required=True)
    run.add_argument("--memory-envelope", type=Path, required=True)
    run.add_argument("--port", type=int, default=8032)
    score = subparsers.add_parser("score")
    score.add_argument("--campaign", type=Path, required=True)
    score.add_argument("--quality-campaign", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    try:
        if args.command in {"arm", "sha-prefetch-arm"}:
            return execute_arm(args)
        if args.command == "memory-probe-arm":
            return execute_memory_probe_arm(args)
        if args.command == "memory-probe":
            return run_memory_probe(args)
        if args.command == "quality-arm":
            return execute_quality_arm(args)
        if args.command == "quality":
            return run_quality_campaign(args)
        if args.command == "run":
            return run_campaign(args)
        if args.command == "score":
            return score_directory(args)
        raise ValueError("unknown command")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"70_glm_rung0_slab_campaign.py: {error}", file=sys.stderr)
        return 1


def score_campaign(
    records: list[dict[str, Any]],
    nll: dict[str, Any],
    *,
    quality_bound: bool = False,
    schedule_flip: bool = False,
) -> dict[str, Any]:
    """Validate raw arms and apply the fixed Rung 0.1 formulas."""
    if quality_bound is not True or not isinstance(schedule_flip, bool):
        raise ValueError("quality evidence was not bound by the authoritative scorer")
    expected_keys = {
        "schema_version",
        "block",
        "sequence",
        "arm",
        "mode",
        "server_instance_id",
        "binary_sha256",
        "configuration_sha256",
        "fixture_sha256",
        "suite_valid",
        "reps",
        "engine",
        "external_io",
        "safety",
    }
    if len(records) != 20:
        raise ValueError("campaign requires exactly 20 arms")

    def sha256(value: Any, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} is not a lowercase SHA-256")
        return value

    def positive(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} is not numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{label} is not positive and finite")
        return result

    validation_rows = []
    binaries: set[str] = set()
    configurations: dict[str, set[str]] = {"off": set(), "on": set()}
    output_signatures: dict[int, set[tuple[Any, ...]]] = {0: set(), 1: set()}
    access_streams: set[str] = set()
    per_row: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    io_throughput: dict[str, list[float]] = {"off": [], "on": []}

    for index, record in enumerate(records):
        if set(record) != expected_keys or record["schema_version"] != 1:
            raise ValueError(f"arm {index} has an invalid schema")
        mode = record["mode"]
        arm = record["arm"]
        if (arm, mode) not in {("A", "off"), ("B", "on")}:
            raise ValueError("arm-to-mode mapping is invalid")
        binary = sha256(record["binary_sha256"], "binary_sha256")
        configuration = sha256(
            record["configuration_sha256"], "configuration_sha256"
        )
        fixture = sha256(record["fixture_sha256"], "fixture_sha256")
        binaries.add(binary)
        configurations[mode].add(configuration)
        validation_rows.append(
            {
                "block": record["block"],
                "sequence": record["sequence"],
                "arm": arm,
                "server_boot_id": record["server_instance_id"],
                "fixture_sha256": fixture,
                "binary_sha256": binary,
                "configuration_sha256": configuration,
            }
        )
        if record["suite_valid"] is not True:
            raise ValueError("speed suite is invalid")
        reps = record["reps"]
        if not isinstance(reps, list) or len(reps) != 2:
            raise ValueError("each arm requires exactly two measured reps")
        decode_rates: list[float] = []
        raw_decode_rates: list[float] = []
        ttfts: list[float] = []
        prompt_rates: list[float] = []
        for rep_index, rep in enumerate(reps):
            if not isinstance(rep, dict) or rep.get("valid") is not True:
                raise ValueError("measured rep is invalid")
            raw_timestamps = rep.get("token_timestamps_ns")
            raw_token_count = rep.get("completion_tokens")
            server_token_count = rep.get("server_completion_tokens")
            token_ids = rep.get("token_ids")
            if (
                rep.get("timing_source") != "server_raw_token_log"
                or not isinstance(raw_timestamps, list)
                or len(raw_timestamps) < 128
                or isinstance(raw_token_count, bool)
                or not isinstance(raw_token_count, int)
                or raw_token_count != server_token_count
                or raw_token_count != len(raw_timestamps)
                or not isinstance(token_ids, list)
                or len(token_ids) != raw_token_count
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_timestamps
                )
                or any(
                    right <= left
                    for left, right in zip(raw_timestamps, raw_timestamps[1:])
                )
            ):
                raise ValueError("raw generated-token timing is incomplete")

            client_timestamps = rep.get("sse_token_timestamps_ns")
            request_started = rep.get("client_request_started_ns")
            first_content = rep.get("client_first_content_ns")
            last_content = rep.get("client_last_content_ns")
            event_count = rep.get("event_completion_tokens")
            client_token_count = rep.get("client_completion_tokens")
            if (
                not isinstance(client_timestamps, list)
                or len(client_timestamps) < 2
                or isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count != len(client_timestamps)
                or isinstance(client_token_count, bool)
                or not isinstance(client_token_count, int)
                or client_token_count <= 0
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in client_timestamps
                )
                or any(
                    right <= left
                    for left, right in zip(client_timestamps, client_timestamps[1:])
                )
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (request_started, first_content, last_content)
                )
                or not request_started < first_content < last_content
                or client_timestamps[0] != first_content
                or client_timestamps[-1] != last_content
            ):
                raise ValueError("independent client timing envelope is incomplete")

            elapsed = (last_content - first_content) / 1_000_000_000
            raw_elapsed = (raw_timestamps[-1] - raw_timestamps[0]) / 1_000_000_000
            observed_ratio = positive(
                rep.get("raw_client_timing_ratio"), "raw/client timing ratio"
            )
            recomputed_ratio = positive(raw_elapsed, "raw decode elapsed") / positive(
                elapsed, "client decode elapsed"
            )
            if (
                not math.isclose(
                    observed_ratio, recomputed_ratio, rel_tol=1e-9, abs_tol=1e-12
                )
                or not 0.75 <= recomputed_ratio <= 1.25
            ):
                raise ValueError("raw/client timing envelope is inconsistent")
            decode_rates.append(
                (raw_token_count - 1) / positive(elapsed, "client decode elapsed")
            )
            raw_decode_rates.append(
                (raw_token_count - 1)
                / positive(raw_elapsed, "raw token decode elapsed")
            )
            ttft = (first_content - request_started) / 1_000_000_000
            reported_ttft = positive(rep.get("ttft_s"), "reported TTFT")
            if not math.isclose(ttft, reported_ttft, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("reported TTFT does not match client endpoints")
            prompt_tokens = rep.get("client_prompt_tokens")
            if (
                isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0
            ):
                raise ValueError("client prompt token count is invalid")
            ttfts.append(ttft)
            prompt_rates.append(prompt_tokens / ttft)
            signature = (
                sha256(rep.get("request_sha256"), "request_sha256"),
                sha256(
                    rep.get("generated_reasoning_sha256"),
                    "generated_reasoning_sha256",
                ),
                sha256(
                    rep.get("generated_content_sha256"),
                    "generated_content_sha256",
                ),
                raw_token_count,
                tuple(token_ids),
            )
            output_signatures[rep_index].add(signature)
        per_row[(record["block"], record["sequence"])] = (
            statistics.fmean(decode_rates),
            statistics.fmean(raw_decode_rates),
            statistics.fmean(ttfts),
            statistics.fmean(prompt_rates),
        )

        engine = record["engine"]
        if not isinstance(engine, dict) or engine.get("slab_mode") != mode:
            raise ValueError("resolved slab mode is invalid")
        reads = engine.get("slab_reads")
        peak_qd = engine.get("slab_peak_qd")
        if (
            isinstance(reads, bool)
            or not isinstance(reads, int)
            or isinstance(peak_qd, bool)
            or not isinstance(peak_qd, int)
            or reads < 0
            or peak_qd < 0
        ):
            raise ValueError("slab counters are invalid")
        if mode == "off" and (reads != 0 or peak_qd != 0):
            raise ValueError("default-off arm performed slab I/O")
        if mode == "on" and (reads <= 0 or peak_qd < 2):
            raise ValueError("slab arm lacks positive concurrent reads")
        if engine.get("arena_pin_ok") is not True or engine.get("trace_lines") != 0:
            raise ValueError("timed engine instrumentation or arena pin is invalid")
        access_streams.add(
            sha256(engine.get("access_stream_sha256"), "access stream")
        )

        external = record["external_io"]
        if not isinstance(external, dict):
            raise ValueError("external I/O record is absent")
        read_bytes = positive(external.get("read_bytes_delta"), "read bytes")
        io_elapsed = positive(external.get("elapsed_seconds"), "I/O elapsed")
        samples = external.get("sample_count")
        external_qd = external.get("peak_read_qd")
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < 2
            or isinstance(external_qd, bool)
            or not isinstance(external_qd, int)
            or external_qd < 0
            or (mode == "on" and external_qd < 2)
        ):
            raise ValueError("external completed-I/O coverage is invalid")
        if mode == "on" and read_bytes < MODEL_BYTES + SLAB_BYTES:
            raise ValueError("slab arm lacks full identity read coverage")
        io_throughput[mode].append(read_bytes / io_elapsed)

        safety = record["safety"]
        if not isinstance(safety, dict):
            raise ValueError("safety evidence is absent")
        if positive(safety.get("minimum_available_gib"), "available memory") < 10:
            raise ValueError("whole-system memory floor was violated")
        for field in (
            "cgroup_high_events",
            "cgroup_max_events",
            "cgroup_oom_events",
            "cgroup_swap_bytes",
        ):
            if safety.get(field) != 0:
                raise ValueError(f"safety evidence has nonzero {field}")
        if safety.get("xid") is not False or safety.get("survivors") != []:
            raise ValueError("Xid or survivor invalidates the arm")
        if safety.get("failures") != []:
            raise ValueError("arm contains a safety failure")

    validate_ab_blocks(validation_rows, flip=schedule_flip)
    if len(binaries) != 1:
        raise ValueError("campaign used more than one binary")
    if any(len(values) != 1 for values in configurations.values()):
        raise ValueError("arm configuration changed between blocks")
    if configurations["off"] == configurations["on"]:
        raise ValueError("campaign arms are identical")
    if any(len(signatures) != 1 for signatures in output_signatures.values()):
        raise ValueError("paired output bytes or token IDs differ")
    if len(access_streams) != 1:
        raise ValueError("expert access streams differ between arms")

    if set(nll) != {
        "case_count",
        "token_weighted_delta_nll",
        "top1_loss_pp",
        "deterministic",
    }:
        raise ValueError("NLL summary schema is invalid")
    if (
        nll["case_count"] != 100
        or nll["token_weighted_delta_nll"] != 0.0
        or nll["top1_loss_pp"] != 0.0
        or nll["deterministic"] is not True
    ):
        raise ValueError("lossless transport requires exact-zero paired NLL")

    decode_off: list[float] = []
    decode_on: list[float] = []
    raw_decode_off: list[float] = []
    raw_decode_on: list[float] = []
    ttft_off: list[float] = []
    ttft_on: list[float] = []
    prompt_rate_off: list[float] = []
    prompt_rate_on: list[float] = []
    for block in range(5):
        for arm, decode_target, raw_decode_target, ttft_target, prompt_target in (
            ("A", decode_off, raw_decode_off, ttft_off, prompt_rate_off),
            ("B", decode_on, raw_decode_on, ttft_on, prompt_rate_on),
        ):
            values = [
                per_row[(block, record["sequence"])]
                for record in records
                if record["block"] == block and record["arm"] == arm
            ]
            if len(values) != 2:
                raise ValueError("block does not contain two instances per arm")
            decode_target.append(statistics.fmean(value[0] for value in values))
            raw_decode_target.append(statistics.fmean(value[1] for value in values))
            ttft_target.append(statistics.fmean(value[2] for value in values))
            prompt_target.append(statistics.fmean(value[3] for value in values))

    client_decode_lower = paired_ratio_bound(decode_on, decode_off, side="lower")
    raw_decode_lower = paired_ratio_bound(
        raw_decode_on, raw_decode_off, side="lower"
    )
    decode_lower = min(client_decode_lower, raw_decode_lower)
    ttft_upper = paired_ratio_bound(ttft_on, ttft_off, side="upper")
    verdict = (
        "PASS"
        if client_decode_lower > 1.0
        and raw_decode_lower > 1.0
        and ttft_upper <= 1.05
        else "FAIL"
    )
    return {
        "scorer_id": "glm.rung0.slab.v3-dual-clock",
        "verdict": verdict,
        "decode_ratio_lower_95": decode_lower,
        "decode_ratio_lower_95_by_clock": {
            "client_wall": client_decode_lower,
            "raw_token": raw_decode_lower,
        },
        "warm_ttft_ratio_upper_95": ttft_upper,
        "decode_tps": {"off": decode_off, "on": decode_on},
        "raw_decode_tps": {"off": raw_decode_off, "on": raw_decode_on},
        "warm_ttft_seconds": {"off": ttft_off, "on": ttft_on},
        "diagnostic_prompt_rate": {
            "label": "client-token-count divided by TTFT; not synchronized prefill",
            "off": prompt_rate_off,
            "on": prompt_rate_on,
        },
        "external_read_bytes_per_second": {
            mode: statistics.fmean(values) for mode, values in io_throughput.items()
        },
        "nll": dict(nll),
    }


if __name__ == "__main__":
    raise SystemExit(main())
