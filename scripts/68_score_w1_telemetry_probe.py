#!/usr/bin/env python3
"""Strictly derive non-authoritative W1 telemetry diagnostics from raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAFE = Path("results/glm52-gates/harness/glm_safe_run.sh")
RUNNER = Path("scripts/67_run_w1_telemetry_probe.py")
SCORER = Path("scripts/68_score_w1_telemetry_probe.py")
LOAD_SOURCE = Path("scripts/fixtures/w1_direct_io_load.c")
PROFILE = Path("configs/glm52-profile.json")
MODEL_PATH = (
    "/home/dsv4/ds4-project/gguf-glm/"
    "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
)
MODEL_SIZE = 211075856448
HASH40 = re.compile(r"^[0-9a-f]{40}$")
HASH64 = re.compile(r"^[0-9a-f]{64}$")
SAMPLE = re.compile(
    r"(\S+) mem_avail_kb=(\d+) eng_rss_kb=(\d+) read_bytes=(\d+)"
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def strict_json_bytes(value: bytes, label: str) -> Any:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} has no explicit timezone")
    return parsed


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=ROOT,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.rstrip("\n")


def git_blob_sha256(commit: str, relative: Path) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "show",
            f"{commit}:{relative}",
        ],
        cwd=ROOT,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return sha256_bytes(completed.stdout)


def read_raw(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line:
            raise ValueError(f"raw evidence line {number} is empty")
        value = strict_json_bytes(line, f"raw evidence line {number}")
        if not isinstance(value, dict):
            raise ValueError(f"raw evidence line {number} is not an object")
        rows.append(value)
    return rows


def validate_invocation(
    record: dict[str, Any], index: int, manifest: dict[str, Any]
) -> None:
    candidate = manifest["candidate_hash"]
    tag = f"w1-telemetry-{candidate[:8]}-{index}"
    source = record["environment"].get("GLM_CANDIDATE_SRC", "")
    fixture = f"{source}/ds4-server"
    expected_environment = {
        "HOME": "/home/bmarti44",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "GLM_CANDIDATE_SRC": source,
        "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
        "GLM_SAFE_EXPECTED_BINARY_SHA256": manifest["artifact_sha256"][
            "artifacts/ds4-server"
        ],
        "GLM_SAFE_KILL_FLOOR_GIB": "40",
        "GLM_SAFE_MIN_START_GIB": "110",
        "GLM_SAFE_TIMEOUT_S": "75",
        "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
    }
    expected_command = [
        "/usr/bin/bash",
        str(ROOT / SAFE),
        "--tag",
        tag,
        "--",
        fixture,
        "60",
    ]
    if (
        not re.fullmatch(
            r"/home/bmarti44/\.cache/glm52-telemetry-[A-Za-z0-9_]+", source
        )
        or record["environment"] != expected_environment
        or record["command"] != expected_command
    ):
        raise ValueError(f"probe {index} invocation is invalid")
    launcher = re.fullmatch(
        rf"SAFE_RUN_DONE rc=0 killed=no dir="
        rf"(/home/bmarti44/\.local/state/glm52-crashlog/"
        rf"\d{{8}}-\d{{6}}-{re.escape(tag)})\n",
        record["launcher_log"],
    )
    if launcher is None:
        raise ValueError(f"probe {index} launcher evidence path is invalid")
    fixture_hash = manifest["artifact_sha256"]["artifacts/ds4-server"]
    provenance = re.search(
        r"^(\S+) candidate_src=(\S+) candidate_binary_sha256=([0-9a-f]{64}) "
        r"candidate_device_inode=(\d+:\d+)$",
        record["main_log"],
        re.MULTILINE,
    )
    execution = re.search(
        r"^(\S+) executed_candidate_verified pid=(\d+) start_ticks=(\d+) "
        r"path=(\S+) executed_binary_sha256=([0-9a-f]{64}) "
        r"device_inode=(\d+:\d+)$",
        record["main_log"],
        re.MULTILINE,
    )
    if (
        provenance is None
        or execution is None
        or provenance.group(2) != source
        or provenance.group(3) != fixture_hash
        or execution.group(4) != fixture
        or execution.group(5) != fixture_hash
        or provenance.group(4) != execution.group(6)
    ):
        raise ValueError(f"probe {index} executable provenance is invalid")


def derive_probe(
    record: dict[str, Any], index: int, manifest: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "record_type",
        "index",
        "started_at",
        "completed_at",
        "command",
        "environment",
        "launcher_log",
        "main_log",
        "samples_log",
        "io_load_log",
        "io_load_returncode",
    }
    if set(record) != required or record["record_type"] != "probe":
        raise ValueError(f"probe {index} schema is invalid")
    if record["index"] != index:
        raise ValueError("probe indices are missing or reordered")
    if (
        not isinstance(record["command"], list)
        or not record["command"]
        or any(not isinstance(item, str) or not item for item in record["command"])
        or not isinstance(record["environment"], dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            for key, value in record["environment"].items()
        )
    ):
        raise ValueError(f"probe {index} invocation is invalid")
    validate_invocation(record, index, manifest)
    if record["io_load_returncode"] != 0:
        raise ValueError(f"probe {index} direct-I/O load failed")
    started = timestamp(record["started_at"], f"probe {index} start")
    completed = timestamp(record["completed_at"], f"probe {index} completion")
    if completed <= started:
        raise ValueError(f"probe {index} wall timing is invalid")
    if not re.search(
        r"^SAFE_RUN_DONE rc=0 killed=no dir=\S+$",
        record["launcher_log"],
        re.MULTILINE,
    ):
        raise ValueError(f"probe {index} launcher did not complete cleanly")
    executed_match = re.search(
        r"^(\S+) executed_candidate_verified pid=(\d+) start_ticks=(\d+)",
        record["main_log"],
        re.MULTILINE,
    )
    completed_match = re.search(
        r"^(\S+) SAFE_RUN end rc=0 killed=no\b",
        record["main_log"],
        re.MULTILINE,
    )
    if not executed_match or not completed_match:
        raise ValueError(f"probe {index} lifecycle evidence is incomplete")
    executed = timestamp(executed_match.group(1), f"probe {index} execution")
    wrapper_completed = timestamp(
        completed_match.group(1), f"probe {index} wrapper completion"
    )
    lifecycle_s = (wrapper_completed - executed).total_seconds()
    if lifecycle_s < 59.0 or lifecycle_s > 61.0:
        raise ValueError(f"probe {index} lifecycle duration is invalid")
    sample_rows = []
    for line in record["samples_log"].splitlines():
        match = SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError(f"probe {index} sample is malformed")
        sample_rows.append(
            (
                timestamp(match.group(1), f"probe {index} sample"),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )
        )
    if len(sample_rows) < 20:
        raise ValueError(f"probe {index} has too few samples")
    times = [row[0] for row in sample_rows]
    gaps = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
    first_delay = (times[0] - executed).total_seconds()
    trailing_delay = (wrapper_completed - times[-1]).total_seconds()
    failed = []
    if any(not math.isfinite(gap) or gap <= 0 or gap > 0.75 for gap in gaps):
        failed.append("max_gap")
    if first_delay > 1.0:
        failed.append("late_first")
    if trailing_delay > 1.0:
        failed.append("early_final")
    io_load = strict_json_bytes(
        record["io_load_log"].encode(), f"probe {index} direct-I/O witness"
    )
    if (
        not isinstance(io_load, dict)
        or set(io_load)
        != {"bytes_read", "direct_io", "elapsed_s", "fcntl_flags", "pid"}
        or io_load["direct_io"] is not True
        or not isinstance(io_load["bytes_read"], int)
        or io_load["bytes_read"] < 1024 * 1024 * 1024
        or not isinstance(io_load["elapsed_s"], (int, float))
        or io_load["elapsed_s"] < 60.0
        or not isinstance(io_load["fcntl_flags"], int)
        or io_load["fcntl_flags"] & os.O_DIRECT == 0
        or not isinstance(io_load["pid"], int)
        or io_load["pid"] <= 1
    ):
        raise ValueError(f"probe {index} direct-I/O witness is invalid")
    if failed:
        raise ValueError(
            f"probe {index} telemetry coverage failed={','.join(failed)}"
        )
    return {
        "index": index,
        "samples": len(sample_rows),
        "minimum_available_gib": min(row[1] for row in sample_rows) / 1048576,
        "min_gap_s": min(gaps),
        "max_gap_s": max(gaps),
        "first_minus_executed_s": first_delay,
        "completed_minus_last_s": trailing_delay,
        "executed_pid": int(executed_match.group(2)),
        "executed_start_ticks": int(executed_match.group(3)),
        "direct_io_bytes": io_load["bytes_read"],
        "direct_io_elapsed_s": io_load["elapsed_s"],
        "verdict": "PASS",
    }


def derive_summary(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3:
        raise ValueError("telemetry confirmation requires exactly three probes")
    probes = [
        derive_probe(row, index, manifest) for index, row in enumerate(rows)
    ]
    return {
        "schema_version": 1,
        "record_type": "w1_telemetry_loaded_confirmation",
        "candidate_hash": manifest["candidate_hash"],
        "formula": {
            "maximum_adjacent_gap_s": 0.75,
            "maximum_first_sample_delay_s": 1.0,
            "maximum_trailing_sample_delay_s": 1.0,
            "minimum_samples": 20,
            "minimum_direct_io_bytes": 1073741824,
            "minimum_direct_io_elapsed_s": 60.0,
        },
        "probes": probes,
        "verdict": "PASS",
    }


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    manifest_path = package / "manifest.json"
    raw_path = package / "raw.jsonl"
    summary_path = package / "summary.json"
    manifest = strict_json_bytes(manifest_path.read_bytes(), "manifest")
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    required = {
        "schema_version",
        "record_type",
        "candidate_hash",
        "candidate_committed_at",
        "started_at",
        "completed_at",
        "artifact_sha256",
        "raw_jsonl_sha256",
        "build",
        "model",
    }
    if (
        set(manifest) != required
        or manifest["schema_version"] != 1
        or manifest["record_type"] != "w1_telemetry_probe_manifest"
        or not isinstance(manifest["candidate_hash"], str)
        or not HASH40.fullmatch(manifest["candidate_hash"])
    ):
        raise ValueError("manifest schema is invalid")
    candidate = manifest["candidate_hash"]
    if git("rev-parse", "--verify", f"{candidate}^{{commit}}") != candidate:
        raise ValueError("candidate commit is unavailable")
    committed_at = timestamp(
        manifest["candidate_committed_at"], "candidate commit timestamp"
    )
    if git("show", "-s", "--format=%cI", candidate) != manifest["candidate_committed_at"]:
        raise ValueError("candidate commit timestamp differs")
    started_at = timestamp(manifest["started_at"], "manifest start")
    completed_at = timestamp(manifest["completed_at"], "manifest completion")
    if started_at < committed_at or completed_at <= started_at:
        raise ValueError("confirmation did not occur after candidate freeze")
    artifacts = manifest["artifact_sha256"]
    expected_artifacts = {
        str(SAFE),
        str(RUNNER),
        str(SCORER),
        str(LOAD_SOURCE),
        str(PROFILE),
        "artifacts/ds4-server",
        "artifacts/w1-direct-io-load",
    }
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != expected_artifacts
        or any(
            not isinstance(value, str) or not HASH64.fullmatch(value)
            for value in artifacts.values()
        )
    ):
        raise ValueError("artifact manifest is invalid")
    for relative in (SAFE, RUNNER, SCORER, LOAD_SOURCE, PROFILE):
        if git_blob_sha256(candidate, relative) != artifacts[str(relative)]:
            raise ValueError(f"candidate artifact differs: {relative}")
    for relative in ("artifacts/ds4-server", "artifacts/w1-direct-io-load"):
        if sha256_file(package / relative) != artifacts[relative]:
            raise ValueError(f"packaged artifact differs: {relative}")
    profile = strict_json_bytes(
        subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "show",
                f"{candidate}:{PROFILE}",
            ],
            cwd=ROOT,
            env={
                "HOME": "/nonexistent",
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "GIT_CONFIG_NOSYSTEM": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout,
        "frozen GLM profile",
    )
    if (
        not isinstance(manifest["model"], dict)
        or set(manifest["model"]) != {"path", "sha256", "size"}
        or manifest["model"]["path"] != MODEL_PATH
        or manifest["model"]["sha256"] != profile.get("model_sha256")
        or manifest["model"]["size"] != MODEL_SIZE
    ):
        raise ValueError("manifest model is not the frozen GLM model")
    if sha256_file(raw_path) != manifest["raw_jsonl_sha256"]:
        raise ValueError("raw evidence hash differs")
    rows = read_raw(raw_path)
    derived = derive_summary(manifest, rows)
    stored = strict_json_bytes(summary_path.read_bytes(), "summary")
    if stored != derived:
        raise ValueError("summary differs from fixed derivation")
    return {
        **derived,
        "acceptance_authority": False,
        "verdict": "DIAGNOSTIC_ONLY",
    }


def write_test_package(package: Path) -> None:
    """Create a complete synthetic package used only by scorer mutations."""
    package.mkdir(exist_ok=True)
    artifacts = package / "artifacts"
    artifacts.mkdir()
    shutil.copy2("/usr/bin/sleep", artifacts / "ds4-server")
    (artifacts / "w1-direct-io-load").write_bytes(b"fixed-test-binary\n")
    candidate = git("rev-parse", "HEAD")
    committed_at = git("show", "-s", "--format=%cI", candidate)
    fixture_hash = sha256_file(artifacts / "ds4-server")
    profile = strict_json_bytes((ROOT / PROFILE).read_bytes(), "test profile")
    rows = []
    for index in range(3):
        source = "/home/bmarti44/.cache/glm52-telemetry-testfixture"
        fixture = f"{source}/ds4-server"
        tag = f"w1-telemetry-{candidate[:8]}-{index}"
        environment = {
            "HOME": "/home/bmarti44",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "GLM_CANDIDATE_SRC": source,
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": fixture_hash,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "75",
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
        }
        sample_start = datetime(2026, 7, 30, tzinfo=timezone.utc)
        samples = "".join(
            f"{(sample_start + timedelta(seconds=sample / 4)).isoformat(timespec='milliseconds')} "
            "mem_avail_kb=90000000 eng_rss_kb=1 read_bytes=1\n"
            for sample in range(241)
        )
        rows.append(
            {
                "record_type": "probe",
                "index": index,
                "started_at": "2026-07-30T00:00:00.000+00:00",
                "completed_at": "2026-07-30T00:00:06.000+00:00",
                "command": [
                    "/usr/bin/bash",
                    str(ROOT / SAFE),
                    "--tag",
                    tag,
                    "--",
                    fixture,
                    "60",
                ],
                "environment": environment,
                "launcher_log": (
                    "SAFE_RUN_DONE rc=0 killed=no dir="
                    f"/home/bmarti44/.local/state/glm52-crashlog/"
                    f"20260730-000000-{tag}\n"
                ),
                "main_log": (
                    "2026-07-30T00:00:00.000+00:00 "
                    f"candidate_src={source} "
                    f"candidate_binary_sha256={fixture_hash} "
                    "candidate_device_inode=1:2\n"
                    f"2026-07-30T00:00:00.000+00:00 "
                    f"executed_candidate_verified pid={100 + index} "
                    f"start_ticks={200 + index} path={fixture} "
                    f"executed_binary_sha256={fixture_hash} "
                    "device_inode=1:2\n"
                    "2026-07-30T00:01:00.000+00:00 "
                    "SAFE_RUN end rc=0 killed=no\n"
                ),
                "samples_log": samples,
                "io_load_log": json.dumps(
                    {
                        "bytes_read": 2 * 1024 * 1024 * 1024,
                        "direct_io": True,
                        "elapsed_s": 61.0,
                        "fcntl_flags": os.O_DIRECT,
                        "pid": 300 + index,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                "io_load_returncode": 0,
            }
        )
    raw = b"".join(canonical(row).rstrip(b"\n") + b"\n" for row in rows)
    (package / "raw.jsonl").write_bytes(raw)
    artifact_hashes = {
        str(relative): git_blob_sha256(candidate, relative)
        for relative in (SAFE, RUNNER, SCORER, LOAD_SOURCE, PROFILE)
    }
    artifact_hashes.update(
        {
            relative: sha256_file(package / relative)
            for relative in (
                "artifacts/ds4-server",
                "artifacts/w1-direct-io-load",
            )
        }
    )
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "record_type": "w1_telemetry_probe_manifest",
        "candidate_hash": candidate,
        "candidate_committed_at": committed_at,
        "started_at": now,
        "completed_at": (
            datetime.now(timezone.utc) + timedelta(seconds=1)
        ).isoformat(),
        "artifact_sha256": artifact_hashes,
        "raw_jsonl_sha256": sha256_bytes(raw),
        "build": {"compiler": "test", "argv": ["test"]},
        "model": {
            "path": MODEL_PATH,
            "sha256": profile["model_sha256"],
            "size": MODEL_SIZE,
        },
    }
    (package / "manifest.json").write_bytes(canonical(manifest))
    summary = derive_summary(manifest, rows)
    (package / "summary.json").write_bytes(canonical(summary))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_package(args.package), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
