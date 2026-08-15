#!/usr/bin/env python3
"""Run the contained four-arm W9 E2M1 100-case fidelity campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run_w9_e2m1_v1.sh"
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
DRAND_VERIFIER = ROOT / "scripts/103_verify_drand_receipt_bundle.mjs"
FLAG = "DS4_GLM_COMPACT_CACHE_E2M1_FAKE"
COMMON_ENGINE_ENVIRONMENT = {
    "DS4_LOCK_FILE": "/run/user/1000/ds4-engine.lock",
    "DS4_CUDA_EXPERT_CACHE_GB": "0",
    "DS4_CUDA_EXPERT_CACHE_PIN": "1",
    "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
    "DS4_CUDA_FETCH_THREADS": "6",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
    "DS4_CUDA_MOE_NO_EXPERT_TILES": "1",
    "DS4_CUDA_STABLE_MODEL_REMAP": "1",
}
FIDELITY_NAMES = (
    "DS4_GLM_COMPACT_CACHE_AFFINE_INT8",
    "DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE",
    "DS4_GLM_COMPACT_CACHE_E4M3_FAKE",
    "DS4_GLM_COMPACT_CACHE_E2M1",
    FLAG,
    "DS4_GLM_COMPACT_CACHE_F16",
    "DS4_GLM_COMPACT_CACHE_INT8_FAKE",
)
PROVENANCE_NAMES = tuple(sorted((*COMMON_ENGINE_ENVIRONMENT, *FIDELITY_NAMES)))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def write_artifact(root: Path, relative: str, data: bytes) -> dict[str, Any]:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path is invalid")
    path = root / relative
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def schedule(seed_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", seed_sha256):
        raise ValueError("seed is invalid")
    return "ABBA" if int(seed_sha256[2:4], 16) % 2 == 0 else "BAAB"


def preflight(*, active_pids: list[int], available_kib: int) -> None:
    if active_pids:
        raise RuntimeError(f"large engine is active: {active_pids}")
    if available_kib < 110 * 1024 * 1024:
        raise RuntimeError("less than 110 GiB MemAvailable")


def environment_sha256(values: dict[str, str]) -> str:
    data = b"".join(
        name.encode("ascii") + b"="
        + values.get(name, "<UNSET>").encode("ascii") + b"\n"
        for name in PROVENANCE_NAMES
    )
    return hashlib.sha256(data).hexdigest()


def runner_closure_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), LAUNCHER, SAFE, DRAND_VERIFIER):
        data = path.read_bytes()
        relative = str(path.relative_to(ROOT)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def fixture_sha256(engine_source: Path, manifest: Path) -> str:
    source = engine_source.resolve()
    digest = hashlib.sha256(manifest.read_bytes())
    seen: set[str] = set()
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(
            (line for line in stream if not line.startswith("#")), delimiter="\t"
        )
        for row in rows:
            if len(row) != 4 or not row[0] or row[0] in seen:
                raise ValueError("fixture manifest is malformed or duplicated")
            seen.add(row[0])
            for relative in row[1:]:
                path = (source / relative).resolve()
                if not path.is_relative_to(source) or not path.is_file():
                    raise ValueError("fixture path escapes or is absent")
                raw_name = relative.encode()
                data = path.read_bytes()
                digest.update(len(raw_name).to_bytes(8, "big"))
                digest.update(raw_name)
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
    if len(seen) != 100:
        raise ValueError(f"fixture requires 100 cases, got {len(seen)}")
    return digest.hexdigest()


def parse_quality_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 100:
        raise ValueError(f"quality output requires 100 cases, got {len(rows)}")
    cases = []
    for row in rows:
        try:
            case = {
                "case_id": row["id"],
                "tokens": int(row["target_tokens"]),
                "nll_sum": float(row["nll"]),
                "top1_correct": int(row["target_top1_correct"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("quality output is malformed") from exc
        if (
            not case["case_id"] or case["tokens"] <= 0
            or not math.isfinite(case["nll_sum"])
            or not 0 <= case["top1_correct"] <= case["tokens"]
        ):
            raise ValueError("quality output value is invalid")
        cases.append(case)
    if len({case["case_id"] for case in cases}) != 100:
        raise ValueError("quality case IDs are duplicated")
    return cases


def _active_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-x", "ds4-server|fio"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return [int(value) for value in result.stdout.split() if value.isdigit()]


def _available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent")


def _verify_drand(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_bytes())
    required = {"round", "randomness", "signature", "previous_signature"}
    if not isinstance(record, dict) or not required <= set(record):
        raise ValueError("drand receipt is malformed")
    command = [
        "node", str(DRAND_VERIFIER), str(record["round"]), record["randomness"],
        record["signature"], record["previous_signature"],
    ]
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or result.stdout != "DRAND_BLS_RECEIPT_OK\n":
        raise ValueError("drand receipt did not verify")
    return {key: record[key] for key in sorted(required)}


def _freeze_binary(source: Path, commit: str, expected_sha256: str) -> Path:
    source_binary = source / "gguf-tools/quality-testing/score_official"
    if sha256_file(source_binary) != expected_sha256:
        raise ValueError("score binary hash differs from the freeze")
    root = Path(f"/home/bmarti44/.cache/glm52-w9-e2m1-{commit[:12]}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / "ds4-server"
    if not target.exists():
        temporary = root / ".ds4-server.tmp"
        shutil.copyfile(source_binary, temporary)
        os.chmod(temporary, 0o500)
        os.replace(temporary, target)
    if sha256_file(target) != expected_sha256:
        raise ValueError("frozen score binary differs")
    return root


def _git_head(source: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    if status.stdout:
        raise ValueError("engine source is dirty")
    return result.stdout.strip()


def _kernel_log(since: str) -> bytes:
    result = subprocess.run(
        ["journalctl", "-k", "--since", since, "--no-pager", "-o", "short-iso"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return result.stdout


def run(args: argparse.Namespace) -> int:
    preflight(active_pids=_active_pids(), available_kib=_available_kib())
    source = args.engine_source.resolve()
    model = args.model.resolve()
    fixture = args.fixture.resolve()
    if _git_head(source) != args.engine_commit:
        raise ValueError("engine commit differs from the freeze")
    if not model.is_file() or not fixture.is_file():
        raise ValueError("model or fixture is absent")
    observed_fixture = fixture_sha256(source, fixture)
    if observed_fixture != args.fixture_sha256:
        raise ValueError("fixture content differs from the freeze")
    if sha256_file(model) != args.model_sha256:
        raise ValueError("model content differs from the freeze")
    drand = _verify_drand(args.drand_json.resolve())
    frozen = _freeze_binary(source, args.engine_commit, args.binary_sha256)
    frozen_binary = frozen / "ds4-server"
    binary_stat = frozen_binary.stat()
    binary_dev_ino = f"{binary_stat.st_dev}:{binary_stat.st_ino}"
    seed = hashlib.sha256(
        (drand["randomness"] + args.engine_commit + args.binary_sha256).encode()
    ).hexdigest()
    candidate_arm = "A" if int(seed[:2], 16) % 2 == 0 else "B"
    arms = schedule(seed)
    baseline_env = dict(COMMON_ENGINE_ENVIRONMENT)
    candidate_env = {**COMMON_ENGINE_ENVIRONMENT, FLAG: "1"}
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    attempt_records = []
    model_identity = (model.stat().st_dev, model.stat().st_ino, model.stat().st_size)
    for sequence, arm in enumerate(arms):
        preflight(active_pids=_active_pids(), available_kib=_available_kib())
        is_candidate = arm == candidate_arm
        engine_env = candidate_env if is_candidate else baseline_env
        expected_env_sha256 = environment_sha256(engine_env)
        work = output.parent / f".{output.name}.attempt-{sequence}"
        work.mkdir(mode=0o700, exist_ok=False)
        quality = work / "quality.tsv"
        environment = os.environ.copy()
        for name in PROVENANCE_NAMES:
            environment.pop(name, None)
        environment.update(engine_env)
        environment.update({
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": args.binary_sha256,
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(PROVENANCE_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": expected_env_sha256,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "9000",
            "GLM_SAFE_VLIMIT_KB": "419430400",
            "GLM_SAFE_FINAL_ARTIFACTS": str(quality),
            "GLM_CANDIDATE_SRC": str(frozen),
        })
        since = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(
            [
                str(LAUNCHER), "--tag", f"w9e2m1-{seed[:8]}-{sequence}-{arm}",
                "--", str(frozen_binary), str(model), str(fixture), str(quality),
                "8192", "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
            ],
            cwd=source, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=9050, check=False,
        )
        launcher_log = result.stdout
        safe_dirs = re.findall(rb"SAFE_RUN_DONE .* dir=(\S+)", launcher_log)
        if result.returncode != 0 or len(safe_dirs) != 1:
            write_artifact(output, f"failed-{sequence}.launcher.log", launcher_log)
            raise RuntimeError(f"attempt {sequence} failed closed; preserved in {output}")
        safe_dir = Path(os.fsdecode(safe_dirs[0]))
        cases = parse_quality_tsv(quality)
        artifact_descriptors = {
            "command_log": write_artifact(
                output, f"attempt-{sequence}.command.log", (safe_dir / "cmd.log").read_bytes()
            ),
            "main_log": write_artifact(
                output, f"attempt-{sequence}.main.log", (safe_dir / "main.log").read_bytes()
            ),
            "samples_log": write_artifact(
                output, f"attempt-{sequence}.samples.log", (safe_dir / "samples.log").read_bytes()
            ),
            "kernel_log": write_artifact(
                output, f"attempt-{sequence}.kernel.log", _kernel_log(since)
            ),
            "cases": write_artifact(
                output, f"attempt-{sequence}.cases.json", canonical_json(cases)
            ),
        }
        attempt_records.append({
            "sequence": sequence,
            "arm": arm,
            "fixture_sha256": observed_fixture,
            "artifacts": artifact_descriptors,
        })
        if (model.stat().st_dev, model.stat().st_ino, model.stat().st_size) != model_identity:
            raise RuntimeError("model identity changed during the campaign")
    manifest = {
        "record_type": "w9_e2m1_fidelity_raw",
        "runner_sha256": runner_closure_sha256(),
        "randomness": drand,
        "engine_candidate_hash": args.engine_commit,
        "seed_sha256": seed,
        "binary_sha256": args.binary_sha256,
        "binary_path": str(frozen_binary),
        "binary_device_inode": binary_dev_ino,
        "baseline_environment_sha256": environment_sha256(baseline_env),
        "candidate_environment_sha256": environment_sha256(candidate_env),
        "fixture_sha256": observed_fixture,
        "candidate_arm": candidate_arm,
        "attempts": attempt_records,
    }
    write_artifact(output, "manifest.json", canonical_json(manifest))
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({
        "output": str(output), "seed_sha256": seed,
        "runner_sha256": manifest["runner_sha256"], "attempts": 4,
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine-source", required=True, type=Path)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--drand-json", required=True, type=Path)
    args = parser.parse_args()
    for name in ("engine_commit", "binary_sha256", "model_sha256", "fixture_sha256"):
        value = getattr(args, name)
        expected = 40 if name == "engine_commit" else 64
        if not re.fullmatch(rf"[0-9a-f]{{{expected}}}", value):
            parser.error(f"--{name.replace('_', '-')} is invalid")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"104_run_w9_e2m1_fidelity.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
