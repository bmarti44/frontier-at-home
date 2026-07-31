#!/usr/bin/env python3
"""Run three post-freeze W1 telemetry probes with raw direct-I/O evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
SCORER_PATH = ROOT / "scripts/68_score_w1_telemetry_probe.py"
LOAD_SOURCE = ROOT / "scripts/fixtures/w1_direct_io_load.c"
PROFILE = ROOT / "configs/glm52-profile.json"
MODEL = Path(
    "/home/dsv4/ds4-project/gguf-glm/"
    "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
)
OUTPUT_ROOT = Path("/home/bmarti44/.local/state/w1-telemetry-probes")
HASH40 = re.compile(r"^[0-9a-f]{40}$")

# Security-significant repository checks are exactly:
#   git status --porcelain
#   git show -s --format=%cI


def load_scorer():
    spec = importlib.util.spec_from_file_location("w1_probe_scorer", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load fixed telemetry scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_scorer()


def run(
    argv: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def git(*arguments: str) -> str:
    completed = run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *arguments],
        environment={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return completed.stdout.rstrip("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not HASH40.fullmatch(args.candidate):
        raise SystemExit("candidate must be an exact 40-character commit")
    if git("rev-parse", "--verify", "HEAD^{commit}") != args.candidate:
        raise SystemExit("candidate is not the current HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("repository is not clean")
    candidate_committed_at = git("show", "-s", "--format=%cI", args.candidate)

    output = args.output.absolute()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if output.parent != OUTPUT_ROOT or output.exists():
        raise SystemExit(f"output must be a new direct child of {OUTPUT_ROOT}")
    output.mkdir(mode=0o700)
    artifacts = output / "artifacts"
    artifacts.mkdir(mode=0o700)
    fixture = artifacts / "ds4-server"
    io_binary = artifacts / "w1-direct-io-load"
    shutil.copy2("/usr/bin/sleep", fixture)
    compiler = run(["/usr/bin/gcc", "--version"])
    if compiler.returncode:
        raise RuntimeError("gcc is unavailable")
    build_argv = [
        "/usr/bin/gcc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-o",
        str(io_binary),
        str(LOAD_SOURCE),
    ]
    built = run(build_argv)
    if built.returncode:
        raise RuntimeError(built.stdout + built.stderr)

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    model_sha256 = sha256_file(MODEL)
    if model_sha256 != profile["model_sha256"]:
        raise RuntimeError("GLM model differs from the frozen profile")
    model_details = MODEL.stat()
    started_at = iso_now()
    raw_rows = []
    with tempfile.TemporaryDirectory(
        prefix="glm52-telemetry-", dir="/home/bmarti44/.cache"
    ) as candidate_directory_text:
        candidate_directory = Path(candidate_directory_text)
        candidate_fixture = candidate_directory / "ds4-server"
        shutil.copy2(fixture, candidate_fixture)
        fixture_sha256 = sha256_file(fixture)
        for index in range(3):
            tag = f"w1-telemetry-{args.candidate[:8]}-{index}"
            environment = {
                "HOME": "/home/bmarti44",
                "PATH": (
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                    "/usr/bin:/sbin:/bin"
                ),
                "LANG": "C.UTF-8",
                "GLM_CANDIDATE_SRC": str(candidate_directory),
                "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
                "GLM_SAFE_EXPECTED_BINARY_SHA256": fixture_sha256,
                "GLM_SAFE_KILL_FLOOR_GIB": "40",
                "GLM_SAFE_MIN_START_GIB": "110",
                "GLM_SAFE_TIMEOUT_S": "75",
                "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            }
            command = [
                "/usr/bin/bash",
                str(SAFE),
                "--tag",
                tag,
                "--",
                str(candidate_fixture),
                "60",
            ]
            probe_started = iso_now()
            io_process = subprocess.Popen(
                [str(io_binary), str(MODEL), "68"],
                cwd=ROOT,
                env={
                    "HOME": "/home/bmarti44",
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wrapper = run(command, environment=environment, timeout=90)
            try:
                io_stdout, io_stderr = io_process.communicate(timeout=75)
            except subprocess.TimeoutExpired:
                io_process.kill()
                io_stdout, io_stderr = io_process.communicate()
                raise RuntimeError("direct-I/O witness timed out")
            probe_completed = iso_now()
            if wrapper.returncode or io_process.returncode:
                raise RuntimeError(
                    "probe failed\n"
                    + wrapper.stdout
                    + wrapper.stderr
                    + io_stdout
                    + io_stderr
                )
            matches = re.findall(
                r"^SAFE_RUN_DONE rc=0 killed=no dir=(\S+)$",
                wrapper.stdout,
                re.MULTILINE,
            )
            if len(matches) != 1:
                raise RuntimeError("safe wrapper did not expose one evidence path")
            crash = Path(matches[0]).resolve(strict=True)
            raw_rows.append(
                {
                    "record_type": "probe",
                    "index": index,
                    "started_at": probe_started,
                    "completed_at": probe_completed,
                    "command": command,
                    "environment": environment,
                    "launcher_log": wrapper.stdout,
                    "main_log": (crash / "main.log").read_text(encoding="utf-8"),
                    "samples_log": (crash / "samples.log").read_text(
                        encoding="utf-8"
                    ),
                    "io_load_log": io_stdout,
                    "io_load_returncode": io_process.returncode,
                }
            )

    raw = b"".join(
        SCORER.canonical(row).rstrip(b"\n") + b"\n" for row in raw_rows
    )
    raw_path = output / "raw.jsonl"
    raw_path.write_bytes(raw)
    artifact_hashes = {
        str(relative): SCORER.git_blob_sha256(args.candidate, relative)
        for relative in (
            SCORER.SAFE,
            SCORER.RUNNER,
            SCORER.SCORER,
            SCORER.LOAD_SOURCE,
            SCORER.PROFILE,
        )
    }
    artifact_hashes.update(
        {
            relative: sha256_file(output / relative)
            for relative in (
                "artifacts/ds4-server",
                "artifacts/w1-direct-io-load",
            )
        }
    )
    manifest = {
        "schema_version": 1,
        "record_type": "w1_telemetry_probe_manifest",
        "candidate_hash": args.candidate,
        "candidate_committed_at": candidate_committed_at,
        "started_at": started_at,
        "completed_at": iso_now(),
        "artifact_sha256": artifact_hashes,
        "raw_jsonl_sha256": SCORER.sha256_bytes(raw),
        "build": {
            "compiler": compiler.stdout.splitlines()[0],
            "argv": build_argv,
        },
        "model": {
            "path": str(MODEL),
            "sha256": model_sha256,
            "size": model_details.st_size,
        },
    }
    (output / "manifest.json").write_bytes(SCORER.canonical(manifest))
    summary = SCORER.derive_summary(manifest, raw_rows)
    (output / "summary.json").write_bytes(SCORER.canonical(summary))
    verified = SCORER.verify_package(output)
    if verified != summary:
        raise RuntimeError("fixed scorer did not reproduce the summary")
    for path in output.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    print(json.dumps(verified, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
