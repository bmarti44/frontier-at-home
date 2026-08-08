#!/usr/bin/env python3
"""Run one hash-bound W4 CUDA microgate confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts/98_score_w4_topk.py"
VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
OBSERVATION_RE = re.compile(
    r"^W4_OBSERVATION block=([0-4]) sequence=([0-3]) arm=([AB]) "
    r"mode=([01]) exact=([01]) ids_sha256=([0-9a-f]{64}) "
    r"elapsed_ms=([0-9]+(?:\.[0-9]+)?)$")
MARKER = "ds4: CUDA exact top-2048 CUB enabled chunk=8192 merge=2"
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30


def verify_digest_bindings(expected: dict[str, str], actual: dict[str, str]) -> None:
    """Require an exact, well-formed digest map for the executable closure."""
    if not isinstance(expected, dict) or set(expected) != set(actual):
        raise ValueError("frozen artifact set differs")
    for name in sorted(expected):
        digest = expected[name]
        observed = actual[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid frozen digest: {name}")
        if observed != digest:
            raise ValueError(f"post-freeze artifact replacement: {name}")


def fail(message: str) -> None:
    raise SystemExit(f"w4 confirmation: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(values: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result
    def constant(value: str) -> None:
        raise ValueError(f"non-finite {value}")
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def expected_ids_sha256() -> str:
    digest = hashlib.sha256()
    mask = 1048576 - 1
    for token in range(8):
        salt = (0x9E3779B9 * (token + 1)) & mask
        for rank in range(2048):
            digest.update(struct.pack("<I", (mask - rank) ^ salt))
    return digest.hexdigest()


def artifact(path: Path, copied_name: str, staging: Path) -> dict:
    destination = staging / copied_name
    shutil.copyfile(path, destination)
    os.chmod(destination, 0o500 if os.access(path, os.X_OK) else 0o400)
    return {"path": copied_name, "sha256": sha256(destination),
            "bytes": destination.stat().st_size}


def artifact_record(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256(path),
            "bytes": path.stat().st_size}


def memory_snapshot() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(value)
    if set(values) != {"MemAvailable", "SwapTotal", "SwapFree"}:
        fail("could not read memory safety state")
    return {"mem_available_kib": values["MemAvailable"],
            "swap_used_kib": values["SwapTotal"] - values["SwapFree"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists():
        fail("run directory already exists")
    for process_name in ("ds4-server", "fio"):
        check = subprocess.run(["/usr/bin/pgrep", "-x", process_name],
                               capture_output=True, check=False)
        if check.returncode == 0:
            fail(f"exclusive process is active: {process_name}")
        if check.returncode != 1:
            fail(f"could not check process: {process_name}")
    freeze = strict_json(args.freeze)
    receipt = strict_json(args.receipt)
    if freeze.get("schema") != "glm52-w4-topk-freeze-v2" or \
            freeze.get("candidate") != 5 or freeze.get("verdict") != "FROZEN":
        fail("freeze record differs")
    candidate_hash = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=args.source_dir,
        text=True, capture_output=True, check=True).stdout.strip()
    if candidate_hash != freeze.get("source", {}).get("implementation_commit"):
        fail("executing source HEAD differs from freeze")
    if subprocess.run(["/usr/bin/git", "diff", "--quiet"],
                      cwd=args.source_dir, check=False).returncode != 0 or \
            subprocess.run(["/usr/bin/git", "diff", "--cached", "--quiet"],
                           cwd=args.source_dir, check=False).returncode != 0:
        fail("source tree has tracked changes")
    frozen_artifacts = freeze.get("source", {}).get("artifact_sha256")
    actual_artifacts = {
        "ds4.c": sha256(args.source_dir / "ds4.c"),
        "ds4_cuda.cu": sha256(args.source_dir / "ds4_cuda.cu"),
        "tests/cuda_topk_w4.cu": sha256(
            args.source_dir / "tests/cuda_topk_w4.cu"),
        "Makefile": sha256(args.source_dir / "Makefile"),
        "scripts/98_score_w4_topk.py": sha256(SCORER),
        "scripts/99_run_w4_topk_confirmation.py": sha256(Path(__file__)),
        "scripts/tests/test_w4_topk_scorer.py": sha256(
            ROOT / "scripts/tests/test_w4_topk_scorer.py"),
        "scripts/89_verify_drand_receipt.mjs": sha256(VERIFIER),
    }
    try:
        verify_digest_bindings(frozen_artifacts, actual_artifacts)
    except ValueError as exc:
        fail(str(exc))
    expected_binary = freeze.get("build", {}).get("test", {}).get("binary_sha256")
    if not isinstance(expected_binary, str) or sha256(args.binary) != expected_binary:
        fail("test binary differs from freeze")
    verified = subprocess.run(
        [str(NODE), str(VERIFIER), str(receipt.get("round")),
         str(receipt.get("randomness")), str(receipt.get("signature")),
         str(receipt.get("previous_signature"))],
        cwd=ROOT, text=True, capture_output=True, timeout=30, check=False,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})
    if verified.returncode != 0 or verified.stdout != "DRAND_BLS_RECEIPT_OK\n":
        fail("drand receipt did not verify")
    round_value = receipt.get("round")
    freeze_time = freeze.get("freeze_time_unix")
    if not isinstance(round_value, int) or not isinstance(freeze_time, int) or \
            DRAND_GENESIS_UNIX + (round_value - 1) * DRAND_PERIOD_SECONDS <= freeze_time:
        fail("drand round was not published strictly after freeze")
    randomness = receipt.get("randomness")
    if not isinstance(randomness, str) or len(randomness) != 64:
        fail("invalid randomness")
    first_schedule = "BAAB" if int(randomness[:2], 16) & 1 else "ABBA"
    environment = {
        "HOME": "/nonexistent",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64",
        "W4_FIRST_SCHEDULE": first_schedule,
    }
    before_memory = memory_snapshot()
    started_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        [str(args.binary)], cwd=args.source_dir, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_stat = Path(f"/proc/{process.pid}/stat").read_text().split()
    if len(process_stat) < 22:
        process.kill()
        fail("could not bind CUDA process start ticks")
    process_start_ticks = int(process_stat[21])
    process_exe = Path(os.readlink(f"/proc/{process.pid}/exe")).resolve()
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        fail("CUDA microgate timed out")
    completed_at = datetime.now(timezone.utc).isoformat()
    after_memory = memory_snapshot()
    completed = subprocess.CompletedProcess(
        [str(args.binary)], process.returncode, stdout, stderr)
    if completed.returncode != 0:
        fail(f"CUDA microgate failed: {completed.stderr[-1000:]}")
    lines = completed.stderr.splitlines()
    parsed = []
    for line in lines:
        match = OBSERVATION_RE.fullmatch(line)
        if match:
            parsed.append((int(match[1]), int(match[2]), match[3],
                           int(match[4]), int(match[5]), match[6],
                           float(match[7])))
    if len(parsed) != 20 or lines.count(MARKER) != 1:
        fail("missing observations or effective-mode marker")
    id_hash = expected_ids_sha256()
    rows = [{
        "schema": "glm52-w4-topk-observation-v1",
        "block": block, "sequence": sequence, "arm": arm,
        "elapsed_ms": elapsed, "ids_sha256": observed_id_hash,
        "ids_identical_to_expected": exact == 1 and observed_id_hash == id_hash,
        "effective_marker_present": mode == 1,
        "n_components": 1048576, "n_tokens": 8, "top_k": 2048,
    } for block, sequence, arm, mode, exact, observed_id_hash, elapsed in parsed]

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.run_dir.name}.",
                                     dir=args.run_dir.parent) as tmp_name:
        staging = Path(tmp_name)
        stderr_path = staging / "microgate.stderr"
        stdout_path = staging / "microgate.stdout"
        stderr_path.write_text(completed.stderr)
        stdout_path.write_text(completed.stdout)
        artifacts = {
            "binary": artifact(args.binary, "binary", staging),
            "ds4.c": artifact(args.source_dir / "ds4.c", "ds4.c", staging),
            "engine.cu": artifact(args.source_dir / "ds4_cuda.cu", "engine.cu", staging),
            "test.cu": artifact(args.source_dir / "tests/cuda_topk_w4.cu", "test.cu", staging),
            "Makefile": artifact(args.source_dir / "Makefile", "Makefile", staging),
            "runner.py": artifact(Path(__file__), "runner.py", staging),
            "scorer.py": artifact(SCORER, "scorer.py", staging),
            "drand-verifier.mjs": artifact(VERIFIER, "drand-verifier.mjs", staging),
            "randomness-receipt.json": artifact(args.receipt, "randomness-receipt.json", staging),
            "freeze.json": artifact(args.freeze, "freeze.json", staging),
            "microgate.stderr": artifact_record(stderr_path),
            "microgate.stdout": artifact_record(stdout_path),
        }
        raw_path = staging / "raw.jsonl"
        raw_path.write_text("".join(json.dumps(row, sort_keys=True,
                                                allow_nan=False) + "\n"
                                    for row in rows))
        gpu = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=name,uuid,driver_version",
             "--format=csv,noheader"], text=True, capture_output=True,
            timeout=10, check=True).stdout.strip().split(", ")
        if len(gpu) != 3 or not all(gpu):
            fail("GPU identity query failed")
        manifest = {
            "schema": "glm52-w4-topk-manifest-v1", "gate": "W4",
            "candidate": 5, "candidate_hash": candidate_hash,
            "freeze_time_unix": freeze_time,
            "binary_sha256": expected_binary,
            "scorer_sha256": sha256(SCORER), "raw_sha256": sha256(raw_path),
            "configuration": {
                "n_components": 1048576, "n_tokens": 8, "top_k": 2048,
                "blocks": 5, "observations_per_block": 4,
                "flag_name": "DS4_CUDA_TOPK2048_CUB", "flag_value": "1",
                "required_speedup_lower_95": 2.0,
            },
            "randomness": {**receipt, "verification": verified.stdout.strip(),
                           "receipt_path": "randomness-receipt.json"},
            "invocation": {"argv": ["binary"],
                           "environment": {"DS4_CUDA_TOPK2048_CUB": "scheduled-per-arm"},
                           "exit_code": completed.returncode},
            "device": {"name": gpu[0], "uuid": gpu[1], "driver": gpu[2]},
            "safety": {
                "started_at": started_at, "completed_at": completed_at,
                "process_pid": process.pid,
                "process_start_ticks": process_start_ticks,
                "process_exe": str(process_exe),
                "executed_binary_sha256": sha256(args.binary),
                "mem_available_before_kib": before_memory["mem_available_kib"],
                "mem_available_after_kib": after_memory["mem_available_kib"],
                "swap_used_before_kib": before_memory["swap_used_kib"],
                "swap_used_after_kib": after_memory["swap_used_kib"],
                "engine_processes_present": False,
                "fio_present": False,
                "failures": [],
            },
            "artifacts": artifacts,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True,
                                            allow_nan=False) + "\n")
        scored = subprocess.run(
            ["/usr/bin/python3", str(staging / "scorer.py"), str(staging)],
            cwd=staging, text=True, capture_output=True, timeout=60,
            check=False, env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})
        if scored.returncode != 0:
            fail(f"fixed scorer failed: {scored.stderr}{scored.stdout}")
        try:
            summary = json.loads(
                scored.stdout,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite scorer output: {value}")),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            fail(f"fixed scorer emitted malformed output: {exc}")
        required_summary = {
            "schema", "formula", "t_0.95_df4", "block_a_ms", "block_b_ms",
            "log_ratios", "speedup_lower_95", "required_speedup_lower_95",
            "selected_ids_sha256", "raw_sha256", "manifest_sha256", "checks",
            "verdict",
        }
        if not isinstance(summary, dict) or set(summary) != required_summary or \
                summary.get("schema") != "glm52-w4-topk-summary-v1" or \
                summary.get("verdict") != "PASS":
            fail("fixed scorer result shape or verdict differs")
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, args.run_dir)
    print(json.dumps({"run_dir": str(args.run_dir), "verdict": "PASS",
                      "speedup_lower_95": summary["speedup_lower_95"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
