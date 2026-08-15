#!/usr/bin/env python3
"""Run one frozen, randomized W6 CUDA microgate confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts/106_score_w6_indexer_tile_reuse.py"
VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30
MIN_MEMORY_KIB = 110 * 1024 * 1024


def fail(message: str) -> None:
    raise SystemExit(f"W6 runner: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           ValueError(f"non-finite value: {value}")))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def memory_snapshot() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(value)
    if set(values) != {"MemAvailable", "SwapTotal", "SwapFree"}:
        fail("memory state unavailable")
    return {"mem_available_kib": values["MemAvailable"],
            "swap_used_kib": values["SwapTotal"] - values["SwapFree"]}


def artifact(source: Path, name: str, staging: Path) -> dict:
    destination = staging / name
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o500 if os.access(source, os.X_OK) else 0o400)
    return {"path": name, "sha256": sha256(destination),
            "bytes": destination.stat().st_size}


def artifact_record(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256(path),
            "bytes": path.stat().st_size}


def schedule_from_randomness(randomness: str) -> list[int]:
    seed = bytes.fromhex(randomness)
    tagged = []
    for width in (1, 2, 4):
        for occurrence in range(5):
            key = hashlib.sha256(seed + f":{width}:{occurrence}".encode()).digest()
            tagged.append((key, width))
    return [width for _, width in sorted(tagged)]


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
            fail(f"exclusive process active: {process_name}")
        if check.returncode != 1:
            fail(f"process check failed: {process_name}")
    before = memory_snapshot()
    if before["mem_available_kib"] < MIN_MEMORY_KIB:
        fail("less than 110 GiB available before CUDA microgate")
    if not args.binary.is_file() or not args.source_dir.is_dir():
        fail("binary or source directory missing")
    freeze = strict_json(args.freeze)
    receipt = strict_json(args.receipt)
    if freeze.get("schema") != "glm52-w6-indexer-tile-reuse-freeze-v2" or \
            freeze.get("candidate") != 2 or freeze.get("verdict") != "FROZEN":
        fail("freeze identity differs")
    source_commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=args.source_dir,
        text=True, capture_output=True, check=True).stdout.strip()
    if source_commit != freeze.get("source_commit"):
        fail("source HEAD differs from freeze")
    for diff_args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["/usr/bin/git", *diff_args], cwd=args.source_dir,
                          check=False).returncode != 0:
            fail("source tree has tracked changes")
    actual = {
        "ds4.c": sha256(args.source_dir / "ds4.c"),
        "ds4_cuda.cu": sha256(args.source_dir / "ds4_cuda.cu"),
        "tests/cuda_indexer_tile_reuse_w6.cu": sha256(
            args.source_dir / "tests/cuda_indexer_tile_reuse_w6.cu"),
        "Makefile": sha256(args.source_dir / "Makefile"),
        "scripts/106_score_w6_indexer_tile_reuse.py": sha256(SCORER),
        "scripts/107_run_w6_indexer_tile_reuse.py": sha256(Path(__file__)),
        "scripts/89_verify_drand_receipt.mjs": sha256(VERIFIER),
    }
    if freeze.get("artifact_sha256") != actual:
        fail("frozen artifact map differs")
    binary_sha = sha256(args.binary)
    if binary_sha != freeze.get("binary_sha256") or \
            args.binary.stat().st_size != freeze.get("binary_bytes"):
        fail("binary differs from freeze")
    if set(receipt) != {"round", "randomness", "signature", "previous_signature"}:
        fail("randomness receipt schema differs")
    verified = subprocess.run(
        [str(NODE), str(VERIFIER), str(receipt.get("round")),
         str(receipt.get("randomness")), str(receipt.get("signature")),
         str(receipt.get("previous_signature"))], cwd=ROOT, text=True,
        capture_output=True, timeout=30, check=False,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})
    if verified.returncode != 0 or verified.stdout != "DRAND_BLS_RECEIPT_OK\n":
        fail("drand receipt did not verify")
    round_value = receipt.get("round")
    freeze_time = freeze.get("freeze_time_unix")
    if not isinstance(round_value, int) or isinstance(round_value, bool) or \
            not isinstance(freeze_time, int) or \
            DRAND_GENESIS_UNIX + (round_value - 1) * DRAND_PERIOD_SECONDS <= freeze_time:
        fail("randomness was not published after freeze")
    randomness = receipt.get("randomness")
    if not isinstance(randomness, str) or len(randomness) != 64 or any(
            char not in "0123456789abcdef" for char in randomness):
        fail("randomness value invalid")
    schedule = schedule_from_randomness(randomness)
    schedule_env = ",".join(map(str, schedule))
    environment = {
        "HOME": "/nonexistent", "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64",
        "W6_TIMING_SCHEDULE": schedule_env,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen([str(args.binary)], cwd=args.source_dir, env=environment,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stat = Path(f"/proc/{process.pid}/stat").read_text().split()
    process_start_ticks = int(stat[21]) if len(stat) >= 22 else 0
    process_exe = Path(os.readlink(f"/proc/{process.pid}/exe")).resolve()
    try:
        stdout, stderr = process.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        process.kill(); process.communicate()
        fail("CUDA microgate timed out")
    completed_at = datetime.now(timezone.utc).isoformat()
    after = memory_snapshot()
    if process.returncode != 0 or stderr != "":
        fail(f"CUDA microgate failed: rc={process.returncode} stderr={stderr[-1000:]}")
    if len(stdout.splitlines()) != 16:
        fail("CUDA microgate output row count differs")

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.run_dir.name}.",
                                     dir=args.run_dir.parent) as tmp:
        staging = Path(tmp)
        stdout_path = staging / "microgate.stdout"
        stderr_path = staging / "microgate.stderr"
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        artifacts = {
            "binary": artifact(args.binary, "binary", staging),
            "ds4.c": artifact(args.source_dir / "ds4.c", "ds4.c", staging),
            "engine.cu": artifact(args.source_dir / "ds4_cuda.cu", "engine.cu", staging),
            "test.cu": artifact(args.source_dir / "tests/cuda_indexer_tile_reuse_w6.cu", "test.cu", staging),
            "Makefile": artifact(args.source_dir / "Makefile", "Makefile", staging),
            "scorer.py": artifact(SCORER, "scorer.py", staging),
            "runner.py": artifact(Path(__file__), "runner.py", staging),
            "drand-verifier.mjs": artifact(VERIFIER, "drand-verifier.mjs", staging),
            "randomness-receipt.json": artifact(args.receipt, "randomness-receipt.json", staging),
            "freeze.json": artifact(args.freeze, "freeze.json", staging),
            "microgate.stdout": artifact_record(stdout_path),
            "microgate.stderr": artifact_record(stderr_path),
        }
        raw_path = staging / "raw.jsonl"
        raw_path.write_text(stdout)
        gpu = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=name,uuid,driver_version",
             "--format=csv,noheader"], text=True, capture_output=True,
            timeout=10, check=True).stdout.strip().split(", ")
        if len(gpu) != 3 or not all(gpu):
            fail("GPU identity unavailable")
        manifest = {
            "schema": "glm52-w6-indexer-tile-reuse-manifest-v1",
            "gate": "W6-indexer-tile-reuse", "candidate": 2,
            "freeze_time_unix": freeze_time, "source_commit": source_commit,
            "binary_sha256": binary_sha, "scorer_sha256": sha256(SCORER),
            "runner_sha256": sha256(Path(__file__)), "raw_sha256": sha256(raw_path),
            "configuration": {
                "rows": 1048576, "ragged_rows": 1048575, "timed_tokens": 64,
                "heads": 32, "head_dim": 128, "top_k": 128,
                "samples_per_width": 5, "supported_widths": [1, 2, 4],
                "required_speedup_lower_95": 1.05,
                "traffic_semantics": "derived_logical_k_bytes_not_physical_traffic",
            },
            "randomness": {**receipt, "verification": verified.stdout.strip(),
                           "receipt_path": "randomness-receipt.json"},
            "schedule": schedule,
            "invocation": {"argv": ["binary"], "schedule_env": schedule_env,
                           "exit_code": process.returncode},
            "device": {"name": gpu[0], "uuid": gpu[1], "driver": gpu[2]},
            "safety": {
                "started_at": started_at, "completed_at": completed_at,
                "process_pid": process.pid, "process_start_ticks": process_start_ticks,
                "process_exe": str(process_exe), "executed_binary_sha256": binary_sha,
                "mem_available_before_kib": before["mem_available_kib"],
                "mem_available_after_kib": after["mem_available_kib"],
                "swap_used_before_kib": before["swap_used_kib"],
                "swap_used_after_kib": after["swap_used_kib"],
                "engine_processes_present": False, "fio_present": False, "failures": [],
            },
            "artifacts": artifacts,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n")
        scored = subprocess.run(
            ["/usr/bin/python3", str(staging / "scorer.py"), str(staging)],
            cwd=staging, text=True, capture_output=True, timeout=60, check=False,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"})
        if scored.returncode != 0:
            fail(f"fixed scorer failed: {scored.stderr}{scored.stdout}")
        summary = json.loads(scored.stdout)
        if summary.get("verdict") not in {"PASS", "NO_RESULT"}:
            fail("fixed scorer emitted no terminal verdict")
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, args.run_dir)
    print(json.dumps({"run_dir": str(args.run_dir), "verdict": summary["verdict"],
                      "selected_width": summary["selected_width"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
