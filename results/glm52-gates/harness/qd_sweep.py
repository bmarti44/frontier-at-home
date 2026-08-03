#!/usr/bin/env python3
"""Read-only fio/io_uring characterization of the frozen GLM expert sidecar."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any


BLOCK_SIZES = (1 << 20, 4 << 20, 9_732_096, 16 << 20)
IODEPTHS = (1, 4, 8, 16, 32)
NUMJOBS = (1, 4)
RUNTIME_SECONDS = 60
MATCHED_RECORD_BYTES = 9_732_096
MATCHED_RECORD_COUNT = 19_200
TAIL_RECORD_BYTES = 12_386_304
SLAB_DATA_OFFSET = 1_560_576
INFERENCE_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
HWMON_ROOT = Path("/sys/class/hwmon")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def describe() -> dict[str, Any]:
    return {
        "block_sizes_bytes": list(BLOCK_SIZES),
        "iodepths": list(IODEPTHS),
        "numjobs": list(NUMJOBS),
        "runtime_seconds": RUNTIME_SECONDS,
        "matched_record_bytes": MATCHED_RECORD_BYTES,
        "matched_record_count": MATCHED_RECORD_COUNT,
        "tail_record_bytes": TAIL_RECORD_BYTES,
        "slab_data_offset": SLAB_DATA_OFFSET,
        "cell_count": len(BLOCK_SIZES) * len(IODEPTHS) * len(NUMJOBS),
        "access": "randread",
        "direct": True,
        "ioengine": "io_uring",
        "target_kind": "regular-file-only",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_target(path: Path) -> dict[str, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"target must be a readable non-symlink: {path}") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("target must be a regular file; raw devices are forbidden")
    if value.st_size <= SLAB_DATA_OFFSET + max(BLOCK_SIZES):
        raise ValueError("target is too small for the preregistered sweep")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def build_fio_argv(
    target: Path, *, block_size: int, iodepth: int, numjobs: int,
    runtime_seconds: int, offset: int, size: int, fio: Path, output: Path,
) -> list[str]:
    return [
        str(fio),
        f"--name=glm-read-bs{block_size}-qd{iodepth}-j{numjobs}",
        f"--filename={target}",
        "--readonly=1",
        "--rw=randread",
        "--direct=1",
        "--ioengine=io_uring",
        f"--bs={block_size}",
        f"--iodepth={iodepth}",
        f"--numjobs={numjobs}",
        f"--runtime={runtime_seconds}",
        "--time_based=1",
        "--group_reporting=1",
        "--randrepeat=1",
        "--norandommap=1",
        "--invalidate=1",
        f"--offset={offset}",
        f"--size={size}",
        "--output-format=json",
        f"--output={output}",
    ]


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def find_nvme_hwmon(root: Path = HWMON_ROOT) -> Path:
    for candidate in sorted(root.glob("hwmon*")):
        try:
            if (candidate / "name").read_text(encoding="utf-8").strip() == "nvme":
                return candidate
        except (FileNotFoundError, PermissionError):
            continue
    raise RuntimeError("readable NVMe hwmon telemetry is unavailable")


def thermal_sample(hwmon: Path) -> dict[str, Any]:
    sensors: dict[str, float] = {}
    for value_path in sorted(hwmon.glob("temp*_input")):
        value = _read_int(value_path)
        if value is not None:
            sensors[value_path.stem] = value / 1000.0
    alarm = _read_int(hwmon / "temp1_alarm")
    maximum = _read_int(hwmon / "temp1_max")
    critical = _read_int(hwmon / "temp1_crit")
    if not sensors or alarm is None:
        raise RuntimeError("incomplete NVMe thermal telemetry")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "temperatures_c": sensors,
        "temp1_alarm": alarm,
        "temp1_max_c": None if maximum is None else maximum / 1000.0,
        "temp1_crit_c": None if critical is None else critical / 1000.0,
    }


def conflicting_processes() -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm == "ds4-server" or (
            "70_glm_rung0_slab_campaign.py" in cmdline and " run " in f" {cmdline} "
        ):
            conflicts.append({"pid": int(entry.name), "comm": comm, "cmdline": cmdline})
    return conflicts


def lock_exclusively(path: Path = INFERENCE_LOCK) -> int:
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def parse_fio_result(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    jobs = document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("fio result has no jobs")
    bandwidth = 0.0
    io_bytes = 0
    runtime_ms = 0
    for job in jobs:
        if job.get("error") not in (0, None):
            raise ValueError("fio job failed")
        read = job.get("read", {})
        write = job.get("write", {})
        if int(write.get("io_bytes", 0)) != 0:
            raise ValueError("fio reported a forbidden write")
        bandwidth += float(read.get("bw_bytes", 0))
        io_bytes += int(read.get("io_bytes", 0))
        runtime_ms = max(runtime_ms, int(read.get("runtime", 0)))
    if not math.isfinite(bandwidth) or bandwidth <= 0 or io_bytes <= 0:
        raise ValueError("fio bandwidth is missing or non-finite")
    return {
        "bandwidth_bytes_s": bandwidth,
        "bandwidth_gb_s": bandwidth / 1e9,
        "read_bytes": io_bytes,
        "runtime_ms": runtime_ms,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cell_geometry(file_size: int, block_size: int) -> tuple[int, int]:
    if block_size == MATCHED_RECORD_BYTES:
        return SLAB_DATA_OFFSET, MATCHED_RECORD_COUNT * MATCHED_RECORD_BYTES
    usable = file_size - SLAB_DATA_OFFSET
    return SLAB_DATA_OFFSET, usable // block_size * block_size


def run_cell(argv: list[str], hwmon: Path) -> tuple[int, list[dict[str, Any]]]:
    samples = [thermal_sample(hwmon)]
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    while process.poll() is None:
        time.sleep(1)
        sample = thermal_sample(hwmon)
        samples.append(sample)
    sample = thermal_sample(hwmon)
    samples.append(sample)
    return process.returncode, samples


def run_sweep(args: argparse.Namespace) -> int:
    # Make the path absolute without resolving its final component: validate_target
    # must see and reject a symlink instead of silently accepting its destination.
    target = Path(os.path.abspath(args.target))
    fio = args.fio.resolve(strict=True)
    if not os.access(fio, os.X_OK) or not stat.S_ISREG(fio.stat().st_mode):
        raise ValueError("fio must be an executable regular file")
    before = validate_target(target)
    if not SHA256_RE.fullmatch(args.target_sha256):
        raise ValueError("target SHA-256 must be 64 lowercase hexadecimal characters")
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError(f"exclusive measurement blocked by: {conflicts}")
    lock_fd = lock_exclusively(args.lock)
    try:
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"exclusive measurement race: {conflicts}")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        cells_dir = output / "cells"
        cells_dir.mkdir()
        raw_log = output / "raw.jsonl"
        hwmon = find_nvme_hwmon(args.hwmon_root)
        rows: list[dict[str, Any]] = []
        for block_size in BLOCK_SIZES:
            offset, size = cell_geometry(before["size"], block_size)
            for iodepth in IODEPTHS:
                for numjobs in NUMJOBS:
                    name = f"bs{block_size}-qd{iodepth}-j{numjobs}"
                    result_path = cells_dir / f"{name}.json"
                    argv = build_fio_argv(
                        target, block_size=block_size, iodepth=iodepth,
                        numjobs=numjobs, runtime_seconds=RUNTIME_SECONDS,
                        offset=offset, size=size, fio=fio, output=result_path,
                    )
                    append_jsonl(raw_log, {"type": "cell_start", "name": name,
                                           "argv": argv, "monotonic_ns": time.monotonic_ns()})
                    returncode, thermals = run_cell(argv, hwmon)
                    if returncode != 0:
                        raise RuntimeError(f"fio failed for {name}: rc={returncode}")
                    metric = parse_fio_result(result_path)
                    if any(sample["temp1_alarm"] != 0 for sample in thermals):
                        raise RuntimeError(f"NVMe thermal alarm during {name}")
                    for sample in thermals:
                        append_jsonl(raw_log, {"type": "thermal", "cell": name, **sample})
                    row = {
                        "name": name, "block_size": block_size,
                        "iodepth": iodepth, "numjobs": numjobs,
                        "offset": offset, "size": size, **metric,
                        "temperature_start_c": thermals[0]["temperatures_c"],
                        "temperature_end_c": thermals[-1]["temperatures_c"],
                        "temperature_peak_c": max(
                            value for sample in thermals
                            for value in sample["temperatures_c"].values()
                        ),
                    }
                    rows.append(row)
                    append_jsonl(raw_log, {"type": "cell_result", **row})
                    if validate_target(target) != before:
                        raise RuntimeError("target identity changed during sweep")
        summary = {
            "schema_version": 1,
            "plan": describe(),
            "target": {
                **before,
                "path": str(target),
                "frozen_sha256": args.target_sha256,
                "digest_semantics": (
                    "caller-supplied frozen digest; stat identity held stable "
                    "throughout this read-only sweep"
                ),
            },
            "fio": {"path": str(fio), "sha256": file_sha256(fio)},
            "hwmon": str(hwmon),
            "cells": rows,
            "verdict": "PASS",
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    finally:
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    describe_parser = commands.add_parser("describe")
    describe_parser.add_argument("--json", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--target", type=Path, required=True)
    run_parser.add_argument("--target-sha256", required=True)
    run_parser.add_argument("--fio", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--lock", type=Path, default=INFERENCE_LOCK)
    run_parser.add_argument("--hwmon-root", type=Path, default=HWMON_ROOT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "describe":
        if args.json:
            print(json.dumps(describe(), sort_keys=True))
        else:
            print(json.dumps(describe(), indent=2, sort_keys=True))
        return 0
    return run_sweep(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"qd_sweep.py: {error}", file=sys.stderr)
        raise SystemExit(1)
