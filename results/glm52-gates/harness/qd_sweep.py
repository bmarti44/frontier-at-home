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
import signal
import stat
import statistics
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
EXPECTED_SLAB_SIZE = 190_028_697_600
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_TARGET_MANIFEST = (
    REPOSITORY_ROOT / "results/glm52-gates/G6-rung0-io-sidecar-build.json"
)
INFERENCE_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
HWMON_ROOT = Path("/sys/class/hwmon")
SYS_DEV_BLOCK = Path("/sys/dev/block")
NVME_CLI = Path("/usr/sbin/nvme")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRIMARY_QD_ORDER = (1, 32, 4, 16, 8)
COOLDOWN_C = 70.0
THERMAL_MARGIN_C = 2.0
QD1_REFERENCE_GB_S = 4.8
QD1_RELATIVE_TOLERANCE = 0.25


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
        "tail_cell_count": 2,
        "sequential_cell_count": 3,
        "total_cell_count": 45,
        "access": "randread",
        "direct": True,
        "ioengine": "io_uring",
        "target_kind": "regular-file-only",
        "target_manifest": str(FROZEN_TARGET_MANIFEST.relative_to(REPOSITORY_ROOT)),
    }


def load_frozen_target() -> dict[str, Any]:
    document = json.loads(FROZEN_TARGET_MANIFEST.read_text(encoding="utf-8"))
    sidecar = document.get("sidecar")
    if (
        document.get("schema_version") != 1 or
        document.get("gate") != "G6-rung0-io-sidecar-build" or
        document.get("verdict") != "PASS" or
        not isinstance(sidecar, dict) or
        sidecar.get("format_version") != 2 or
        sidecar.get("records") != 19_456 or
        sidecar.get("bytes") != EXPECTED_SLAB_SIZE or
        not isinstance(sidecar.get("path"), str) or
        not SHA256_RE.fullmatch(str(sidecar.get("generated_content_sha256", "")))
    ):
        raise ValueError("committed G6 frozen-sidecar manifest is malformed")
    return {
        "path": Path(sidecar["path"]),
        "sha256": sidecar["generated_content_sha256"],
        "bytes": sidecar["bytes"],
        "manifest": str(FROZEN_TARGET_MANIFEST),
        "manifest_sha256": file_sha256(FROZEN_TARGET_MANIFEST),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_target_sha256(path: Path, identity: dict[str, int]) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    offset = 0
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_size != identity["size"]:
            raise ValueError("target identity changed before full hash")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while True:
                chunk = stream.read(16 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                offset += len(chunk)
                if (hasattr(os, "posix_fadvise") and
                        offset % (256 << 20) < len(chunk)):
                    os.posix_fadvise(
                        descriptor, max(0, offset - (256 << 20)), 256 << 20,
                        os.POSIX_FADV_DONTNEED,
                    )
                if offset % (1 << 30) < len(chunk):
                    conflicts = conflicting_processes()
                    if conflicts:
                        raise RuntimeError(
                            f"exclusive workload appeared during identity hash: {conflicts}"
                        )
        after = os.fstat(descriptor)
        observed = {
            "device": after.st_dev, "inode": after.st_ino,
            "size": after.st_size, "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        }
        if observed != identity:
            raise RuntimeError("target identity changed during full hash")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


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
    if value.st_size != EXPECTED_SLAB_SIZE:
        raise ValueError(
            f"target size is not the frozen slab size: {value.st_size} != "
            f"{EXPECTED_SLAB_SIZE}"
        )
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
    access: str = "randread", name: str | None = None,
) -> list[str]:
    if access not in ("randread", "read"):
        raise ValueError("unsupported fio access pattern")
    job_name = name or f"glm-read-bs{block_size}-qd{iodepth}-j{numjobs}"
    return [
        str(fio),
        f"--name={job_name}",
        f"--filename={target}",
        "--readonly",
        f"--rw={access}",
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


def resolve_nvme_controller(
    device: int, sys_dev_block: Path = SYS_DEV_BLOCK,
) -> tuple[Path, Path]:
    major_minor = f"{os.major(device)}:{os.minor(device)}"
    block = (sys_dev_block / major_minor).resolve(strict=True)
    controller = next(
        (parent for parent in (block, *block.parents)
         if re.fullmatch(r"nvme\d+", parent.name)),
        None,
    )
    if controller is None:
        raise RuntimeError(f"target device {major_minor} is not backed by NVMe")
    return block, controller


def find_nvme_hwmon(controller: Path) -> Path:
    for candidate in sorted(controller.glob("hwmon*")):
        try:
            if (candidate / "name").read_text(encoding="utf-8").strip() == "nvme":
                return candidate
        except (FileNotFoundError, PermissionError):
            continue
    raise RuntimeError("readable NVMe hwmon telemetry is unavailable")


def controller_identity(controller: Path) -> dict[str, str]:
    result: dict[str, str] = {"sysfs_path": str(controller.resolve())}
    for key, filename in (
        ("name", "name"), ("model", "model"),
        ("firmware_rev", "firmware_rev"), ("state", "state"),
    ):
        try:
            result[key] = (controller / filename).read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError):
            result[key] = "unavailable"
    device = controller / "device"
    for key, filename in (
        ("current_link_speed", "current_link_speed"),
        ("current_link_width", "current_link_width"),
    ):
        try:
            result[key] = (device / filename).read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError):
            result[key] = "unavailable"
    return result


def read_diskstats(block: Path) -> dict[str, int]:
    fields = (block / "stat").read_text(encoding="utf-8").split()
    if len(fields) < 11:
        raise RuntimeError("incomplete target NVMe diskstats")
    return {
        "reads_completed": int(fields[0]),
        "sectors_read": int(fields[2]),
        "io_in_flight": int(fields[8]),
        "io_time_ms": int(fields[9]),
    }


def smart_log_sample(controller: Path, nvme_cli: Path = NVME_CLI) -> dict[str, Any]:
    device = Path("/dev") / controller.name
    try:
        result = subprocess.run(
            [str(nvme_cli), "smart-log", str(device), "-o", "json"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}
    if result.returncode != 0:
        return {
            "available": False,
            "reason": "permission-or-command-failure",
            "output_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        }
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"available": False, "reason": "malformed-json"}
    wanted = (
        "critical_warning", "temperature", "warning_temp_time",
        "critical_comp_time", "thm_temp1_trans_count", "thm_temp2_trans_count",
        "thm_temp1_total_time", "thm_temp2_total_time",
    )
    return {"available": True, "values": {key: data.get(key) for key in wanted}}


def validate_smart_pair(before: dict[str, Any], after: dict[str, Any]) -> None:
    if bool(before.get("available")) != bool(after.get("available")):
        raise ValueError("NVMe SMART availability changed during cell")
    if not before.get("available"):
        return
    before_values = before.get("values")
    after_values = after.get("values")
    if not isinstance(before_values, dict) or not isinstance(after_values, dict):
        raise ValueError("NVMe SMART values are malformed")

    def numeric(values: dict[str, Any], key: str) -> int:
        value = values.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value, 0)
        raise ValueError(f"NVMe SMART field is missing: {key}")

    if numeric(after_values, "critical_warning") != 0:
        raise ValueError("NVMe SMART critical warning is nonzero")
    for key in (
        "warning_temp_time", "critical_comp_time",
        "thm_temp1_trans_count", "thm_temp2_trans_count",
        "thm_temp1_total_time", "thm_temp2_total_time",
    ):
        first = numeric(before_values, key)
        second = numeric(after_values, key)
        if second != first:
            raise ValueError(f"NVMe SMART thermal counter changed: {key}")


def thermal_sample(hwmon: Path, block: Path | None = None) -> dict[str, Any]:
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
    result = {
        "monotonic_ns": time.monotonic_ns(),
        "temperatures_c": sensors,
        "temp1_alarm": alarm,
        "temp1_max_c": None if maximum is None else maximum / 1000.0,
        "temp1_crit_c": None if critical is None else critical / 1000.0,
    }
    if block is not None:
        result["disk"] = read_diskstats(block)
    return result


def wait_for_thermal_window(
    hwmon: Path, block: Path, *, timeout_seconds: int = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"exclusive workload appeared during cooldown: {conflicts}")
        sample = thermal_sample(hwmon, block)
        composite = sample["temperatures_c"].get("temp1_input")
        maximum = sample["temp1_max_c"]
        if (
            composite is not None and maximum is not None and
            sample["temp1_alarm"] == 0 and composite <= COOLDOWN_C and
            composite <= maximum - THERMAL_MARGIN_C
        ):
            return sample
        if time.monotonic() >= deadline:
            raise RuntimeError("NVMe did not return to the preregistered thermal window")
        time.sleep(5)


def conflicting_processes(
    *,
    allowed_process_group: int | None = None,
    allowed_fio_argv: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            if allowed_process_group is not None and os.getpgid(int(entry.name)) == allowed_process_group:
                continue
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            raw_cmdline = (entry / "cmdline").read_bytes()
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            close_paren = stat_line.rfind(")")
            if close_paren < 0:
                continue
            process_state = stat_line[close_paren + 2:].split()[0]
            command = tuple(
                part.decode("utf-8", errors="replace")
                for part in raw_cmdline.split(b"\0")
                if part
            )
            cmdline = raw_cmdline.replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # A zombie has no address space and cannot submit I/O. /proc can retain
        # comm while cmdline has already been cleared during normal fio exit.
        if process_state == "Z":
            continue
        if comm == "fio" and allowed_fio_argv is not None and command == allowed_fio_argv:
            continue
        accelerator = False
        try:
            for descriptor in (entry / "fd").iterdir():
                try:
                    destination = os.readlink(descriptor)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if (
                    destination.startswith("/dev/nvidia") or
                    destination == "/dev/kfd" or
                    destination.startswith("/dev/dri/render")
                ):
                    accelerator = True
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        conflicting_comms = {
            "ds4", "ds4-server", "ds4-bench", "ds4-eval",
            "llama-server", "fio",
        }
        conflicting_markers = (
            "70_glm_rung0_slab_campaign.py",
            "glm_safe_run.sh",
            "glm_cgroup_run.sh",
            "68_measure_cuda_bandwidth.sh",
            "regression-suite.py",
        )
        if (comm in conflicting_comms or accelerator or
                any(marker in cmdline for marker in conflicting_markers)):
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


def parse_fio_result(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document.get("fio version"), str):
        raise ValueError("fio result has no version")
    jobs = document.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("fio result must contain exactly one grouped job")
    bandwidth = 0.0
    io_bytes = 0
    runtime_ms = 0
    achieved_iodepth_level: dict[str, float] = {}
    for job in jobs:
        if job.get("error") != 0:
            raise ValueError("fio job failed")
        if expected is not None and job.get("jobname") != expected["name"]:
            raise ValueError("fio job name does not match the requested cell")
        read = job.get("read")
        write = job.get("write")
        trim = job.get("trim")
        if not isinstance(read, dict) or not isinstance(write, dict) or not isinstance(trim, dict):
            raise ValueError("fio result is missing operation records")
        if int(write.get("io_bytes", -1)) != 0 or int(write.get("total_ios", -1)) != 0:
            raise ValueError("fio reported a forbidden write")
        if int(trim.get("io_bytes", -1)) != 0 or int(trim.get("total_ios", -1)) != 0:
            raise ValueError("fio reported a forbidden trim")
        bandwidth += float(read.get("bw_bytes", 0))
        io_bytes += int(read.get("io_bytes", 0))
        runtime_ms = max(runtime_ms, int(read.get("runtime", 0)))
        total_ios = int(read.get("total_ios", 0))
        if total_ios <= 0:
            raise ValueError("fio completed no reads")
        if expected is not None and total_ios * expected["block_size"] != int(read["io_bytes"]):
            raise ValueError("fio read count and byte count disagree")
        levels = job.get("iodepth_level")
        if levels is not None:
            if not isinstance(levels, dict):
                raise ValueError("fio achieved queue-depth distribution is malformed")
            expected_depth_buckets = {"1", "2", "4", "8", "16", "32", ">=64"}
            if set(levels) != expected_depth_buckets:
                raise ValueError(
                    "fio achieved queue depth distribution has missing or extra buckets"
                )
            for key, value in levels.items():
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0 or numeric > 100.1:
                    raise ValueError("fio achieved queue-depth percentage is invalid")
                achieved_iodepth_level[key] = numeric
            achieved_total = sum(achieved_iodepth_level.values())
            if not 99.0 <= achieved_total <= 101.0:
                raise ValueError(
                    "fio achieved queue depth distribution total is implausible"
                )
        if expected is not None and "iodepth" in expected:
            requested = str(expected["iodepth"])
            if (not achieved_iodepth_level or
                    achieved_iodepth_level.get(requested, 0.0) < 99.0):
                raise ValueError(
                    "fio achieved queue depth does not match requested depth"
                )
    if not math.isfinite(bandwidth) or bandwidth <= 0 or io_bytes <= 0:
        raise ValueError("fio bandwidth is missing or non-finite")
    if runtime_ms < 59_000 or runtime_ms > 75_000:
        raise ValueError("fio runtime is outside the 60-second tolerance")
    observed = io_bytes / (runtime_ms / 1000.0)
    if abs(observed - bandwidth) / bandwidth > 0.15:
        raise ValueError("fio bandwidth disagrees with bytes/runtime")
    return {
        "bandwidth_bytes_s": bandwidth,
        "bandwidth_gb_s": bandwidth / 1e9,
        "read_bytes": io_bytes,
        "runtime_ms": runtime_ms,
        "achieved_iodepth_level": achieved_iodepth_level,
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


def cell_specs(file_size: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for block_index, block_size in enumerate(BLOCK_SIZES):
        offset, size = cell_geometry(file_size, block_size)
        qd_order = PRIMARY_QD_ORDER[block_index:] + PRIMARY_QD_ORDER[:block_index]
        for q_index, iodepth in enumerate(qd_order):
            jobs = (NUMJOBS if (block_index + q_index) % 2 == 0
                    else tuple(reversed(NUMJOBS)))
            for numjobs in jobs:
                specs.append({
                    "kind": "primary", "access": "randread",
                    "name": f"primary-bs{block_size}-qd{iodepth}-j{numjobs}",
                    "block_size": block_size, "iodepth": iodepth,
                    "numjobs": numjobs, "offset": offset, "size": size,
                })
    tail_offset = SLAB_DATA_OFFSET + MATCHED_RECORD_COUNT * MATCHED_RECORD_BYTES
    tail_size = 256 * TAIL_RECORD_BYTES
    for iodepth in (1, 16):
        specs.append({
            "kind": "tail", "access": "randread",
            "name": f"tail-bs{TAIL_RECORD_BYTES}-qd{iodepth}-j1",
            "block_size": TAIL_RECORD_BYTES, "iodepth": iodepth,
            "numjobs": 1, "offset": tail_offset, "size": tail_size,
        })
    sequential_size = (file_size - SLAB_DATA_OFFSET) // (16 << 20) * (16 << 20)
    for iodepth in (1, 16, 32):
        specs.append({
            "kind": "sequential", "access": "read",
            "name": f"sequential-bs{16 << 20}-qd{iodepth}-j1",
            "block_size": 16 << 20, "iodepth": iodepth,
            "numjobs": 1, "offset": SLAB_DATA_OFFSET, "size": sequential_size,
        })
    return specs


def telemetry_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 30:
        raise ValueError("insufficient one-second telemetry coverage")
    rates: list[float] = []
    for first, second in zip(samples, samples[1:]):
        elapsed = (second["monotonic_ns"] - first["monotonic_ns"]) / 1e9
        sectors = second["disk"]["sectors_read"] - first["disk"]["sectors_read"]
        if elapsed < 0.5 or elapsed > 2.5 or sectors < 0:
            raise ValueError("invalid diskstats interval or sampling cadence")
        rates.append(sectors * 512 / elapsed / 1e9)
    if len(rates) < 20:
        raise ValueError("insufficient diskstats rate intervals")
    split = len(rates) // 2
    head = sum(rates[:split]) / split
    tail_values = rates[split:]
    tail = sum(tail_values) / len(tail_values)
    mean = sum(rates) / len(rates)
    peak_c = max(
        value for sample in samples for value in sample["temperatures_c"].values()
    )
    max_limits = [sample["temp1_max_c"] for sample in samples
                  if sample["temp1_max_c"] is not None]
    if any(sample["temp1_alarm"] != 0 for sample in samples):
        raise ValueError("NVMe thermal alarm during cell")
    if not max_limits or peak_c > min(max_limits) - THERMAL_MARGIN_C:
        raise ValueError("NVMe temperature entered the throttle-margin window")
    if head > 0 and tail < 0.85 * head:
        raise ValueError("sustained device bandwidth decayed during the cell")
    return {
        "device_head_gb_s": head,
        "device_tail_gb_s": tail,
        "device_mean_gb_s": mean,
        "device_interval_min_gb_s": min(rates),
        "device_interval_max_gb_s": max(rates),
        "temperature_peak_c": peak_c,
        "temperature_start_c": samples[0]["temperatures_c"],
        "temperature_end_c": samples[-1]["temperatures_c"],
    }


def score_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = cell_specs(EXPECTED_SLAB_SIZE)
    expected_names = {cell["name"] for cell in expected}
    names = [row.get("name") for row in rows]
    if len(rows) != len(expected) or len(set(names)) != len(names) or set(names) != expected_names:
        return {"verdict": "FAIL", "reason": "missing, duplicate, or unexpected cells"}
    qd1 = next(row for row in rows if
               row["kind"] == "primary" and
               row["block_size"] == MATCHED_RECORD_BYTES and
               row["iodepth"] == 1 and row["numjobs"] == 1)
    low = QD1_REFERENCE_GB_S * (1.0 - QD1_RELATIVE_TOLERANCE)
    high = QD1_REFERENCE_GB_S * (1.0 + QD1_RELATIVE_TOLERANCE)
    qd1_sustained = min(qd1["bandwidth_gb_s"], qd1["device_tail_gb_s"])
    if not low <= qd1_sustained <= high:
        return {
            "verdict": "NO_RESULT",
            "reason": "matched QD1 cell did not reproduce the committed method",
            "qd1_expected_gb_s": QD1_REFERENCE_GB_S,
            "qd1_allowed_gb_s": [low, high],
            "qd1_observed_gb_s": qd1_sustained,
            "qd1_fio_average_gb_s": qd1["bandwidth_gb_s"],
            "qd1_device_tail_gb_s": qd1["device_tail_gb_s"],
        }
    matched = [row for row in rows if
               row["kind"] == "primary" and
               row["block_size"] == MATCHED_RECORD_BYTES and
               row["iodepth"] in (16, 32)]
    sequential = [row for row in rows if
                  row["kind"] == "sequential" and row["iodepth"] in (16, 32)]

    def matched_reference(candidates: list[dict[str, Any]]) -> tuple[float, list[str]]:
        temperatures = [row["temperature_start_c"]["temp1_input"] for row in candidates]
        center = statistics.median(temperatures)
        eligible = [row for row in candidates if
                    abs(row["temperature_start_c"]["temp1_input"] - center) <= 2.0]
        if len(eligible) < 2 or {row["iodepth"] for row in eligible} != {16, 32}:
            raise ValueError("QD16/QD32 cells lack thermally matched observations")
        values = [min(row["bandwidth_gb_s"], row["device_tail_gb_s"])
                  for row in eligible]
        return statistics.median(values), [row["name"] for row in eligible]

    try:
        matched_value, matched_cells = matched_reference(matched)
        sequential_value, sequential_cells = matched_reference(sequential)
    except (KeyError, TypeError, ValueError) as error:
        return {"verdict": "NO_RESULT", "reason": str(error)}
    return {
        "verdict": "PASS",
        "qd1_expected_gb_s": QD1_REFERENCE_GB_S,
        "qd1_allowed_gb_s": [low, high],
        "qd1_observed_gb_s": qd1_sustained,
        "qd1_fio_average_gb_s": qd1["bandwidth_gb_s"],
        "qd1_device_tail_gb_s": qd1["device_tail_gb_s"],
        "matched_sustained_reference_gb_s": matched_value,
        "matched_reference_cells": matched_cells,
        "future_engine_80pct_target_gb_s": 0.8 * matched_value,
        "sequential_sustained_reference_gb_s": sequential_value,
        "sequential_reference_cells": sequential_cells,
        "future_identity_scan_80pct_target_gb_s": 0.8 * sequential_value,
        "sustained_formula": (
            "median(min(fio_average,device_tail_average)) over start-temperature-"
            "matched QD16/QD32 cells"
        ),
    }


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    else:
        process.wait()
    deadline = time.monotonic() + 5
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            time.sleep(0.1)
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            raise RuntimeError("fio process group survived SIGKILL")
        time.sleep(0.1)


def run_cell(
    argv: list[str], hwmon: Path, raw_log: Path | None = None,
    *, block: Path | None = None, cell_name: str = "test-cell",
    stderr_path: Path | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    samples = [thermal_sample(hwmon, block)]
    stderr_stream = stderr_path.open("xb") if stderr_path is not None else None
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=(stderr_stream if stderr_stream is not None else subprocess.DEVNULL),
            start_new_session=True,
        )
    except BaseException:
        if stderr_stream is not None:
            stderr_stream.close()
        raise
    try:
        while process.poll() is None:
            time.sleep(1)
            conflicts = conflicting_processes(
                allowed_process_group=process.pid,
                allowed_fio_argv=tuple(argv),
            )
            if conflicts:
                raise RuntimeError(f"exclusive workload appeared during cell: {conflicts}")
            samples.append(thermal_sample(hwmon, block))
        return process.returncode, samples
    finally:
        terminate_process_group(process)
        if stderr_stream is not None:
            stderr_stream.close()
        if raw_log is not None:
            for sample in samples:
                append_jsonl(raw_log, {"type": "telemetry", "cell": cell_name, **sample})
            if stderr_path is not None:
                append_jsonl(raw_log, {
                    "type": "cell_stderr", "cell": cell_name,
                    "bytes": stderr_path.stat().st_size,
                    "sha256": file_sha256(stderr_path),
                })


def run_sweep(args: argparse.Namespace) -> int:
    frozen_target = load_frozen_target()
    # Make the committed path absolute without resolving its final component:
    # validate_target must see and reject a symlink rather than its destination.
    target = Path(os.path.abspath(frozen_target["path"]))
    fio = args.fio.resolve(strict=True)
    if not os.access(fio, os.X_OK) or not stat.S_ISREG(fio.stat().st_mode):
        raise ValueError("fio must be an executable regular file")
    before = validate_target(target)
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError(f"exclusive measurement blocked by: {conflicts}")
    lock_fd = lock_exclusively(INFERENCE_LOCK)
    try:
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"exclusive measurement race: {conflicts}")
        before_digest = verified_target_sha256(target, before)
        if validate_target(target) != before or before_digest != frozen_target["sha256"]:
            raise ValueError("target stat or full SHA-256 does not match the frozen slab")
        block, controller = resolve_nvme_controller(before["device"])
        hwmon = find_nvme_hwmon(controller)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        cells_dir = output / "cells"
        cells_dir.mkdir()
        raw_log = output / "raw.jsonl"
        rows: list[dict[str, Any]] = []
        base_summary = {
            "schema_version": 2,
            "plan": describe(),
            "target": {
                **before, "path": str(target),
                "frozen_sha256": frozen_target["sha256"],
                "frozen_manifest": frozen_target["manifest"],
                "frozen_manifest_sha256": frozen_target["manifest_sha256"],
                "verified_sha256_before": before_digest,
            },
            "fio": {"path": str(fio), "sha256": file_sha256(fio)},
            "nvme_controller": controller_identity(controller),
            "block_sysfs": str(block),
            "hwmon": str(hwmon),
        }
        try:
            for spec in cell_specs(before["size"]):
                conflicts = conflicting_processes()
                if conflicts:
                    raise RuntimeError(f"exclusive workload before cell: {conflicts}")
                cooldown = wait_for_thermal_window(hwmon, block)
                smart_before = smart_log_sample(controller)
                name = spec["name"]
                result_path = cells_dir / f"{name}.json"
                argv = build_fio_argv(
                    target, block_size=spec["block_size"],
                    iodepth=spec["iodepth"], numjobs=spec["numjobs"],
                    runtime_seconds=RUNTIME_SECONDS, offset=spec["offset"],
                    size=spec["size"], fio=fio, output=result_path,
                    access=spec["access"], name=name,
                )
                config_sha256 = hashlib.sha256(
                    json.dumps(argv, separators=(",", ":")).encode()
                ).hexdigest()
                append_jsonl(raw_log, {
                    "type": "cell_start", "name": name, "argv": argv,
                    "config_sha256": config_sha256, "cooldown": cooldown,
                    "smart_before": smart_before,
                    "monotonic_ns": time.monotonic_ns(),
                })
                returncode, samples = run_cell(
                    argv, hwmon, raw_log, block=block, cell_name=name,
                    stderr_path=cells_dir / f"{name}.stderr",
                )
                if returncode != 0:
                    raise RuntimeError(f"fio failed for {name}: rc={returncode}")
                metric = parse_fio_result(result_path, spec)
                telemetry = telemetry_metrics(samples)
                ratio = metric["bandwidth_gb_s"] / telemetry["device_mean_gb_s"]
                if not 0.80 <= ratio <= 1.20:
                    raise RuntimeError(
                        f"fio/device read accounting mismatch for {name}: {ratio:.3f}"
                    )
                smart_after = smart_log_sample(controller)
                validate_smart_pair(smart_before, smart_after)
                row = {
                    **spec, **metric, **telemetry,
                    "fio_device_bandwidth_ratio": ratio,
                    "config_sha256": config_sha256,
                    "smart_before": smart_before,
                    "smart_after": smart_after,
                }
                rows.append(row)
                append_jsonl(raw_log, {"type": "cell_result", **row})
                if validate_target(target) != before:
                    raise RuntimeError("target identity changed during sweep")
            after_digest = verified_target_sha256(target, before)
            if validate_target(target) != before or after_digest != before_digest:
                raise RuntimeError("target stat or full SHA-256 changed during sweep")
            score = score_sweep(rows)
            summary = {
                **base_summary,
                "target": {**base_summary["target"],
                           "verified_sha256_after": after_digest},
                "cells": rows,
                "score": score,
                "verdict": score["verdict"],
                "smart_limitation": (
                    "NVMe SMART is recorded when the unprivileged device permits it; "
                    "controller-bound hwmon, alarm/max margin, diskstats, and "
                    "within-cell sustained-bandwidth decay are mandatory regardless."
                ),
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0 if score["verdict"] == "PASS" else 1
        except BaseException as error:
            append_jsonl(raw_log, {
                "type": "terminal_failure", "error_type": type(error).__name__,
                "error": str(error), "completed_cells": len(rows),
            })
            failure = {
                **base_summary, "cells": rows, "verdict": "FAIL",
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
            (output / "summary.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise
    finally:
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    describe_parser = commands.add_parser("describe")
    describe_parser.add_argument("--json", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--fio", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
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
