#!/usr/bin/env python3
"""Contract tests for the read-only GLM NVMe characterization gate."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "results/glm52-gates/harness/qd_sweep.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("glm_nvme_characterization", PROBE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GlmNvmeCharacterizationTests(unittest.TestCase):
    def test_describe_preregisters_exact_matrix_without_touching_target(self):
        result = subprocess.run(
            [sys.executable, str(PROBE), "describe", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["block_sizes_bytes"], [
            1 << 20, 4 << 20, 9_732_096, 16 << 20,
        ])
        self.assertEqual(plan["iodepths"], [1, 4, 8, 16, 32])
        self.assertEqual(plan["numjobs"], [1, 4])
        self.assertEqual(plan["runtime_seconds"], 60)
        self.assertEqual(plan["matched_record_bytes"], 9_732_096)
        self.assertEqual(plan["tail_record_bytes"], 12_386_304)
        self.assertEqual(plan["cell_count"], 40)
        self.assertEqual(plan["tail_cell_count"], 2)
        self.assertEqual(plan["sequential_cell_count"], 3)
        self.assertEqual(plan["total_cell_count"], 45)

    def test_fio_arguments_are_read_only_direct_and_machine_parseable(self):
        probe = load_probe()
        argv = probe.build_fio_argv(
            Path("/evidence/slab"), block_size=9_732_096,
            iodepth=16, numjobs=4, runtime_seconds=60,
            offset=1_560_576, size=186_856_243_200,
            fio=Path("/opt/fio"), output=Path("/evidence/cell.json"),
        )
        joined = "\n".join(argv)
        for required in (
            "--readonly", "--rw=randread", "--direct=1",
            "--ioengine=io_uring", "--output-format=json",
            "--bs=9732096", "--iodepth=16", "--numjobs=4",
            "--runtime=60", "--time_based=1",
        ):
            self.assertIn(required, joined)
        self.assertNotIn("--readonly=1", argv)
        self.assertNotIn("write", "\n".join(argv).lower())

    def test_failed_cell_preserves_stderr(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr_path = root / "fio.stderr"
            with mock.patch.object(
                probe, "thermal_sample", return_value={"temperatures_c": {}}
            ):
                returncode, _ = probe.run_cell(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stderr.write('fio-readonly-error\\n'); sys.exit(3)",
                    ],
                    root,
                    stderr_path=stderr_path,
                )
            self.assertEqual(returncode, 3)
            self.assertEqual(stderr_path.read_text(), "fio-readonly-error\n")

    def test_exclusivity_allows_only_exact_launched_fio_command(self):
        probe = load_probe()
        code = (
            "import ctypes,time; "
            "ctypes.CDLL(None).prctl(15, b'fio', 0, 0, 0); "
            "time.sleep(30)"
        )
        argv = [sys.executable, "-c", code]
        process = subprocess.Popen(argv, start_new_session=True)
        try:
            for _ in range(100):
                if Path(f"/proc/{process.pid}/comm").read_text().strip() == "fio":
                    break
                time.sleep(0.01)
            conflicts = probe.conflicting_processes(
                allowed_process_group=-1,
                allowed_fio_argv=tuple(argv),
            )
            self.assertNotIn(process.pid, {row["pid"] for row in conflicts})
            conflicts = probe.conflicting_processes(allowed_process_group=-1)
            self.assertIn(process.pid, {row["pid"] for row in conflicts})
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def test_exclusivity_ignores_reaped_fio_zombie(self):
        probe = load_probe()
        code = (
            "import ctypes,os; "
            "ctypes.CDLL(None).prctl(15, b'fio', 0, 0, 0); "
            "os._exit(0)"
        )
        process = subprocess.Popen([sys.executable, "-c", code])
        try:
            for _ in range(200):
                try:
                    fields = Path(f"/proc/{process.pid}/stat").read_text().split()
                except FileNotFoundError:
                    self.fail("test process was unexpectedly reaped")
                if fields[2] == "Z":
                    break
                time.sleep(0.01)
            else:
                self.fail("test process did not enter zombie state")
            conflicts = probe.conflicting_processes(allowed_process_group=-1)
            self.assertNotIn(process.pid, {row["pid"] for row in conflicts})
        finally:
            process.wait()

    def test_target_validation_rejects_symlinks_and_non_regular_files(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "slab"
            target.touch()
            target.write_bytes(b"x")
            with target.open("r+b") as stream:
                stream.truncate(190_028_697_600)
            link = root / "link"
            link.symlink_to(target)
            identity = probe.validate_target(target)
            self.assertEqual(identity["size"], 190_028_697_600)
            wrong_size = root / "wrong-size"
            wrong_size.write_bytes(b"x")
            with wrong_size.open("r+b") as stream:
                stream.truncate(32 << 20)
            with self.assertRaises(ValueError):
                probe.validate_target(wrong_size)
            with self.assertRaises(ValueError):
                probe.validate_target(link)
            with self.assertRaises(ValueError):
                probe.validate_target(Path("/dev/null"))

    def test_gate_requires_exclusive_lock_no_engine_and_thermal_samples(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("inference.lock", source)
        self.assertIn("LOCK_EX | fcntl.LOCK_NB", source)
        self.assertIn('"ds4-server"', source)
        self.assertIn("/sys/class/hwmon", source)
        self.assertIn("temp1_alarm", source)
        self.assertIn("math.isfinite", source)

    def test_parser_rejects_nonfinite_bandwidth_and_any_write(self):
        probe = load_probe()
        base = {
            "fio version": "fio-3.36",
            "jobs": [{"error": 0, "read": {
                "bw_bytes": 12_000_000_000, "io_bytes": 720_000_000_000,
                "runtime": 60_000, "total_ios": 720_000,
            }, "write": {"io_bytes": 0, "total_ios": 0},
                "trim": {"io_bytes": 0, "total_ios": 0}}]
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "fio.json"
            result.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(probe.parse_fio_result(result)["bandwidth_gb_s"], 12.0)
            base["jobs"][0]["read"]["bw_bytes"] = float("nan")
            result.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.parse_fio_result(result)
            base["jobs"][0]["read"]["bw_bytes"] = 12_000_000_000
            base["jobs"][0]["write"]["io_bytes"] = 4096
            result.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.parse_fio_result(result)

    def test_parser_rejects_short_or_missing_status(self):
        probe = load_probe()
        malformed = {"jobs": [{
            "read": {"bw_bytes": 1, "io_bytes": 1, "runtime": 0},
            "write": {},
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "fio.json"
            result.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.parse_fio_result(result)

    def test_production_lock_cannot_be_overridden(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--lock"', source)
        self.assertNotIn('add_argument("--hwmon-root"', source)
        self.assertNotIn('add_argument("--target"', source)
        self.assertNotIn('add_argument("--target-sha256"', source)
        self.assertIn("G6-rung0-io-sidecar-build.json", source)

    def test_cell_schedule_covers_primary_tail_and_sequential_once(self):
        probe = load_probe()
        cells = probe.cell_specs(probe.EXPECTED_SLAB_SIZE)
        self.assertEqual(len(cells), 45)
        self.assertEqual(len({cell["name"] for cell in cells}), 45)
        self.assertEqual(sum(cell["kind"] == "primary" for cell in cells), 40)
        tails = [cell for cell in cells if cell["kind"] == "tail"]
        self.assertEqual(len(tails), 2)
        self.assertEqual({cell["block_size"] for cell in tails}, {12_386_304})
        self.assertEqual({cell["size"] for cell in tails}, {256 * 12_386_304})
        sequential = [cell for cell in cells if cell["kind"] == "sequential"]
        self.assertEqual(len(sequential), 3)
        self.assertEqual({cell["access"] for cell in sequential}, {"read"})
        orders = []
        for block_size in probe.BLOCK_SIZES:
            orders.append(tuple(
                cell["iodepth"] for cell in cells
                if cell["kind"] == "primary" and
                cell["block_size"] == block_size and cell["numjobs"] == 1
            ))
        self.assertEqual(len(set(orders)), len(probe.BLOCK_SIZES))

    def test_fixed_scorer_rejects_bad_qd1_and_duplicate_cells(self):
        probe = load_probe()
        rows = []
        for cell in probe.cell_specs(probe.EXPECTED_SLAB_SIZE):
            bandwidth = 4.8 if (
                cell["kind"] == "primary" and
                cell["block_size"] == probe.MATCHED_RECORD_BYTES and
                cell["iodepth"] == 1 and cell["numjobs"] == 1
            ) else 12.0
            rows.append({**cell, "bandwidth_gb_s": bandwidth,
                         "device_tail_gb_s": bandwidth,
                         "temperature_start_c": {"temp1_input": 55.0}})
        score = probe.score_sweep(rows)
        self.assertEqual(score["verdict"], "PASS")
        self.assertAlmostEqual(score["future_engine_80pct_target_gb_s"], 9.6)
        bad = [dict(row) for row in rows]
        for row in bad:
            if (row["kind"] == "primary" and
                    row["block_size"] == probe.MATCHED_RECORD_BYTES and
                    row["iodepth"] == 1 and row["numjobs"] == 1):
                row["bandwidth_gb_s"] = 1.0
        self.assertEqual(probe.score_sweep(bad)["verdict"], "NO_RESULT")
        self.assertEqual(probe.score_sweep(rows[:-1] + [rows[0]])["verdict"], "FAIL")

        high_qd = [row for row in rows if
                   row["kind"] == "primary" and
                   row["block_size"] == probe.MATCHED_RECORD_BYTES and
                   row["iodepth"] in (16, 32)]
        high_qd[-1]["bandwidth_gb_s"] = 100.0
        high_qd[-1]["device_tail_gb_s"] = 100.0
        robust = probe.score_sweep(rows)
        self.assertLess(robust["matched_sustained_reference_gb_s"], 20.0)

    def test_target_device_resolves_to_its_own_nvme_controller(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "devices" / "nvme" / "nvme7"
            block = controller / "nvme7n1" / "nvme7n1p2"
            block.mkdir(parents=True)
            dev = root / "dev-block"
            dev.mkdir()
            (dev / "259:2").symlink_to(block)
            resolved_block, resolved_controller = probe.resolve_nvme_controller(
                os.makedev(259, 2), dev,
            )
            self.assertEqual(resolved_block, block)
            self.assertEqual(resolved_controller, controller)

    def test_near_zero_terminal_diskstats_interval_is_rejected(self):
        probe = load_probe()
        samples = []
        for second in range(31):
            samples.append({
                "monotonic_ns": second * 1_000_000_000,
                "temperatures_c": {"temp1_input": 50.0},
                "temp1_alarm": 0, "temp1_max_c": 82.0,
                "disk": {"sectors_read": second * 1_000_000},
            })
        final = dict(samples[-1])
        final["monotonic_ns"] += 1
        final["disk"] = {"sectors_read": samples[-1]["disk"]["sectors_read"]}
        samples.append(final)
        with self.assertRaises(ValueError):
            probe.telemetry_metrics(samples)

    def test_smart_counter_increase_and_availability_drift_fail(self):
        probe = load_probe()
        values = {
            "critical_warning": 0, "warning_temp_time": 0,
            "critical_comp_time": 0, "thm_temp1_trans_count": 0,
            "thm_temp2_trans_count": 0, "thm_temp1_total_time": 0,
            "thm_temp2_total_time": 0,
        }
        probe.validate_smart_pair(
            {"available": True, "values": dict(values)},
            {"available": True, "values": dict(values)},
        )
        changed = dict(values)
        changed["thm_temp1_total_time"] = 1
        with self.assertRaises(ValueError):
            probe.validate_smart_pair(
                {"available": True, "values": dict(values)},
                {"available": True, "values": changed},
            )
        with self.assertRaises(ValueError):
            probe.validate_smart_pair({"available": False},
                                      {"available": True, "values": dict(values)})

    def test_telemetry_failure_terminates_and_reaps_fio(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "pid"
            raw = root / "raw.jsonl"
            code = (
                "import os,time,pathlib; "
                f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)"
            )
            calls = 0
            original = probe.thermal_sample

            def failing_thermal(_hwmon, _block=None):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise RuntimeError("telemetry disappeared")
                return {"monotonic_ns": 1, "temperatures_c": {"temp1_input": 40.0},
                        "temp1_alarm": 0}

            probe.thermal_sample = failing_thermal
            try:
                with self.assertRaises(RuntimeError):
                    probe.run_cell([sys.executable, "-c", code], root, raw)
                pid = int(pidfile.read_text(encoding="utf-8"))
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                probe.thermal_sample = original
                if pidfile.exists():
                    pid = int(pidfile.read_text(encoding="utf-8"))
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_fake_fio_full_schedule_reaches_pass_only_with_45_valid_cells(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "slab"
            target.touch()
            fio = root / "fio"
            fio.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fio.chmod(0o755)
            output = root / "evidence"
            identity = {
                "device": os.makedev(259, 2), "inode": 7,
                "size": probe.EXPECTED_SLAB_SIZE,
                "mtime_ns": 11, "ctime_ns": 12,
            }
            original = {
                name: getattr(probe, name) for name in (
                    "load_frozen_target", "validate_target", "verified_target_sha256",
                    "conflicting_processes", "lock_exclusively",
                    "resolve_nvme_controller", "find_nvme_hwmon",
                    "controller_identity", "file_sha256",
                    "wait_for_thermal_window", "smart_log_sample", "run_cell",
                )
            }

            def fake_run_cell(argv, _hwmon, _raw, **_kwargs):
                options = dict(
                    part[2:].split("=", 1)
                    for part in argv
                    if part.startswith("--") and "=" in part
                )
                block_size = int(options["bs"])
                desired = 4.8 if (
                    block_size == probe.MATCHED_RECORD_BYTES and
                    int(options["iodepth"]) == 1 and int(options["numjobs"]) == 1
                ) else 12.0
                total_ios = round(desired * 1e9 * 60 / block_size)
                io_bytes = total_ios * block_size
                bandwidth = io_bytes / 60
                document = {
                    "fio version": "fio-3.36",
                    "jobs": [{
                        "jobname": options["name"], "error": 0,
                        "read": {"bw_bytes": bandwidth, "io_bytes": io_bytes,
                                 "runtime": 60_000, "total_ios": total_ios},
                        "write": {"io_bytes": 0, "total_ios": 0},
                        "trim": {"io_bytes": 0, "total_ios": 0},
                    }],
                }
                Path(options["output"]).write_text(json.dumps(document), encoding="utf-8")
                sectors_per_second = round(bandwidth / 512)
                samples = []
                for second in range(61):
                    samples.append({
                        "monotonic_ns": second * 1_000_000_000,
                        "temperatures_c": {"temp1_input": 50.0},
                        "temp1_alarm": 0, "temp1_max_c": 82.0,
                        "temp1_crit_c": 85.0,
                        "disk": {"reads_completed": second * total_ios // 60,
                                 "sectors_read": second * sectors_per_second,
                                 "io_in_flight": 0, "io_time_ms": second * 1000},
                    })
                return 0, samples

            try:
                probe.load_frozen_target = lambda: {
                    "path": target, "sha256": "a" * 64,
                    "bytes": probe.EXPECTED_SLAB_SIZE,
                    "manifest": str(root / "manifest.json"),
                    "manifest_sha256": "c" * 64,
                }
                probe.validate_target = lambda _path: dict(identity)
                probe.verified_target_sha256 = lambda _path, _identity: "a" * 64
                probe.conflicting_processes = lambda **_kwargs: []
                probe.lock_exclusively = lambda _path: os.open(root / "lock", os.O_CREAT | os.O_RDWR)
                probe.resolve_nvme_controller = lambda _device: (root / "block", root / "nvme0")
                probe.find_nvme_hwmon = lambda _controller: root / "hwmon0"
                probe.controller_identity = lambda _controller: {"name": "nvme0"}
                probe.file_sha256 = lambda _path: "b" * 64
                probe.wait_for_thermal_window = lambda *_args, **_kwargs: {"ok": True}
                probe.smart_log_sample = lambda _controller: {"available": False}
                probe.run_cell = fake_run_cell
                rc = probe.run_sweep(SimpleNamespace(
                    fio=fio, output=output,
                ))
                self.assertEqual(rc, 0)
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["verdict"], "PASS")
                self.assertEqual(len(summary["cells"]), 45)
            finally:
                for name, value in original.items():
                    setattr(probe, name, value)


if __name__ == "__main__":
    unittest.main()
