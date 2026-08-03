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
import unittest


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
            "--readonly=1", "--rw=randread", "--direct=1",
            "--ioengine=io_uring", "--output-format=json",
            "--bs=9732096", "--iodepth=16", "--numjobs=4",
            "--runtime=60", "--time_based=1",
        ):
            self.assertIn(required, joined)
        self.assertNotIn("write", "\n".join(argv).lower())

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
            "jobs": [{"error": 0, "read": {
                "bw_bytes": 12_000_000_000, "io_bytes": 720_000_000_000,
                "runtime": 60_000,
            }, "write": {"io_bytes": 0}}]
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "fio.json"
            result.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(probe.parse_fio_result(result)["bandwidth_gb_s"], 12.0)
            base["jobs"][0]["read"]["bw_bytes"] = float("nan")
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

            def failing_thermal(_hwmon):
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
            base["jobs"][0]["read"]["bw_bytes"] = 12_000_000_000
            base["jobs"][0]["write"]["io_bytes"] = 4096
            result.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.parse_fio_result(result)


if __name__ == "__main__":
    unittest.main()
