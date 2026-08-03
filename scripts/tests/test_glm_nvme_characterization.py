#!/usr/bin/env python3
"""Contract tests for the read-only GLM NVMe characterization gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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
            target.write_bytes(b"x" * 8192)
            link = root / "link"
            link.symlink_to(target)
            identity = probe.validate_target(target)
            self.assertEqual(identity["size"], 8192)
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


if __name__ == "__main__":
    unittest.main()
