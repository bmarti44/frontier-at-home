#!/usr/bin/env python3
"""scripts/04_host_facts.py --host-file writes a resolver-ready host file."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver as resolver  # noqa: E402

spec = importlib.util.spec_from_file_location("host_facts", ROOT / "scripts/04_host_facts.py")
host_facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_facts)


def facts(**overrides) -> dict:
    base = {"os": "linux", "arch": "x86_64", "ram_gib": 128, "gpus": [],
            "amd_gfx": None, "backends_available": ["cpu"],
            "host_class_guess": "any", "tier_guess_gib": 128}
    base.update(overrides)
    return base


class HostFileFromFacts(unittest.TestCase):
    def build(self, **overrides) -> dict:
        return host_facts.host_file(facts(**overrides), home="/home/someone",
                                    hostname="box", repo="/home/someone/frontier")

    def test_strix_halo(self) -> None:
        host = self.build(amd_gfx="gfx1151", backends_available=["rocm", "cpu"],
                          host_class_guess="strix")
        self.assertEqual((host["hardware_class"], host["backends"], host["ram_gib"]),
                         ("strix", ["rocm", "cpu"], 128))
        self.assertEqual(host["paths"]["cache_root"], "/home/someone/.cache")

    def test_dgpu_tier_is_vram(self) -> None:
        # dgpu profiles carry the VRAM tier in ram_tier_gib (hardware-matrix).
        host = self.build(gpus=[{"vendor": "nvidia", "name": "RTX 4090",
                                 "vram_bytes": 24 * 2**30}],
                          backends_available=["cuda", "cpu"],
                          host_class_guess="dgpu", tier_guess_gib=24)
        self.assertEqual((host["hardware_class"], host["ram_gib"]), ("dgpu", 24))

    def test_unknown_class_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.build(host_class_guess=None)


class HostFileCli(unittest.TestCase):
    def test_writes_a_loadable_host_file_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "host.json"
            command = [sys.executable, str(ROOT / "scripts/04_host_facts.py"),
                       "--host-file", str(path)]
            subprocess.run(command, check=True, capture_output=True)
            host = resolver.load_host(path)
            self.assertIn("cpu", host["backends"])
            self.assertEqual(json.loads(path.read_text())["ports"]["production"], 8013)
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("exists", again.stderr)


if __name__ == "__main__":
    unittest.main()
