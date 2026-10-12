#!/usr/bin/env python3
"""The qualification kit (94) and its memory-reading cells run off Linux.

docs/QUALIFY-OFFHOST.md sends community hosts through 94_qualify_profile.py.
Its context and soak cells sampled /proc/meminfo, which macOS lacks; the
macOS watchdog's availability definition (vm_stat free + inactive +
purgeable) is the one they must use there.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import host_memory  # noqa: E402
import profile_resolver as resolver  # noqa: E402

DARWIN = sys.platform == "darwin"

VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    96788.
Pages active:                                3764855.
Pages inactive:                              3279601.
Pages speculative:                            610820.
Pages wired down:                             521807.
Pages purgeable:                               63861.
"""


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VmStatAvailability(unittest.TestCase):
    def test_counts_free_inactive_and_purgeable_pages(self) -> None:
        # Same definition as 06_memwatch_macos.sh / 07_memory_guard_macos.py.
        pages = 96788 + 3279601 + 63861
        self.assertEqual(host_memory.parse_vm_stat_kib(VM_STAT), pages * 16384 // 1024)

    def test_unrecognizable_output_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            host_memory.parse_vm_stat_kib("Mach Virtual Memory Statistics\n")

    def test_linux_reads_meminfo(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            meminfo = Path(scratch) / "meminfo"
            meminfo.write_text("MemTotal:       131072000 kB\nMemAvailable:   12582912 kB\n")
            self.assertEqual(host_memory.available_kib(meminfo), 12582912)
            self.assertEqual(host_memory.total_kib(meminfo), 131072000)


class KitCellsReadMemoryOnThisHost(unittest.TestCase):
    def test_soak_samples_available_memory(self) -> None:
        soak = load_script("soak_kit", "35_soak.py")
        self.assertGreater(soak.mem_available_gib(), 0)

    def test_context_probe_samples_and_reports_total(self) -> None:
        probe = load_script("probe_kit", "50_probe_context.py")
        sampler = probe.MemorySampler(interval_s=0.01)
        sampler.start()
        sampler.stop.set()
        sampler.join(timeout=5)
        self.assertTrue(sampler.samples)
        self.assertNotIn("error", sampler.samples[0])
        self.assertGreater(sampler.summary(floor_gib=0)["min_mem_available_gib"], 0)
        self.assertGreater(probe.host_facts()["mem_total_kib"], 0)

    def test_qualify_manifest_records_host_memory(self) -> None:
        kit = load_script("qualify_kit", "94_qualify_profile.py")
        self.assertGreater(kit.host_facts()["mem_total_gib"], 0)

    @unittest.skipUnless(DARWIN, "vm_stat")
    def test_darwin_uses_vm_stat_definition(self) -> None:
        with mock.patch.object(host_memory, "_vm_stat", return_value=VM_STAT):
            self.assertEqual(host_memory.available_kib(), host_memory.parse_vm_stat_kib(VM_STAT))


class CommunityAttestation(unittest.TestCase):
    def test_qualified_status_accepts_the_attestation_field(self) -> None:
        # docs/QUALIFY-OFFHOST.md section 5 tells community hosts to write
        # status.attestation; the resolver must accept that promotion PR.
        source = resolver.PROFILES_ROOT / "qwen3.8-27b"
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "qwen3.8-27b"
            shutil.copytree(source, directory)
            path = directory / "apple-silicon-128g.json"
            profile = json.loads(path.read_text())
            profile["status"] = {
                "state": "qualified", "qualified_at": "2026-10-12",
                "evidence": "results/qwen38-gates/x/",
                "attestation": "community-hardware; unreproduced on reference host",
            }
            path.write_text(json.dumps(profile))
            with mock.patch.object(resolver, "PROFILES_ROOT", Path(scratch)):
                loaded = resolver.load_profile("qwen3.8-27b", "apple-silicon-128g.json")
        self.assertEqual(loaded["status"]["attestation"],
                         "community-hardware; unreproduced on reference host")


if __name__ == "__main__":
    unittest.main()
