#!/usr/bin/env python3
"""Contract for the mandatory, bounded direct-I/O loader patch."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "results" / "dsv4-cold-load" / "llama-direct-io-required.patch"
FAULT_SOURCE = ROOT / "results" / "dsv4-cold-load" / "fault_io.c"
FAULT_PROBE = ROOT / "results" / "dsv4-cold-load" / "direct_io_fault_probe.cpp"
PINNED_COMMIT = "0dc74e332edee2616e4d8d9ab3b68dfc340fc14a"
PATCHED_FILES = {
    "common/arg.cpp", "common/common.cpp", "common/common.h", "include/llama.h",
    "src/llama.cpp", "src/llama-mmap.cpp", "src/llama-mmap.h",
    "src/llama-model-loader.cpp", "src/llama-model-loader.h",
    "src/llama-model.cpp", "src/llama-quant.cpp",
}


class DirectIoPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        clean = os.environ.get("DSV4_FUSION_CLEAN_SOURCE_ROOT")
        patched = os.environ.get("DSV4_FUSION_PATCHED_SOURCE_ROOT")
        self.clean_source = Path(clean).resolve() if clean else None
        self.patched_source = Path(patched).resolve() if patched else None
        self.patch_text = PATCH.read_text(encoding="utf-8")

    def test_patch_applies_to_selected_source(self) -> None:
        if self.clean_source is None or self.patched_source is None:
            self.skipTest("exact clean and patched fusion source paths are required")
        for source in (self.clean_source, self.patched_source):
            head = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(head, PINNED_COMMIT)
        with tempfile.TemporaryDirectory() as raw:
            checkout = Path(raw) / "source"
            shutil.copytree(self.clean_source, checkout, ignore=shutil.ignore_patterns("build*", ".git"))
            result = subprocess.run(
                ["git", "apply", "--check", "--unidiff-zero", str(PATCH)],
                cwd=checkout,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run(
                ["git", "apply", "--unidiff-zero", str(PATCH)], cwd=checkout, check=True
            )
            for relative in PATCHED_FILES:
                self.assertEqual(
                    (checkout / relative).read_bytes(),
                    (self.patched_source / relative).read_bytes(),
                    relative,
                )
        changed = set(subprocess.check_output(
            ["git", "-C", str(self.patched_source), "diff", "--name-only"], text=True
        ).splitlines())
        self.assertEqual(changed, PATCHED_FILES)

    def test_patch_adds_distinct_required_mode_and_fatal_fallbacks(self) -> None:
        self.assertIn("--direct-io-required", self.patch_text)
        self.assertIn("require_direct_io", self.patch_text)
        self.assertIn("required direct I/O open failed", self.patch_text)
        self.assertIn("required direct I/O read failed", self.patch_text)
        self.assertIn("required direct I/O uploader unavailable", self.patch_text)
        if self.patched_source is not None:
            mmap_source = (self.patched_source / "src/llama-mmap.cpp").read_text(encoding="utf-8")
            self.assertIn("Falling back to buffered I/O", mmap_source)
        self.assertIn("throw std::runtime_error", self.patch_text)

    def test_patch_bounds_alignment_headroom(self) -> None:
        self.assertIn("upload_chunk_size", self.patch_text)
        self.assertIn("host_buffer_size", self.patch_text)
        self.assertIn("alignment - 1", self.patch_text)
        self.assertIn("direct_io_staging_limit", self.patch_text)

    def test_exact_source_test_has_no_vendor_fallback(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        self.assertIn("DSV4_FUSION_CLEAN_SOURCE_ROOT", source)
        self.assertIn("DSV4_FUSION_PATCHED_SOURCE_ROOT", source)
        forbidden = 'return ROOT / "ven' + 'dor" / "llama.cpp"'
        self.assertNotIn(forbidden, source)

    def test_fault_injector_covers_open_read_and_partial_read(self) -> None:
        source = FAULT_SOURCE.read_text(encoding="utf-8")
        for marker in (
            "DSV4_FAULT_OPEN_SUFFIX",
            "DSV4_FAULT_READ_ERRNO",
            "DSV4_FAULT_PARTIAL_BYTES",
            "O_DIRECT",
        ):
            self.assertIn(marker, source)
        probe = FAULT_PROBE.read_text(encoding="utf-8")
        self.assertIn('llama_file file(argv[2], "rb", true, required)', probe)
        self.assertIn('std::strcmp(argv[1], "optional")', probe)
        self.assertIn("file.read_raw", probe)


if __name__ == "__main__":
    unittest.main()
