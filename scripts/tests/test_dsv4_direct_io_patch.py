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
PINNED_COMMIT = "0dc74e332edee2616e4d8d9ab3b68dfc340fc14a"


def source_root() -> Path:
    configured = os.environ.get("DSV4_FUSION_SOURCE_ROOT")
    if configured:
        return Path(configured).resolve()
    return ROOT / "vendor" / "llama.cpp"


class DirectIoPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = source_root()
        self.patch_text = PATCH.read_text(encoding="utf-8")

    def test_patch_applies_to_selected_source(self) -> None:
        if os.environ.get("DSV4_FUSION_SOURCE_ROOT"):
            head = subprocess.check_output(
                ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(head, PINNED_COMMIT)
        with tempfile.TemporaryDirectory() as raw:
            checkout = Path(raw) / "source"
            shutil.copytree(self.source, checkout, ignore=shutil.ignore_patterns("build*", ".git"))
            result = subprocess.run(
                ["git", "apply", "--check", "--unidiff-zero", str(PATCH)],
                cwd=checkout,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_patch_adds_distinct_required_mode_and_fatal_fallbacks(self) -> None:
        self.assertIn("--direct-io-required", self.patch_text)
        self.assertIn("required direct I/O open failed", self.patch_text)
        self.assertIn("required direct I/O read failed", self.patch_text)
        self.assertIn("throw std::runtime_error", self.patch_text)

    def test_patch_bounds_alignment_headroom(self) -> None:
        self.assertIn("upload_chunk_size", self.patch_text)
        self.assertIn("host_buffer_size", self.patch_text)
        self.assertIn("alignment - 1", self.patch_text)

    def test_fault_injector_covers_open_read_and_partial_read(self) -> None:
        source = FAULT_SOURCE.read_text(encoding="utf-8")
        for marker in (
            "DSV4_FAULT_OPEN_SUFFIX",
            "DSV4_FAULT_READ_ERRNO",
            "DSV4_FAULT_PARTIAL_BYTES",
            "O_DIRECT",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
