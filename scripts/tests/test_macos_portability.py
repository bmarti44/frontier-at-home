#!/usr/bin/env python3
"""Off-Spark (macOS) portability of the portable serve and build paths.

The apple-silicon profiles were authored without ever running on a Mac.
These tests pin the pieces that must agree for a Mac host to build the
pinned engine where its profile looks for it.
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN38_BUILD = ROOT / "scripts/12_build_qwen38_llamacpp.sh"


def _readonly_block(script: Path) -> str:
    """The script's leading readonly path constants, as shell source."""
    lines = [line for line in script.read_text().splitlines()
             if re.match(r"^readonly (CACHE_ROOT|SOURCE_DIR|BUILD_DIR)=", line)]
    return "\n".join(lines)


class Qwen38BuildCacheRootTests(unittest.TestCase):
    def _cache_root(self, home: str) -> str:
        source = _readonly_block(QWEN38_BUILD) + '\nprintf "%s" "$CACHE_ROOT"\n'
        return subprocess.run(
            ["bash", "-c", source], env={"HOME": home, "PATH": os.environ["PATH"]},
            capture_output=True, text=True, check=True).stdout

    def test_cache_root_follows_home(self):
        # The profile's binary_path is {cache_root}/llamacpp-qwen38-9d77fa17/...;
        # a hardcoded Linux home makes the Mac build unwritable.
        self.assertEqual(self._cache_root("/Users/someone"),
                         "/Users/someone/.cache/llamacpp-qwen38-9d77fa17")

    def test_spark_cache_root_is_unchanged(self):
        self.assertEqual(self._cache_root("/home/bmarti44"),
                         "/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17")


if __name__ == "__main__":
    unittest.main()
