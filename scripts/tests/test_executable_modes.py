#!/usr/bin/env python3
"""Scripts that other components execute directly must stay executable.

The exec bit on 03_memory_guard.py has been lost twice (it is invoked
directly — not via python3 — by 52_engine_switch.sh and the serve
scripts, and a root context then fails with a bare Permission denied at
switch/restore time). File modes regress silently through checkouts and
patch flows, so pin them here against the Git index.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Files some caller executes directly (shebang execution, root contexts
# included). Add a line when a new script is invoked without an explicit
# interpreter.
MUST_BE_EXECUTABLE = [
    "scripts/00_preflight.sh",
    "scripts/01_memwatch.sh",
    "scripts/03_guard.sh",
    "scripts/03_memory_guard.py",
    "scripts/13_build_laguna_llamacpp.sh",
    "scripts/21_serve_llamacpp.sh",
    "scripts/22_serve_qwen38.sh",
    "scripts/23_serve_qwen38_sglang.sh",
    "scripts/25_serve_laguna.sh",
    "scripts/42_verify_exposure.sh",
    "scripts/52_engine_switch.sh",
    "scripts/90_scaffold_model.sh",
    "scripts/lint_secrets.sh",
]


class ExecutableModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        listing = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-s", "--", "scripts/"],
            check=True, stdout=subprocess.PIPE, text=True,
        ).stdout
        cls.index_modes = {}
        for line in listing.splitlines():
            mode, _oid, _stage, path = line.replace("\t", " ").split(" ", 3)
            cls.index_modes[path] = mode

    def test_directly_executed_scripts_are_executable_in_the_index(self):
        for path in MUST_BE_EXECUTABLE:
            with self.subTest(path=path):
                self.assertIn(
                    path, self.index_modes,
                    f"{path} is not tracked; update MUST_BE_EXECUTABLE",
                )
                self.assertEqual(
                    self.index_modes[path], "100755",
                    f"{path} lost its exec bit in the Git index; restore "
                    f"with: chmod +x {path} && "
                    f"git update-index --chmod=+x {path}",
                )

    def test_directly_executed_scripts_are_executable_on_disk(self):
        import os
        for path in MUST_BE_EXECUTABLE:
            with self.subTest(path=path):
                self.assertTrue(
                    os.access(ROOT / path, os.X_OK),
                    f"{path} is not executable on disk (chmod +x)",
                )


if __name__ == "__main__":
    unittest.main()
