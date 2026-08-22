#!/usr/bin/env python3
"""Every encoder the accuracy harness can select must exist and be tracked.

vendor/ is gitignored as a whole; encoders are carved back in via negation
patterns in .gitignore. This test fails if a registered encoder module is
missing from the working tree or absent from the Git index — the failure
mode where a new model's encoder silently never gets committed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_accuracy", ROOT / "scripts" / "31_bench_accuracy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncoderRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = _load_bench_module()

    def test_every_registered_encoder_exists(self):
        for name, path in self.bench.ENCODER_PATHS.items():
            self.assertTrue(
                Path(path).is_file(),
                f"encoder {name!r} is registered but missing: {path}",
            )

    def test_every_registered_encoder_is_git_tracked_or_pinned(self):
        # DSV4's encoder is not tracked: it is fetched from the official
        # upstream pin by scripts/14_fetch_encoder.sh. Every other encoder
        # must be tracked in Git so a clean checkout can load it.
        import json

        pin = json.loads(
            (ROOT / "configs" / "pins" / "official-encoding.json")
            .read_text(encoding="utf-8")
        )
        pinned = {
            str(Path("vendor/official-encoding") / item["path"])
            for item in pin["files"]
        }
        for name, path in self.bench.ENCODER_PATHS.items():
            relative = Path(path).resolve().relative_to(ROOT)
            if str(relative) in pinned:
                continue
            result = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                 str(relative)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0,
                f"encoder {name!r} is neither pinned in "
                f"configs/pins/official-encoding.json nor tracked by Git "
                f"(vendor/ is gitignored; stage it with `git add -f "
                f"{relative}` and check the .gitignore negation patterns): "
                f"{relative}",
            )

    def test_every_registered_encoder_has_an_effort_contract(self):
        self.assertEqual(
            set(self.bench.ENCODER_PATHS),
            set(self.bench.ENCODER_REASONING_EFFORTS),
            "ENCODER_PATHS and ENCODER_REASONING_EFFORTS must cover the "
            "same encoder names",
        )


if __name__ == "__main__":
    sys.exit(unittest.main())
