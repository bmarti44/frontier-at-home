#!/usr/bin/env python3

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/86_run_w7_equivalence.py"


class W7EquivalenceLauncherTest(unittest.TestCase):
    def test_all_runtime_programs_are_kernel_sealed(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--self-test"], cwd=ROOT, text=True,
            capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("W7_EQUIVALENCE_SEALS_OK", result.stdout)
        for name in ("harness", "scorer", "trace_scorer"):
            self.assertIn(f'"{name}":', result.stdout)

    def test_seed_is_runtime_input_not_frozen_before_launcher(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--seed-sha256")', source)
        self.assertIn("W7_RANDOM_SEED_SHA256=seed_sha256", source)
        self.assertNotIn("RANDOM_SEED_SHA256 =", source)


if __name__ == "__main__":
    unittest.main()
