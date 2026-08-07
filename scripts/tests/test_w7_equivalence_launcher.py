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
        self.assertIn("W7_SEALED_RUNTIME_OK", result.stdout)
        for name in ("harness", "scorer", "trace_scorer"):
            self.assertIn(f'"{name}":', result.stdout)

    def test_seed_is_derived_from_latest_public_randomness(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn('parser.add_argument("--seed-sha256")', source)
        self.assertIn('f"https://{host}/public/latest"', source)
        self.assertIn("hashlib.sha256(bytes.fromhex(signature))", source)
        self.assertIn("W7_RANDOM_SEED_SHA256=seed_sha256", source)


if __name__ == "__main__":
    unittest.main()
