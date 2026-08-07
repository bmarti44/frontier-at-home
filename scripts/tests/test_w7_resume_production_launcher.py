#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/88_run_w7_resume_production.py"
SPEC = importlib.util.spec_from_file_location("w7_launcher", LAUNCHER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7ProductionLauncherTest(unittest.TestCase):
    def test_all_runtime_programs_are_kernel_sealed(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--self-test"], cwd=ROOT, text=True,
            capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("W7_PRODUCTION_EQUIVALENCE_SEALS_OK", result.stdout)
        self.assertIn("W7_SEALED_RUNTIME_OK", result.stdout)
        for name in ("harness", "scorer", "trace_scorer"):
            self.assertIn(f'"{name}":', result.stdout)

    def test_seed_is_derived_from_latest_public_randomness(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn('parser.add_argument("--seed-sha256")', source)
        self.assertNotIn("/public/latest", source)
        self.assertIn('f"https://{host}/public/{DRAND_TARGET_ROUND}"', source)
        self.assertIn('round_number != DRAND_TARGET_ROUND', source)
        self.assertIn("hashlib.sha256(bytes.fromhex(signature))", source)
        self.assertIn("W7_RANDOM_SEED_SHA256=seed_sha256", source)

    def test_public_randomness_rejects_stale_forged_and_disagreement(self) -> None:
        signature = "61" * 96
        record = {
            "round": MODULE.DRAND_TARGET_ROUND,
            "randomness": hashlib.sha256(bytes.fromhex(signature)).hexdigest(),
            "signature": signature,
            "previous_signature": "62" * 96,
        }

        def responses(values):
            return [
                subprocess.CompletedProcess([], 0, json.dumps(value).encode(), b"")
                for value in values
            ]

        with mock.patch.object(MODULE.subprocess, "run", side_effect=responses([record] * 3)):
            seed, receipt = MODULE._public_randomness()
        self.assertEqual(len(seed), 64)
        self.assertEqual(json.loads(receipt)["round"], record["round"])

        stale = {**record, "round": MODULE.DRAND_TARGET_ROUND - 1}
        forged = {**record, "randomness": "0" * 64}
        disagree = {**record, "round": record["round"] + 1}
        for values in ([stale] * 3, [forged] * 3, [record, record, disagree]):
            with self.subTest(values=values):
                with mock.patch.object(
                    MODULE.subprocess, "run", side_effect=responses(values)
                ):
                    with self.assertRaises(SystemExit):
                        MODULE._public_randomness()


if __name__ == "__main__":
    unittest.main()
