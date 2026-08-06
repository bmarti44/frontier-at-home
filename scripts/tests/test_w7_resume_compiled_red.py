import json
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_compiled_red_v1.sh"


class W7CompiledRedHarnessTests(unittest.TestCase):
    def test_frozen_production_geometry_and_binary_are_available(self):
        completed = subprocess.run(
            ["/usr/bin/bash", str(HARNESS), "--self-test"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "W7_RED_SELFTEST_OK")

    def test_acceptance_is_resume_not_guard_bypass(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("GLM sync start=5028 prompt=5048 suffix=20", source)
        self.assertIn("strict_guard_cold_restart", source)
        self.assertIn("readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock", source)
        self.assertIn("DS4_LOCK_EXPECTED_DEV_INO=$engine_lock_identity", source)
        assignments = [
            line for line in source.splitlines()
            if line.strip().startswith("DS4_GLM_RESUME_GUARD_OFF=")
        ]
        self.assertEqual(assignments, [])

    def test_preregistered_plan_authorizes_compiled_red(self):
        plan = json.loads(
            (ROOT / "results/glm52-gates/W7-resume-correctness-plan-v7.json").read_text()
        )
        self.assertEqual(plan["status"], "COMPILED_RED_AUTHORIZED_NOT_EXECUTED")
        self.assertEqual(
            plan["compiled_red_classification"]["geometry"],
            {"selected": 5028, "common": 5029, "live": 5037, "prompt": 5048},
        )


if __name__ == "__main__":
    unittest.main()
