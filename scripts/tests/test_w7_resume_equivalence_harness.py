#!/usr/bin/env python3

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_equivalence_v1.sh"


class W7HarnessTest(unittest.TestCase):
    def test_self_test_passes_without_starting_model(self) -> None:
        result = subprocess.run(
            ["/usr/bin/bash", str(HARNESS), "--self-test"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("W7_EQUIVALENCE_SELFTEST_OK", result.stdout)

    def test_all_arms_use_hardened_containment(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn('"$CGROUP" --tag "$tag"', source)
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER=1", source)
        self.assertIn("GLM_SAFE_MIN_START_GIB=110", source)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=24", source)
        self.assertIn("DS4_GLM_LOGIT_DUMP_ALL=1", source)
        self.assertIn("DS4_GLM_RESTORED_FRONTIER_DIAGNOSTIC=1", source)

    def test_arm_roles_are_explicit_and_fresh(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        for arm in ("strict", "candidate", "cold"):
            self.assertIn(arm, source)
        self.assertIn('mkdir "$arm_out/kv"', source)
        self.assertIn("kv-before.sha256", source)
        self.assertIn("kv-after.sha256", source)


if __name__ == "__main__":
    unittest.main()
