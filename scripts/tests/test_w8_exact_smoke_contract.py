import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w8_exact_smoke_v1.sh"
SCORER = ROOT / "scripts/90_score_w8_exact_smoke.py"


class W8ExactSmokeContractTests(unittest.TestCase):
    def test_scorer_mutations(self):
        result = subprocess.run(
            ["python3", str(SCORER), "--self-test"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_harness_pins_reviewed_runtime_and_preserves_failures(self):
        source = HARNESS.read_text(encoding="utf-8")
        for needle in (
            "W8-exact-smoke-review-r236.json",
            "drand_min_round",
            "verify_reviewed_components",
            "finalize_attempt",
            "--failure-reason",
            "randomness-receipt.json",
        ):
            self.assertIn(needle, source)

    def test_scorer_requires_io_counters_and_cgroup_events(self):
        source = SCORER.read_text(encoding="utf-8")
        for needle in (
            "W8 exact request complete",
            "checksum_validations",
            "direct_slot_calls",
            "oom_group_kill",
            "executed_candidate_verified",
            "artifact_inventory",
            "terminal-receipt.json",
        ):
            self.assertIn(needle, source)


if __name__ == "__main__":
    unittest.main()
