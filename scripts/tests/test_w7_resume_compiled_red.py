import json
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_compiled_red_v1.sh"


class W7CompiledRedHarnessTests(unittest.TestCase):
    def test_fixture_pool_binds_completion_parser_rendered_wire(self):
        pool = json.loads(
            (ROOT / "results/glm52-gates/harness/w7-production-fixture-pool-v1.json").read_text()
        )
        self.assertEqual(pool["schema"], "glm52-w7-production-fixture-pool-v2")
        self.assertEqual(
            pool["render_contract"],
            {
                "api": "/v1/completions",
                "context_tokens": 8192,
                "model": "default",
                "reasoning_effort": "high",
                "thinking": True,
                "system": "You are a helpful assistant",
                "oracle": "frozen-ds4-server-c-parser",
            },
        )
        primary = pool["variants"][0]
        self.assertEqual(primary["variant"], "primary-fixed")
        self.assertEqual(
            {
                "selected": primary["selected_tokens"],
                "common": primary["common_tokens"],
                "live": primary["live_tokens"],
                "prompt": primary["prompt_tokens"],
            },
            {"selected": 5044, "common": 5045, "live": 5055, "prompt": 5066},
        )

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
        self.assertIn("GLM sync start=5044 prompt=5066 suffix=22", source)
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
            (ROOT / "results/glm52-gates/W7-resume-correctness-plan-v8.json").read_text()
        )
        self.assertEqual(plan["status"], "C_PARSER_FIXTURE_CORRECTION_AUTHORIZED_NOT_EXECUTED")
        self.assertEqual(
            plan["compiled_red_classification"]["geometry"],
            {"selected": 5044, "common": 5045, "live": 5055, "prompt": 5066},
        )


if __name__ == "__main__":
    unittest.main()
