#!/usr/bin/env python3

from pathlib import Path
import hashlib
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_production_v1.sh"


class W7ProductionHarnessTest(unittest.TestCase):
    def test_self_test_passes_without_starting_model(self) -> None:
        result = subprocess.run(
            ["/usr/bin/bash", str(HARNESS), "--self-test"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("W7_PRODUCTION_EQUIVALENCE_SELFTEST_OK", result.stdout)

    def test_all_arms_use_hardened_containment(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn('"$CGROUP" --tag "$tag"', source)
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER=1", source)
        self.assertIn("GLM_SAFE_MIN_START_GIB=110", source)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=24", source)
        self.assertIn("DS4_GLM_LOGIT_DUMP_ALL=1", source)
        self.assertNotIn("DS4_GLM_RESTORED_FRONTIER_DIAGNOSTIC", source)

    def test_strict_reference_and_production_arms_are_distinct_and_bound(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("readonly STRICT_BIN=", source)
        self.assertIn("readonly STRICT_BINARY_SHA256=", source)
        resolved = {}
        for arm in ("strict", "candidate", "cold"):
            result = subprocess.run(
                ["/usr/bin/bash", str(HARNESS), "--resolved-arm-config", arm],
                cwd=ROOT, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved[arm] = result.stdout.strip()
        self.assertNotEqual(resolved["strict"], resolved["candidate"])
        self.assertEqual(resolved["candidate"], resolved["cold"])

    def test_configuration_is_emitted_and_recomputed(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn('"$attempt_out/configuration.json"', source)
        self.assertNotIn("readonly CONFIGURATION_SHA256=", source)

    def test_arm_roles_are_explicit_and_fresh(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        for arm in ("strict", "candidate", "cold"):
            self.assertIn(arm, source)
        self.assertIn('mkdir "$arm_out/kv"', source)
        self.assertIn("kv-before.sha256", source)
        self.assertIn("kv-after.sha256", source)

    def test_real_arm_tags_pass_the_frozen_launcher_validator(self) -> None:
        for arm in ("strict", "candidate", "cold"):
            result = subprocess.run(
                ["/usr/bin/bash", str(HARNESS), "--validate-tag", arm,
                 "0123456789abcdef0123456789abcdef"],
                cwd=ROOT, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tag = result.stdout.strip()
            self.assertLessEqual(len(tag), 40)

    def test_model_and_executables_are_bound_before_run(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn('verify_file "$MODEL" "$MODEL_SHA256"', source)
        self.assertIn('/proc/$$/fd/$harness_fd', source)
        self.assertIn('/proc/$$/fd/$scorer_fd', source)

        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw) / "model.gguf"
            model.write_bytes(b"reviewed-model")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            passed = subprocess.run(
                ["/usr/bin/bash", str(HARNESS), "--verify-model", str(model), digest],
                cwd=ROOT, capture_output=True, timeout=10,
            )
            self.assertEqual(passed.returncode, 0)
            model.write_bytes(b"mutated-model!")  # equal byte length
            failed = subprocess.run(
                ["/usr/bin/bash", str(HARNESS), "--verify-model", str(model), digest],
                cwd=ROOT, capture_output=True, timeout=10,
            )
            self.assertNotEqual(failed.returncode, 0)

    def test_runtime_inputs_require_kernel_sealed_descriptors(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("F_GET_SEALS", source)
        self.assertIn("F_SEAL_WRITE", source)
        self.assertIn("W7_SEALED_HARNESS_FD", source)
        self.assertNotIn('exec {harness_fd}<"$INVOKED_SCRIPT"', source)

    def test_safe_wrapper_and_driver_bind_same_binary_inode(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        binary = re.search(r"^readonly BIN=(\S+)$", source, re.M)
        candidate_source = re.search(
            r"^\s+GLM_CANDIDATE_SRC=(\S+) \\$", source, re.M
        )
        self.assertIsNotNone(binary)
        self.assertIsNotNone(candidate_source)
        expected = Path(candidate_source.group(1)) / "ds4-server"
        self.assertTrue(expected.samefile(Path(binary.group(1))))

    def test_run_arm_does_not_expand_unassigned_locals(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertNotIn(
            "local arm=$1 root=$2 arm_out=$root/$arm tag rc", source
        )
        self.assertIn("local arm=$1 root=$2", source)
        self.assertIn("local arm_out=$root/$arm", source)

    def test_cold_control_uses_branch_matched_5044_boundary(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("local boundary_trim=8", source)
        self.assertIn('[[ $arm == cold ]] && boundary_trim=20', source)
        self.assertIn(
            '--kv-cache-boundary-align-tokens 4 '
            '--kv-cache-boundary-trim-tokens "$boundary_trim"',
            source,
        )

    def test_required_evidence_contract_is_emitted(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("manifest.json", source)
        self.assertIn("raw.jsonl", source)
        self.assertIn('"$arm_out/safety"', source)


if __name__ == "__main__":
    unittest.main()
