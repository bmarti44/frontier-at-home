#!/usr/bin/env python3
"""Production-path acceptance tests for scripts/glm52_goal.py."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "glm52_goal.py"


def load_goal_module():
    spec = importlib.util.spec_from_file_location("glm52_goal", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.goal = load_goal_module()

    def test_decode_formula_uses_first_and_last_token(self):
        timestamps = [1.0 + index / 10.0 for index in range(128)]
        self.assertAlmostEqual(
            self.goal.decode_tokens_per_second(timestamps), 10.0, places=12
        )

    def test_decode_formula_rejects_short_or_nonfinite_input(self):
        for timestamps in (
            [1.0] * 127,
            [1.0] * 128,
            [1.0] * 127 + [math.nan],
            [1.0] * 127 + [math.inf],
        ):
            with self.subTest(timestamps=timestamps[-1]):
                with self.assertRaises(ValueError):
                    self.goal.decode_tokens_per_second(timestamps)

    def test_ratio_confidence_bounds_are_directional(self):
        candidate = [8.0, 8.1, 8.2, 8.3, 8.4]
        reference = [10.0, 10.1, 10.2, 10.3, 10.4]
        lower = self.goal.paired_ratio_bound(candidate, reference, side="lower")
        upper = self.goal.paired_ratio_bound(candidate, reference, side="upper")
        self.assertLessEqual(lower, sum(candidate) / sum(reference))
        self.assertGreaterEqual(upper, sum(candidate) / sum(reference))

    def test_ratio_rejects_bad_or_unpaired_samples(self):
        bad = (
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([1.0, math.nan], [1.0, 2.0]),
            ([1.0, 0.0], [1.0, 2.0]),
            ([1.0, 2.0], [1.0, math.inf]),
        )
        for candidate, reference in bad:
            with self.subTest(candidate=candidate, reference=reference):
                with self.assertRaises(ValueError):
                    self.goal.paired_ratio_bound(candidate, reference, side="lower")

    def test_performance_acceptance_formula(self):
        passing = {
            "decode_glm": [8.4] * 5,
            "decode_dsv4": [10.0] * 5,
            "prefill_glm": [84.0] * 5,
            "prefill_dsv4": [100.0] * 5,
            "prefill_time_glm": [12.0] * 5,
            "prefill_time_dsv4": [10.0] * 5,
            "warm_ttft_glm": [1.15] * 5,
            "warm_ttft_dsv4": [1.0] * 5,
            "cold_ttft_glm": [11.5] * 5,
            "cold_ttft_dsv4": [10.0] * 5,
        }
        verdict = self.goal.performance_verdict(passing)
        self.assertEqual(verdict["verdict"], "PASS")
        mutated = dict(passing)
        mutated["decode_glm"] = [7.0] * 5
        self.assertEqual(self.goal.performance_verdict(mutated)["verdict"], "FAIL")

    def test_long_context_requires_real_processing_and_safety(self):
        passing = {
            "context_cap": 1_048_576,
            "processed_tokens": 1_000_000,
            "retrieval_pass": True,
            "negative_control_pass": True,
            "completed_generation": True,
            "truncated": False,
            "oom": False,
            "xid": False,
            "available_memory_gib": 10.0,
        }
        self.assertEqual(self.goal.context_verdict(passing)["verdict"], "PASS")
        for key, value in (
            ("processed_tokens", 999_999),
            ("retrieval_pass", False),
            ("negative_control_pass", False),
            ("truncated", True),
            ("oom", True),
            ("xid", True),
            ("available_memory_gib", 9.99),
        ):
            mutated = dict(passing)
            mutated[key] = value
            with self.subTest(key=key):
                self.assertEqual(
                    self.goal.context_verdict(mutated)["verdict"], "FAIL"
                )
        for key, value in (
            ("context_cap", "1048576"),
            ("processed_tokens", 1_000_000.9),
            ("processed_tokens", True),
        ):
            malformed = dict(passing)
            malformed[key] = value
            with self.subTest(malformed=key):
                with self.assertRaises(ValueError):
                    self.goal.context_verdict(malformed)

    def test_lossy_quality_formula_and_mutations(self):
        passing = [
            {
                "tokens": 100,
                "baseline_nll_sum": 200.0,
                "candidate_nll_sum": 200.5,
                "baseline_top1_correct": 70,
                "candidate_top1_correct": 70,
            }
            for _ in range(100)
        ]
        verdict = self.goal.quality_verdict(passing)
        self.assertEqual(verdict["verdict"], "PASS")
        high_nll = [dict(case) for case in passing]
        high_nll[0]["candidate_nll_sum"] = 400.0
        self.assertEqual(self.goal.quality_verdict(high_nll)["verdict"], "FAIL")
        top1_loss = [dict(case) for case in passing]
        for case in top1_loss:
            case["candidate_top1_correct"] = 69
        self.assertEqual(self.goal.quality_verdict(top1_loss)["verdict"], "FAIL")
        for mutation in (
            passing[:99],
            passing + [dict(passing[0])],
            [{**passing[0], "tokens": 0}] * 100,
            [{**passing[0], "candidate_nll_sum": math.nan}] * 100,
        ):
            with self.assertRaises(ValueError):
                self.goal.quality_verdict(mutation)

    def test_raw_arm_validation_fails_closed(self):
        valid = {
            "arm": "A",
            "fixture_sha256": "a" * 64,
            "binary_sha256": "b" * 64,
            "token_timestamps": [float(i) for i in range(128)],
            "evaluated_tokens": 1000,
            "prefill_seconds": 10.0,
            "failures": [],
        }
        self.goal.validate_raw_record(valid)
        mutations = (
            {"arm": ""},
            {"fixture_sha256": "bad"},
            {"binary_sha256": "bad"},
            {"token_timestamps": [float(i) for i in range(127)]},
            {"token_timestamps": [0.0] * 128},
            {"evaluated_tokens": 0},
            {"prefill_seconds": math.nan},
            {"failures": ["timeout"]},
        )
        for mutation in mutations:
            record = dict(valid)
            record.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.validate_raw_record(record)

    def test_five_fresh_server_blocks_require_abba_baab_and_distinct_arms(self):
        records = []
        for block in range(5):
            order = "ABBA" if block % 2 == 0 else "BAAB"
            for sequence, arm in enumerate(order):
                records.append(
                    {
                        "block": block,
                        "sequence": sequence,
                        "arm": arm,
                        "server_boot_id": f"boot-{block}",
                        "fixture_sha256": "a" * 64,
                        "binary_sha256": ("b" if arm == "A" else "c") * 64,
                        "configuration_sha256": ("d" if arm == "A" else "e") * 64,
                    }
                )
        self.goal.validate_ab_blocks(records)
        rotating = [dict(item) for item in records]
        for item in rotating:
            item["binary_sha256"] = (
                f"{item['block'] + (0 if item['arm'] == 'A' else 8):x}" * 64
            )[:64]
        with self.assertRaises(ValueError):
            self.goal.validate_ab_blocks(rotating)
        for mutation in ("same_binary", "same_boot", "wrong_order", "unequal_fixture"):
            broken = [dict(item) for item in records]
            if mutation == "same_binary":
                for item in broken:
                    item["binary_sha256"] = "b" * 64
                    item["configuration_sha256"] = "d" * 64
            elif mutation == "same_boot":
                broken[-1]["server_boot_id"] = broken[0]["server_boot_id"]
            elif mutation == "wrong_order":
                broken[0]["arm"] = "B"
            else:
                broken[-1]["fixture_sha256"] = "f" * 64
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.validate_ab_blocks(broken)

    def test_attempt_requires_manifest_raw_and_fixed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            artifacts = {}
            manifest = {"gate": "foundation", "candidate_hash": "a" * 40}
            for index, field in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                artifact = attempt / f"{field}.artifact"
                artifact.write_bytes(f"artifact-{index}".encode())
                artifacts[field] = artifact.name
                manifest[f"{field}_sha256"] = hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest()
            manifest["artifacts"] = artifacts
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(
                json.dumps({"event": "diagnostic", "failures": []}) + "\n"
            )
            (attempt / "summary.json").write_text(
                json.dumps(
                    {
                        "formula_version": 1,
                        "verdict": "NO_RESULT",
                        "reason": "diagnostic-only",
                    }
                )
            )
            self.goal.validate_attempt(attempt)
            (attempt / "raw.jsonl").write_text("")
            with self.assertRaises(ValueError):
                self.goal.validate_attempt(attempt)

    def test_attempt_rejects_fabricated_pass_and_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            manifest = {
                f"{name}_sha256": str(index) * 64
                for index, name in enumerate(
                    (
                        "source",
                        "diff",
                        "binary",
                        "scorer",
                        "model",
                        "tokenizer",
                        "fixture",
                        "configuration",
                    )
                )
            }
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text('{"event":"fabricated","failures":[]}\n')
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"PASS"}'
            )
            with self.assertRaises(ValueError):
                self.goal.validate_attempt(attempt)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                '{"formula_version":1,"verdict":"FAIL","verdict":"PASS"}'
            )
            with self.assertRaises(ValueError):
                self.goal._read_strict_json(path)


class ControllerTests(unittest.TestCase):
    def run_cli(self, state_dir: Path, *args: str):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--state-dir", str(state_dir), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_status_initializes_all_required_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(Path(tmp), "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            expected = {"foundation", "switch", "parity", "review"} | {
                f"W{i}" for i in range(1, 12)
            }
            self.assertEqual(set(state["gates"]), expected)
            self.assertTrue(
                all(
                    gate["status"] == "PENDING"
                    for gate in state["gates"].values()
                )
            )

    def test_resume_selects_highest_value_unfinished_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            first = self.run_cli(state_dir, "resume")
            self.assertEqual(first.returncode, 0, first.stderr)
            event = json.loads(first.stdout)
            self.assertEqual(event["selected_gate"], "foundation")

    def test_state_rejects_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            state["gates"]["W1"]["status"] = "CLAIMED_PASS"
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "status", "--json")
            self.assertNotEqual(result.returncode, 0)

    def test_state_rejects_terminal_status_without_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            for gate in state["gates"].values():
                gate["status"] = "PASS"
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "resume")
            self.assertNotEqual(result.returncode, 0)

    def test_state_rejects_terminal_status_when_attempt_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            state["gates"]["foundation"] = {
                "status": "PASS",
                "attempts": ["foundation/ghost-attempt"],
                "reason": None,
            }
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "status", "--json")
            self.assertNotEqual(result.returncode, 0)

    def test_attempt_manifest_gate_must_match_directory_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "W1" / "attempt-001"
            attempt.mkdir(parents=True)
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = attempt / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text('{"event":"diagnostic"}\n')
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"NO_RESULT","reason":"fake"}'
            )
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            self.assertEqual(state["gates"]["W1"]["status"], "FAIL")
            self.assertIn("does not match", state["gates"]["W1"]["reason"])

    def test_release_check_fails_closed_on_incomplete_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "release-check", "--json")
            self.assertNotEqual(result.returncode, 0)
            value = json.loads(result.stdout)
            self.assertFalse(value["release_qualified"])
            self.assertIn("W11", value["failed_requirements"])

    def test_release_does_not_trust_no_go_summary_boolean(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('summary.get("independently_reviewed") is True', source)
        self.assertIn("parity_no_go = False", source)

    def test_resume_ingests_valid_attempt_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "foundation" / "attempt-001"
            attempt.mkdir(parents=True)
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = attempt / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(
                json.dumps({"event": "bounded falsifier", "failures": []}) + "\n"
            )
            (attempt / "summary.json").write_text(
                json.dumps(
                    {
                        "formula_version": 1,
                        "verdict": "NO_RESULT",
                        "reason": "bounded attempt exhausted",
                    }
                )
            )
            result = self.run_cli(state_dir, "resume")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "W1")
            state = json.loads((state_dir / "state.json").read_text())
            self.assertEqual(state["gates"]["foundation"]["status"], "NO_RESULT")
            self.assertEqual(
                state["gates"]["foundation"]["attempts"], ["foundation/attempt-001"]
            )

    def test_malformed_attempt_fails_gate_and_does_not_get_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "foundation" / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "manifest.json").write_text("{}")
            (attempt / "raw.jsonl").write_text("{}\n")
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"PASS"}'
            )
            result = self.run_cli(state_dir, "resume")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "W1")
            state = json.loads((state_dir / "state.json").read_text())
            self.assertEqual(state["gates"]["foundation"]["status"], "FAIL")
            self.assertIn("invalid", state["gates"]["foundation"]["reason"])

    def test_run_executes_registered_runner_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            seed = state_dir / "seed-attempt"
            seed.mkdir()
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = seed / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (seed / "manifest.json").write_text(json.dumps(manifest))
            (seed / "raw.jsonl").write_text(
                '{"event":"bounded_runner","failures":[]}\n'
            )
            (seed / "summary.json").write_text(
                '{"formula_version":1,"verdict":"NO_RESULT","reason":"bounded"}'
            )
            runners = state_dir / "runners"
            runners.mkdir()
            runner = runners / "foundation"
            runner.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "mkdir -p \"$1/foundation\"\n"
                "cp -R \"$1/seed-attempt\" \"$1/foundation/attempt-001\"\n"
                "echo ran > \"$1/runner.marker\"\n"
            )
            runner.chmod(0o700)
            result = self.run_cli(state_dir, "run")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "W1")
            self.assertEqual((state_dir / "runner.marker").read_text(), "ran\n")


if __name__ == "__main__":
    unittest.main()
