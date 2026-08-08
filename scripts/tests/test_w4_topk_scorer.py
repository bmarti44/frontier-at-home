#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = ROOT / "scripts/98_score_w4_topk.py"
SPEC = importlib.util.spec_from_file_location("w4_scorer", SCORER_PATH)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "w4_runner", ROOT / "scripts/99_run_w4_topk_confirmation.py")
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
RECEIPT = json.loads((
    ROOT / "results/glm52-gates/W7-resume-production-drand-verifier-fixture.json"
).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class W4TopkScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        for name, data in (("binary", b"binary"), ("engine.cu", b"source"),
                           ("test.cu", b"test"), ("runner.py", b"runner")):
            (self.run_dir / name).write_bytes(data)
        (self.run_dir / "randomness-receipt.json").write_text(
            json.dumps(RECEIPT, sort_keys=True) + "\n")
        (self.run_dir / "drand-verifier.mjs").write_bytes(
            (ROOT / "scripts/89_verify_drand_receipt.mjs").read_bytes())
        (self.run_dir / "scorer.py").write_bytes(SCORER_PATH.read_bytes())
        first_baab = int(RECEIPT["randomness"][:2], 16) & 1
        self.rows = []
        for block in range(5):
            schedule = "BAAB" if bool(first_baab) ^ bool(block & 1) else "ABBA"
            for sequence, arm in enumerate(schedule):
                self.rows.append({
                    "schema": "glm52-w4-topk-observation-v1",
                    "block": block,
                    "sequence": sequence,
                    "arm": arm,
                    "elapsed_ms": 1.6 if arm == "B" else 4.9,
                    "ids_sha256": "1" * 64,
                    "ids_identical_to_expected": True,
                    "effective_marker_present": arm == "B",
                    "n_components": 1048576,
                    "n_tokens": 8,
                    "top_k": 2048,
                })
        self._write_raw()
        artifacts = {}
        for name in ("binary", "engine.cu", "test.cu", "runner.py",
                     "scorer.py", "randomness-receipt.json",
                     "drand-verifier.mjs"):
            path = self.run_dir / name
            artifacts[name] = {"path": name, "sha256": sha256(path),
                               "bytes": path.stat().st_size}
        self.manifest = {
            "schema": "glm52-w4-topk-manifest-v1",
            "gate": "W4",
            "candidate": 2,
            "candidate_hash": "a" * 40,
            "freeze_time_unix": 1,
            "binary_sha256": sha256(self.run_dir / "binary"),
            "scorer_sha256": sha256(SCORER_PATH),
            "raw_sha256": sha256(self.run_dir / "raw.jsonl"),
            "configuration": {
                "n_components": 1048576, "n_tokens": 8, "top_k": 2048,
                "blocks": 5, "observations_per_block": 4,
                "flag_name": "DS4_CUDA_TOPK2048_CUB", "flag_value": "1",
                "required_speedup_lower_95": 2.0,
            },
            "randomness": {
                **RECEIPT,
                "verification": "DRAND_BLS_RECEIPT_OK",
                "receipt_path": "randomness-receipt.json",
            },
            "invocation": {
                "argv": ["binary"],
                "environment": {"DS4_CUDA_TOPK2048_CUB": "scheduled-per-arm"},
                "exit_code": 0,
            },
            "device": {"name": "NVIDIA GB10", "uuid": "GPU-test", "driver": "test"},
            "artifacts": artifacts,
        }
        self._write_manifest()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_raw(self) -> None:
        (self.run_dir / "raw.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows))

    def _write_manifest(self) -> None:
        self.manifest["raw_sha256"] = sha256(self.run_dir / "raw.jsonl")
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest, sort_keys=True) + "\n")

    def _score(self) -> dict:
        return SCORER.score_run(self.run_dir)

    def test_valid_evidence_passes_fixed_formula(self) -> None:
        result = self._score()
        self.assertEqual(result["verdict"], "PASS")
        self.assertGreater(result["speedup_lower_95"], 2.0)

    def test_missing_duplicate_or_reordered_observation_fails(self) -> None:
        mutations = (
            self.rows[:-1],
            self.rows + [copy.deepcopy(self.rows[-1])],
            [self.rows[1], self.rows[0], *self.rows[2:]],
        )
        for rows in mutations:
            with self.subTest(rows=len(rows)):
                self.rows = rows
                self._write_raw()
                self._write_manifest()
                with self.assertRaises(SCORER.ScoreError):
                    self._score()
        
    def test_nonfinite_or_nonpositive_timing_fails(self) -> None:
        original_rows = copy.deepcopy(self.rows)
        for value in (math.nan, math.inf, 0.0, -1.0):
            with self.subTest(value=value):
                self.rows = copy.deepcopy(original_rows)
                self.rows[0]["elapsed_ms"] = value
                self._write_raw()
                self._write_manifest()
                with self.assertRaises(SCORER.ScoreError):
                    self._score()

    def test_wrong_ids_missing_marker_and_disabled_flag_fail(self) -> None:
        mutations = ("ids", "marker", "flag")
        original_rows = copy.deepcopy(self.rows)
        original_manifest = copy.deepcopy(self.manifest)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.rows = copy.deepcopy(original_rows)
                self.manifest = copy.deepcopy(original_manifest)
                if mutation == "ids":
                    self.rows[0]["ids_identical_to_expected"] = False
                elif mutation == "marker":
                    next(r for r in self.rows if r["arm"] == "B")[
                        "effective_marker_present"] = False
                else:
                    self.manifest["configuration"]["flag_value"] = "0"
                self._write_raw()
                self._write_manifest()
                with self.assertRaises(SCORER.ScoreError):
                    self._score()

    def test_stale_binary_and_malformed_data_fail(self) -> None:
        self.manifest["artifacts"]["binary"]["sha256"] = "0" * 64
        self._write_manifest()
        with self.assertRaises(SCORER.ScoreError):
            self._score()

    def test_runner_rejects_post_freeze_scorer_replacement(self) -> None:
        reviewed = hashlib.sha256(b"reviewed scorer").hexdigest()
        RUNNER.verify_digest_bindings(
            {"scorer": reviewed}, {"scorer": reviewed})
        with self.assertRaises(ValueError):
            RUNNER.verify_digest_bindings(
                {"scorer": reviewed},
                {"scorer": hashlib.sha256(b"replaced scorer").hexdigest()},
            )
        self.manifest["artifacts"]["binary"]["sha256"] = sha256(
            self.run_dir / "binary")
        (self.run_dir / "raw.jsonl").write_text("{bad json\n")
        self.manifest["raw_sha256"] = sha256(self.run_dir / "raw.jsonl")
        self._write_manifest()
        with self.assertRaises(SCORER.ScoreError):
            self._score()


if __name__ == "__main__":
    unittest.main()
