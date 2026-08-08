#!/usr/bin/env python3
"""Production-path contract for the real-capture W9 FP4 falsifier."""

from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/93_score_w9_fp4_falsifier.py"
SPEC = importlib.util.spec_from_file_location("w9_fp4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W9Fp4FalsifierTests(unittest.TestCase):
    def test_e2m1_quantizer_uses_exact_blocks_and_finite_values(self) -> None:
        rows = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]],
                        dtype=np.float32)
        quantized = MODULE.e2m1_quantize(rows, block_width=8)
        np.testing.assert_array_equal(quantized, rows)
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.e2m1_quantize(np.array([[np.nan]], dtype=np.float32), 1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MODULE.e2m1_quantize(rows, block_width=3)

    def test_scale_search_closes_amax_counterexample(self) -> None:
        row = np.zeros((1, 32), dtype=np.float32)
        row[0, :2] = [4.1, 3.0]
        quantized = MODULE.e2m1_quantize(row)
        observed_sse = float(np.sum((quantized - row) ** 2))
        scale_one = row.copy()
        scale_one[0, :2] = [4.0, 3.0]
        representative_sse = float(np.sum((scale_one - row) ** 2))
        self.assertLessEqual(observed_sse, representative_sse + 1e-7)
        rng = np.random.default_rng(12)
        rows = rng.normal(size=(16, 32)).astype(np.float32)
        candidate = MODULE.e2m1_quantize(rows)
        amax = np.max(np.abs(rows), axis=1, keepdims=True)
        scale = np.where(amax > 0, amax / 6.0, 1.0)
        normalized = rows / scale
        codes = MODULE.E2M1_LEVELS[
            np.searchsorted(MODULE.E2M1_MIDPOINTS, normalized, side="left")]
        baseline = codes * scale
        self.assertLessEqual(float(np.sum((candidate - rows) ** 2)),
                             float(np.sum((baseline - rows) ** 2)) + 1e-5)

    def test_hadamard_rotation_preserves_dot_products(self) -> None:
        rng = np.random.default_rng(8)
        left = rng.normal(size=(7, 8)).astype(np.float32)
        right = rng.normal(size=(5, 8)).astype(np.float32)
        signs = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)
        rotated_left = MODULE.hadamard_rotate(left, signs)
        rotated_right = MODULE.hadamard_rotate(right, signs)
        np.testing.assert_allclose(rotated_left @ rotated_right.T,
                                   left @ right.T, rtol=2e-6, atol=2e-6)

    def test_split_is_seeded_disjoint_complete_and_stable(self) -> None:
        first = MODULE.split_indices(32, bytes.fromhex("11" * 32), b"keys")
        second = MODULE.split_indices(32, bytes.fromhex("11" * 32), b"keys")
        self.assertEqual(first, second)
        calibration, heldout = first
        self.assertEqual(len(calibration), 16)
        self.assertEqual(set(calibration) & set(heldout), set())
        self.assertEqual(set(calibration) | set(heldout), set(range(32)))

    def test_channel_correction_fits_calibration_rows_only(self) -> None:
        quantized = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        reference = quantized * np.array([2.0, 0.5], dtype=np.float32)
        alpha = MODULE.fit_channel_correction(reference, quantized)
        np.testing.assert_allclose(alpha, [2.0, 0.5], rtol=0, atol=1e-7)

    def test_query_weighted_error_uses_only_heldout_selected_pairs(self) -> None:
        keys = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [8.0, 8.0]],
                        dtype=np.float32)
        candidate = keys.copy()
        candidate[1] = [0.0, 1.0]
        candidate[3] = [100.0, 100.0]  # not selected and must be irrelevant
        queries = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        selected = np.array([[0, 4, 4], [1, 2, 4]], dtype=np.uint32)
        metric = MODULE.query_weighted_error(
            queries, keys, candidate, selected, selected_sentinel=4,
            heldout_queries=np.array([False, True]),
            heldout_keys=np.array([True, True, True, False]),
        )
        # Held-out logits are [2, 1], errors [-1, 0].
        self.assertEqual(metric["pairs"], 2)
        self.assertAlmostEqual(metric["relative_rmse"], math.sqrt(1.0 / 5.0), places=7)

    def test_selected_rows_reject_duplicates_and_out_of_range_ids(self) -> None:
        queries = np.zeros((1, 1, 2), dtype=np.float32)
        keys = np.zeros((2, 2), dtype=np.float32)
        mask_q = np.array([True])
        mask_k = np.array([True, True])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.query_weighted_error(
                queries, keys, keys, np.array([[0, 0]], dtype=np.uint32), 2,
                mask_q, mask_k)
        with self.assertRaisesRegex(ValueError, "range"):
            MODULE.query_weighted_error(
                queries, keys, keys, np.array([[3, 2]], dtype=np.uint32), 2,
                mask_q, mask_k)

    def test_fixed_gate_threshold_and_candidates_are_source_bound(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "MAXIMUM_RELATIVE_RMSE = 0.05",
            '"plain_e2m1_multistart_f32_scale"',
            '"hadamard_e2m1_multistart_f32_scale"',
            '"hadamard_e2m1_multistart_f32_scale_channel_correction"',
            "manifest.json", "raw.jsonl", "summary.json",
            '("ds4-server", "ds4", "fio")',
            "W9-fp4-falsifier-review-r253.json",
            "W9-fp4-falsifier-candidate3-freeze.json",
            '"NO_RESULT"',
        ):
            self.assertIn(token, source)
        self.assertNotIn('add_argument("--candidate-commit"', source)
        self.assertNotIn('add_argument("--minimum-drand-round"', source)

    def test_review_receipt_rejects_caller_floor_and_blockers(self) -> None:
        valid = {
            "schema": "glm52-w9-fp4-falsifier-review-v1",
            "candidate_hash": "a" * 40,
            "review_round": 253,
            "critical": [],
            "high": [],
            "verdict": "PASS_RUNTIME_ALLOWED",
            "drand_min_round": 6357227,
        }
        self.assertEqual(MODULE.validate_review_receipt(valid), ("a" * 40, 6357227))
        for mutation in (
            {**valid, "drand_min_round": 1},
            {**valid, "critical": ["x"]},
            {**valid, "candidate_hash": "main:" + "a" * 40},
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_review_receipt(mutation)

    def test_runtime_tree_snapshot_rejects_transitive_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "package").mkdir()
            (root / "package" / "module.py").write_text("VALUE = 1\n")
            (root / "native.so").write_bytes(b"linked-numerical-library")
            snapshot = MODULE.snapshot_runtime_tree(root)
            MODULE.verify_runtime_tree(root, snapshot)
            (root / "native.so").write_bytes(b"changed-numerical-library")
            with self.assertRaisesRegex(ValueError, "runtime tree"):
                MODULE.verify_runtime_tree(root, snapshot)

    def test_runtime_contract_isolated_and_deterministic(self) -> None:
        expected = {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        with mock.patch.dict(os.environ, expected, clear=False):
            MODULE.verify_execution_environment(
                isolated=True, no_site=True, safe_path=True)
        with mock.patch.dict(os.environ, {**expected, "OPENBLAS_NUM_THREADS": "4"}, clear=False):
            with self.assertRaisesRegex(ValueError, "thread"):
                MODULE.verify_execution_environment(
                    isolated=True, no_site=True, safe_path=True)

    def test_launcher_clears_environment_and_fixes_capture(self) -> None:
        launcher = (ROOT / "results/glm52-gates/harness/w9_fp4_falsifier_v1.sh").read_text()
        for token in ("/usr/bin/env -i", "/usr/bin/python3 -I -S -B",
                      "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                      "MKL_NUM_THREADS=1", "BLIS_NUM_THREADS=1",
                      "NUMEXPR_NUM_THREADS=1",
                      "attempt-73838408ccb1d126ade7b67c8d86fa00/on/capture"):
            self.assertIn(token, launcher)
        self.assertNotIn("PYTHONPATH=", launcher)
        self.assertNotIn("candidate-commit", launcher)
        self.assertNotIn("minimum-drand-round", launcher)

    def test_bound_input_rejects_path_replacement_and_in_place_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "input.bin"
            original = b"authoritative-generation"
            path.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            with self.assertRaisesRegex(ValueError, "generation"):
                with MODULE.BoundInput(path, len(original), digest):
                    replacement = root / "replacement.bin"
                    replacement.write_bytes(b"replacement-generation!"[:len(original)])
                    os.replace(replacement, path)
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "generation"):
                with MODULE.BoundInput(path, len(original), digest):
                    path.write_bytes(b"x" * len(original))

    def test_terminal_verifier_rejects_post_publication_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            MODULE.publish_evidence(
                root,
                {"schema": "manifest"},
                [{"record_type": "row", "value": 1}],
                {"schema": "summary", "verdict": "PASS"},
            )
            terminal = MODULE.verify_terminal(root)
            self.assertEqual(terminal["verdict"], "PASS")
            (root / "summary.json").write_text(
                json.dumps({"schema": "summary", "verdict": "NO_RESULT"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "terminal"):
                MODULE.verify_terminal(root)

    def test_terminal_verifier_holds_all_artifacts_through_cross_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            work = root / "work"
            evidence = root / "evidence"
            work.mkdir()
            evidence.mkdir()
            manifest, rows, summary = self._compact_evidence(work)
            MODULE.publish_evidence(evidence, manifest, rows, summary)
            original = MODULE.strict_json_bytes

            def mutate_during_summary(value: bytes, label: str):
                parsed = original(value, label)
                if label == "terminal summary":
                    (evidence / "manifest.json").write_bytes(b"{}\n" + b" " * 4094)
                return parsed

            with mock.patch.object(MODULE, "strict_json_bytes", side_effect=mutate_during_summary):
                with self.assertRaisesRegex(ValueError, "terminal"):
                    MODULE.verify_terminal(evidence)

    def test_terminal_verifier_recomputes_aggregate_and_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            work = root / "work"
            evidence = root / "evidence"
            work.mkdir()
            evidence.mkdir()
            manifest, rows, summary = self._compact_evidence(work)
            summary["query_weighted_error"] = 0.0
            MODULE.publish_evidence(evidence, manifest, rows, summary)
            with self.assertRaisesRegex(ValueError, "aggregate|metric|summary"):
                MODULE.verify_terminal(evidence)

    def test_three_relay_randomness_requires_exact_agreement(self) -> None:
        record = {
            "round": 6357190,
            "randomness": "11" * 32,
            "signature": "22" * 96,
            "previous_signature": "33" * 96,
        }
        receipt = {
            "schema": "glm52-drand-three-relay-v1",
            "relay_urls": list(MODULE.RELAY_URLS),
            "relay_records": [record, record, record],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with MODULE.BoundInput(path, None, None) as bound:
                with mock.patch.object(MODULE.subprocess, "run") as run:
                    self.assertEqual(MODULE._verify_randomness(bound, 6357189), record)
                    run.assert_called_once()
            receipt["relay_records"][2] = {**record, "round": 6357191}
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with MODULE.BoundInput(path, None, None) as bound:
                with self.assertRaisesRegex(ValueError, "agree"):
                    MODULE._verify_randomness(bound, 6357189)

    def _compact_evidence(self, root: pathlib.Path):
            rng = np.random.default_rng(19)
            keys = rng.normal(size=(8, 32)).astype("<f4")
            queries = rng.normal(size=(2, 1, 32)).astype("<f4")
            candidate_commit = "a" * 40
            randomness = None
            for value in range(256):
                proposed = f"{value:064x}"
                master = hashlib.sha256(
                    b"GLM52-W9-FP4-SPLIT-V1\0" + bytes.fromhex(proposed)
                    + bytes.fromhex(MODULE.CAPTURE_HASHES["kv.f32"])
                    + bytes.fromhex(MODULE.CAPTURE_HASHES["query.f32"])
                    + bytes.fromhex(candidate_commit)
                ).digest()
                if MODULE.split_indices(2, master, b"queries/0")[1] == (1,):
                    randomness = proposed
                    break
            self.assertIsNotNone(randomness)
            selected = np.array([[0, 9, 9, 9, 9, 9, 9, 9], list(range(8))], dtype="<u4")
            paths = {
                "kv.f32": keys.tobytes(),
                "query.f32": queries.tobytes(),
                "selected.u32": selected.tobytes(),
            }
            with contextlib.ExitStack() as stack:
                bound = {}
                for name, value in paths.items():
                    path = root / name
                    path.write_bytes(value)
                    bound[name] = stack.enter_context(MODULE.BoundInput(
                        path, len(value), hashlib.sha256(value).hexdigest()))
                metadata = {"selected_padding_sentinel": 9}
                with (mock.patch.object(MODULE, "LAYERS", (0,)),
                      mock.patch.object(MODULE, "KV_ROWS", 8),
                      mock.patch.object(MODULE, "QUERY_ROWS", 2),
                      mock.patch.object(MODULE, "QUERY_HEADS", 1),
                      mock.patch.object(MODULE, "WIDTH", 32),
                      mock.patch.object(MODULE, "SELECTED_CAPACITY", 8)):
                    manifest, rows, summary = MODULE._evaluate(
                        root, bound, metadata,
                        {"round": 6357190, "randomness": randomness}, "b" * 64,
                        candidate_commit, 6357189, {"source": "c" * 64},
                        {"runtime": "d" * 64}, "e" * 64,
                    )
            return manifest, rows, summary

    def test_compact_evaluate_exercises_all_candidates_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest, rows, summary = self._compact_evidence(root)
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(summary["candidates"]), set(MODULE.CANDIDATES))
        self.assertIn(summary["verdict"], {"PASS", "NO_RESULT"})
        self.assertEqual(manifest["drand_round"], 6357190)


if __name__ == "__main__":
    unittest.main()
