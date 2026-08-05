#!/usr/bin/env python3
"""Acceptance tests for train-only GLM probe CV metric aggregation."""

from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/79_glm_union_probe_cv.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe_cv", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CVMetricTests(unittest.TestCase):
    def test_runtime_fault_scan_rejects_xid_and_oom_without_false_positive(self) -> None:
        clean = "NVRM: GPU initialized\ntorch allocator ready\n"
        self.assertEqual(MODULE.runtime_fault_lines(clean), [])
        faulted = clean + "NVRM: Xid (PCI:000f:01:00): 31, pid=2\noom-kill:constraint=CONSTRAINT_NONE\n"
        self.assertEqual(len(MODULE.runtime_fault_lines(faulted)), 2)

    def checkpoint_fixture(self):
        events = {"2": 8, "4": 6, "8": 2}
        def metric():
            return {
                str(k): {
                    str(budget): {"1": {
                        "recall_sum": 0.0,
                        "precision_sum": 0.0,
                        "wasted_sum": float(budget * events[str(k)]),
                        "coverage_sum": 0.0,
                        "events": events[str(k)],
                    }} for budget in MODULE.BUDGETS
                } for k in MODULE.K_VALUES
            }
        training = {
            str(rank): {
                str(fold): {
                    "fit_rows": 5 + fold,
                    "rank": rank,
                    "epochs": 8,
                    "batch_rows": 512,
                    "seed": 20260805,
                    "positive_weights": [10.0, 11.0, 12.0],
                    "epoch_losses": [1.0] * 8,
                    "epoch_k_losses": [[1.0, 1.0, 1.0]] * 8,
                    "deterministic_algorithms": True,
                } for fold in range(3)
            } for rank in MODULE.RANKS
        }
        identity = {
            "repository_head": "1" * 40,
            "driver_sha256": "2" * 64,
            "probe_sha256": "3" * 64,
            "training_source_binding_sha256": "4" * 64,
        }
        contract = {
            "layer": 3,
            "source_rows": 10,
            "prediction_rows": 8,
            "requests": 1,
            "request_events": {str(k): {"1": events[str(k)]} for k in MODULE.K_VALUES},
            "fit_rows_by_fold": {str(fold): 5 + fold for fold in range(3)},
        }
        checkpoint = {
            "schema_version": 1,
            "classification": "TRAIN_ONLY_LAYER_CV",
            **identity,
            "previous_checkpoint_sha256": "5" * 64,
            "layer": 3,
            "source_rows": 10,
            "prediction_rows": 8,
            "requests": 1,
            "event_evidence_file": "layer-003-events.npz",
            "event_evidence_sha256": "6" * 64,
            "event_evidence_bytes": 123,
            "frequency": metric(),
            "probe": {str(rank): metric() for rank in MODULE.RANKS},
            "training": training,
        }
        return checkpoint, contract, identity

    def test_layer_checkpoint_rejects_metric_and_identity_mutations(self) -> None:
        checkpoint, contract, identity = self.checkpoint_fixture()
        MODULE.validate_layer_checkpoint(checkpoint, contract, identity, "5" * 64)
        checkpoint["frequency"]["4"]["32"]["1"]["recall_sum"] = 999.0
        with self.assertRaises(ValueError):
            MODULE.validate_layer_checkpoint(checkpoint, contract, identity, "5" * 64)
        checkpoint, contract, identity = self.checkpoint_fixture()
        checkpoint["driver_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            MODULE.validate_layer_checkpoint(checkpoint, contract, identity, "5" * 64)
        checkpoint, contract, identity = self.checkpoint_fixture()
        mutations = []
        missing_request = copy.deepcopy(checkpoint)
        missing_request["probe"]["8"]["2"]["16"].pop("1")
        mutations.append(missing_request)
        nonfinite = copy.deepcopy(checkpoint)
        nonfinite["frequency"]["2"]["16"]["1"]["precision_sum"] = float("nan")
        mutations.append(nonfinite)
        wrong_events = copy.deepcopy(checkpoint)
        wrong_events["frequency"]["8"]["64"]["1"]["events"] = 3
        mutations.append(wrong_events)
        missing_rank = copy.deepcopy(checkpoint)
        missing_rank["probe"].pop("32")
        mutations.append(missing_rank)
        extra_key = copy.deepcopy(checkpoint)
        extra_key["unexpected"] = True
        mutations.append(extra_key)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.validate_layer_checkpoint(mutation, contract, identity, "5" * 64)

    def test_production_main_runs_only_after_metric_definitions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertGreater(source.rfind('if __name__ == "__main__"'), source.rfind("def aggregate_request_metrics"))

    def test_aggregation_is_request_macro_not_event_or_layer_macro(self) -> None:
        requests = np.asarray([1, 1, 2], dtype=np.uint16)
        targets = np.zeros((3, 256), dtype=np.bool_)
        targets[0, [0, 1]] = True
        targets[1, [2, 3]] = True
        targets[2, [9, 10]] = True
        rankings = np.tile(np.arange(256, dtype=np.uint16), (3, 1))
        rankings[2] = np.concatenate(([9, 10], np.arange(0, 9), np.arange(11, 256)))
        first = MODULE.accumulate_request_metrics(requests, targets, rankings, budgets=(1, 2))
        second = MODULE.accumulate_request_metrics(requests, targets, rankings, budgets=(1, 2))
        result = MODULE.aggregate_request_metrics([first, second])
        self.assertEqual(result["2"]["requests"], 2)
        self.assertEqual(result["2"]["events"], 6)
        self.assertEqual(result["2"]["macro_request_recall"], 0.75)
        self.assertEqual(result["2"]["event_weighted_recall"], 2 / 3)

    def test_integer_event_evidence_is_replayable_and_rejects_impossible_hits(self) -> None:
        requests = np.asarray([1, 2], dtype=np.uint16)
        targets = np.zeros((2, 256), dtype=np.bool_)
        targets[0, [1, 2]] = True
        targets[1, [8, 9, 10]] = True
        rankings = np.tile(np.arange(256, dtype=np.uint16), (2, 1))
        observed = MODULE.event_evidence(requests, targets, rankings, budgets=(2, 4))
        np.testing.assert_array_equal(observed["target_size"], [2, 3])
        np.testing.assert_array_equal(observed["hits"], [[1, 2], [0, 0]])
        scored = MODULE.score_event_evidence(
            observed["request"], observed["target_size"], observed["hits"], budgets=(2, 4),
        )
        self.assertEqual(scored["4"]["1"]["coverage_sum"], 1.0)
        broken = observed["hits"].copy()
        broken[0, 0] = 3
        with self.assertRaises(ValueError):
            MODULE.score_event_evidence(
                observed["request"], observed["target_size"], broken, budgets=(2, 4),
            )
        nonmonotonic = observed["hits"].copy()
        nonmonotonic[0] = [2, 1]
        with self.assertRaises(ValueError):
            MODULE.score_event_evidence(
                observed["request"], observed["target_size"], nonmonotonic,
                budgets=(2, 4),
            )

    def test_event_archive_rejects_writer_side_input_mutation(self) -> None:
        arrays = {"k2_hits": np.asarray([[1, 2, 2]], dtype=np.uint8)}
        original_savez = MODULE.np.savez

        def mutate_then_save(handle, **values):
            values["k2_hits"][0, 0] = 2
            return original_savez(handle, **values)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "events.npz"
            with mock.patch.object(MODULE.np, "savez", side_effect=mutate_then_save):
                with self.assertRaises(ValueError):
                    MODULE._write_npz_exclusive(output, arrays)
            self.assertFalse(output.exists())

    def test_aggregation_rejects_duplicate_rankings_and_request_drift(self) -> None:
        requests = np.asarray([1], dtype=np.uint16)
        targets = np.zeros((1, 256), dtype=np.bool_)
        targets[0, 1] = True
        rankings = np.tile(np.arange(256, dtype=np.uint16), (1, 1))
        rankings[0, 2] = rankings[0, 1]
        with self.assertRaises(ValueError):
            MODULE.accumulate_request_metrics(requests, targets, rankings)
        valid = np.tile(np.arange(256, dtype=np.uint16), (1, 1))
        one = MODULE.accumulate_request_metrics(requests, targets, valid)
        changed = {budget: {"2": values["1"]} for budget, values in one.items()}
        with self.assertRaises(ValueError):
            MODULE.aggregate_request_metrics([one, changed])

    def test_fold_weights_exclude_validation_lengths_and_equalize_each_k(self) -> None:
        request = np.asarray([1] * 10 + [2] * 4 + [3] * 6, dtype=np.uint16)
        rows = np.arange(20, dtype=np.int64)
        valid = np.ones((20, 3), dtype=np.bool_)
        valid[[8, 9, 18, 19], 2] = False
        folds = np.asarray([0] * 10 + [1] * 10, dtype=np.uint8)
        weights, fitting = MODULE.fold_training_weights(request, rows, valid, folds, 0)
        np.testing.assert_array_equal(fitting, np.arange(10, 20))
        self.assertTrue(np.all(weights[:10] == 0))
        for k_index in range(3):
            active = (weights[:, k_index] > 0)
            if not active.any():
                continue
            masses = [
                float(weights[(request == identity) & active, k_index].sum())
                for identity in np.unique(request[active])
            ]
            self.assertAlmostEqual(min(masses), max(masses))
        mutated = valid.copy()
        mutated[:10, 1:] = False
        changed, _ = MODULE.fold_training_weights(request, rows, mutated, folds, 0)
        np.testing.assert_array_equal(weights[10:], changed[10:])


if __name__ == "__main__":
    unittest.main()
