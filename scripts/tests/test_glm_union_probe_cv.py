#!/usr/bin/env python3
"""Acceptance tests for train-only GLM probe CV metric aggregation."""

from __future__ import annotations

import importlib.util
import copy
import json
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
    def production_fixture(self):
        request = np.repeat(np.asarray([1, 2, 3], dtype=np.uint16), 12)
        layer = np.full(request.size, 3, dtype=np.uint16)
        position = np.tile(np.arange(12, dtype=np.uint32), 3)
        selected = np.asarray([
            (np.arange(8, dtype=np.uint16) + int(pos) * 8) % 256
            for pos in position
        ], dtype=np.uint8)
        source = {
            "request_index": request,
            "layer": layer,
            "token_position": position,
            "selected_ids": selected,
            "hidden_q4": np.zeros((request.size, 1), dtype=np.uint8),
            "hidden_scale": np.ones((request.size, 1), dtype=np.float16),
        }
        groups = {}
        for identity in (1, 2, 3):
            for candidate in range(1000):
                group = f"fixture-{identity}-{candidate}"
                if MODULE.PROBE.grouped_fold(group) == identity - 1:
                    groups[identity] = group
                    break
        return [source], groups

    def fake_layer_run(self, data, groups, layer_id):
        rows, targets, valid = MODULE.PROBE.multi_k_targets(
            data["request_index"], data["layer"], data["token_position"], data["selected_ids"],
        )
        evidence = {}
        frequency = {}
        for k_index, k in enumerate(MODULE.K_VALUES):
            active = valid[:, k_index]
            event_rows = rows[active]
            rankings = np.tile(np.arange(256, dtype=np.uint16), (int(active.sum()), 1))
            raw = MODULE.event_evidence(
                data["request_index"][event_rows], targets[active, k_index], rankings,
            )
            evidence[f"k{k}_row"] = event_rows.astype(np.uint32)
            evidence[f"k{k}_request"] = raw["request"]
            evidence[f"k{k}_target_size"] = raw["target_size"]
            evidence[f"k{k}_frequency_hits"] = raw["hits"]
            for rank in MODULE.RANKS:
                evidence[f"k{k}_probe_{rank}_hits"] = raw["hits"].copy()
            frequency[str(k)] = MODULE.score_event_evidence(
                raw["request"], raw["target_size"], raw["hits"], MODULE.BUDGETS,
            )
        contract = MODULE.expected_layer_contract(data, groups, layer_id)
        training = {}
        for rank in MODULE.RANKS:
            training[str(rank)] = {}
            for fold in range(3):
                training[str(rank)][str(fold)] = {
                    "fit_rows": contract["fit_rows_by_fold"][str(fold)],
                    "rank": rank,
                    "epochs": 8,
                    "batch_rows": 512,
                    "seed": 20260805,
                    "positive_weights": [1.0, 1.0, 1.0],
                    "epoch_losses": [1.0] * 8,
                    "epoch_k_losses": [[1.0, 1.0, 1.0]] * 8,
                    "deterministic_algorithms": True,
                }
        return {
            "schema_version": 1,
            "classification": "TRAIN_ONLY_LAYER_CV",
            "layer": layer_id,
            "source_rows": int(data["request_index"].size),
            "prediction_rows": int(rows.size),
            "requests": 3,
            "frequency": frequency,
            "probe": {
                str(rank): copy.deepcopy(frequency) for rank in MODULE.RANKS
            },
            "training": training,
        }, evidence

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
            "event_evidence_schema": {},
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

    def test_fresh_production_execute_reopens_and_rejects_mutated_outputs(self) -> None:
        sources, groups = self.production_fixture()
        source_binding = {"fixture": "bounded-production-path"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cv"
            with (
                mock.patch.object(MODULE, "LAYERS", (3,)),
                mock.patch.object(MODULE.PROBE, "_tracked_bytes", side_effect=lambda path: Path(path).read_bytes()),
                mock.patch.object(MODULE.PROBE, "validate_training_sources", return_value=source_binding),
                mock.patch.object(MODULE, "_load_authorized_sources", return_value=(sources, groups)),
                mock.patch.object(MODULE, "run_layer", side_effect=self.fake_layer_run),
                mock.patch.object(MODULE, "_repository_head", return_value="1" * 40),
                mock.patch.object(MODULE, "_gpu_snapshot", return_value={"gpu": "ok", "compute_applications": ""}),
                mock.patch.object(MODULE, "_mem_available_kib", return_value=120_000_000),
                mock.patch.object(MODULE, "_kernel_log_since", return_value="clean kernel\n"),
            ):
                self.assertEqual(MODULE.execute("run", output), 0)
                manifest = MODULE._read_json_snapshot(output / "manifest.json")
                identity = {
                    "repository_head": manifest["repository_head"],
                    "driver_sha256": manifest["driver_sha256"],
                    "probe_sha256": manifest["probe_sha256"],
                    "training_source_binding_sha256": manifest["training_source_binding_sha256"],
                }
                MODULE.validate_completed_output(output, source_binding, sources, groups, identity)
                substituted = [{name: value.copy() for name, value in sources[0].items()}]
                substituted[0]["selected_ids"] = (
                    (substituted[0]["selected_ids"].astype(np.uint16) + 128) % 256
                ).astype(np.uint8)
                with self.assertRaises(ValueError):
                    MODULE.validate_completed_output(
                        output, source_binding, substituted, groups, identity,
                    )
                targets = [output / "runtime-start.json", output / "layer-003-events.npz"]
                for target in targets:
                    original = target.read_bytes()
                    target.write_bytes(original + b" ")
                    with self.subTest(target=target.name), self.assertRaises(ValueError):
                        MODULE.validate_completed_output(output, source_binding, sources, groups, identity)
                    target.write_bytes(original)
                summary_path = output / "summary.json"
                summary = MODULE._read_json_snapshot(summary_path)
                summary["selected_rank"] = 16
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.validate_completed_output(output, source_binding, sources, groups, identity)
                with self.assertRaises(FileExistsError):
                    MODULE.execute("run", output)

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
