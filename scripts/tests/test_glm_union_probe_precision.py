#!/usr/bin/env python3
"""Tests for the frozen GLM probe feature-precision comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/80_glm_union_probe_precision.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe_precision", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrecisionDiagnosticTests(unittest.TestCase):
    def bounded_sources(self):
        request = np.repeat(np.asarray([1, 2, 3], dtype=np.uint16), 12)
        position = np.tile(np.arange(12, dtype=np.uint32), 3)
        source = {
            "request_index": request,
            "layer": np.full(request.size, 4, dtype=np.uint16),
            "token_position": position,
            "selected_ids": np.asarray([
                (np.arange(8, dtype=np.uint16) + int(value) * 8) % 256 for value in position
            ], dtype=np.uint8),
            "hidden_q4": np.full((request.size, 3072), 0x88, dtype=np.uint8),
            "hidden_scale": np.ones((request.size, 192), dtype=np.float16),
        }
        return [source], {1: "a", 2: "b", 3: "c"}

    def bounded_diagnostic(self):
        position = np.arange(12, dtype=np.uint32)
        return {
            "request_index": np.ones(12, dtype=np.uint16),
            "layer": np.full(12, 4, dtype=np.uint16),
            "token_position": position,
            "selected_ids": np.asarray([
                (np.arange(8, dtype=np.uint16) + int(value) * 8) % 256 for value in position
            ], dtype=np.uint8),
            "hidden_q4": np.full((12, 3072), 0x88, dtype=np.uint8),
            "hidden_scale": np.ones((12, 192), dtype=np.float16),
            "top_ids": np.tile(np.arange(32, dtype=np.uint8), (12, 1)),
            "top_logits": np.zeros((12, 32), dtype=np.float16),
            "hidden_fp16_holdout_row": np.asarray([0, 1], dtype=np.uint32),
            "hidden_fp16_holdout": np.zeros((2, 6144), dtype=np.float16),
        }

    def sparse_layer_diagnostic(self):
        request = np.repeat(np.asarray([1, 2], dtype=np.uint16), 10)
        layer = np.repeat(np.asarray([4, 5], dtype=np.uint16), 10)
        position = np.tile(np.arange(10, dtype=np.uint32), 2)
        selected = np.asarray([
            (np.arange(8, dtype=np.uint16) + int(value) * 8) % 256 for value in position
        ], dtype=np.uint8)
        return {
            "request_index": request,
            "layer": layer,
            "token_position": position,
            "selected_ids": selected,
            "hidden_q4": np.full((20, 3072), 0x88, dtype=np.uint8),
            "hidden_scale": np.ones((20, 192), dtype=np.float16),
            "top_ids": np.tile(np.arange(32, dtype=np.uint8), (20, 1)),
            "top_logits": np.zeros((20, 32), dtype=np.float16),
            "hidden_fp16_holdout_row": np.asarray([0, 10], dtype=np.uint32),
            "hidden_fp16_holdout": np.zeros((2, 6144), dtype=np.float16),
        }

    def test_paired_metrics_preserve_rows_and_measure_top32_overlap(self) -> None:
        requests = np.asarray([1, 1, 2], dtype=np.uint16)
        targets = np.zeros((3, 256), dtype=np.bool_)
        targets[0, [0, 1]] = True
        targets[1, [2, 3]] = True
        targets[2, [4, 5]] = True
        q4 = np.tile(-np.arange(256, dtype=np.float32), (3, 1))
        fp16 = q4.copy()
        q4[1, 2] = 1000
        fp16[2] = np.roll(fp16[2], 64)
        result = MODULE.diagnostic_pair_metrics(requests, targets, q4, fp16)
        self.assertEqual(result["q4"]["32"]["1"]["events"], 2)
        self.assertEqual(result["fp16"]["32"]["2"]["events"], 1)
        self.assertEqual(result["top32_overlap"]["1"]["events"], 2)
        self.assertEqual(result["top32_overlap"]["2"]["overlap_sum"], 0)
        np.testing.assert_array_equal(result["evidence"]["request"], requests)

    def test_paired_metrics_reject_unpaired_nonfinite_or_empty_inputs(self) -> None:
        requests = np.asarray([1], dtype=np.uint16)
        targets = np.zeros((1, 256), dtype=np.bool_)
        targets[0, 1] = True
        logits = np.zeros((1, 256), dtype=np.float32)
        mutations = [
            (requests[:0], targets[:0], logits[:0], logits[:0]),
            (requests, targets, logits[:, :-1], logits),
            (requests, targets, logits, np.full_like(logits, np.nan)),
        ]
        for values in mutations:
            with self.subTest(shapes=[value.shape for value in values]), self.assertRaises(ValueError):
                MODULE.diagnostic_pair_metrics(*values)

    def test_overlap_aggregation_is_request_macro_and_event_weighted(self) -> None:
        result = MODULE.aggregate_overlap([
            {"1": {"overlap_sum": 32, "events": 1}},
            {"1": {"overlap_sum": 0, "events": 1}, "2": {"overlap_sum": 96, "events": 3}},
        ])
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["events"], 5)
        self.assertEqual(result["event_weighted_overlap"], 0.8)
        self.assertEqual(result["macro_request_overlap"], 0.75)

    def test_replay_aggregates_sparse_layer_request_coverage(self) -> None:
        diagnostic = self.sparse_layer_diagnostic()
        evidence = {}
        with mock.patch.object(MODULE.CV, "LAYERS", (4, 5)):
            for layer_id in MODULE.CV.LAYERS:
                _data, contracts, _selected, _scorable = MODULE.diagnostic_layer_contract(
                    diagnostic, layer_id,
                )
                for k in MODULE.CV.K_VALUES:
                    contract = contracts[str(k)]
                    prefix = f"layer{layer_id}_k{k}_"
                    hits = np.minimum(
                        contract["target_size"][:, None],
                        np.asarray(MODULE.CV.BUDGETS, dtype=np.uint8)[None, :],
                    ).astype(np.uint8)
                    for field in ("global_row", "local_row", "prediction_row", "request", "target_size"):
                        evidence[prefix + field] = contract[field]
                    evidence[prefix + "q4_hits"] = hits
                    evidence[prefix + "fp16_hits"] = hits.copy()
                    evidence[prefix + "top32_overlap_count"] = np.full(
                        contract["request"].size, 32, dtype=np.uint8,
                    )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "events.npz"
                binding = MODULE.CV._write_npz_exclusive(path, evidence)
                replayed = MODULE.replay_diagnostic_events(path, binding, diagnostic)
        for k in map(str, MODULE.CV.K_VALUES):
            self.assertEqual(replayed["q4"][k]["32"]["requests"], 2)
            self.assertEqual(replayed["q4"][k]["32"]["events"], 2)
            self.assertEqual(replayed["q4"][k]["32"]["macro_request_recall"], 1.0)
            self.assertEqual(replayed["q4"][k]["32"]["event_weighted_recall"], 1.0)

    def test_diagnostic_contract_rejects_row_and_expert_mutations(self) -> None:
        rows = 12225
        arrays = {
            "request_index": np.ones(rows, dtype=np.uint16),
            "layer": np.full(rows, 4, dtype=np.uint16),
            "token_position": np.arange(rows, dtype=np.uint32),
            "selected_ids": np.tile(np.arange(8, dtype=np.uint8), (rows, 1)),
            "hidden_q4": np.full((rows, 3072), 0x88, dtype=np.uint8),
            "hidden_scale": np.ones((rows, 192), dtype=np.float16),
            "top_ids": np.tile(np.arange(32, dtype=np.uint8), (rows, 1)),
            "top_logits": np.zeros((rows, 32), dtype=np.float16),
            "hidden_fp16_holdout_row": np.arange(203, dtype=np.uint32),
            "hidden_fp16_holdout": np.zeros((203, 6144), dtype=np.float16),
        }
        MODULE.validate_diagnostic_arrays(arrays)
        duplicate_row = {name: value.copy() for name, value in arrays.items()}
        duplicate_row["hidden_fp16_holdout_row"][1] = 0
        duplicate_expert = {name: value.copy() for name, value in arrays.items()}
        duplicate_expert["selected_ids"][0, 1] = 0
        wrong_dtype = dict(arrays)
        wrong_dtype["layer"] = wrong_dtype["layer"].astype(np.uint8)
        for mutation in (duplicate_row, duplicate_expert, wrong_dtype):
            with self.assertRaises(ValueError):
                MODULE.validate_diagnostic_arrays(mutation)

    def test_bounded_execute_replays_and_rejects_output_mutations(self) -> None:
        sources, groups = self.bounded_sources()
        diagnostic = self.bounded_diagnostic()
        source_binding = {"fixture": "bounded"}
        diagnostic_binding = {"fixture": "precision"}
        state = {
            "down.weight": np.full((32, 6400), 0.25, dtype=np.float32),
            "up.weight": np.full((768, 32), 0.125, dtype=np.float32),
            "up.bias": np.full(768, 0.0625, dtype=np.float32),
        }
        report = {
            "fit_rows": 30, "rank": 32, "epochs": 8, "batch_rows": 512,
            "seed": 20260805, "positive_weights": [1.0, 1.0, 1.0],
            "epoch_losses": [1.0] * 8, "epoch_k_losses": [[1.0] * 3] * 8,
            "deterministic_algorithms": True,
        }
        logits = np.tile(-np.arange(256, dtype=np.float32), (2, 3, 1))
        operations = []

        def predict(*_args, **_kwargs):
            operations.append("predict")
            return logits

        def kernel_log(_start):
            operations.append("fault-scan")
            return "clean\n"

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "precision"
            with (
                mock.patch.object(MODULE.CV, "LAYERS", (4,)),
                mock.patch.object(MODULE.PROBE, "_tracked_bytes", side_effect=lambda path: Path(path).read_bytes()),
                mock.patch.object(MODULE.PROBE, "validate_training_sources", return_value=source_binding),
                mock.patch.object(MODULE.CV, "_load_authorized_sources", return_value=(sources, groups)),
                mock.patch.object(MODULE, "_load_diagnostic", return_value=(diagnostic, diagnostic_binding)),
                mock.patch.object(MODULE.PROBE, "train_probe_head", return_value=(state, report)),
                mock.patch.object(MODULE.PROBE, "predict_probe_head", side_effect=predict),
                mock.patch.object(MODULE.CV, "_repository_head", return_value="1" * 40),
                mock.patch.object(MODULE.CV, "_gpu_snapshot", return_value={"gpu": "ok", "compute_applications": ""}),
                mock.patch.object(MODULE.CV, "_mem_available_kib", return_value=120_000_000),
                mock.patch.object(MODULE.CV, "_kernel_log_since", side_effect=kernel_log),
            ):
                self.assertEqual(MODULE.execute(output), 0)
                self.assertEqual(operations[-1], "fault-scan")
                MODULE.validate_completed_output(output)
                summary_path = output / "summary.json"
                original_summary = summary_path.read_bytes()
                summary = json.loads(original_summary)
                summary["verdict"] = "FAIL"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.validate_completed_output(output)
                summary_path.write_bytes(original_summary)
                event_path = output / "diagnostic-events.npz"
                original_event = event_path.read_bytes()
                event_path.write_bytes(original_event + b"x")
                with self.assertRaises(ValueError):
                    MODULE.validate_completed_output(output)
                event_path.write_bytes(original_event)
                with np.load(event_path, allow_pickle=False) as archive:
                    dropped = {name: archive[name].copy() for name in archive.files}
                for name in list(dropped):
                    if name.startswith("layer4_k4_"):
                        dropped[name] = dropped[name][1:]
                dropped_path = output.parent / "dropped-events.npz"
                dropped_binding = MODULE.CV._write_npz_exclusive(dropped_path, dropped)
                with self.assertRaises(ValueError):
                    MODULE.replay_diagnostic_events(
                        dropped_path, dropped_binding, diagnostic,
                    )
                with np.load(event_path, allow_pickle=False) as archive:
                    fabricated = {name: archive[name].copy() for name in archive.files}
                for name in fabricated:
                    if name.endswith(("_q4_hits", "_fp16_hits")):
                        fabricated[name].fill(0)
                    elif name.endswith("_top32_overlap_count"):
                        fabricated[name].fill(32)
                original_event_path = output.parent / "original-events.npz"
                event_path.rename(original_event_path)
                fabricated_binding = MODULE.CV._write_npz_exclusive(event_path, fabricated)
                fabricated_replay = MODULE.replay_diagnostic_events(
                    event_path, fabricated_binding, diagnostic,
                )
                fabricated_summary = MODULE.build_precision_summary(
                    "1" * 40,
                    MODULE._sha256(output / "manifest.json"),
                    MODULE._sha256(output / "model-manifest.json"),
                    diagnostic_binding,
                    fabricated_binding,
                    fabricated_replay,
                    MODULE._sha256(output / "runtime-final.json"),
                )
                summary_path.write_text(
                    json.dumps(fabricated_summary, sort_keys=True, indent=2), encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "precision semantic event differs"):
                    MODULE.validate_completed_output(output)
                event_path.unlink()
                original_event_path.rename(event_path)
                summary_path.write_bytes(original_summary)

                model_manifest_path = output / "model-manifest.json"
                model_manifest = json.loads(model_manifest_path.read_bytes())
                record = model_manifest["layers"]["4"]
                state_path = output / record["file"]
                original_state_path = output.parent / "original-state.npz"
                state_path.rename(original_state_path)
                zero_state = {
                    name: np.zeros_like(value) for name, value in state.items()
                }
                zero_binding = MODULE.CV._write_npz_exclusive(state_path, zero_state)
                record.update(zero_binding)
                model_manifest_path.write_text(
                    json.dumps(model_manifest, sort_keys=True, indent=2), encoding="utf-8",
                )
                original = json.loads(original_summary)
                rebound_summary = MODULE.build_precision_summary(
                    "1" * 40,
                    MODULE._sha256(output / "manifest.json"),
                    MODULE._sha256(model_manifest_path),
                    diagnostic_binding,
                    original["diagnostic_event_binding"],
                    MODULE.replay_diagnostic_events(
                        event_path, original["diagnostic_event_binding"], diagnostic,
                    ),
                    MODULE._sha256(output / "runtime-final.json"),
                )
                summary_path.write_text(
                    json.dumps(rebound_summary, sort_keys=True, indent=2), encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "precision trained model differs"):
                    MODULE.validate_completed_output(output)
                state_path = output / "layer-004-rank32.npz"
                original_state = state_path.read_bytes()
                state_path.write_bytes(original_state + b"x")
                with self.assertRaises(ValueError):
                    MODULE.validate_completed_output(output)


if __name__ == "__main__":
    unittest.main()
