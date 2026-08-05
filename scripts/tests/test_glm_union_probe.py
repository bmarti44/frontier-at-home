#!/usr/bin/env python3
"""Acceptance tests for exact future-union construction and scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/78_glm_union_probe.py"
SPEC = importlib.util.spec_from_file_location("glm_union_probe", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnionTargetTests(unittest.TestCase):
    def fixture(self):
        request = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint16)
        layer = np.asarray([3] * 9, dtype=np.uint16)
        position = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.uint32)
        selected = np.asarray([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [10, 11], [11, 12], [12, 13], [13, 14],
        ], dtype=np.uint8)
        return request, layer, position, selected

    def test_future_union_is_exact_and_never_crosses_request_boundary(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        np.testing.assert_array_equal(rows, [0, 1, 2, 5, 6])
        expected = [
            {1, 2, 3}, {2, 3, 4}, {3, 4, 5},
            {11, 12, 13}, {12, 13, 14},
        ]
        self.assertEqual(target.shape, (5, 256))
        for row, wanted in zip(target, expected):
            self.assertEqual(set(np.flatnonzero(row)), wanted)

    def test_future_union_rejects_noncontiguous_or_reordered_rows(self) -> None:
        request, layer, position, selected = self.fixture()
        for mutation in ("gap", "reorder", "bad_k", "duplicate_expert"):
            with self.subTest(mutation=mutation):
                changed_position = position.copy()
                changed_selected = selected.copy()
                k = 2
                if mutation == "gap":
                    changed_position[2] = 7
                elif mutation == "reorder":
                    changed_position[[1, 2]] = changed_position[[2, 1]]
                elif mutation == "bad_k":
                    k = 3
                else:
                    changed_selected[0] = [1, 1]
                with self.assertRaises(ValueError):
                    MODULE.future_union_targets(
                        request, layer, changed_position, changed_selected, k=k,
                    )

    def test_scoring_uses_unweighted_request_macro_and_reports_waste(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        rankings = np.tile(np.arange(256, dtype=np.uint16), (len(rows), 1))
        result = MODULE.score_rankings(
            rows, target, rankings, request, budgets=(2, 4),
        )
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["events"], 5)
        self.assertEqual(result["budgets"], [2, 4])
        self.assertEqual(result["by_budget"]["2"]["macro_request_recall"], 1 / 18)
        self.assertEqual(result["by_budget"]["4"]["macro_request_recall"], 1 / 3)
        self.assertEqual(result["by_budget"]["2"]["event_weighted_wasted_experts"], 9 / 5)

    def test_frequency_prior_is_per_layer_count_ranked_with_stable_ties(self) -> None:
        layer = np.asarray([3, 3, 4], dtype=np.uint16)
        selected = np.asarray([[2, 1], [2, 3], [9, 8]], dtype=np.uint8)
        rankings = MODULE.frequency_prior_by_layer(layer, selected)
        self.assertEqual(set(rankings), {3, 4})
        np.testing.assert_array_equal(rankings[3][:5], [2, 1, 3, 0, 4])
        np.testing.assert_array_equal(rankings[4][:4], [8, 9, 0, 1])
        self.assertEqual(np.unique(rankings[3]).size, 256)
        duplicate = selected.copy()
        duplicate[0] = [2, 2]
        with self.assertRaises(ValueError):
            MODULE.frequency_prior_by_layer(layer, duplicate)

    def test_probe_feature_transforms_are_exact_causal_and_group_stable(self) -> None:
        packed = np.asarray([[0xF1, 0x97]], dtype=np.uint8)
        scale = np.asarray([[2.0, 1.0]], dtype=np.float16)
        np.testing.assert_array_equal(
            MODULE.unpack_probe_hidden(packed, scale, width=4),
            np.asarray([[-14.0, 14.0, -1.0, 1.0]], dtype=np.float32),
        )
        request = np.asarray([1, 1, 1, 2, 2], dtype=np.uint16)
        layer = np.asarray([3, 3, 3, 3, 3], dtype=np.uint16)
        position = np.asarray([0, 1, 2, 0, 1], dtype=np.uint32)
        selected = np.asarray([[1], [2], [3], [8], [9]], dtype=np.uint8)
        history = MODULE.causal_expert_history(request, layer, position, selected)
        self.assertEqual(history[2, 3], 1.0)
        self.assertEqual(history[2, 2], 0.5)
        self.assertEqual(history[2, 1], 0.25)
        self.assertEqual(history[3, 8], 1.0)
        self.assertEqual(history[3, 3], 0.0)
        self.assertEqual(MODULE.grouped_fold("case_001"), MODULE.grouped_fold("case_001"))
        self.assertIn(MODULE.grouped_fold("case_001"), range(3))
        gapped = position.copy()
        gapped[2] = 4
        with self.assertRaises(ValueError):
            MODULE.causal_expert_history(request, layer, gapped, selected)

    def test_scoring_rejects_duplicate_rankings_or_cross_row_mismatch(self) -> None:
        request, layer, position, selected = self.fixture()
        rows, target = MODULE.future_union_targets(
            request, layer, position, selected, k=2,
        )
        rankings = np.tile(np.arange(256, dtype=np.uint16), (len(rows), 1))
        rankings[0, 1] = rankings[0, 0]
        with self.assertRaises(ValueError):
            MODULE.score_rankings(rows, target, rankings, request)
        with self.assertRaises(ValueError):
            MODULE.score_rankings(rows[:-1], target, rankings, request)

    def test_partition_is_request_grouped_complete_and_disjoint(self) -> None:
        request = np.asarray([1, 1, 2, 3, 3, 4], dtype=np.uint16)
        metadata = [
            {"request_index": 1, "case_id": "a", "split": "train-fit"},
            {"request_index": 2, "case_id": "b", "split": "train-precision-diagnostic"},
            {"request_index": 3, "case_id": "c", "split": "calibration"},
            {"request_index": 4, "case_id": "d", "split": "test"},
        ]
        expected = {name: 1 for name in MODULE.SPLIT_COUNTS}
        observed = MODULE.partition_request_rows(request, metadata, expected)
        np.testing.assert_array_equal(observed["train-fit"], [0, 1])
        np.testing.assert_array_equal(observed["train-precision-diagnostic"], [2])
        np.testing.assert_array_equal(observed["calibration"], [3, 4])
        np.testing.assert_array_equal(observed["test"], [5])
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(observed.values()))), np.arange(6),
        )

    def test_partition_rejects_missing_duplicate_or_wrong_split_metadata(self) -> None:
        request = np.asarray([1, 2, 3, 4], dtype=np.uint16)
        valid = [
            {"request_index": 1, "case_id": "a", "split": "train-fit"},
            {"request_index": 2, "case_id": "b", "split": "train-precision-diagnostic"},
            {"request_index": 3, "case_id": "c", "split": "calibration"},
            {"request_index": 4, "case_id": "d", "split": "test"},
        ]
        expected = {name: 1 for name in MODULE.SPLIT_COUNTS}
        mutations = [
            valid[:-1],
            [valid[0], valid[0], *valid[2:]],
            [{**valid[0], "split": "test"}, *valid[1:]],
            [{**valid[0], "case_id": "b"}, *valid[1:]],
        ]
        for metadata in mutations:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                MODULE.partition_request_rows(request, metadata, expected)

    def test_partition_rejects_balanced_relabel_and_cross_split_group(self) -> None:
        request = np.asarray([1, 2, 3, 4], dtype=np.uint16)
        valid = [
            {"request_index": 1, "case_id": "a", "group_id": "a", "split": "train-fit"},
            {"request_index": 2, "case_id": "b", "group_id": "b", "split": "train-precision-diagnostic"},
            {"request_index": 3, "case_id": "c", "group_id": "c", "split": "calibration"},
            {"request_index": 4, "case_id": "d", "group_id": "d", "split": "test"},
        ]
        expected_counts = {name: 1 for name in MODULE.SPLIT_COUNTS}
        expected_mapping = {str(row["case_id"]): str(row["split"]) for row in valid}
        balanced = [dict(row) for row in valid]
        balanced[0]["split"], balanced[2]["split"] = balanced[2]["split"], balanced[0]["split"]
        duplicate_group = [dict(row) for row in valid]
        duplicate_group[2]["group_id"] = duplicate_group[0]["group_id"]
        for metadata in (balanced, duplicate_group):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                MODULE.partition_request_rows(
                    request, metadata, expected_counts, expected_mapping,
                )

    def test_split_arrays_remaps_fp16_holdout_without_leaking_rows(self) -> None:
        rows = {
            "train-fit": np.asarray([0, 1]),
            "train-precision-diagnostic": np.asarray([2]),
            "calibration": np.asarray([3, 4]),
            "test": np.asarray([5]),
        }
        arrays = {
            "request_index": np.asarray([1, 1, 2, 3, 3, 4], dtype=np.uint16),
            "layer": np.asarray([3] * 6, dtype=np.uint16),
            "token_position": np.asarray([0, 1, 0, 0, 1, 0], dtype=np.uint32),
            "selected_ids": np.arange(12, dtype=np.uint8).reshape(6, 2),
            "hidden_fp16_holdout_row": np.asarray([0, 2, 5], dtype=np.uint32),
            "hidden_fp16_holdout": np.asarray([[10], [20], [30]], dtype=np.float16),
        }
        split = MODULE.split_compact_arrays(arrays, rows)
        np.testing.assert_array_equal(split["train-fit"]["request_index"], [1, 1])
        np.testing.assert_array_equal(split["calibration"]["request_index"], [3, 3])
        np.testing.assert_array_equal(
            split["train-fit"]["hidden_fp16_holdout_row"], [0],
        )
        np.testing.assert_array_equal(
            split["train-precision-diagnostic"]["hidden_fp16_holdout"], [[20]],
        )
        np.testing.assert_array_equal(split["test"]["hidden_fp16_holdout_row"], [0])
        self.assertEqual(sum(value["layer"].size for value in split.values()), 6)

    def test_split_arrays_rejects_bad_holdout_or_row_coverage(self) -> None:
        base = {
            "request_index": np.asarray([1, 2], dtype=np.uint16),
            "layer": np.asarray([3, 3], dtype=np.uint16),
            "hidden_fp16_holdout_row": np.asarray([0], dtype=np.uint32),
            "hidden_fp16_holdout": np.asarray([[1]], dtype=np.float16),
        }
        rows = {
            "train-fit": np.asarray([0]),
            "train-precision-diagnostic": np.asarray([], dtype=np.int64),
            "calibration": np.asarray([], dtype=np.int64),
            "test": np.asarray([1]),
        }
        mutations = []
        duplicate = {key: value.copy() for key, value in base.items()}
        duplicate["hidden_fp16_holdout_row"] = np.asarray([0, 0], dtype=np.uint32)
        duplicate["hidden_fp16_holdout"] = np.asarray([[1], [1]], dtype=np.float16)
        mutations.append(duplicate)
        out_of_range = {key: value.copy() for key, value in base.items()}
        out_of_range["hidden_fp16_holdout_row"] = np.asarray([2], dtype=np.uint32)
        mutations.append(out_of_range)
        short = {key: value.copy() for key, value in base.items()}
        short["layer"] = short["layer"][:1]
        mutations.append(short)
        for arrays in mutations:
            with self.subTest(arrays=arrays), self.assertRaises(ValueError):
                MODULE.split_compact_arrays(arrays, rows)

    def test_split_archive_publishes_four_isolated_bound_shards(self) -> None:
        compactor = MODULE._load_compactor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            arrays = {
                "request_index": np.asarray([1, 2, 3, 4], dtype=np.uint16),
                "layer": np.asarray([3, 3, 3, 3], dtype=np.uint16),
                "token_position": np.asarray([0, 0, 0, 0], dtype=np.uint32),
                "selected_ids": np.tile(np.arange(8, dtype=np.uint8), (4, 1)),
                "top_ids": np.tile(np.arange(32, dtype=np.uint8), (4, 1)),
                "top_logits": np.zeros((4, 32), dtype=np.float16),
                "hidden_q4": np.zeros((4, 2), dtype=np.uint8),
                "hidden_scale": np.ones((4, 1), dtype=np.float16),
                "hidden_fp16_holdout_row": np.asarray([0, 3], dtype=np.uint32),
                "hidden_fp16_holdout": np.ones((2, 4), dtype=np.float16),
            }
            splits = list(MODULE.SPLIT_COUNTS)
            metadata = [
                {"request_index": index + 1, "case_id": f"case_{index:03d}",
                 "group_id": f"case_{index:03d}", "split": split}
                for index, split in enumerate(splits)
            ]
            source_manifest = compactor.publish_bundle(source, arrays, {
                "schema_version": 1,
                "format": "glm52-union-p0-npz-v2",
                "requests": 4,
                "rows": 4,
                "request_metadata": metadata,
                "raw_source_retained": str(root / "raw"),
            })
            receipt_path = root / "receipt.json"
            receipt = {
                "manifest_sha256": MODULE._sha256(source / "manifest.json"),
                "output_sha256": source_manifest["output_sha256"],
                "output_bytes": source_manifest["output_bytes"],
                "retained_raw_directory": str(root / "raw"),
                "observed": {"requests": 4, "rows": 4},
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            plan_path = root / "plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            case_mapping = {str(row["case_id"]): str(row["split"]) for row in metadata}
            counts = {split: 1 for split in splits}

            def trusted_read(path):
                return Path(path).read_bytes()

            output = root / "output"
            with (
                mock.patch.object(MODULE, "QUALITY_COMPACTION_RECEIPT", receipt_path),
                mock.patch.object(MODULE, "SPLIT_PLAN", plan_path),
                mock.patch.object(MODULE, "SPLIT_COUNTS", counts),
                mock.patch.object(MODULE, "_tracked_bytes", side_effect=trusted_read),
                mock.patch.object(MODULE, "_repository_head", return_value="1" * 40),
                mock.patch.object(MODULE, "expected_case_splits", return_value=case_mapping),
            ):
                result = MODULE.split_archive(source, output)
            self.assertEqual(set(result["splits"]), set(splits))
            self.assertEqual(result["repository_head"], "1" * 40)
            self.assertEqual(result["splitter_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            for split in splits:
                child = json.loads((output / split / "manifest.json").read_text())
                self.assertEqual(child["split"], split)
                self.assertEqual(child["requests"], 1)
                self.assertEqual(child["rows"], 1)
                with np.load(output / split / "records.npz", allow_pickle=False) as shard:
                    self.assertEqual(shard["request_index"].size, 1)
            self.assertFalse(any(root.glob(".output.tmp.*")))

            calls = 0
            def fail_second_publish(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected publisher failure")
                return compactor.publish_bundle(*args, **kwargs)
            failing_publisher = mock.Mock(publish_bundle=mock.Mock(side_effect=fail_second_publish))
            with (
                mock.patch.object(MODULE, "QUALITY_COMPACTION_RECEIPT", receipt_path),
                mock.patch.object(MODULE, "SPLIT_PLAN", plan_path),
                mock.patch.object(MODULE, "SPLIT_COUNTS", counts),
                mock.patch.object(MODULE, "_tracked_bytes", side_effect=trusted_read),
                mock.patch.object(MODULE, "_repository_head", return_value="1" * 40),
                mock.patch.object(MODULE, "expected_case_splits", return_value=case_mapping),
                mock.patch.object(MODULE, "_load_compactor", return_value=failing_publisher),
                self.assertRaises(RuntimeError),
            ):
                MODULE.split_archive(source, root / "atomic-fail")
            self.assertFalse((root / "atomic-fail").exists())
            self.assertFalse(any(root.glob(".atomic-fail.tmp.*")))

            broken_receipt = dict(receipt)
            broken_receipt["output_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(broken_receipt), encoding="utf-8")
            with (
                mock.patch.object(MODULE, "QUALITY_COMPACTION_RECEIPT", receipt_path),
                mock.patch.object(MODULE, "SPLIT_PLAN", plan_path),
                mock.patch.object(MODULE, "SPLIT_COUNTS", counts),
                mock.patch.object(MODULE, "_tracked_bytes", side_effect=trusted_read),
                mock.patch.object(MODULE, "_repository_head", return_value="1" * 40),
                mock.patch.object(MODULE, "expected_case_splits", return_value=case_mapping),
                self.assertRaises(ValueError),
            ):
                MODULE.split_archive(source, root / "must-not-exist")
            self.assertFalse((root / "must-not-exist").exists())

    def test_tracked_input_rejects_worktree_bytes_that_differ_from_head(self) -> None:
        completed = mock.Mock(returncode=0, stdout=b"")
        committed = mock.Mock(returncode=0, stdout=b"different")
        with mock.patch.object(MODULE.subprocess, "run", side_effect=[completed, committed]):
            with self.assertRaises(ValueError):
                MODULE._tracked_bytes(SCRIPT)

    def test_training_source_gate_accepts_only_train_fit_and_two_bound_long_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            train.mkdir()
            (train / "records.npz").write_bytes(b"train-records")
            train_manifest = {"format": "glm52-union-p1-split-npz-v1", "split": "train-fit"}
            (train / "manifest.json").write_text(json.dumps(train_manifest), encoding="utf-8")
            longs = []
            shards = []
            for index in (1, 2):
                directory = root / f"long-{index}"
                directory.mkdir()
                (directory / "records.npz").write_bytes(f"long-{index}".encode())
                (directory / "manifest.json").write_text(
                    json.dumps({"format": "glm52-union-p0-npz-v1"}), encoding="utf-8",
                )
                longs.append(directory)
                shards.append({
                    "directory": str(directory),
                    "manifest_sha256": MODULE._sha256(directory / "manifest.json"),
                    "output_sha256": MODULE._sha256(directory / "records.npz"),
                    "records_bytes": (directory / "records.npz").stat().st_size,
                })
            split_receipt = root / "split-receipt.json"
            split_receipt.write_text(json.dumps({"observed": {"splits": {"train-fit": {
                "manifest_sha256": MODULE._sha256(train / "manifest.json"),
                "output_sha256": MODULE._sha256(train / "records.npz"),
                "output_bytes": (train / "records.npz").stat().st_size,
            }}}}), encoding="utf-8")
            long_receipt = root / "long-receipt.json"
            long_receipt.write_text(json.dumps({"shards": shards}), encoding="utf-8")
            with (
                mock.patch.object(MODULE, "P1_SPLIT_RECEIPT", split_receipt),
                mock.patch.object(MODULE, "LONG_COMPACTION_RECEIPT", long_receipt),
                mock.patch.object(MODULE, "_tracked_bytes", side_effect=lambda path: Path(path).read_bytes()),
            ):
                accepted = MODULE.validate_training_sources(train, longs)
                self.assertEqual(len(accepted["long_train"]), 2)
                changed = dict(train_manifest)
                changed["split"] = "calibration"
                (train / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                split_data = json.loads(split_receipt.read_text())
                split_data["observed"]["splits"]["train-fit"]["manifest_sha256"] = MODULE._sha256(
                    train / "manifest.json"
                )
                split_receipt.write_text(json.dumps(split_data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.validate_training_sources(train, longs)
                with self.assertRaises(ValueError):
                    MODULE.validate_training_sources(train, longs[:1])


if __name__ == "__main__":
    unittest.main()
