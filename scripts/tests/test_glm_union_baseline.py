#!/usr/bin/env python3
"""Mutation tests for the frozen GLM held-out baseline capture scorer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/81_glm_union_baseline.py"
SPEC = importlib.util.spec_from_file_location("glm_union_baseline_tests", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureBundleTests(unittest.TestCase):
    def publish(self, directory: Path, source: int = 4) -> None:
        base = directory / f"source-{source:08d}"
        gate = np.zeros((78, 256), dtype="<f4")
        shared = np.zeros((78, 256), dtype="<f4")
        mtp = np.zeros((8, 78, 256), dtype="<f4")
        gate_selected = np.full((78, 8), -1, dtype="<i4")
        shared_selected = np.full((78, 8), -1, dtype="<i4")
        mtp_selected = np.full((8, 78, 8), -1, dtype="<i4")
        for layer in range(4, 78):
            for rank, expert in enumerate(range(8)):
                gate[layer, expert] = np.float32(20 - rank)
                shared[layer, expert] = np.float32(20 - rank)
                gate_selected[layer, rank] = expert
                shared_selected[layer, rank] = expert
                for step in range(8):
                    mtp[step, layer, expert] = np.float32(20 - rank)
                    mtp_selected[step, layer, rank] = expert
        artifacts = {
            "-gate-scores.f32": gate,
            "-gate-selected.i32": gate_selected,
            "-shared-scores.f32": shared,
            "-shared-selected.i32": shared_selected,
            "-mtp-scores.f32": mtp,
            "-mtp-selected.i32": mtp_selected,
            "-predicted.i32": np.arange(8, dtype="<i4"),
        }
        for suffix, value in artifacts.items():
            Path(f"{base}{suffix}").write_bytes(value.tobytes(order="C"))
        base.with_suffix(".json").write_text(json.dumps({
            "format": "glm52-p1-baseline-source-v1",
            "source_position": source,
            "mtp_min_position": source,
            "prompt_tokens": 20,
            "layers_first": 4,
            "layers_last": 77,
            "experts": 256,
            "selected": 8,
            "K": 8,
            "source_ready_ms": 20.0,
            "mtp_ms": [1.0] * 7,
            "target_ms": [2.0] * 8,
            "cumulative_ms": [23.0 + 3.0 * index for index in range(8)],
            "elapsed_ms": 123.5,
            "predicted_tokens": list(range(8)),
        }) + "\n", encoding="utf-8")

    def test_accepts_complete_exact_capture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.publish(directory)
            result = MODULE.load_capture_source(directory, 4)
            self.assertEqual(result["metadata"]["source_position"], 4)
            self.assertEqual(result["mtp_scores"].shape, (8, 78, 256))

    def test_rejects_missing_short_nan_or_top8_mismatch(self) -> None:
        for mutation in ("missing", "short", "nan", "top8"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                self.publish(directory)
                base = directory / "source-00000004"
                if mutation == "missing":
                    Path(f"{base}-mtp-scores.f32").unlink()
                elif mutation == "short":
                    path = Path(f"{base}-gate-scores.f32")
                    path.write_bytes(path.read_bytes()[:-4])
                elif mutation == "nan":
                    path = Path(f"{base}-shared-scores.f32")
                    values = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
                    values[4 * 256] = np.nan
                    path.write_bytes(values.tobytes())
                else:
                    path = Path(f"{base}-gate-selected.i32")
                    values = np.frombuffer(path.read_bytes(), dtype="<i4").copy()
                    values[4 * 8] = 9
                    path.write_bytes(values.tobytes())
                with self.assertRaises((FileNotFoundError, ValueError)):
                    MODULE.load_capture_source(directory, 4)

    def test_rejects_metadata_token_lineage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.publish(directory)
            path = directory / "source-00000004.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["predicted_tokens"][3] = 99
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_capture_source(directory, 4)

    def test_rejects_mtp_cache_origin_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.publish(directory)
            path = directory / "source-00000004.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["mtp_min_position"] = 0
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_capture_source(directory, 4)


class BaselineTableTests(unittest.TestCase):
    def fixture(self):
        requests = np.repeat(np.arange(1, 21, dtype=np.uint16), 2)
        targets = {}
        rankings = {
            method: {} for method in
            ("frequency", "gate_replay", "shared_correction", "mtp", "probe")
        }
        base = np.tile(np.arange(256, dtype=np.uint16), (requests.size, 1))
        better = base.copy()
        better[:, 0], better[:, 200] = base[:, 200], base[:, 0]
        for k in (2, 4, 8):
            target = np.zeros((requests.size, 256), dtype=np.bool_)
            target[:, 200] = True
            targets[k] = target
            for method in rankings:
                rankings[method][k] = (
                    better.copy() if method == "probe" else base.copy()
                )
        return requests, targets, rankings

    def test_probe_decision_uses_paired_request_bootstrap_and_all_cells(self) -> None:
        requests, targets, rankings = self.fixture()
        result = MODULE.score_baseline_table(
            requests, targets, rankings, bootstrap_resamples=1000,
        )
        self.assertEqual(result["decision"]["verdict"], "PASS")
        self.assertTrue(result["decision"]["probe_point_higher_all_nine_cells"])
        self.assertGreater(
            result["paired_probe_minus_frequency"]["4"]["32"]["one_sided_95_lower"],
            0.0,
        )

    def test_one_nonpositive_secondary_cell_stops_probe(self) -> None:
        requests, targets, rankings = self.fixture()
        rankings["probe"][8] = rankings["frequency"][8].copy()
        result = MODULE.score_baseline_table(
            requests, targets, rankings, bootstrap_resamples=1000,
        )
        self.assertEqual(result["decision"]["verdict"], "STOP_PROBE")
        self.assertFalse(result["decision"]["probe_point_higher_all_nine_cells"])

    def test_rejects_unequal_request_event_coverage_or_partial_rankings(self) -> None:
        requests, targets, rankings = self.fixture()
        bad_requests = requests.copy()
        bad_requests[-1] = 19
        with self.assertRaises(ValueError):
            MODULE.score_baseline_table(
                bad_requests, targets, rankings, bootstrap_resamples=10,
            )
        rankings["mtp"][4] = rankings["mtp"][4][:, :64]
        with self.assertRaises(ValueError):
            MODULE.score_baseline_table(
                requests, targets, rankings, bootstrap_resamples=10,
            )


if __name__ == "__main__":
    unittest.main()
