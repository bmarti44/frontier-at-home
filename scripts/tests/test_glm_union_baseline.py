#!/usr/bin/env python3
"""Mutation tests for the frozen GLM held-out baseline capture scorer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
        gate = np.zeros((79, 256), dtype="<f4")
        shared = np.zeros((79, 256), dtype="<f4")
        mtp = np.zeros((8, 79, 256), dtype="<f4")
        gate_selected = np.full((79, 8), -1, dtype="<i4")
        shared_selected = np.full((79, 8), -1, dtype="<i4")
        mtp_selected = np.full((8, 79, 8), -1, dtype="<i4")
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
            "-prompt-tokens.i32": np.arange(20, dtype="<i4"),
        }
        for suffix, value in artifacts.items():
            Path(f"{base}{suffix}").write_bytes(value.tobytes(order="C"))
        base.with_suffix(".json").write_text(json.dumps({
            "format": "glm52-p1-baseline-source-v1",
            "source_position": source,
            "mtp_min_position": source,
            "prompt_tokens": 20,
            "layers_total": 79,
            "vocab": 1000,
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
            self.assertEqual(result["mtp_scores"].shape, (8, 79, 256))

    def test_rejects_missing_short_nan_or_top8_mismatch(self) -> None:
        for mutation in ("missing", "short", "extra", "nan", "top8"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                self.publish(directory)
                base = directory / "source-00000004"
                if mutation == "missing":
                    Path(f"{base}-mtp-scores.f32").unlink()
                elif mutation == "short":
                    path = Path(f"{base}-gate-scores.f32")
                    path.write_bytes(path.read_bytes()[:-4])
                elif mutation == "extra":
                    path = Path(f"{base}-gate-scores.f32")
                    path.write_bytes(path.read_bytes() + b"\0\0\0\0")
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

    def test_rejects_prompt_token_artifact_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.publish(directory)
            path = directory / "source-00000004-prompt-tokens.i32"
            path.write_bytes(path.read_bytes()[:-4])
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

    def test_rejects_duplicate_or_extra_metadata(self) -> None:
        for mutation in ("duplicate", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                self.publish(directory)
                path = directory / "source-00000004.json"
                if mutation == "duplicate":
                    payload = path.read_text(encoding="utf-8")
                    path.write_text(
                        payload.replace('"source_position": 4',
                                        '"source_position": 999, "source_position": 4'),
                        encoding="utf-8",
                    )
                else:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["unexpected"] = True
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.load_capture_source(directory, 4)

    def test_authenticated_module_executes_snapshot_not_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "module.py"
            path.write_text("raise RuntimeError('path was reopened')\n", encoding="utf-8")
            module = MODULE._execute_module_snapshot(
                "snapshot_fixture", path, b"VALUE = 17\n",
            )
            self.assertEqual(module.VALUE, 17)

    def test_authenticated_dependency_graph_never_executes_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            probe_path = directory / "probe.py"
            cv_path = directory / "cv.py"
            precision_path = directory / "precision.py"
            for path in (probe_path, cv_path, precision_path):
                path.write_text("raise RuntimeError('path was reopened')\n", encoding="utf-8")
            probe = MODULE._execute_module_snapshot(
                "probe_fixture", probe_path, b"VALUE = 23\n",
            )
            cv = MODULE._execute_module_snapshot(
                "cv_fixture",
                cv_path,
                b"def _load_probe_module():\n"
                b"    raise RuntimeError('nested probe path was reopened')\n"
                b"PROBE = _load_probe_module()\n",
                injected={"__authenticated_probe__": probe},
                substitutions=((
                    b"PROBE = _load_probe_module()\n",
                    b"PROBE = __authenticated_probe__\n",
                ),),
            )
            precision = MODULE._execute_module_snapshot(
                "precision_fixture",
                precision_path,
                b"def _load_module(name, path):\n"
                b"    raise RuntimeError('nested CV path was reopened')\n"
                b"CV = _load_module('cv', 'malicious-path')\n"
                b"PROBE = CV.PROBE\n",
                injected={"__authenticated_cv__": cv},
                substitutions=((
                    b"CV = _load_module('cv', 'malicious-path')\n",
                    b"CV = __authenticated_cv__\n",
                ),),
            )
            self.assertIs(cv.PROBE, probe)
            self.assertIs(precision.CV, cv)
            self.assertIs(precision.PROBE, probe)

    def test_authenticated_dependency_edge_must_occur_exactly_once(self) -> None:
        edge = b"PROBE = _load_probe_module()\n"
        for payload in (b"VALUE = 1\n", edge + edge):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MODULE._execute_module_snapshot(
                    "bad_edge_fixture",
                    Path("bad-edge.py"),
                    payload,
                    injected={"__authenticated_probe__": object()},
                    substitutions=((edge, b"PROBE = __authenticated_probe__\n"),),
                )

    def test_missing_capture_root_fails_before_heldout_open(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-capture"
            with mock.patch.object(
                MODULE, "_load_frozen_module_graph", return_value=(object(), object(), object()),
            ), mock.patch.object(
                MODULE, "_load_test_archive", side_effect=AssertionError("held-out opened"),
            ) as opener:
                with self.assertRaises(FileNotFoundError):
                    MODULE.score_heldout(missing, device="cpu")
            opener.assert_not_called()


class AtomicLifecycleTests(unittest.TestCase):
    def test_safe_run_attestation_accepts_real_clean_kernel_forms(self) -> None:
        environment = "1" * 64
        main = b"\n".join((
            f"candidate_binary_sha256={MODULE.FROZEN_BINARY_SHA256}".encode(),
            f"executed_environment_sha256={environment}".encode(),
            b"executed candidate was verified alive at least once",
            b"SAFE_RUN end rc=0 killed=no",
        ))
        stdout = b"SAFE_RUN_DONE rc=0 killed=no dir=/tmp/log\n"
        for kernel in (b"NO_KERNEL_FAULTS\n", b"-- No entries --\n"):
            with self.subTest(kernel=kernel):
                MODULE._validate_safe_run_artifacts(main, kernel, stdout, environment)

    def test_safe_run_attestation_rejects_fault_or_missing_identity(self) -> None:
        environment = "2" * 64
        good = b"\n".join((
            f"candidate_binary_sha256={MODULE.FROZEN_BINARY_SHA256}".encode(),
            f"executed_environment_sha256={environment}".encode(),
            b"executed candidate was verified alive at least once",
            b"SAFE_RUN end rc=0 killed=no",
        ))
        stdout = b"SAFE_RUN_DONE rc=0 killed=no dir=/tmp/log\n"
        for main, kernel in (
            (good, b"NVRM: Xid 31\n"),
            (good.replace(b"verified alive", b"not sampled"), b"-- No entries --\n"),
            (good + b"\nFATAL replacement", b"-- No entries --\n"),
        ):
            with self.subTest(main=main, kernel=kernel), self.assertRaises(RuntimeError):
                MODULE._validate_safe_run_artifacts(main, kernel, stdout, environment)

    def test_attempt_precedes_single_open_and_success_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "result"
            events: list[str] = []

            def preflight():
                events.append("preflight")
                return {"candidate": "bound"}

            def open_heldout(staging):
                attempt = json.loads((staging / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(attempt["status"], "STARTED")
                events.append("open")
                return {"secret": 17}

            def capture(opened, staging):
                self.assertEqual(opened, {"secret": 17})
                events.append("capture")
                return {"capture": "bound"}

            def score(opened, captured, staging):
                self.assertEqual(captured, {"capture": "bound"})
                events.append("score")
                return {"verdict": "PASS"}

            result = MODULE._run_atomic_lifecycle(
                output, preflight, open_heldout, capture, score,
            )
            self.assertEqual(result, {"verdict": "PASS"})
            self.assertEqual(events, ["preflight", "open", "capture", "score"])
            self.assertEqual(
                json.loads((output / "attempt.json").read_text(encoding="utf-8"))["status"],
                "COMPLETE",
            )
            self.assertEqual(
                json.loads((output / "summary.json").read_text(encoding="utf-8")), result,
            )

    def test_post_open_failure_is_sealed_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "failed"
            open_count = 0

            def open_heldout(_staging):
                nonlocal open_count
                open_count += 1
                return {"secret": 17}

            def fail_capture(_opened, _staging):
                raise RuntimeError("injected capture failure")

            with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
                MODULE._run_atomic_lifecycle(
                    output, lambda: {"candidate": "bound"}, open_heldout,
                    fail_capture, lambda *_args: {"verdict": "PASS"},
                )
            self.assertEqual(open_count, 1)
            attempt = json.loads((output / "attempt.json").read_text(encoding="utf-8"))
            self.assertEqual(attempt["status"], "FAILED")
            self.assertEqual(attempt["failure_type"], "RuntimeError")
            self.assertFalse((output / "summary.json").exists())

    def test_authorized_gate_wires_real_phases_through_atomic_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "authorized"
            configuration = {"output_root": output, "device": "cpu"}
            with mock.patch.object(
                MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
            ) as preflight, mock.patch.object(
                MODULE, "_open_authorized_heldout", return_value={"opened": True},
            ) as opener, mock.patch.object(
                MODULE, "_capture_authorized_set", return_value={"captured": True},
            ) as capture, mock.patch.object(
                MODULE, "_score_authorized_gate", return_value={"verdict": "PASS"},
            ) as scorer:
                result = MODULE._run_authorized_gate(configuration)
            self.assertEqual(result, {"verdict": "PASS"})
            preflight.assert_called_once_with(configuration)
            opener.assert_called_once()
            capture.assert_called_once()
            scorer.assert_called_once()
            attempt = json.loads((output / "attempt.json").read_text(encoding="utf-8"))
            self.assertEqual(attempt["status"], "COMPLETE")

    def test_authorized_gate_is_globally_one_shot_across_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = root / "global-ledger.json"
            calls = 0

            def opener(*_args):
                nonlocal calls
                calls += 1
                return {"opened": True}

            with mock.patch.object(
                MODULE, "AUTHORIZED_LEDGER_PATH", ledger, create=True,
            ), mock.patch.object(
                MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
            ), mock.patch.object(
                MODULE, "_open_authorized_heldout", side_effect=opener,
            ), mock.patch.object(
                MODULE, "_capture_authorized_set", return_value={"captured": True},
            ), mock.patch.object(
                MODULE, "_score_authorized_gate", return_value={"verdict": "PASS"},
            ):
                MODULE._run_authorized_gate({"output_root": root / "first", "device": "cpu"})
                with self.assertRaises(FileExistsError):
                    MODULE._run_authorized_gate({
                        "output_root": root / "alternate", "device": "cpu",
                    })
            self.assertEqual(calls, 1)

    def test_abrupt_death_after_open_permanently_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = root / "global-ledger.json"
            marker = root / "opened"
            configuration = {"output_root": root / "attempt", "device": "cpu"}
            child = os.fork()
            if child == 0:
                with mock.patch.object(
                    MODULE, "AUTHORIZED_LEDGER_PATH", ledger, create=True,
                ), mock.patch.object(
                    MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
                ), mock.patch.object(
                    MODULE, "_open_authorized_heldout",
                    side_effect=lambda *_args: (marker.write_text("opened"), os._exit(91)),
                ):
                    MODULE._run_authorized_gate(configuration)
                os._exit(92)
            _pid, status = os.waitpid(child, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 91)
            self.assertTrue(marker.exists())
            with mock.patch.object(
                MODULE, "AUTHORIZED_LEDGER_PATH", ledger, create=True,
            ), mock.patch.object(
                MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
            ), mock.patch.object(
                MODULE, "_open_authorized_heldout", return_value={"opened": True},
            ), mock.patch.object(
                MODULE, "_capture_authorized_set", return_value={"captured": True},
            ), mock.patch.object(
                MODULE, "_score_authorized_gate", return_value={"verdict": "PASS"},
            ):
                with self.assertRaises(FileExistsError):
                    MODULE._run_authorized_gate({
                        "output_root": root / "retry", "device": "cpu",
                    })


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
