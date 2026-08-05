#!/usr/bin/env python3
"""Mutation tests for the frozen GLM held-out baseline capture scorer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
REAL_RESERVE_GLOBAL_AUTHORITY = MODULE._reserve_global_authority


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
    def setUp(self) -> None:
        self.authority_patch = mock.patch.object(
            MODULE, "_reserve_global_authority",
            return_value={"classification": "TEST_EXTERNAL_AUTHORITY"},
        )
        self.authority_patch.start()
        self.addCleanup(self.authority_patch.stop)

    def test_safe_run_attestation_accepts_real_clean_kernel_forms(self) -> None:
        environment = "1" * 64
        main = b"\n".join((
            f"candidate_binary_sha256={MODULE.FROZEN_BINARY_SHA256}".encode(),
            (
                "memory_guard_descriptor_path=/proc/123/fd/9 memory_guard_sha256=" +
                MODULE.FROZEN_SCRIPT_HASHES[MODULE.MEMORY_GUARD_PATH]
            ).encode(),
            f"executed_environment_sha256={environment}".encode(),
            b"executed candidate was verified alive at least once",
            b"SAFE_RUN end rc=0 killed=no",
        ))
        stdout = b"SAFE_RUN_DONE rc=0 killed=no dir=/tmp/log\n"
        for kernel in (b"NO_KERNEL_FAULTS\n", b"-- No entries --\n"):
            with self.subTest(kernel=kernel):
                MODULE._validate_safe_run_artifacts(main, kernel, stdout, environment)

    def test_root_managed_journal_reservation_is_visible_before_return(self) -> None:
        emitted: dict[str, str] = {}

        def emit(fields):
            emitted.update(fields)

        def records():
            if not emitted:
                return []
            return [{
                "MESSAGE_ID": MODULE.AUTHORITY_MESSAGE_ID,
                "GLM52_P1_GATE": MODULE.AUTHORITY_GATE_ID,
                "GLM52_P1_EVENT": "STARTED",
                "PRIORITY": "2",
                "SYSLOG_IDENTIFIER": MODULE.AUTHORITY_IDENTIFIER,
                **emitted,
                "__CURSOR": "test-cursor",
                "_BOOT_ID": "test-boot",
                "__REALTIME_TIMESTAMP": "123456789",
            }]

        with mock.patch.object(
            MODULE, "_journal_authority_records", side_effect=records,
        ), mock.patch.object(
            MODULE, "_emit_journal_authority", side_effect=emit,
        ), mock.patch.object(
            MODULE, "_reserve_root_tombstone",
            return_value={"status": "RESERVED"},
        ) as publisher:
            result = REAL_RESERVE_GLOBAL_AUTHORITY(
                {"candidate": "bound"}, Path("/tmp/nonheldout-test"), 123,
            )
        publisher.assert_called_once()
        self.assertEqual(result["journal_cursor"], "test-cursor")
        self.assertEqual(result["started_epoch_ns"], 123)
        self.assertRegex(result["preflight_sha256"], r"^[0-9a-f]{64}$")

    def test_existing_journal_reservation_blocks_before_publication(self) -> None:
        with mock.patch.object(
            MODULE, "_journal_authority_records", return_value=[{"existing": True}],
        ), mock.patch.object(MODULE, "_emit_journal_authority") as publisher:
            with self.assertRaises(FileExistsError):
                REAL_RESERVE_GLOBAL_AUTHORITY(
                    {"candidate": "bound"}, Path("/tmp/retry"), 124,
                )
        publisher.assert_not_called()

    def test_root_tombstone_helper_response_is_exactly_bound(self) -> None:
        installed_sha256 = MODULE._hash_regular(MODULE.ROOT_SUBMITTER_PATH)[0]
        controller_sha256 = MODULE._hash_regular(Path(MODULE.__file__).resolve())[0]
        candidate = "1" * 40
        fields = {
            "GLM52_P1_PREFLIGHT_SHA256": "2" * 64,
            "GLM52_P1_OUTPUT_SHA256": "3" * 64,
            "GLM52_P1_STARTED_NS": "123",
        }
        reservation_payload = json.dumps({
            "schema_version": 1,
            "classification": "GLM52_P1_PERMANENT_RESERVATION_REQUEST",
            "candidate_hash": candidate,
            **fields,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        reservation = MODULE._sha256_bytes(reservation_payload)
        authority = {
            "schema_version": 1,
            "status": "APPROVED",
            "candidate_hash": candidate,
            "controller_sha256": controller_sha256,
            "approval_sha256": "7" * 64,
            "approval_device": 8,
            "approval_inode": 9,
        }
        response = {
            "schema_version": 1,
            "status": "RESERVED",
            "candidate_hash": candidate,
            "reservation_sha256": reservation,
            "marker_sha256": "4" * 64,
            "marker_device": 5,
            "marker_inode": 6,
            "approved_controller_sha256": controller_sha256,
            "approval_sha256": authority["approval_sha256"],
            "approval_device": authority["approval_device"],
            "approval_inode": authority["approval_inode"],
        }
        authority_completed = subprocess.CompletedProcess(
            ["sudo"], 0, json.dumps(authority, sort_keys=True) + "\n", "",
        )
        reserve_completed = subprocess.CompletedProcess(
            ["sudo"], 0, json.dumps(response, sort_keys=True) + "\n", "",
        )
        with mock.patch.object(
            MODULE, "FROZEN_ROOT_SUBMITTER_SHA256", installed_sha256,
        ), mock.patch.object(
            MODULE.subprocess, "run",
            side_effect=(authority_completed, reserve_completed),
        ) as runner:
            observed = MODULE._reserve_root_tombstone(
                {"harness_commit": candidate}, fields,
            )
        self.assertEqual(observed, {**response, "root_approval": authority})
        self.assertEqual(runner.call_args_list[0].args[0], [
            "/usr/bin/sudo", "-n", str(MODULE.ROOT_SUBMITTER_PATH), "p1-authority",
        ])
        command = runner.call_args_list[1].args[0]
        self.assertEqual(command[:4], [
            "/usr/bin/sudo", "-n", str(MODULE.ROOT_SUBMITTER_PATH), "reserve-p1",
        ])
        self.assertEqual(command[4:], [candidate, reservation])

    def test_existing_root_tombstone_fails_closed(self) -> None:
        installed_sha256 = MODULE._hash_regular(MODULE.ROOT_SUBMITTER_PATH)[0]
        controller_sha256 = MODULE._hash_regular(Path(MODULE.__file__).resolve())[0]
        authority = {
            "schema_version": 1,
            "status": "APPROVED",
            "candidate_hash": "1" * 40,
            "controller_sha256": controller_sha256,
            "approval_sha256": "4" * 64,
            "approval_device": 5,
            "approval_inode": 6,
        }
        with mock.patch.object(
            MODULE, "FROZEN_ROOT_SUBMITTER_SHA256", installed_sha256,
        ), mock.patch.object(
            MODULE.subprocess, "run",
            side_effect=(
                subprocess.CompletedProcess(
                    ["sudo"], 0, json.dumps(authority) + "\n", "",
                ),
                subprocess.CompletedProcess(["sudo"], 17, "", ""),
            ),
        ), self.assertRaises(FileExistsError):
            MODULE._reserve_root_tombstone(
                {"harness_commit": "1" * 40},
                {
                    "GLM52_P1_PREFLIGHT_SHA256": "2" * 64,
                    "GLM52_P1_OUTPUT_SHA256": "3" * 64,
                    "GLM52_P1_STARTED_NS": "123",
                },
            )

    def test_root_authority_rejects_changed_executing_controller(self) -> None:
        installed_sha256 = MODULE._hash_regular(MODULE.ROOT_SUBMITTER_PATH)[0]
        authority = {
            "schema_version": 1,
            "status": "APPROVED",
            "candidate_hash": "1" * 40,
            "controller_sha256": "9" * 64,
            "approval_sha256": "4" * 64,
            "approval_device": 5,
            "approval_inode": 6,
        }
        with mock.patch.object(
            MODULE, "FROZEN_ROOT_SUBMITTER_SHA256", installed_sha256,
        ), mock.patch.object(
            MODULE.subprocess, "run",
            return_value=subprocess.CompletedProcess(
                ["sudo"], 0, json.dumps(authority) + "\n", "",
            ),
        ) as runner, self.assertRaisesRegex(RuntimeError, "executing controller"):
            MODULE._reserve_root_tombstone(
                {"harness_commit": "1" * 40},
                {
                    "GLM52_P1_PREFLIGHT_SHA256": "2" * 64,
                    "GLM52_P1_OUTPUT_SHA256": "3" * 64,
                    "GLM52_P1_STARTED_NS": "123",
                },
            )
        self.assertEqual(runner.call_count, 1)

    def test_root_authority_rejects_unapproved_executing_commit(self) -> None:
        installed_sha256 = MODULE._hash_regular(MODULE.ROOT_SUBMITTER_PATH)[0]
        controller_sha256 = MODULE._hash_regular(Path(MODULE.__file__).resolve())[0]
        authority = {
            "schema_version": 1,
            "status": "APPROVED",
            "candidate_hash": "1" * 40,
            "controller_sha256": controller_sha256,
            "approval_sha256": "4" * 64,
            "approval_device": 5,
            "approval_inode": 6,
        }
        with mock.patch.object(
            MODULE, "FROZEN_ROOT_SUBMITTER_SHA256", installed_sha256,
        ), mock.patch.object(
            MODULE.subprocess, "run",
            return_value=subprocess.CompletedProcess(
                ["sudo"], 0, json.dumps(authority) + "\n", "",
            ),
        ) as runner, self.assertRaisesRegex(RuntimeError, "approved candidate"):
            MODULE._reserve_root_tombstone(
                {"harness_commit": "2" * 40},
                {
                    "GLM52_P1_PREFLIGHT_SHA256": "3" * 64,
                    "GLM52_P1_OUTPUT_SHA256": "5" * 64,
                    "GLM52_P1_STARTED_NS": "123",
                },
            )
        self.assertEqual(runner.call_count, 1)

    def test_journal_eviction_cannot_reopen_permanent_authority(self) -> None:
        retained: list[dict[str, object]] = []
        permanent = False

        def emit(fields):
            retained.append({
                "MESSAGE_ID": MODULE.AUTHORITY_MESSAGE_ID,
                "GLM52_P1_GATE": MODULE.AUTHORITY_GATE_ID,
                "GLM52_P1_EVENT": "STARTED",
                "PRIORITY": "2",
                "SYSLOG_IDENTIFIER": MODULE.AUTHORITY_IDENTIFIER,
                **fields,
                "__CURSOR": "first-cursor",
                "_BOOT_ID": "first-boot",
                "__REALTIME_TIMESTAMP": "123456789",
            })

        def tombstone(*_args):
            nonlocal permanent
            if permanent:
                raise FileExistsError("permanent root tombstone exists")
            permanent = True
            return {"status": "RESERVED"}

        with mock.patch.object(
            MODULE, "_journal_authority_records", side_effect=lambda: list(retained),
        ), mock.patch.object(
            MODULE, "_emit_journal_authority", side_effect=emit,
        ), mock.patch.object(
            MODULE, "_reserve_root_tombstone", side_effect=tombstone,
        ):
            REAL_RESERVE_GLOBAL_AUTHORITY(
                {"candidate": "bound"}, Path("/tmp/first"), 125,
            )
            retained.clear()  # Simulate ordinary journal rotation/vacuum.
            with self.assertRaises(FileExistsError):
                REAL_RESERVE_GLOBAL_AUTHORITY(
                    {"candidate": "bound"}, Path("/tmp/after-rotation"), 126,
                )

    def test_safe_run_attestation_rejects_fault_or_missing_identity(self) -> None:
        environment = "2" * 64
        good = b"\n".join((
            f"candidate_binary_sha256={MODULE.FROZEN_BINARY_SHA256}".encode(),
            (
                "memory_guard_descriptor_path=/proc/123/fd/9 memory_guard_sha256=" +
                MODULE.FROZEN_SCRIPT_HASHES[MODULE.MEMORY_GUARD_PATH]
            ).encode(),
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
            ledger = Path(raw) / "global-ledger.json"
            configuration = {"output_root": output, "device": "cpu"}
            with mock.patch.object(
                MODULE, "AUTHORIZED_LEDGER_PATH", ledger,
            ), mock.patch.object(
                MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
            ) as preflight, mock.patch.object(
                MODULE, "_open_authorized_heldout", return_value={"opened": True},
            ) as opener, mock.patch.object(
                MODULE, "_capture_authorized_set", return_value={"captured": True},
            ) as capture, mock.patch.object(
                MODULE, "_score_authorized_gate", return_value={"verdict": "PASS"},
            ) as scorer, mock.patch.object(
                MODULE, "validate_completed_result",
                side_effect=lambda _root, *, expected_summary, **_kwargs: expected_summary,
            ):
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
            ), mock.patch.object(
                MODULE, "validate_completed_result",
                side_effect=lambda _root, *, expected_summary, **_kwargs: expected_summary,
            ):
                MODULE._run_authorized_gate({"output_root": root / "first", "device": "cpu"})
                with self.assertRaises(FileExistsError):
                    MODULE._run_authorized_gate({
                        "output_root": root / "alternate", "device": "cpu",
                    })
            self.assertEqual(calls, 1)

    def test_deleting_local_ledger_cannot_reopen_authorized_gate(self) -> None:
        """The user-owned receipt must not be the global one-shot authority."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = root / "deletable-local-ledger.json"
            calls = 0
            reserved = False

            def opener(*_args):
                nonlocal calls
                calls += 1
                return {"opened": True}

            def reserve(*_args):
                nonlocal reserved
                if reserved:
                    raise FileExistsError("external authority already reserved")
                reserved = True
                return {"classification": "TEST_EXTERNAL_AUTHORITY"}

            patches = (
                mock.patch.object(MODULE, "AUTHORIZED_LEDGER_PATH", ledger, create=True),
                mock.patch.object(
                    MODULE, "_preflight_authorized_gate", return_value={"frozen": True},
                ),
                mock.patch.object(
                    MODULE, "_open_authorized_heldout", side_effect=opener,
                ),
                mock.patch.object(
                    MODULE, "_capture_authorized_set", return_value={"captured": True},
                ),
                mock.patch.object(
                    MODULE, "_score_authorized_gate", return_value={"verdict": "PASS"},
                ),
                mock.patch.object(
                    MODULE, "validate_completed_result",
                    side_effect=lambda _root, *, expected_summary, **_kwargs: expected_summary,
                ),
                mock.patch.object(
                    MODULE, "_reserve_global_authority", side_effect=reserve,
                ),
            )
            with (
                patches[0], patches[1], patches[2], patches[3], patches[4],
                patches[5], patches[6],
            ):
                MODULE._run_authorized_gate({
                    "output_root": root / "first", "device": "cpu",
                })
                ledger.unlink()
                with self.assertRaises(FileExistsError):
                    MODULE._run_authorized_gate({
                        "output_root": root / "second", "device": "cpu",
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

    def test_concurrent_output_names_share_one_global_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = root / "global-ledger.json"
            marker = root / "open-count"
            read_end, write_end = os.pipe()
            children = []
            for index in range(2):
                child = os.fork()
                if child == 0:
                    os.close(write_end)
                    os.read(read_end, 1)

                    def opener(*_args):
                        descriptor = os.open(
                            marker, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600,
                        )
                        try:
                            os.write(descriptor, b"opened\n")
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
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
                    ), mock.patch.object(
                        MODULE, "validate_completed_result",
                        side_effect=lambda _root, *, expected_summary, **_kwargs: expected_summary,
                    ):
                        try:
                            MODULE._run_authorized_gate({
                                "output_root": root / f"concurrent-{index}",
                                "device": "cpu",
                            })
                        except FileExistsError:
                            pass
                    os._exit(0)
                children.append(child)
            os.close(read_end)
            os.write(write_end, b"xx")
            os.close(write_end)
            for child in children:
                _pid, status = os.waitpid(child, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "opened\n")

    def test_launch_environment_discards_all_ambient_values(self) -> None:
        ds4_values = {
            "DS4_CUDA_FETCH_THREADS": "8",
            "DS4_LOCK_FILE": "/run/lock/frontier-at-home/inference.lock",
        }
        injected = {
            "LD_PRELOAD": "/tmp/injected.so",
            "LD_LIBRARY_PATH": "/tmp/injected",
            "CUDA_VISIBLE_DEVICES": "none",
            "DS4_GLM_PREFETCH": "1",
            "PYTHONPATH": "/tmp/injected",
            "PATH": "/tmp/injected",
        }
        with mock.patch.dict(os.environ, injected, clear=True):
            environment = MODULE._build_launch_environment(
                ds4_values, Path("/home/bmarti44/.cache/glm52-candidate"),
            )
        self.assertTrue(set(ds4_values).issubset(environment))
        self.assertEqual(environment["PATH"], MODULE.FIXED_LAUNCH_PATH)
        for name in injected:
            if name != "PATH":
                self.assertNotIn(name, environment)
        self.assertEqual(
            environment["GLM_SAFE_EXPECTED_ENV_SHA256"],
            MODULE._environment_sha256(ds4_values, list(ds4_values)),
        )

    def test_launch_environment_binds_descriptor_memory_guard(self) -> None:
        environment = MODULE._build_launch_environment(
            {"DS4_LOCK_FILE": "/run/lock/frontier-at-home/inference.lock"},
            Path("/home/bmarti44/.cache/glm52-candidate"),
            {
                "memory_guard_path": "/proc/123/fd/9",
                "memory_guard_sha256": "a" * 64,
            },
        )
        self.assertEqual(environment["GLM_SAFE_MEMORY_GUARD_PATH"], "/proc/123/fd/9")
        self.assertEqual(environment["GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256"], "a" * 64)

    def test_authenticated_runtime_uses_sealed_script_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapper_payload = b"#!/bin/bash\necho bound\n"
            guard_payload = b"print('bound')\n"
            runtime = {
                "safe_run_payload": wrapper_payload,
                "memory_guard_payload": guard_payload,
            }
            wrapper = MODULE._publish_authenticated_runtime(root, runtime)
            try:
                guard = Path(runtime["memory_guard_path"])
                self.assertEqual(wrapper.read_bytes(), wrapper_payload)
                self.assertEqual(guard.read_bytes(), guard_payload)
                expected_seals = (
                    MODULE.fcntl.F_SEAL_WRITE | MODULE.fcntl.F_SEAL_GROW |
                    MODULE.fcntl.F_SEAL_SHRINK | MODULE.fcntl.F_SEAL_SEAL
                )
                for name in ("safe_run_descriptor", "memory_guard_descriptor"):
                    self.assertEqual(
                        MODULE.fcntl.fcntl(
                            runtime[name], MODULE.fcntl.F_GET_SEALS,
                        ),
                        expected_seals,
                    )
            finally:
                os.close(runtime["safe_run_descriptor"])
                os.close(runtime["memory_guard_descriptor"])

    def test_authenticated_runtime_cannot_be_replaced_after_publication(self) -> None:
        """Published script authority must survive same-user pathname mutation."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = {
                "safe_run_payload": b"#!/bin/bash\necho bound\n",
                "memory_guard_payload": b"print('bound')\n",
            }
            wrapper = MODULE._publish_authenticated_runtime(root, runtime)
            try:
                guard = Path(runtime["memory_guard_path"])
                with self.assertRaises(OSError):
                    wrapper.unlink()
                with self.assertRaises(OSError):
                    guard.unlink()
                with self.assertRaises(OSError):
                    wrapper.write_bytes(b"#!/bin/bash\necho forged\n")
                with self.assertRaises(OSError):
                    guard.write_bytes(b"print('forged')\n")
                replacement = root / "forged-wrapper"
                replacement.write_bytes(b"#!/bin/bash\necho forged\n")
                with self.assertRaises(OSError):
                    os.replace(replacement, wrapper)
                with self.assertRaises(OSError):
                    os.rename(wrapper, root / "moved-wrapper")
                with self.assertRaises(OSError):
                    os.link(wrapper, root / "linked-wrapper")
                with self.assertRaises(OSError):
                    os.symlink(root / "forged-wrapper", wrapper)
            finally:
                os.close(runtime["safe_run_descriptor"])
                os.close(runtime["memory_guard_descriptor"])

    def test_authenticated_binary_is_private_named_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source-binary"
            source.write_bytes(b"authenticated executable bytes")
            descriptor = MODULE._open_regular(source)
            try:
                digest, identity = MODULE._hash_open_descriptor(descriptor, source)
                candidate_root, binary = MODULE._publish_authenticated_binary(
                    {
                        "candidate_root": str(root),
                        "binary_sha256": digest,
                        "binary_stat": list(identity),
                    },
                    {"binary_descriptor": descriptor},
                    parent=root,
                )
                self.assertEqual(binary.name, "ds4-server")
                self.assertEqual(binary.parent, candidate_root)
                self.assertEqual(binary.read_bytes(), b"authenticated executable bytes")
                self.assertEqual(MODULE._hash_regular(binary)[0], digest)
                self.assertEqual(candidate_root.stat().st_mode & 0o777, 0o700)
            finally:
                os.close(descriptor)

    def test_engine_capture_uses_normal_glm_prompt_templating(self) -> None:
        command = MODULE._engine_capture_command(
            Path("/private/glm_safe_run.sh"), "tag", Path("/private/ds4-server"),
            Path("/proc/controller/fd/model"), Path("/private/prompt.txt"),
        )
        self.assertNotIn("--raw-prompt", command)
        self.assertEqual(
            command[command.index("--prompt-file") + 1], "/private/prompt.txt",
        )

    def test_retained_model_descriptor_defeats_path_swap_restore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = root / "model.gguf"
            displaced = root / "approved.gguf"
            model.write_bytes(b"approved-model")
            descriptor = MODULE._open_regular(model)
            try:
                expected = MODULE._hash_open_descriptor(descriptor, model)
                model.rename(displaced)
                model.write_bytes(b"replacement-model")
                observed = MODULE._hash_open_descriptor(descriptor, model)
                self.assertEqual(observed[0], expected[0])
                self.assertEqual(observed[1][:3], expected[1][:3])
                self.assertNotEqual(observed[1], expected[1])
                self.assertNotEqual(model.read_bytes(), displaced.read_bytes())
            finally:
                os.close(descriptor)


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

    def test_two_control_contract_rejects_state_or_continuation_drift(self) -> None:
        record = {
            "schema_version": 1,
            "request_index": 1,
            "request_id": "1" * 64,
            "stage_order": ["control_before", "diagnostic", "control_after"],
            "diagnostic": {
                "fresh_process": True,
                "resident_arena_bytes": 0,
                "cache_namespace": "diagnostic-1",
                "exit_code": 0,
            },
            "control_before": {
                "fresh_process": True,
                "cache_namespace": "control-before-1",
                "continuation_sha256": "2" * 64,
                "token_ids_sha256": "3" * 64,
                "exit_code": 0,
            },
            "control_after": {
                "fresh_process": True,
                "cache_namespace": "control-after-1",
                "continuation_sha256": "2" * 64,
                "token_ids_sha256": "3" * 64,
                "exit_code": 0,
            },
            "failure_injection": {
                "stages": [
                    "mtp_call", "target_eval", "route_capture", "disposal",
                ],
                "all_destroyed": True,
                "all_control_continuations_equal": True,
            },
        }
        MODULE.validate_two_control_record(record)
        for mutation in ("continuation", "namespace", "failure"):
            changed = json.loads(json.dumps(record))
            if mutation == "continuation":
                changed["control_after"]["continuation_sha256"] = "9" * 64
            elif mutation == "namespace":
                changed["control_after"]["cache_namespace"] = "diagnostic-1"
            else:
                changed["failure_injection"]["all_destroyed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.validate_two_control_record(changed)

    def test_two_control_sequence_brackets_diagnostic_in_distinct_processes(self) -> None:
        calls = []
        continuation = "2" * 64
        tokens = "3" * 64

        def run_arm(name):
            calls.append(name)
            if name == "diagnostic":
                return {
                    "fresh_process": True,
                    "resident_arena_bytes": 0,
                    "cache_namespace": "diagnostic-1",
                    "exit_code": 0,
                }
            return {
                "fresh_process": True,
                "cache_namespace": f"{name}-1",
                "continuation_sha256": continuation,
                "token_ids_sha256": tokens,
                "exit_code": 0,
            }

        failure = {
            "stages": ["mtp_call", "target_eval", "route_capture", "disposal"],
            "all_destroyed": True,
            "all_control_continuations_equal": True,
        }
        record = MODULE.run_two_control_sequence(
            1, "1" * 64, run_arm, failure,
        )
        self.assertEqual(calls, ["control_before", "diagnostic", "control_after"])
        self.assertEqual(record["failure_injection"], failure)
        MODULE.validate_two_control_record(record)

    def test_control_fingerprint_uses_exact_tokens_logits_and_rejects_short_output(self):
        selected = [
            f'ds4: decode-consistency selected[{index}]={{"token":{index}}}'.encode()
            for index in range(8)
        ]
        payload = b"\n".join([
            b"unrelated timing line",
            *selected,
            b"ds4: decode-consistency compared prefix_tokens=32 vocab=64 max_abs=0 at token=0 live=1 fresh=1 rms=0",
            b"ds4: live_top: {\"token\":1}@1",
            b"ds4: fresh_top: {\"token\":1}@1",
        ]) + b"\n"
        continuation, tokens = MODULE._control_fingerprint(payload)
        self.assertRegex(continuation, r"^[0-9a-f]{64}$")
        self.assertRegex(tokens, r"^[0-9a-f]{64}$")
        with self.assertRaises(RuntimeError):
            MODULE._control_fingerprint(payload.replace(selected[-1] + b"\n", b""))

    def test_cost_table_requires_five_matched_cold_warm_blocks_and_fails_closed(self) -> None:
        rows = []
        for block in range(5):
            for temperature in ("cold", "warm"):
                for method, milliseconds, persistent, temporary, expert_bytes in (
                    ("gate_replay", 1.0, 0, 1024, 0),
                    ("shared_correction", 1.2, 0, 2048, 0),
                    ("mtp", 10.0, 4096, 8192, 100000),
                    ("probe", 2.0, 2048, 4096, 0),
                ):
                    rows.append({
                        "block": block,
                        "temperature": temperature,
                        "method": method,
                        "completed_ms": milliseconds + block / 100.0,
                        "persistent_bytes": persistent,
                        "peak_temporary_bytes": temporary,
                        "target_expert_bytes_read": expert_bytes,
                        "completed_events": 32,
                        "synchronized": True,
                    })
        table = MODULE.score_cost_table(rows)
        self.assertEqual(table["verdict"], "PASS")
        self.assertFalse(table["mtp_equal_cost_to_probe"])
        for mutation in ("missing", "nan", "unsynchronized", "unequal"):
            changed = json.loads(json.dumps(rows))
            if mutation == "missing":
                changed.pop()
            elif mutation == "nan":
                changed[0]["completed_ms"] = float("nan")
            elif mutation == "unsynchronized":
                changed[0]["synchronized"] = False
            else:
                changed[0]["completed_events"] = 31
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.score_cost_table(changed)

    def test_mtp_fold_requires_equal_cost_and_strict_recall_dominance(self) -> None:
        requests, targets, rankings = self.fixture()
        metrics = MODULE.score_baseline_table(
            requests, targets, rankings, bootstrap_resamples=100,
        )
        cost = {
            "mtp_equal_cost_to_gate_replay": True,
            "mtp_equal_cost_to_shared_correction": True,
            "mtp_equal_cost_to_probe": True,
        }
        self.assertFalse(MODULE.decide_mtp_fold(metrics, cost)["fold_into_mtp"])
        for k in (2, 4, 8):
            for budget in (16, 32, 64):
                metrics["methods"]["probe"][str(k)][str(budget)][
                    "macro_request_recall"
                ] = 0.9
                metrics["methods"]["mtp"][str(k)][str(budget)][
                    "macro_request_recall"
                ] = 1.0
        self.assertTrue(MODULE.decide_mtp_fold(metrics, cost)["fold_into_mtp"])
        cost["mtp_equal_cost_to_probe"] = False
        self.assertFalse(MODULE.decide_mtp_fold(metrics, cost)["fold_into_mtp"])

    def test_completed_result_is_independently_reconstructed_and_strict_reopened(self) -> None:
        requests, targets, rankings = self.fixture()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical = root / "canonical.npz"
            MODULE.write_canonical_scorer_input(canonical, requests, targets, rankings)
            summary = MODULE.score_baseline_table(
                requests, targets, rankings, bootstrap_resamples=1000,
            )
            summary.update({
                "schema_version": 1,
                "classification": "P1_HELD_OUT_BASELINE_SCORE",
            })
            cost_rows = []
            for block in range(5):
                for temperature in ("cold", "warm"):
                    for method, milliseconds in (
                        ("gate_replay", 1.0), ("shared_correction", 1.2),
                        ("mtp", 10.0), ("probe", 2.0),
                    ):
                        cost_rows.append({
                            "block": block,
                            "temperature": temperature,
                            "method": method,
                            "completed_ms": milliseconds + block / 100.0,
                            "persistent_bytes": 0,
                            "peak_temporary_bytes": 1024,
                            "target_expert_bytes_read": (
                                100000 if method == "mtp" else 0
                            ),
                            "completed_events": 32,
                            "synchronized": True,
                        })
            (root / "cost.json").write_text(json.dumps({
                "schema_version": 1, "rows": cost_rows,
            }) + "\n", encoding="utf-8")
            summary["cost"] = MODULE.score_cost_table(cost_rows)
            failure = {
                "stages": ["mtp_call", "target_eval", "route_capture", "disposal"],
                "all_destroyed": True,
                "all_control_continuations_equal": True,
            }
            runtime_logs = []
            for request in range(1, 21):
                control = {
                    "schema_version": 1,
                    "request_index": request,
                    "request_id": f"{request:064x}",
                    "stage_order": ["control_before", "diagnostic", "control_after"],
                    "diagnostic": {
                        "fresh_process": True, "resident_arena_bytes": 0,
                        "cache_namespace": f"diagnostic-{request}", "exit_code": 0,
                    },
                    "control_before": {
                        "fresh_process": True,
                        "cache_namespace": f"before-{request}",
                        "continuation_sha256": "2" * 64,
                        "token_ids_sha256": "3" * 64, "exit_code": 0,
                    },
                    "control_after": {
                        "fresh_process": True,
                        "cache_namespace": f"after-{request}",
                        "continuation_sha256": "2" * 64,
                        "token_ids_sha256": "3" * 64, "exit_code": 0,
                    },
                    "failure_injection": failure,
                }
                runtime_logs.append({"two_control": control})
            (root / "raw.json").write_text(
                json.dumps({"runtime_logs": runtime_logs}) + "\n", encoding="utf-8",
            )
            summary["two_control"] = {
                "requests": 20,
                "all_isolated": True,
                "all_continuations_equal": True,
                "failure_injection_stages": failure["stages"],
            }
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            MODULE.validate_completed_result(
                root, bootstrap_resamples=1000, expected_summary=summary,
            )
            changed = json.loads(summary_path.read_text(encoding="utf-8"))
            changed["decision"]["verdict"] = "STOP_PROBE"
            summary_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.validate_completed_result(
                    root, bootstrap_resamples=1000, expected_summary=summary,
                )


if __name__ == "__main__":
    unittest.main()
