#!/usr/bin/env python3
"""Acceptance tests for lossless-label, measured-loss feature compaction."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/77_compact_glm_union_trace.py"
SPEC = importlib.util.spec_from_file_location("glm_union_trace_compact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCORE_SCRIPT = ROOT / "scripts/75_glm_union_trace_score.py"
SCORE_SPEC = importlib.util.spec_from_file_location("glm_union_trace_score_for_compact", SCORE_SCRIPT)
assert SCORE_SPEC and SCORE_SPEC.loader
SCORE_MODULE = importlib.util.module_from_spec(SCORE_SPEC)
SCORE_SPEC.loader.exec_module(SCORE_MODULE)


class CompactArrayTests(unittest.TestCase):
    def fixture(self):
        hidden = np.array([
            [-7.0, -3.5, -1.0, 0.0, 1.0, 3.5, 6.0, 7.0],
            [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0],
        ], dtype=np.float32)
        logits = np.array([
            [0.1, 0.7, -0.2, 1.0, 0.3, 0.9, -0.4, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        selected = np.array([[7, 6], [6, 7]], dtype=np.int32)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return hidden, logits, probabilities.astype(np.float32), bias, selected

    def test_preserves_labels_and_stable_effective_score_topk(self) -> None:
        hidden, logits, probabilities, bias, selected = self.fixture()
        compact, metrics = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=4, router_probs=probabilities,
        )
        np.testing.assert_array_equal(compact["selected_ids"], selected.astype(np.uint8))
        np.testing.assert_array_equal(compact["top_ids"][0], [7, 6, 3, 5])
        np.testing.assert_array_equal(compact["top_ids"][1], [6, 7, 0, 1])
        gathered = np.take_along_axis(logits, compact["top_ids"], axis=1)
        np.testing.assert_array_equal(compact["top_logits"], gathered.astype(np.float16))
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["hidden_values"], 16)

    def test_int4_round_trip_obeys_per_row_half_step_bound(self) -> None:
        hidden, logits, probabilities, bias, selected = self.fixture()
        compact, metrics = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=4, hidden_group_size=4,
            router_probs=probabilities,
        )
        self.assertEqual(compact["hidden_scale"].shape, (2, 2))
        self.assertEqual(metrics["hidden_group_size"], 4)
        restored = MODULE.unpack_hidden_int4(
            compact["hidden_q4"], compact["hidden_scale"], hidden.shape[1],
        )
        error = np.abs(restored - hidden)
        bound = np.repeat(compact["hidden_scale"].astype(np.float32), 4, axis=1) / 2 + 1e-6
        self.assertTrue(np.all(error <= bound[:, :hidden.shape[1]]))
        self.assertAlmostEqual(metrics["hidden_max_abs_error"], float(error.max()), places=6)
        self.assertGreaterEqual(metrics["hidden_nrmse"], 0.0)

    def test_captured_probabilities_preserve_fp32_tie_order(self) -> None:
        hidden = np.array([[0.0, 1.0]], dtype=np.float32)
        logits = np.zeros((1, 4), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        probabilities = np.full((1, 4), 0.5, dtype=np.float32)
        probabilities[0, 1] = np.nextafter(np.float32(0.5), np.float32(1.0))
        selected = np.array([[1, 0]], dtype=np.int32)
        compact, _ = MODULE.compact_arrays(
            hidden, logits, bias, selected, top_k=2,
            router_probs=probabilities,
        )
        np.testing.assert_array_equal(compact["top_ids"], [[1, 0]])

    def test_rejects_nonfinite_and_malformed_inputs(self) -> None:
        hidden, logits, probabilities, bias, selected = self.fixture()
        for mutation in ("nan", "rows", "selected", "top_k"):
            with self.subTest(mutation=mutation):
                h, g, b, s = hidden.copy(), logits.copy(), bias.copy(), selected.copy()
                top_k = 4
                if mutation == "nan":
                    h[0, 0] = np.nan
                elif mutation == "rows":
                    g = g[:1]
                elif mutation == "selected":
                    s[0, 0] = 9
                else:
                    top_k = 9
                with self.assertRaises(ValueError):
                    MODULE.compact_arrays(
                        h, g, b, s, top_k=top_k, router_probs=probabilities,
                    )

    def test_captured_probabilities_are_mandatory(self) -> None:
        hidden, logits, _, bias, selected = self.fixture()
        with self.assertRaises(ValueError):
            MODULE.compact_arrays(hidden, logits, bias, selected, top_k=4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QualifiedBundleTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        trace = source / "on" / "trace"
        trace.mkdir(parents=True)
        for pos, rows in ((0, 2), (2, 1)):
            hidden = (
                (np.arange(rows * SCORE_MODULE.N_EMBD, dtype=np.float32).reshape(rows, -1) % 31)
                / np.float32(100.0) + np.float32(pos) / np.float32(1000.0)
            )
            base_logits = np.linspace(-2.0, 2.0, SCORE_MODULE.N_EXPERT, dtype=np.float32)
            logits = np.stack([
                np.roll(base_logits, pos + row) for row in range(rows)
            ]).astype(np.float32)
            probabilities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
            bias = np.zeros(SCORE_MODULE.N_EXPERT, dtype=np.float32)
            selected = np.argsort(-(probabilities + bias), axis=1, kind="stable")[
                :, :SCORE_MODULE.N_EXPERT_USED
            ].astype(np.int32)
            payloads = {
                "ffn_norm": ("f32", hidden), "router_logits": ("f32", logits),
                "router_probs": ("f32", probabilities),
                "router_selected": ("i32", selected), "router_bias": ("f32", bias),
            }
            for kind, (ext, values) in payloads.items():
                path = trace / f"request_glm_indexed_{kind}-4_pos{pos}.{ext}"
                path.write_bytes(values.tobytes())
        server_log = source / "on" / "server.log"
        server_log.write_text(
            "GLM_UNION_TRACE_OK path=full_indexed_batch_ffn layer=4 pos=0 rows=2\n"
            "GLM_UNION_TRACE_OK path=full_indexed_batch_ffn layer=4 pos=2 rows=1\n"
        )
        trace_score = SCORE_MODULE.score_trace(
            trace, server_log, max_bytes=10**7,
            expected_layers={4}, expected_chunks=[(0, 2), (2, 1)],
        )
        self.assertEqual(trace_score["verdict"], "PASS")
        artifacts = trace_score["artifacts"]
        arm = {
            "mode": "on", "binary_sha256": "a" * 64, "model_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64, "fixture_sha256": "d" * 64,
            "configuration_sha256": "e" * 64, "environment_sha256": "1" * 64,
            "prompt_tokens": 3,
            "full_indexed_chunks": [[0, 2], [2, 1]], "trace_files": 10,
            "trace_bytes": sum(item["bytes"] for item in artifacts),
            "response_signature": {
                "request_sha256": "d" * 64, "completion_tokens": 128,
                "token_ids": [1, 2], "generated_content_sha256": "f" * 64,
            },
            "server_log_sha256": sha256(server_log),
        }
        result_path = source / "on" / "result.json"
        result_path.write_text("{}\n")
        arm["result_sha256"] = sha256(result_path)
        arm_path = source / "on" / "arm.json"
        arm_path.write_text(json.dumps(arm, sort_keys=True, indent=2) + "\n")
        off_dir = source / "off"
        off_dir.mkdir()
        off_server_log = off_dir / "server.log"
        off_server_log.write_text("no trace\n")
        off_result = off_dir / "result.json"
        off_result.write_text("{}\n")
        off_arm = dict(arm)
        off_arm.update({
            "mode": "off", "environment_sha256": "2" * 64, "trace_files": 0,
            "trace_bytes": 0, "server_log_sha256": sha256(off_server_log),
            "result_sha256": sha256(off_result),
        })
        off_arm_path = off_dir / "arm.json"
        off_arm_path.write_text(json.dumps(off_arm, sort_keys=True, indent=2) + "\n")
        containment = {
            "clean": True, "crash_directory": "/tmp/test", "kernel_sha256": "3" * 64,
            "main_sha256": "4" * 64, "samples_sha256": "5" * 64,
        }
        off_containment = source / "off.containment.json"
        on_containment = source / "on.containment.json"
        off_containment.write_text(json.dumps(containment, sort_keys=True) + "\n")
        on_containment.write_text(json.dumps(containment, sort_keys=True) + "\n")
        summary = {
            "schema_version": 1, "scope": "short_single_indexed_batch_only",
            "high_row_2048_status": "OPEN", "verdict": "PASS",
            "candidate_hash": "1" * 40,
            "engine_commit": "2" * 40, "binary_sha256": "a" * 64,
            "model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "seed": 7, "context_level": 4096,
            "off_arm_sha256": sha256(off_arm_path), "on_arm_sha256": sha256(arm_path),
            "off_containment_sha256": sha256(off_containment),
            "on_containment_sha256": sha256(on_containment),
            "max_trace_bytes": 10**7, "trace_score": trace_score,
            "checks": {
                "arm_modes": True, "byte_and_token_identity": True,
                "containment_clean": True, "frozen_identity": True,
                "matched_indexed_chunks": True, "off_emitted_no_trace": True,
                "on_emitted_trace": True, "prompt_tokens_and_exact_coverage": True,
                "trace_score_passed": True,
            },
        }
        summary_path = source / "summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        receipt = {
            "schema_version": 1, "classification": "PASS", "candidate_hash": "1" * 40,
            "engine_commit": "2" * 40, "scope": "short_single_indexed_batch_only",
            "high_row_2048_status": "OPEN",
            "summary_sha256": sha256(summary_path), "on_arm_sha256": sha256(arm_path),
            "off_arm_sha256": sha256(off_arm_path),
            "off_result_sha256": sha256(off_result), "on_result_sha256": sha256(result_path),
            "off_server_log_sha256": sha256(off_server_log),
            "on_server_log_sha256": sha256(server_log),
            "observed": {
                "context_level": 4096, "prompt_tokens": 3,
                "completion_tokens_per_arm": 128,
                "full_indexed_chunks": [[0, 2], [2, 1]],
                "byte_and_token_identity": True, "containment_clean": True,
                "off_trace_files": 0, "on_trace_files": 10,
                "on_trace_bytes": arm["trace_bytes"], "trace_events": 2,
                "trace_score_verdict": "PASS",
            },
            "pre_runtime_authorization_review": {
                "round": 1, "gap_reviewer_score": 100,
                "adversarial_reviewer_score": 100, "critical": [], "high": [],
            },
            "post_runtime_review": {
                "round": 2, "gap_reviewer_score": 100,
                "adversarial_reviewer_score": 100, "critical": [], "high": [],
            },
            "conclusion": "test fixture",
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return source, receipt_path

    def make_corpus_bundle(
        self, root: Path, *, layers: range = range(3, 78), minimum_events: int = 300,
    ) -> tuple[Path, Path]:
        source = root / "corpus-source"
        trace = source / "on" / "trace"
        trace.mkdir(parents=True)
        request_hashes = {1: "d" * 64, 2: "e" * 64}
        for request_id in (1, 2):
            rows = 2
            hidden = (
                np.arange(rows * SCORE_MODULE.N_EMBD, dtype=np.float32).reshape(rows, -1)
                / np.float32(1000.0) + np.float32(request_id)
            )
            base_logits = np.linspace(-2.0, 2.0, SCORE_MODULE.N_EXPERT, dtype=np.float32)
            logits = np.stack([
                np.roll(base_logits, request_id + row) for row in range(rows)
            ]).astype(np.float32)
            probabilities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
            bias = np.zeros(SCORE_MODULE.N_EXPERT, dtype=np.float32)
            selected = np.argsort(-probabilities, axis=1, kind="stable")[
                :, :SCORE_MODULE.N_EXPERT_USED
            ].astype(np.int32)
            payloads = {
                "ffn_norm": ("f32", hidden), "router_logits": ("f32", logits),
                "router_probs": ("f32", probabilities),
                "router_selected": ("i32", selected), "router_bias": ("f32", bias),
            }
            for layer in layers:
                for kind, (ext, values) in payloads.items():
                    path = trace / (
                        f"request_r{request_id:08d}_glm_indexed_{kind}-{layer}_pos0.{ext}"
                    )
                    path.write_bytes(values.tobytes())
        server_log = source / "on" / "server.log"
        server_log.write_text("".join(
            f"GLM_UNION_TRACE_OK path=full_indexed_batch_ffn request={request_id} "
            f"layer={layer} pos=0 rows=2\n"
            for request_id in (1, 2) for layer in layers
        ))
        trace_score = SCORE_MODULE.score_trace(
            trace, server_log, max_bytes=10**8, expected_layers=set(layers), expected_chunks=[],
            expected_requests={1: [(0, 2)], 2: [(0, 2)]},
        )
        self.assertEqual(trace_score["verdict"], "PASS")

        def signature(request_id: int) -> dict[str, object]:
            return {
                "request_sha256": request_hashes[request_id],
                "completion_tokens": 128,
                "token_ids": [request_id, 9],
                "generated_reasoning_sha256": str(request_id) * 64,
                "generated_reasoning_bytes": 2,
                "generated_content_sha256": "f" * 64,
                "generated_content_bytes": 0,
            }

        result_paths: dict[tuple[str, int], Path] = {}
        for mode in ("off", "on"):
            directory = source / mode
            directory.mkdir(exist_ok=True)
            for request_id in (1, 2):
                path = directory / f"result-{request_id}.json"
                path.write_text(json.dumps({"mode": mode, "request": request_id}) + "\n")
                result_paths[(mode, request_id)] = path
        corpus_requests = {
            mode: [
                {
                    "request_id": request_id,
                    "seed": 7 + request_id - 1,
                    "prompt_tokens": 2,
                    "full_indexed_chunks": [[0, 2]],
                    "response_signature": signature(request_id),
                    "result_sha256": sha256(result_paths[(mode, request_id)]),
                }
                for request_id in (1, 2)
            ]
            for mode in ("off", "on")
        }
        fixture_digest = hashlib.sha256(
            b"".join(bytes.fromhex(request_hashes[index]) for index in (1, 2))
        ).hexdigest()
        arms: dict[str, dict[str, object]] = {}
        arm_paths: dict[str, Path] = {}
        for mode in ("off", "on"):
            mode_log = source / mode / "server.log"
            if mode == "on":
                mode_log = server_log
            else:
                mode_log.write_text("no trace\n")
            result_digest = hashlib.sha256(b"".join(
                bytes.fromhex(sha256(result_paths[(mode, request_id)]))
                for request_id in (1, 2)
            )).hexdigest()
            arm = {
                "mode": mode, "binary_sha256": "a" * 64,
                "model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
                "fixture_sha256": fixture_digest, "configuration_sha256": "9" * 64,
                "environment_sha256": ("1" if mode == "on" else "2") * 64,
                "response_signature": [signature(1), signature(2)],
                "prompt_tokens": 2, "full_indexed_chunks": [[0, 2]],
                "trace_files": len(layers) * 10 if mode == "on" else 0,
                "trace_bytes": trace_score["total_bytes"] if mode == "on" else 0,
                "result_sha256": result_digest, "server_log_sha256": sha256(mode_log),
                "corpus_requests": corpus_requests[mode],
                "expert_cache_budget": "32GB", "cuda_expert_cache_gb": "56",
                "cuda_cache_runtime": {"slots": 5754, "arena_gib": 52.15},
            }
            arms[mode] = arm
            arm_path = source / mode / "arm.json"
            arm_path.write_text(json.dumps(arm, sort_keys=True, indent=2) + "\n")
            arm_paths[mode] = arm_path
        containment = {
            "clean": True, "crash_directory": "/tmp/test", "kernel_sha256": "3" * 64,
            "main_sha256": "4" * 64, "samples_sha256": "5" * 64,
        }
        containment_paths = {}
        for mode in ("off", "on"):
            path = source / f"{mode}.containment.json"
            path.write_text(json.dumps(containment, sort_keys=True) + "\n")
            containment_paths[mode] = path
        checks = {
            "arm_modes": True, "byte_and_token_identity": True,
            "containment_clean": True, "corpus_cuda_cache": True,
            "corpus_event_floor": True, "corpus_request_scope": True,
            "frozen_identity": True, "matched_indexed_chunks": True,
            "off_emitted_no_trace": True, "on_emitted_trace": True,
            "prompt_tokens_and_exact_coverage": True, "trace_score_passed": True,
        }
        summary = {
            "schema_version": 1, "scope": "multi_request_all_routed_layer_corpus_smoke",
            "high_row_2048_status": "OPEN", "candidate_hash": "1" * 40,
            "engine_commit": "2" * 40, "binary_sha256": "a" * 64,
            "model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "seed": 7, "context_level": 2, "max_trace_bytes": 10**8,
            "minimum_token_layer_events": minimum_events,
            "off_arm_sha256": sha256(arm_paths["off"]),
            "on_arm_sha256": sha256(arm_paths["on"]),
            "off_containment_sha256": sha256(containment_paths["off"]),
            "on_containment_sha256": sha256(containment_paths["on"]),
            "trace_score": trace_score, "checks": checks, "verdict": "PASS",
        }
        summary_path = source / "summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        receipt = {
            "schema_version": 1, "candidate_hash": "1" * 40,
            "engine_commit": "2" * 40, "classification": "PASS",
            "scope": "multi_request_all_routed_layer_corpus_smoke",
            "high_row_2048_status": "OPEN", "summary_sha256": sha256(summary_path),
            "off_arm_sha256": sha256(arm_paths["off"]),
            "on_arm_sha256": sha256(arm_paths["on"]),
            "off_result_1_sha256": sha256(result_paths[("off", 1)]),
            "off_result_2_sha256": sha256(result_paths[("off", 2)]),
            "on_result_1_sha256": sha256(result_paths[("on", 1)]),
            "on_result_2_sha256": sha256(result_paths[("on", 2)]),
            "off_server_log_sha256": sha256(source / "off/server.log"),
            "on_server_log_sha256": sha256(server_log),
            "off_containment_sha256": sha256(containment_paths["off"]),
            "on_containment_sha256": sha256(containment_paths["on"]),
            "observed": {
                "context_level": 2, "requests": 2,
                "prompt_tokens_per_request": [2, 2],
                "completion_tokens_per_request": [128, 128],
                "full_indexed_chunks_per_request": [[[0, 2]], [[0, 2]]],
                "distinct_request_fixtures": 2, "byte_and_token_identity": True,
                "containment_clean": True, "streaming_cache_budget": "32GB",
                "cuda_cache_environment_gb": 56, "cuda_cache_slots": 5754,
                "cuda_cache_arena_gib": 52.15, "off_trace_files": 0,
                "on_trace_files": len(layers) * 10,
                "on_trace_bytes": trace_score["total_bytes"],
                "trace_events": len(layers) * 2,
                "token_layer_events": len(layers) * 4,
                "routed_layer_first": layers.start, "routed_layer_last": layers.stop - 1,
                "minimum_available_memory_gib": {"off": 32.0, "on": 32.0},
                "maximum_cgroup_memory_bytes": {"off": 1, "on": 1},
                "maximum_cgroup_swap_bytes": 0, "kernel_oom_or_xid": False,
                "trace_score_verdict": "PASS",
            },
            "pre_runtime_authorization_review": {
                "round": 1, "gap_reviewer_score": 100,
                "adversarial_reviewer_score": 100, "critical": [], "high": [],
            },
            "post_runtime_review": {
                "round": 2, "gap_reviewer_score": 100,
                "adversarial_reviewer_score": 100, "critical": [], "high": [],
            },
            "retained_directory": str(source), "conclusion": "test corpus fixture",
        }
        receipt_path = root / "corpus-receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return source, receipt_path

    def test_accepts_exact_qualified_bundle_and_emits_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_bundle(Path(directory))
            validated = MODULE.validate_source_bundle(
                source, receipt, repository_root=Path(directory), require_tracked_receipt=False,
                minimum_prompt_tokens=1,
            )
            self.assertEqual(validated["layers"], [4])
            self.assertEqual(validated["chunks"], [(0, 2), (2, 1)])
            for field in (
                "source_receipt_sha256", "source_summary_sha256", "source_arm_sha256",
                "candidate_hash", "engine_commit", "binary_sha256", "model_sha256",
                "tokenizer_sha256", "fixture_sha256", "configuration_sha256",
                "request_id", "seed", "scorer_sha256", "repository_head",
            ):
                self.assertIn(field, validated["lineage"])

    def test_accepts_corpus_receipt_and_selects_one_request_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_corpus_bundle(Path(directory))
            with mock.patch.object(MODULE, "CORPUS_MIN_TOKEN_LAYER_EVENTS", 300, create=True):
                validated = MODULE.validate_source_bundle(
                    source, receipt, repository_root=Path(directory),
                    require_tracked_receipt=False, minimum_prompt_tokens=1,
                    request_index=2,
                )
            self.assertEqual(validated["layers"], list(range(3, 78)))
            self.assertEqual(validated["chunks"], [(0, 2)])
            self.assertEqual(validated["lineage"]["request_index"], 2)
            self.assertEqual(validated["lineage"]["request_id"], "e" * 64)
            self.assertEqual(len(validated["files"]), 375)

    def test_corpus_source_requires_explicit_valid_request_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_corpus_bundle(Path(directory))
            for request_index in (None, 0, 3):
                with self.subTest(request_index=request_index), self.assertRaises(ValueError):
                    with mock.patch.object(MODULE, "CORPUS_MIN_TOKEN_LAYER_EVENTS", 300, create=True):
                        MODULE.validate_source_bundle(
                            source, receipt, repository_root=Path(directory),
                            require_tracked_receipt=False, minimum_prompt_tokens=1,
                            request_index=request_index,
                        )

    def test_corpus_rejects_incomplete_layer_range_and_lowered_event_floor(self) -> None:
        for name, layers, minimum_events in (
            ("missing_first_layer", range(4, 78), 296),
            ("missing_last_layer", range(3, 77), 296),
            ("lowered_floor", range(3, 78), 299),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source, receipt = self.make_corpus_bundle(
                    Path(directory), layers=layers, minimum_events=minimum_events,
                )
                with mock.patch.object(MODULE, "CORPUS_MIN_TOKEN_LAYER_EVENTS", 300, create=True):
                    with self.assertRaises(ValueError):
                        MODULE.validate_source_bundle(
                            source, receipt, repository_root=Path(directory),
                            require_tracked_receipt=False, minimum_prompt_tokens=1,
                            request_index=1,
                        )

    def test_corpus_rejects_boolean_cache_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt_path = self.make_corpus_bundle(Path(directory))
            arms = {}
            for mode in ("off", "on"):
                arm_path = source / mode / "arm.json"
                arm = json.loads(arm_path.read_text())
                arm["cuda_cache_runtime"]["arena_gib"] = True
                arm_path.write_text(json.dumps(arm, sort_keys=True, indent=2) + "\n")
                arms[mode] = arm_path
            summary_path = source / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["off_arm_sha256"] = sha256(arms["off"])
            summary["on_arm_sha256"] = sha256(arms["on"])
            summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
            receipt = json.loads(receipt_path.read_text())
            receipt["off_arm_sha256"] = sha256(arms["off"])
            receipt["on_arm_sha256"] = sha256(arms["on"])
            receipt["summary_sha256"] = sha256(summary_path)
            receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            with mock.patch.object(MODULE, "CORPUS_MIN_TOKEN_LAYER_EVENTS", 300, create=True):
                with self.assertRaises(ValueError):
                    MODULE.validate_source_bundle(
                        source, receipt_path, repository_root=Path(directory),
                        require_tracked_receipt=False, minimum_prompt_tokens=1, request_index=1,
                    )

    def test_corpus_rejects_mutated_safety_observations(self) -> None:
        mutations = {
            "kernel_fault": lambda value: value.__setitem__("kernel_oom_or_xid", True),
            "low_available": lambda value: value.__setitem__(
                "minimum_available_memory_gib", {"off": 9.0, "on": 32.0},
            ),
            "swap": lambda value: value.__setitem__("maximum_cgroup_swap_bytes", 1),
            "cache_slots": lambda value: value.__setitem__("cuda_cache_slots", 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source, receipt_path = self.make_corpus_bundle(Path(directory))
                receipt = json.loads(receipt_path.read_text())
                mutate(receipt["observed"])
                receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
                with mock.patch.object(MODULE, "CORPUS_MIN_TOKEN_LAYER_EVENTS", 300, create=True):
                    with self.assertRaises(ValueError):
                        MODULE.validate_source_bundle(
                            source, receipt_path, repository_root=Path(directory),
                            require_tracked_receipt=False, minimum_prompt_tokens=1,
                            request_index=1,
                        )

    def test_cli_exposes_request_scoped_corpus_shards(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('result.add_argument("--request-index", type=int)', source)
        self.assertIn("request_index=args.request_index", source)

    def test_consumes_the_exact_tracked_receipt_snapshot(self) -> None:
        """A path replacement after the HEAD comparison cannot change authority."""
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_bundle(Path(directory))
            committed = receipt.read_bytes()

            def verified_snapshot(path: Path, repository_root: Path) -> bytes:
                self.assertEqual(path, receipt.resolve())
                receipt.write_text('{"substituted":true}\n')
                return committed

            with mock.patch.object(
                MODULE, "_require_tracked_receipt", side_effect=verified_snapshot,
            ):
                validated = MODULE.validate_source_bundle(
                    source, receipt, repository_root=Path(directory),
                    require_tracked_receipt=True, require_tracked_scorer=False,
                    minimum_prompt_tokens=1,
                )
            self.assertEqual(
                validated["lineage"]["source_receipt_sha256"],
                hashlib.sha256(committed).hexdigest(),
            )

    def test_rejects_ambiguous_unknown_symlink_and_lineage_mutations(self) -> None:
        for mutation in (
            "duplicate", "unknown", "symlink", "extension", "bias", "receipt", "chunks",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                source, receipt = self.make_bundle(Path(directory))
                trace = source / "on" / "trace"
                if mutation == "duplicate":
                    (trace / "evil_glm_indexed_router_probs-4_pos0.f32").write_bytes(b"duplicate")
                elif mutation == "unknown":
                    (trace / "unknown.bin").write_bytes(b"unknown")
                elif mutation == "symlink":
                    target = trace / "request_glm_indexed_router_probs-4_pos0.f32"
                    target.unlink()
                    target.symlink_to("request_glm_indexed_router_probs-4_pos2.f32")
                elif mutation == "extension":
                    old = trace / "request_glm_indexed_router_selected-4_pos0.i32"
                    new = trace / "request_glm_indexed_router_selected-4_pos0.f32"
                    old.rename(new)
                    summary_path = source / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    for artifact in summary["trace_score"]["artifacts"]:
                        if artifact["name"] == old.name:
                            artifact["name"] = new.name
                    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
                    data = json.loads(receipt.read_text())
                    data["summary_sha256"] = sha256(summary_path)
                    receipt.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
                elif mutation == "bias":
                    (trace / "request_glm_indexed_router_bias-4_pos2.f32").write_bytes(b"different")
                elif mutation == "receipt":
                    data = json.loads(receipt.read_text())
                    data["candidate_hash"] = "9" * 40
                    receipt.write_text(json.dumps(data))
                else:
                    data = json.loads((source / "on/arm.json").read_text())
                    data["full_indexed_chunks"] = [[2, 1], [0, 2]]
                    (source / "on/arm.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    MODULE.validate_source_bundle(
                        source, receipt, repository_root=Path(directory),
                        require_tracked_receipt=False,
                        minimum_prompt_tokens=1,
                    )

    def test_fixed_scorer_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_bundle(Path(directory))
            with mock.patch.object(MODULE.TRACE_SCORER, "score_trace", return_value={
                "verdict": "FAIL", "checks": {"mutation": False},
            }):
                with self.assertRaises(ValueError):
                    MODULE.validate_source_bundle(
                        source, receipt, repository_root=Path(directory),
                        require_tracked_receipt=False,
                        minimum_prompt_tokens=1,
                    )

    def test_consumed_tensor_bytes_remain_bound_after_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_bundle(Path(directory))
            validated = MODULE.validate_source_bundle(
                source, receipt, repository_root=Path(directory),
                require_tracked_receipt=False, minimum_prompt_tokens=1,
            )
            key = (4, 0, "router_probs")
            path = validated["files"][key]
            payload = bytearray(path.read_bytes())
            payload[0] ^= 1
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(payload)
            replacement.replace(path)
            with self.assertRaises(ValueError):
                MODULE._read_bound_array(
                    path, "<f4", (2, SCORE_MODULE.N_EXPERT),
                    validated["artifacts"][path.name],
                )

    def test_trusted_receipt_must_equal_tracked_head_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            receipt = repository / "results/glm52-gates/receipt.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}\n")
            for command in (
                ["git", "init", "-q"],
                ["git", "config", "user.email", "test@example.invalid"],
                ["git", "config", "user.name", "Test"],
                ["git", "add", str(receipt.relative_to(repository))],
                ["git", "commit", "-qm", "receipt"],
            ):
                subprocess.run(command, cwd=repository, check=True)
            MODULE._require_tracked_receipt(receipt, repository)
            receipt.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                MODULE._require_tracked_receipt(receipt, repository)

    def test_atomic_bundle_publication_refuses_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "published"
            arrays = {"value": np.array([1, 2], dtype=np.uint8)}
            MODULE.publish_bundle(destination, arrays, {"schema_version": 1})
            self.assertTrue((destination / "records.npz").is_file())
            self.assertTrue((destination / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                MODULE.publish_bundle(destination, arrays, {"schema_version": 1})
            dangling = Path(directory) / "dangling"
            dangling.symlink_to(Path(directory) / "redirected")
            with self.assertRaises(FileExistsError):
                MODULE.publish_bundle(dangling, arrays, {"schema_version": 1})
            self.assertFalse((Path(directory) / "redirected").exists())

    def test_atomic_bundle_publication_loses_no_replace_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "published"
            arrays = {"value": np.array([1, 2], dtype=np.uint8)}
            original_savez = MODULE.np.savez

            def racing_savez(*args, **kwargs):
                destination.mkdir()
                return original_savez(*args, **kwargs)

            with mock.patch.object(MODULE.np, "savez", side_effect=racing_savez):
                with self.assertRaises(FileExistsError):
                    MODULE.publish_bundle(destination, arrays, {"schema_version": 1})
            self.assertTrue(destination.is_dir())


if __name__ == "__main__":
    unittest.main()
