#!/usr/bin/env python3
"""Acceptance tests for lossless-label, measured-loss feature compaction."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/77_compact_glm_union_trace.py"
SPEC = importlib.util.spec_from_file_location("glm_union_trace_compact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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
        artifacts = []
        for pos, rows in ((0, 2), (2, 1)):
            for kind, ext in (
                ("ffn_norm", "f32"), ("router_logits", "f32"),
                ("router_probs", "f32"), ("router_selected", "i32"),
                ("router_bias", "f32"),
            ):
                path = trace / f"request_glm_indexed_{kind}-4_pos{pos}.{ext}"
                payload = b"same-bias" if kind == "router_bias" else f"{kind}:{pos}:{rows}".encode()
                path.write_bytes(payload)
                artifacts.append({"name": path.name, "bytes": len(payload), "sha256": sha256(path)})
        server_log = source / "on" / "server.log"
        server_log.write_text("qualified\n")
        arm = {
            "mode": "on", "binary_sha256": "a" * 64, "model_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64, "fixture_sha256": "d" * 64,
            "configuration_sha256": "e" * 64, "prompt_tokens": 3,
            "full_indexed_chunks": [[0, 2], [2, 1]], "trace_files": 10,
            "trace_bytes": sum(item["bytes"] for item in artifacts),
            "response_signature": {"request_sha256": "f" * 64},
            "server_log_sha256": sha256(server_log),
        }
        arm_path = source / "on" / "arm.json"
        arm_path.write_text(json.dumps(arm, sort_keys=True, indent=2) + "\n")
        summary = {
            "verdict": "PASS", "candidate_hash": "1" * 40,
            "engine_commit": "2" * 40, "binary_sha256": "a" * 64,
            "model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "seed": 7, "on_arm_sha256": sha256(arm_path),
            "trace_score": {
                "verdict": "PASS", "checks": {"all": True}, "events": 2,
                "total_rows": 3, "total_bytes": arm["trace_bytes"], "artifacts": artifacts,
            },
        }
        summary_path = source / "summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        receipt = {
            "classification": "PASS", "candidate_hash": "1" * 40,
            "summary_sha256": sha256(summary_path), "on_arm_sha256": sha256(arm_path),
            "on_server_log_sha256": sha256(server_log),
            "observed": {"full_indexed_chunks": [[0, 2], [2, 1]]},
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return source, receipt_path

    def test_accepts_exact_qualified_bundle_and_emits_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt = self.make_bundle(Path(directory))
            validated = MODULE.validate_source_bundle(source, receipt)
            self.assertEqual(validated["layers"], [4])
            self.assertEqual(validated["chunks"], [(0, 2), (2, 1)])
            for field in (
                "source_receipt_sha256", "source_summary_sha256", "source_arm_sha256",
                "candidate_hash", "engine_commit", "binary_sha256", "model_sha256",
                "tokenizer_sha256", "fixture_sha256", "configuration_sha256",
                "request_id", "seed",
            ):
                self.assertIn(field, validated["lineage"])

    def test_rejects_ambiguous_unknown_symlink_and_lineage_mutations(self) -> None:
        for mutation in ("duplicate", "unknown", "symlink", "bias", "receipt", "chunks"):
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
                    MODULE.validate_source_bundle(source, receipt)

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


if __name__ == "__main__":
    unittest.main()
