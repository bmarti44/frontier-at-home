#!/usr/bin/env python3
"""Mutation tests for the R0b union-trace scorer."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/75_glm_union_trace_score.py"
SPEC = importlib.util.spec_from_file_location("glm_union_trace_score", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

N_EMBD = 7168
N_EXPERT = 256
N_USED = 8


class UnionTraceScoreTests(unittest.TestCase):
    def make_attempt(self, root: Path, *, rows: int = 2) -> tuple[Path, Path]:
        trace = root / "trace"
        trace.mkdir()
        prefix = trace / "request-a"
        stem = f"{prefix}_"
        norm = [0.25] * (rows * N_EMBD)
        norm[-1] = 0.5
        (Path(stem + "glm_indexed_ffn_norm-4_pos0.f32")).write_bytes(
            struct.pack(f"<{rows * N_EMBD}f", *norm)
        )
        logits = [-(abs(expert - row) / 10.0)
                  for row in range(rows) for expert in range(N_EXPERT)]
        (Path(stem + "glm_indexed_router_logits-4_pos0.f32")).write_bytes(
            struct.pack(f"<{rows * N_EXPERT}f", *logits)
        )
        probs = [1.0 / (1.0 + math.exp(-value)) for value in logits]
        (Path(stem + "glm_indexed_router_probs-4_pos0.f32")).write_bytes(
            struct.pack(f"<{rows * N_EXPERT}f", *probs)
        )
        bias = [0.0] * N_EXPERT
        (Path(stem + "glm_indexed_router_bias-4_pos0.f32")).write_bytes(
            struct.pack(f"<{N_EXPERT}f", *bias)
        )
        selected = []
        for row in range(rows):
            scores = [probs[row * N_EXPERT + expert] for expert in range(N_EXPERT)]
            selected.extend(sorted(range(N_EXPERT), key=lambda expert: (-scores[expert], expert))[:N_USED])
        (Path(stem + "glm_indexed_router_selected-4_pos0.i32")).write_bytes(
            struct.pack(f"<{len(selected)}i", *selected)
        )
        log = root / "server.log"
        log.write_text(
            "GLM_UNION_TRACE_OK path=full_indexed_batch_ffn layer=4 pos=0 rows=2\n",
            encoding="utf-8",
        )
        return trace, log

    def score(self, trace: Path, log: Path, *, max_bytes: int = 1_000_000):
        return MODULE.score_trace(
            trace, log, max_bytes=max_bytes,
            expected_layers={4}, expected_chunks=[(0, 2)],
        )

    def test_accepts_one_complete_finite_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            result = self.score(trace, log)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["events"], 1)
        self.assertEqual(len(result["artifacts"]), 5)

    def test_rejects_missing_truncated_and_trailing_files(self) -> None:
        for mutation in ("missing", "truncated", "trailing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                target = next(trace.glob("*router_logits*"))
                if mutation == "missing":
                    target.unlink()
                elif mutation == "truncated":
                    target.write_bytes(target.read_bytes()[:-4])
                else:
                    target.write_bytes(target.read_bytes() + b"xxxx")
                self.assertEqual(
                    self.score(trace, log)["verdict"],
                    "FAIL",
                )

    def test_rejects_duplicate_log_key_and_trace_error(self) -> None:
        for extra in (
            "GLM_UNION_TRACE_OK path=full_indexed_batch_ffn layer=4 pos=0 rows=2\n",
            "GLM_UNION_TRACE_ERROR stage=capture layer=4 pos=0 rows=2\n",
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                log.write_text(log.read_text() + extra, encoding="utf-8")
                self.assertEqual(
                    self.score(trace, log)["verdict"],
                    "FAIL",
                )

    def test_rejects_nonfinite_float_and_bad_selected_ids(self) -> None:
        for mutation in ("nan", "range", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                if mutation == "nan":
                    target = next(trace.glob("*router_logits*"))
                    data = bytearray(target.read_bytes())
                    data[:4] = struct.pack("<f", math.nan)
                else:
                    target = next(trace.glob("*router_selected*"))
                    data = bytearray(target.read_bytes())
                    data[:4] = struct.pack("<i", 256 if mutation == "range" else 1)
                target.write_bytes(data)
                self.assertEqual(
                    self.score(trace, log)["verdict"],
                    "FAIL",
                )

    def test_rejects_unknown_partial_symlink_and_byte_budget(self) -> None:
        for mutation in ("unknown", "partial", "symlink", "budget"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                max_bytes = 1_000_000
                if mutation == "unknown":
                    (trace / "unexpected").write_text("x")
                elif mutation == "partial":
                    (trace / "request-a.tmp.123").write_text("x")
                elif mutation == "symlink":
                    target = next(trace.glob("*ffn_norm*"))
                    saved = trace / "saved"
                    target.rename(saved)
                    target.symlink_to(saved)
                else:
                    max_bytes = 1
                self.assertEqual(
                    self.score(trace, log, max_bytes=max_bytes)["verdict"],
                    "FAIL",
                )

    def test_rejects_swapped_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            renames = {
                next(trace.glob("*ffn_norm*")): ".i32",
                next(trace.glob("*router_logits*")): ".i32",
                next(trace.glob("*router_selected*")): ".f32",
            }
            for source, extension in renames.items():
                source.rename(source.with_suffix(extension))
            self.assertEqual(self.score(trace, log)["verdict"], "FAIL")

    def test_rejects_wrong_path_layer_or_chunk_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            for expected_layers, expected_chunks in (
                ({5}, [(0, 2)]),
                ({4}, [(1, 2)]),
                ({4}, [(0, 1)]),
            ):
                result = MODULE.score_trace(
                    trace, log, max_bytes=1_000_000,
                    expected_layers=expected_layers,
                    expected_chunks=expected_chunks,
                )
                self.assertEqual(result["verdict"], "FAIL")
            log.write_text(log.read_text().replace("full_indexed_batch_ffn", "verify_rows"))
            self.assertEqual(self.score(trace, log)["verdict"], "FAIL")

    def test_rejects_overlapping_ranges_and_degenerate_values(self) -> None:
        for mutation in ("overlap", "zero_norm", "zero_logits", "fixed_ids"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                if mutation == "overlap":
                    log.write_text(log.read_text() +
                        "GLM_UNION_TRACE_OK path=full_indexed_batch_ffn layer=4 pos=1 rows=2\n")
                elif mutation == "zero_norm":
                    target = next(trace.glob("*ffn_norm*"))
                    target.write_bytes(bytes(target.stat().st_size))
                elif mutation == "zero_logits":
                    target = next(trace.glob("*router_logits*"))
                    target.write_bytes(bytes(target.stat().st_size))
                else:
                    target = next(trace.glob("*router_selected*"))
                    target.write_bytes(struct.pack("<16i", *(list(range(8)) * 2)))
                self.assertEqual(self.score(trace, log)["verdict"], "FAIL")

    def test_rejects_selected_ids_that_disagree_with_logits_and_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            target = next(trace.glob("*router_selected*"))
            values = list(struct.unpack("<16i", target.read_bytes()))
            values[0], values[1] = values[1], values[0]
            target.write_bytes(struct.pack("<16i", *values))
            result = self.score(trace, log)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["selected_matches_router_formula"])

    def test_rejects_double_precision_reordering_of_a_float32_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            probs_path = next(trace.glob("*router_probs*"))
            probs = list(struct.unpack("<512f", probs_path.read_bytes()))
            probs[0] = probs[1] = struct.unpack("<f", struct.pack("<f", 0.75))[0]
            probs_path.write_bytes(struct.pack("<512f", *probs))
            selected_path = next(trace.glob("*router_selected*"))
            selected = list(struct.unpack("<16i", selected_path.read_bytes()))
            selected[0], selected[1] = 1, 0
            selected_path.write_bytes(struct.pack("<16i", *selected))
            result = self.score(trace, log)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["selected_matches_router_formula"])


if __name__ == "__main__":
    unittest.main()
