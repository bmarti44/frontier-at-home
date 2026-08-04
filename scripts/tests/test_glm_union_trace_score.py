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
        (Path(stem + "glm_indexed_ffn_norm-4_pos0.f32")).write_bytes(
            struct.pack(f"<{rows * N_EMBD}f", *([0.25] * (rows * N_EMBD)))
        )
        (Path(stem + "glm_indexed_router_logits-4_pos0.f32")).write_bytes(
            struct.pack(f"<{rows * N_EXPERT}f", *([0.5] * (rows * N_EXPERT)))
        )
        selected = list(range(N_USED)) * rows
        (Path(stem + "glm_indexed_router_selected-4_pos0.i32")).write_bytes(
            struct.pack(f"<{len(selected)}i", *selected)
        )
        log = root / "server.log"
        log.write_text("GLM_UNION_TRACE_OK layer=4 pos=0 rows=2\n", encoding="utf-8")
        return trace, log

    def test_accepts_one_complete_finite_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace, log = self.make_attempt(Path(tmp))
            result = MODULE.score_trace(trace, log, max_bytes=1_000_000)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["events"], 1)
        self.assertEqual(len(result["artifacts"]), 3)

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
                    MODULE.score_trace(trace, log, max_bytes=1_000_000)["verdict"],
                    "FAIL",
                )

    def test_rejects_duplicate_log_key_and_trace_error(self) -> None:
        for extra in (
            "GLM_UNION_TRACE_OK layer=4 pos=0 rows=2\n",
            "GLM_UNION_TRACE_ERROR stage=capture layer=4 pos=0 rows=2\n",
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                trace, log = self.make_attempt(Path(tmp))
                log.write_text(log.read_text() + extra, encoding="utf-8")
                self.assertEqual(
                    MODULE.score_trace(trace, log, max_bytes=1_000_000)["verdict"],
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
                    MODULE.score_trace(trace, log, max_bytes=1_000_000)["verdict"],
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
                    MODULE.score_trace(trace, log, max_bytes=max_bytes)["verdict"],
                    "FAIL",
                )


if __name__ == "__main__":
    unittest.main()
