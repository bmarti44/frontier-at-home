#!/usr/bin/env python3
"""Acceptance contract for the contained R0b trace smoke."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/76_run_glm_union_trace_smoke.py"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SPEC = importlib.util.spec_from_file_location("glm_union_trace_smoke", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arm(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "binary_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "fixture_sha256": "d" * 64,
        "configuration_sha256": "e" * 64,
        "response_signature": {"request_sha256": "d" * 64, "token_ids": [1, 2]},
        "full_indexed_chunks": [[0, 2048], [2048, 2048], [4096, 40]],
        "trace_files": 0 if mode == "off" else 9,
    }


class UnionTraceSmokeVerdictTests(unittest.TestCase):
    def test_accepts_bound_identical_contained_arms(self) -> None:
        result = MODULE.smoke_verdict(
            arm("off"), arm("on"), {"verdict": "PASS", "events": 3},
            {"clean": True}, {"clean": True},
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_rejects_identity_output_chunk_trace_and_containment_mutations(self) -> None:
        for mutation in (
            "binary", "model", "tokenizer", "fixture", "configuration",
            "output", "chunks", "off_trace", "score", "containment",
        ):
            with self.subTest(mutation=mutation):
                off, on = arm("off"), arm("on")
                score = {"verdict": "PASS", "events": 3}
                off_c, on_c = {"clean": True}, {"clean": True}
                if mutation in {"binary", "model", "tokenizer", "fixture", "configuration"}:
                    on[f"{mutation}_sha256"] = "f" * 64
                elif mutation == "output":
                    on["response_signature"] = {"request_sha256": "d" * 64, "token_ids": [9]}
                elif mutation == "chunks":
                    on["full_indexed_chunks"] = [[0, 2048]]
                elif mutation == "off_trace":
                    off["trace_files"] = 1
                elif mutation == "score":
                    score["verdict"] = "FAIL"
                else:
                    on_c["clean"] = False
                self.assertEqual(
                    MODULE.smoke_verdict(off, on, score, off_c, on_c)["verdict"],
                    "FAIL",
                )


class UnionTraceSmokeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.cgroup = CGROUP.read_text(encoding="utf-8")

    def test_runner_uses_existing_containment_and_fixed_large_enough_fixture(self) -> None:
        for marker in (
            "glm_cgroup_run.sh", '"--context-levels", "4096"',
            '"--max-tokens", "128"', "GLM_SAFE_MEMORY_HIGH_GIB",
            "GLM_SAFE_KILL_FLOOR_GIB", "GLM_SAFE_MIN_START_GIB",
        ):
            self.assertIn(marker, self.runner)

    def test_runner_has_prewrite_disk_bound_and_preservation_reserve(self) -> None:
        for marker in (
            "MAX_TRACE_BYTES", "TRACE_DISK_RESERVE_BYTES", "shutil.disk_usage",
            "DS4_METAL_GRAPH_DUMP_LAYER", '"4"',
        ):
            self.assertIn(marker, self.runner)

    def test_trace_environment_is_forwarded_by_containment(self) -> None:
        for name in (
            "DS4_GLM_SYNC_TRACE", "DS4_METAL_GRAPH_DUMP_PREFIX",
            "DS4_METAL_GRAPH_DUMP_NAME", "DS4_METAL_GRAPH_DUMP_LAYER",
        ):
            self.assertIn(name, self.runner)
            self.assertIn(name, self.cgroup)


if __name__ == "__main__":
    unittest.main()
