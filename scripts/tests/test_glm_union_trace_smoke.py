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
        "prompt_tokens": 573,
        "full_indexed_chunks": [[0, 573]],
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
            "prompt_tokens", "shared_truncation",
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
                elif mutation == "prompt_tokens":
                    on["prompt_tokens"] = 500
                elif mutation == "shared_truncation":
                    off["full_indexed_chunks"] = on["full_indexed_chunks"] = [[0, 1]]
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

    def test_high_row_requires_exact_contiguous_multichunk_coverage(self) -> None:
        off, on = arm("off"), arm("on")
        for candidate in (off, on):
            candidate["prompt_tokens"] = 4157
            candidate["full_indexed_chunks"] = [[0, 2048], [2048, 2048], [4096, 61]]
        result = MODULE.smoke_verdict(
            off, on, {"verdict": "PASS", "events": 3},
            {"clean": True}, {"clean": True},
            min_prompt_tokens=2049, require_multichunk=True,
        )
        self.assertEqual(result["verdict"], "PASS")
        for bad_chunks in (
            [[0, 4157]],
            [[0, 2048], [2049, 2108]],
            [[0, 2048], [2048, 2048]],
        ):
            with self.subTest(chunks=bad_chunks):
                bad_off, bad_on = copy.deepcopy(off), copy.deepcopy(on)
                bad_off["full_indexed_chunks"] = bad_on["full_indexed_chunks"] = bad_chunks
                self.assertEqual(
                    MODULE.smoke_verdict(
                        bad_off, bad_on, {"verdict": "PASS", "events": 3},
                        {"clean": True}, {"clean": True},
                        min_prompt_tokens=2049, require_multichunk=True,
                    )["verdict"],
                    "FAIL",
                )

    def test_randomness_must_postdate_freeze_commit(self) -> None:
        self.assertTrue(MODULE.randomness_is_after_freeze(100, 1595431050))
        self.assertFalse(MODULE.randomness_is_after_freeze(1, 1595431050))
        self.assertFalse(MODULE.randomness_is_after_freeze(2, 1595431080))

    def test_committed_randomness_postdates_committed_freeze(self) -> None:
        MODULE.validate_randomness_order()


class UnionTraceSmokeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.cgroup = CGROUP.read_text(encoding="utf-8")

    def test_runner_uses_existing_containment_and_bounded_fixture(self) -> None:
        for marker in (
            "glm_cgroup_run.sh", '"--context-levels", str(args.context_level)',
            '"--max-tokens", "128"', "GLM_SAFE_MEMORY_HIGH_GIB",
            "GLM_SAFE_KILL_FLOOR_GIB", "GLM_SAFE_MIN_START_GIB",
        ):
            self.assertIn(marker, self.runner)

    def test_short_smoke_does_not_claim_2048_row_coverage(self) -> None:
        self.assertIn('"scope": "short_single_indexed_batch_only"', self.runner)
        self.assertIn('"high_row_2048_status": "OPEN"', self.runner)

    def test_high_row_mode_is_explicit_and_requires_multichunk(self) -> None:
        for marker in (
            '"scope": "high_row_multichunk"',
            '"high_row_2048_status": "PASS"',
            'public.add_argument("--context-level", type=int, default=512)',
            'public.add_argument("--require-multichunk", action="store_true")',
            '"--require-multichunk" if args.require_multichunk else',
        ):
            self.assertIn(marker, self.runner)

    def test_runner_has_prewrite_disk_bound_and_preservation_reserve(self) -> None:
        for marker in (
            "MAX_TRACE_BYTES", "TRACE_DISK_RESERVE_BYTES", "shutil.disk_usage",
            "DS4_METAL_GRAPH_DUMP_LAYER", '"4"',
            "(6144 + 256 + 256 + 8 + 256) * 4",
        ):
            self.assertIn(marker, self.runner)

    def test_trace_environment_is_forwarded_by_containment(self) -> None:
        for name in (
            "DS4_GLM_SYNC_TRACE", "DS4_METAL_GRAPH_DUMP_PREFIX",
            "DS4_METAL_GRAPH_DUMP_NAME", "DS4_METAL_GRAPH_DUMP_LAYER",
        ):
            self.assertIn(name, self.runner)
            self.assertIn(name, self.cgroup)

    def test_freeze_binds_runtime_transitive_dependencies(self) -> None:
        for relative in (
            "scripts/73_run_glm_shared_router_probe.py",
            "scripts/30_bench_speed.py",
            "scripts/glm52_goal.py",
            "scripts/03_memory_guard.py",
            "results/glm52-gates/harness/glm_safe_run.sh",
            "fixtures/ctx-32k.txt",
        ):
            self.assertIn(relative, self.runner)

    def test_summary_binds_both_containment_records(self) -> None:
        self.assertIn('"off_containment_sha256"', self.runner)
        self.assertIn('"on_containment_sha256"', self.runner)


if __name__ == "__main__":
    unittest.main()
