#!/usr/bin/env python3
"""Acceptance contract for the contained R0b trace smoke."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
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
        "cuda_expert_cache_gb": "60",
        "cuda_cache_runtime": {"slots": 6200, "arena_gib": 56.25},
    }


class UnionTraceSmokeVerdictTests(unittest.TestCase):
    def contained_receipt(
        self,
        directory: Path,
        *,
        kernel_text: str = "kernel clean\n",
        main_suffix: str = "",
        command_text: str = "command clean\n",
    ) -> Path:
        (directory / "main.log").write_text(
            "executed candidate was verified alive at least once\n"
            "cgroup_final current_bytes=1 peak_bytes=2 swap_current_bytes=0 events=\n"
            f"{main_suffix}"
            "SAFE_RUN end rc=0 killed=no\n",
            encoding="utf-8",
        )
        (directory / "samples.log").write_text("sample clean\n", encoding="utf-8")
        (directory / "kernel.log").write_text(kernel_text, encoding="utf-8")
        (directory / "cmd.log").write_text(command_text, encoding="utf-8")
        stdout = directory / "stdout.log"
        stdout.write_text(
            f"SAFE_RUN_DONE rc=0 killed=no dir={directory}\n",
            encoding="utf-8",
        )
        return stdout

    def test_python_containment_rejects_driver_and_host_oom_mutations(self) -> None:
        mutations = (
            "NVRM: Xid (PCI:0000:0f:00): 31, pid=1\n",
            "NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory "
            "[NV_ERR_NO_MEMORY] (0x00000051)\n",
            "kernel: oom-kill:constraint=CONSTRAINT_MEMCG\n",
            "kernel: Out of memory: Killed process 123 (ds4-server)\n",
        )
        for index, marker in enumerate(mutations):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as temporary:
                receipt = self.contained_receipt(Path(temporary), kernel_text=marker)
                with self.assertRaisesRegex(ValueError, "GPU|OOM|kernel"):
                    MODULE.SHARED.containment_record(receipt)

    def test_python_containment_rejects_cuda_userspace_oom_mutations(self) -> None:
        mutations = (
            "CUDA_ERROR_OUT_OF_MEMORY\n",
            "cudaErrorMemoryAllocation: out of memory\n",
            "CUDA allocation failed while creating expert buffer\n",
        )
        for location in ("main", "command"):
            for marker in mutations:
                with self.subTest(location=location, marker=marker), tempfile.TemporaryDirectory() as temporary:
                    kwargs = ({"main_suffix": marker} if location == "main" else
                              {"command_text": marker})
                    receipt = self.contained_receipt(Path(temporary), **kwargs)
                    with self.assertRaisesRegex(ValueError, "GPU|OOM|CUDA"):
                        MODULE.SHARED.containment_record(receipt)

    def test_python_containment_accepts_benign_nvidia_runtime_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self.contained_receipt(
                Path(temporary),
                kernel_text="NVRM: loading NVIDIA UNIX Open Kernel Module 2.0\n",
                command_text="CUDA device 0 initialized successfully\n",
            )
            self.assertEqual(
                MODULE.SHARED.containment_record(receipt)["crash_directory"],
                str(Path(temporary).resolve()),
            )

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

    def test_corpus_verdict_binds_two_requests_and_event_floor(self) -> None:
        off, on = arm("off"), arm("on")
        requests = [
            {
                "request_id": request_id,
                "seed": 99 + request_id,
                "prompt_tokens": 512,
                "full_indexed_chunks": [[0, 512]],
                "response_signature": {
                    "request_sha256": str(request_id) * 64,
                    "token_ids": [request_id, 9],
                },
            }
            for request_id in (1, 2)
        ]
        off["corpus_requests"] = copy.deepcopy(requests)
        on["corpus_requests"] = copy.deepcopy(requests)
        off["expert_cache_budget"] = on["expert_cache_budget"] = "32GB"
        score = {
            "verdict": "PASS", "events": 150, "requests": 2,
            "token_layer_events": 76800,
        }
        result = MODULE.smoke_verdict(
            off, on, score, {"clean": True}, {"clean": True},
            expected_corpus_seed=100,
        )
        self.assertEqual(result["verdict"], "PASS")
        for mutation in (
            "id", "output", "event_floor", "seed", "duplicate_fixture",
            "empty_chunks", "short_chunk", "null_chunks", "scalar_chunk",
            "cache_budget", "cuda_cache_environment", "cuda_cache_runtime",
        ):
            with self.subTest(mutation=mutation):
                bad_on = copy.deepcopy(on)
                bad_score = copy.deepcopy(score)
                if mutation == "id":
                    bad_on["corpus_requests"][1]["request_id"] = 3
                elif mutation == "output":
                    bad_on["corpus_requests"][1]["response_signature"]["token_ids"] = [7]
                elif mutation == "event_floor":
                    bad_score["token_layer_events"] = 76799
                elif mutation == "seed":
                    bad_on["corpus_requests"][1]["seed"] = 100
                elif mutation == "duplicate_fixture":
                    duplicate = bad_on["corpus_requests"][0]["response_signature"]["request_sha256"]
                    bad_on["corpus_requests"][1]["response_signature"]["request_sha256"] = duplicate
                elif mutation == "empty_chunks":
                    bad_on["corpus_requests"][1]["full_indexed_chunks"] = []
                elif mutation == "short_chunk":
                    bad_on["corpus_requests"][1]["full_indexed_chunks"] = [[0]]
                elif mutation == "null_chunks":
                    bad_on["corpus_requests"][1]["full_indexed_chunks"] = None
                elif mutation == "scalar_chunk":
                    bad_on["corpus_requests"][1]["full_indexed_chunks"] = [1]
                elif mutation == "cuda_cache_environment":
                    bad_on["cuda_expert_cache_gb"] = "68"
                elif mutation == "cuda_cache_runtime":
                    bad_on["cuda_cache_runtime"] = {"slots": 6987, "arena_gib": 63.33}
                else:
                    bad_on["expert_cache_budget"] = "40GB"
                self.assertEqual(
                    MODULE.smoke_verdict(
                        off, bad_on, bad_score,
                        {"clean": True}, {"clean": True},
                        expected_corpus_seed=100,
                    )["verdict"],
                    "FAIL",
                )

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
            '["--require-multichunk"] if args.require_multichunk else []',
        ):
            self.assertIn(marker, self.runner)

    def test_runner_has_prewrite_disk_bound_and_preservation_reserve(self) -> None:
        for marker in (
            "max_trace_bytes", "TRACE_DISK_RESERVE_BYTES", "shutil.disk_usage",
            "DS4_METAL_GRAPH_DUMP_LAYER", '"4"',
            "(6144 + 256 + 256 + 8 + 256) * 4",
        ):
            self.assertIn(marker, self.runner)

    def test_trace_environment_is_forwarded_by_containment(self) -> None:
        for name in (
            "DS4_GLM_SYNC_TRACE", "DS4_METAL_GRAPH_DUMP_PREFIX",
            "DS4_METAL_GRAPH_DUMP_NAME", "DS4_METAL_GRAPH_DUMP_LAYER",
            "DS4_GLM_UNION_TRACE_CORPUS",
        ):
            self.assertIn(name, self.runner)
            self.assertIn(name, self.cgroup)

    def test_runner_has_bounded_two_request_corpus_mode(self) -> None:
        for marker in (
            'public.add_argument("--corpus-smoke", action="store_true")',
            '"DS4_GLM_UNION_TRACE_CORPUS"',
            '"DS4_METAL_GRAPH_DUMP_LAYER": "all"',
            'for request_index in range(2 if args.corpus_smoke else 1)',
            '"minimum_token_layer_events": 76800',
        ):
            self.assertIn(marker, self.runner)
        self.assertIn('CORPUS_CACHE_EXPERTS = "32GB"', self.runner)
        self.assertIn('CORPUS_CUDA_CACHE_GB = "60"', self.runner)
        self.assertIn("cache_experts=(CORPUS_CACHE_EXPERTS", self.runner)
        self.assertIn('values["DS4_CUDA_EXPERT_CACHE_GB"] = CORPUS_CUDA_CACHE_GB', self.runner)
        self.assertIn("cuda_cache_runtime", self.runner)

    def test_corpus_scores_exact_zero_based_main_routed_layers(self) -> None:
        self.assertIn("expected_layers=set(range(3, 78))", self.runner)
        self.assertNotIn("expected_layers=set(range(4, 79))", self.runner)

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

    def test_server_log_is_checked_for_userspace_gpu_oom(self) -> None:
        self.assertIn("SHARED.require_no_gpu_fault", self.runner)
        self.assertIn("server_log", self.runner)


if __name__ == "__main__":
    unittest.main()
