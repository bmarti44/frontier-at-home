#!/usr/bin/env python3
"""Acceptance contract for the contained R0b trace smoke."""

from __future__ import annotations

import copy
import hashlib
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
        "cuda_expert_cache_gb": "48",
        "cuda_cache_runtime": {"slots": 4800, "arena_gib": 43.50},
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

    def test_cuda_cache_runtime_requires_one_bounded_resolved_arena(self) -> None:
        marker = (
            "ds4: CUDA persistent expert cache enabled: "
            "4800 slots x 9.28 MiB = 43.50 GiB (fixed arena)\n"
        )
        self.assertEqual(
            MODULE.cuda_cache_runtime(marker),
            {"slots": 4800, "arena_gib": 43.50},
        )
        for bad in (
            "",
            marker + marker,
            marker + "ds4: CUDA persistent expert cache enabled: malformed\n",
            marker.rstrip("\n") + " unexpected suffix\n",
            "ds4: CUDA persistent expert cache enabled: malformed\n",
            "ds4: CUDA persistent expert cache enabled: "
            "6987 slots x 9.28 MiB = 63.33 GiB (fixed arena)\n",
            "ds4: CUDA persistent expert cache enabled: "
            "6000 slots x 9.28 MiB = 50.00 GiB (fixed arena)\n",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.cuda_cache_runtime(bad)

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

    def test_corpus_randomness_postdates_corpus_freeze(self) -> None:
        MODULE.validate_randomness_order(MODULE.CORPUS_FREEZE, MODULE.CORPUS_RANDOMNESS)

    def test_quality_prompt_render_is_independently_tokenizable(self) -> None:
        self.assertEqual(
            MODULE.render_quality_prompt("Hello"),
            "[gMASK]<sop><|system|>Reasoning Effort: High"
            "<|system|>You are a helpful assistant"
            "<|user|>Hello<|assistant|><think>",
        )

    def test_quality_verdict_requires_frozen_expected_token_coverage(self) -> None:
        ledger = [
            {
                "case_id": "case_001", "group_id": "case_001", "split": "train-fit",
                "request_id": 1, "request_sha256": "1" * 64,
                "expected_prompt_tokens": 2, "token_ids": [11, 12],
            },
            {
                "case_id": "case_002", "group_id": "case_002", "split": "test",
                "request_id": 2, "request_sha256": "2" * 64,
                "expected_prompt_tokens": 3, "token_ids": [21, 22, 23],
            },
        ]
        requests = [
            {
                "case_id": row["case_id"], "group_id": row["group_id"],
                "split": row["split"], "request_id": row["request_id"],
                "request_sha256": row["request_sha256"],
                "prompt_tokens": row["expected_prompt_tokens"],
                "full_indexed_chunks": [[0, row["expected_prompt_tokens"]]],
                "completion_tokens": 8,
                "generated_reasoning_sha256": "a" * 64,
                "generated_reasoning_bytes": 8,
                "generated_content_sha256": "b" * 64,
                "generated_content_bytes": 0,
                "token_ids": list(range(8)),
                "sse_content_events": 5,
            }
            for row in ledger
        ]
        score = {"verdict": "PASS", "requests": 2, "token_layer_events": 375}
        result = MODULE.quality_capture_verdict(
            ledger, requests, copy.deepcopy(requests), score,
            {"clean": True}, {"clean": True},
        )
        self.assertEqual(result["verdict"], "PASS")

        mutations = (
            "consistent_truncation", "missing_case", "relabel", "output", "rows",
            "empty_output", "false_usage", "missing_output",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                off, on, bad_score = copy.deepcopy(requests), copy.deepcopy(requests), copy.deepcopy(score)
                if mutation == "consistent_truncation":
                    for arm_requests in (off, on):
                        arm_requests[0]["prompt_tokens"] = 1
                        arm_requests[0]["full_indexed_chunks"] = [[0, 1]]
                    bad_score["token_layer_events"] = 300
                elif mutation == "missing_case":
                    off.pop()
                    on.pop()
                    bad_score["requests"] = 1
                    bad_score["token_layer_events"] = 150
                elif mutation == "relabel":
                    on[1]["case_id"] = "case_001"
                elif mutation == "output":
                    on[1]["generated_reasoning_sha256"] = "c" * 64
                elif mutation == "empty_output":
                    for arm_requests in (off, on):
                        arm_requests[0]["token_ids"] = []
                        arm_requests[0]["generated_reasoning_sha256"] = hashlib.sha256(b"").hexdigest()
                        arm_requests[0]["generated_content_sha256"] = hashlib.sha256(b"").hexdigest()
                elif mutation == "false_usage":
                    for arm_requests in (off, on):
                        arm_requests[0]["completion_tokens"] = 7
                        arm_requests[0]["token_ids"] = list(range(7))
                elif mutation == "missing_output":
                    for arm_requests in (off, on):
                        arm_requests[0].pop("generated_content_sha256")
                else:
                    bad_score["token_layer_events"] = 374
                self.assertEqual(
                    MODULE.quality_capture_verdict(
                        ledger, off, on, bad_score,
                        {"clean": True}, {"clean": True},
                    )["verdict"],
                    "FAIL",
                )

    def test_quality_arm_identity_rejects_copied_and_mutated_arms(self) -> None:
        expected = {
            "binary_sha256": "1" * 64,
            "model_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
            "fixture_sha256": "4" * 64,
            "split_plan_sha256": "5" * 64,
            "configuration_sha256": "6" * 64,
            "off_environment_sha256": "7" * 64,
            "on_environment_sha256": "8" * 64,
        }
        off = {
            **{key: value for key, value in expected.items() if not key.endswith("environment_sha256")},
            "mode": "off", "environment_sha256": expected["off_environment_sha256"],
            "trace_files": 0, "trace_bytes": 0,
        }
        on = {
            **{key: value for key, value in expected.items() if not key.endswith("environment_sha256")},
            "mode": "on", "environment_sha256": expected["on_environment_sha256"],
            "trace_files": 10, "trace_bytes": 1000,
        }
        score = {"verdict": "PASS", "artifacts": [{}] * 10, "total_bytes": 1000}
        self.assertTrue(all(MODULE.quality_arm_identity_checks(off, on, score, expected).values()))
        for mutation in ("copied", "identity", "off_trace", "on_count", "on_bytes"):
            with self.subTest(mutation=mutation):
                bad_off, bad_on, bad_score = copy.deepcopy(off), copy.deepcopy(on), copy.deepcopy(score)
                if mutation == "copied":
                    bad_off = copy.deepcopy(on)
                elif mutation == "identity":
                    bad_on["binary_sha256"] = "9" * 64
                elif mutation == "off_trace":
                    bad_off["trace_files"] = 1
                elif mutation == "on_count":
                    bad_on["trace_files"] = 9
                else:
                    bad_on["trace_bytes"] = 999
                self.assertFalse(all(MODULE.quality_arm_identity_checks(
                    bad_off, bad_on, bad_score, expected,
                ).values()))

    def test_final_artifact_receipts_reject_post_containment_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "arm.json"
            artifact.write_text("original", encoding="utf-8")
            stat = artifact.stat()
            main = root / "main.log"
            main.write_text(
                "final_artifact_verified path=" + str(artifact) +
                " sha256=" + hashlib.sha256(b"original").hexdigest() +
                f" device_inode={stat.st_dev}:{stat.st_ino}:{stat.st_size}\n",
                encoding="utf-8",
            )
            containment = {"crash_directory": str(root), "main_sha256": MODULE.SHARED.sha256(main)}
            MODULE.verify_final_artifact_receipts(containment, [artifact])
            artifact.write_text("replacement", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final artifact"):
                MODULE.verify_final_artifact_receipts(containment, [artifact])

    def test_quality_windows_never_cross_case_or_split_boundaries(self) -> None:
        rows = [
            {"case_id": "a", "split": "train-fit", "layer": 3, "position": position}
            for position in range(9)
        ] + [
            {"case_id": "b", "split": "test", "layer": 3, "position": position}
            for position in range(9)
        ]
        windows = MODULE.quality_window_indices(rows, horizon=8)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(len(window) == 9 for window in windows))
        for window in windows:
            self.assertEqual(len({rows[index]["case_id"] for index in window}), 1)
            self.assertEqual(len({rows[index]["split"] for index in window}), 1)

    def test_quality_probe_ledger_is_one_frozen_case_only(self) -> None:
        bundle = {
            "schema_version": 1, "split_plan_sha256": "a" * 64,
            "fixture_content_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "seed": 7, "total_expected_prompt_tokens": 5,
            "expected_token_layer_events": 375,
            "cases": [
                {"case_id": "case_1", "expected_prompt_tokens": 2},
                {"case_id": "case_2", "expected_prompt_tokens": 3},
            ],
            "_prompts": {"case_1": "one", "case_2": "two"},
        }
        probe = MODULE.quality_probe_ledger(bundle)
        self.assertEqual(probe["cases"], [bundle["cases"][0]])
        self.assertEqual(probe["_prompts"], {"case_1": "one"})
        self.assertEqual(probe["total_expected_prompt_tokens"], 2)
        self.assertEqual(probe["expected_token_layer_events"], 150)

    def test_quality_raw_output_accepts_exact_bytes_with_noncanonical_bpe(self) -> None:
        tokenizer = MODULE.Tokenizer.from_file(str(MODULE.SHARED.TOKENIZER))
        raw_ids = [8507, 111, 198, 154842, 8507, 111, 271, 91]
        self.assertEqual(
            MODULE.quality_raw_visible_output_errors(
                tokenizer, raw_ids, "害\n", "害\n\n|",
            ),
            [],
        )
        self.assertTrue(MODULE.quality_raw_visible_output_errors(
            tokenizer, raw_ids, "different", "害\n\n|",
        ))
        self.assertTrue(MODULE.quality_raw_visible_output_errors(
            tokenizer, raw_ids[:-1] + [tokenizer.get_vocab_size() + 1],
            "害\n", "害\n\n|",
        ))
        for invalid in (-1, True, "8507"):
            with self.subTest(invalid=invalid):
                self.assertTrue(MODULE.quality_raw_visible_output_errors(
                    tokenizer, [invalid, *raw_ids[1:]], "害\n", "害\n\n|",
                ))
        self.assertTrue(MODULE.quality_raw_visible_output_errors(
            tokenizer, [*raw_ids[:3], 91, *raw_ids[4:]], "害\n", "害\n\n|",
        ))

    def test_quality_raw_output_uses_first_reasoning_boundary(self) -> None:
        tokenizer = MODULE.Tokenizer.from_file(str(MODULE.SHARED.TOKENIZER))
        raw_ids = [8507, 81272, 13, 154842, 154842, 154842, 154842, 154842]
        self.assertEqual(
            MODULE.quality_raw_visible_output_errors(
                tokenizer, raw_ids, "守权.", "</think></think></think></think>",
            ),
            [],
        )
        self.assertTrue(MODULE.quality_raw_visible_output_errors(
            tokenizer, raw_ids, "守权.", "",
        ))


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

    def test_corpus_capture_has_room_above_the_measured_70_gib_peak(self) -> None:
        self.assertIn('CORPUS_MEMORY_HIGH_GIB = "71"', self.runner)
        self.assertIn('"GLM_SAFE_MEMORY_HIGH_GIB": (', self.runner)
        self.assertIn('CORPUS_MEMORY_HIGH_GIB if large_corpus else "69"', self.runner)

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
            "DS4_GLM_UNION_TRACE_CORPUS", "DS4_GLM_STREAMING_TOKEN_PREFILL_MAX",
        ):
            self.assertIn(name, self.runner)
            self.assertIn(name, self.cgroup)

    def test_quality_capture_forces_the_indexed_prefill_path(self) -> None:
        for mode in ("off", "on"):
            values = MODULE.trace_environment(
                mode, Path("/tmp/quality-indexed"), quality_corpus=True,
            )
            self.assertEqual(values["DS4_GLM_STREAMING_TOKEN_PREFILL_MAX"], "0")
        self.assertNotIn(
            "DS4_GLM_STREAMING_TOKEN_PREFILL_MAX",
            MODULE.trace_environment("off", Path("/tmp/non-quality"), corpus_smoke=True),
        )

    def test_runner_has_bounded_two_request_corpus_mode(self) -> None:
        for marker in (
            'public.add_argument("--corpus-smoke", action="store_true")',
            '"DS4_GLM_UNION_TRACE_CORPUS"',
            '"DS4_METAL_GRAPH_DUMP_LAYER": "all"',
            'for request_index in range(0 if args.quality_corpus else (2 if args.corpus_smoke else 1))',
            '"minimum_token_layer_events": 76800',
        ):
            self.assertIn(marker, self.runner)
        self.assertIn('CORPUS_CACHE_EXPERTS = "32GB"', self.runner)
        self.assertIn('CORPUS_CUDA_CACHE_GB = "48"', self.runner)
        self.assertIn("cache_experts=(CORPUS_CACHE_EXPERTS", self.runner)
        self.assertIn('values["DS4_CUDA_EXPERT_CACHE_GB"] = CORPUS_CUDA_CACHE_GB', self.runner)
        self.assertIn("cuda_cache_runtime", self.runner)

    def test_quality_mode_is_exactly_bounded_and_frozen(self) -> None:
        for marker in (
            'public.add_argument("--quality-corpus", action="store_true")',
            'QUALITY_REQUEST_COUNT = 100',
            'QUALITY_DISK_MAX_TOKENS = 512',
            'QUALITY_FIXTURE_CONTENT_SHA256',
            'quality fixture content differs from the preregistered freeze',
            '"scope": "quality_100_case_all_routed_layer_corpus"',
            'QUALITY_FREEZE',
            'QUALITY_RANDOMNESS',
            '"GLM_SAFE_TIMEOUT_S": "7200" if args.quality_corpus else "3600"',
            'str(out / "ledger.json"), str(out / "responses.json")',
            'set(range(3, 78))',
        ):
            self.assertIn(marker, self.runner)

    def test_quality_probe_is_explicit_one_case_safety_only(self) -> None:
        for marker in (
            'public.add_argument("--quality-probe", action="store_true")',
            'internal.add_argument("--quality-probe", action="store_true")',
            '"scope": "quality_one_case_safety_probe"',
            'quality probe requires quality corpus mode',
            'request_count = 1 if args.quality_probe else',
            '(["--quality-probe"] if args.quality_probe else [])',
        ):
            self.assertIn(marker, self.runner)

    def test_quality_arms_bind_the_same_complete_ledger(self) -> None:
        for marker in (
            'raise ValueError("quality arm ledgers differ")',
            'arm.get("quality_ledger_sha256")',
            'arm.get("expected_token_layer_events")',
            'arm.get("fixture_sha256")',
        ):
            self.assertIn(marker, self.runner)

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
