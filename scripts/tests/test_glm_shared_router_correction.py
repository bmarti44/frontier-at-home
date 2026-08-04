#!/usr/bin/env python3
"""Acceptance and source contract for R0-UPGRADE a."""

from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/72_glm_shared_router_score.py"
PATCH = ROOT / "results/glm52-gates/harness/ds4-shared-router-correction.patch"
RUNNER = ROOT / "scripts/73_run_glm_shared_router_probe.py"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SPEC = importlib.util.spec_from_file_location("shared_router_score", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location("shared_router_runner", RUNNER)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def valid_response_signature() -> dict[str, object]:
    return {
        "request_sha256": "a" * 64,
        "token_ids": list(range(128)),
        "completion_tokens": 128,
        "generated_reasoning_sha256": "e" * 64,
        "generated_reasoning_bytes": 512,
        "generated_content_sha256": "f" * 64,
        "generated_content_bytes": 0,
    }

def row(event: int, position: int, layer: int, actual: range, baseline: range, shared: range) -> str:
    values = lambda items: " ".join(str(item) for item in items)
    return (
        f"PREDPAIR E{event} P{position} L{layer} actual: {values(actual)}"
        f" base: {values(baseline)} shared: {values(shared)}"
    )


class SharedRouterScorerTests(unittest.TestCase):
    def score_rows(self, rows: list[str]) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return MODULE.score(path)

    def test_accepts_matched_trace_with_two_point_recall_gain(self) -> None:
        rows = [row(index + 1, index // 74, 4 + index % 74,
                    range(8), range(8), range(8)) for index in range(1036)]
        # Lower the baseline by two of eight experts in enough samples to
        # cross the preregistered 0.02 absolute-recall threshold.
        for index in range(83):
            rows[index] = row(index + 1, index // 74, 4 + index % 74,
                              range(8), range(2, 10), range(8))
        result = self.score_rows(rows)
        self.assertEqual(result["verdict"], "PASS")
        self.assertGreaterEqual(result["absolute_recall_gain"], 0.02)

    def test_rejects_too_few_rows(self) -> None:
        result = self.score_rows([
            row(index + 1, index // 74, 4 + index % 74,
                range(8), range(2, 10), range(8)) for index in range(999)
        ])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["minimum_samples"])

    def test_rejects_no_gain(self) -> None:
        result = self.score_rows([
            row(index + 1, index // 74, 4 + index % 74,
            range(8), range(8), range(8)) for index in range(1036)
        ])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["shared_recall_gain"])

    def test_rejects_malformed_or_duplicate_ids(self) -> None:
        rows = [row(index + 1, index // 74, 4 + index % 74,
                    range(8), range(2, 10), range(8)) for index in range(1036)]
        rows[0] = "PREDPAIR E1 P0 L4 actual: 0 0 1 2 3 4 5 6 base: 2 3 4 5 6 7 8 9 shared: 0 1 2 3 4 5 6 7"
        result = self.score_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["malformed_rows"], 1)

    def test_rejects_a_duplicated_favorable_observation(self) -> None:
        favorable = row(1, 0, 4, range(8), range(2, 10), range(8))
        result = self.score_rows([favorable] * 1036)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["unique_event_keys"])

    def test_rejects_one_token_layer_sweep_replayed_to_sample_floor(self) -> None:
        sweep = [row(index + 1, 0, 4 + index, range(8), range(2, 10), range(8))
                 for index in range(74)]
        result = self.score_rows((sweep * 14)[:1000])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["position_coverage"])

    def test_rejects_unique_events_with_shuffled_incomplete_sweeps(self) -> None:
        rows = [row(index + 1, index % 14, 4 + index % 74,
                    range(8), range(2, 10), range(8)) for index in range(1036)]
        result = self.score_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["complete_position_sweeps"])


class SharedRouterSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PATCH.read_text(encoding="utf-8") if PATCH.exists() else ""
        cls.added_source = "\n".join(
            line[1:] for line in cls.source.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

    def test_correction_is_explicit_and_default_off(self) -> None:
        self.assertIn('getenv("DS4_GLM_PREFETCH_SHARED_CORRECTION")', self.source)
        self.assertIn("shared-expert router correction enabled", self.source)

    def test_correction_uses_shared_residual_and_next_layer_norm(self) -> None:
        for marker in (
            "pf_corrected_state, after_attn, ffn_sum",
            "lnext->ffn_norm->abs_offset",
            "ds4_gpu_rms_norm_weight_tensor(",
            "pf_corrected_norm, pf_corrected_state",
            "g->batch_router_logits",
        ):
            self.assertIn(marker, self.source)

    def test_probe_logs_matched_actual_baseline_and_shared_sets(self) -> None:
        self.assertIn("PREDPAIR E%llu P%u L%u actual:", self.source)
        self.assertIn('getenv("DS4_GLM_PREDACC_SHARED")', self.source)

    def test_pending_pair_state_is_graph_scoped(self) -> None:
        self.assertIn("predacc_pair_layer", self.source)
        self.assertIn("predacc_pair_event", self.source)
        self.assertNotIn("static uint32_t pair_layer", self.source)

    def test_predictor_stops_at_normal_layer_boundary_not_mtp_layer(self) -> None:
        self.assertIn(
            "const uint32_t normal_layers = glm_graph_normal_layer_count();",
            self.added_source,
        )
        self.assertEqual(
            self.added_source.count("il + 1u < normal_layers"), 4
        )
        self.assertNotIn("il + 1u < DS4_N_LAYER", self.added_source)

    def test_probe_failures_close_and_rebalance_command_ownership(self) -> None:
        for marker in (
            "if (!ended || !read_actual || !restarted)",
            "if (!pf_base_valid) ok = false",
            "corrected_ok = corrected_ok && corrected_ended && corrected_restarted",
        ):
            self.assertIn(marker, self.source)

    def test_prefetch_hint_stays_after_current_selected_load(self) -> None:
        self.assertIn("shared correction waits for current selected load", self.source)

    def test_union_probe_triplet_is_in_live_batch_ffn_helper(self) -> None:
        """Reject hooks placed only in the compile-time-dead indexed fallback."""
        marker = 'glm_union_probe_dump_triplet('
        self.assertIn(marker, self.added_source)
        call_at = self.added_source.rfind(marker)
        hunk_at = self.added_source.rfind("@@", 0, call_at)
        hunk_end = self.added_source.find("\n", hunk_at)
        self.assertIn(
            "glm_graph_encode_ffn_batch",
            self.added_source[hunk_at:hunk_end],
        )
        self.assertNotIn(
            'metal_graph_debug_dump_tensor("glm_indexed_ffn_norm"',
            self.added_source,
        )

    def test_union_probe_triplet_is_fail_closed_and_immutable(self) -> None:
        for marker in (
            '"glm_indexed_ffn_norm"',
            '"glm_indexed_router_logits"',
            '"glm_indexed_router_selected"',
            "O_CREAT | O_EXCL",
            "fsync(fd)",
            "GLM_UNION_TRACE_ERROR",
        ):
            self.assertIn(marker, self.added_source)
        self.assertIn("if (ok) ok = glm_union_probe_dump_triplet(", self.added_source)


class SharedRouterRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8") if RUNNER.exists() else ""

    def test_runner_uses_existing_containment_and_one_request_slot(self) -> None:
        self.assertIn("glm_cgroup_run.sh", self.source)
        self.assertIn("SINGLE_REQUEST_SLOT = 1", self.source)
        self.assertIn("--batched-sessions", self.source)
        self.assertIn('result["DS4_GLM_PREDACC_SHARED"] = "1"', self.source)
        self.assertIn('"DS4_TOKEN_TIMING_LOG": "1"', self.source)
        self.assertIn('"--max-tokens", "128", "--min-completion-tokens", "128"', self.source)
        self.assertIn('"DS4_LOCK_FILE"', self.source)
        self.assertIn('out / "runtime.lock"', self.source)

    def test_runner_binds_candidate_and_safety_artifacts(self) -> None:
        for marker in (
            "GLM_SAFE_EXPECTED_BINARY_SHA256",
            "GLM_SAFE_EXPECTED_ENV_SHA256",
            "GLM_SAFE_FINAL_ARTIFACTS",
            "memory.events.local",
            "kernel.log",
            "generated_token_ids",
        ):
            self.assertIn(marker, self.source)

    def test_public_runner_does_not_accept_identity_or_seed_overrides(self) -> None:
        parser = RUNNER_MODULE.parser()
        subparsers = next(action for action in parser._actions
                          if isinstance(action.choices, dict))
        destinations = {action.dest for action in subparsers.choices["run"]._actions}
        self.assertEqual(destinations, {"help", "tag", "port"})

    def test_runner_verifies_the_committed_public_randomness_round(self) -> None:
        self.assertIn("https://api.drand.sh/public/{round}", self.source)
        self.assertIn('public.get("randomness") != raw', self.source)

    def test_small_perf_gate_is_fixed_at_five_percent_non_regression(self) -> None:
        passed = RUNNER_MODULE.performance_verdict(2.0, 1.9, True)
        failed = RUNNER_MODULE.performance_verdict(2.0, 1.899, True)
        unequal = RUNNER_MODULE.performance_verdict(2.0, 2.1, False)
        nonfinite = RUNNER_MODULE.performance_verdict(2.0, float("nan"), True)
        zero = RUNNER_MODULE.performance_verdict(0.0, 2.1, True)
        self.assertEqual(passed["verdict"], "PASS")
        self.assertEqual(passed["decode_ratio"], 0.95)
        self.assertEqual(failed["verdict"], "FAIL")
        self.assertEqual(unequal["verdict"], "FAIL")
        self.assertEqual(nonfinite["verdict"], "FAIL")
        self.assertEqual(zero["verdict"], "FAIL")

    def test_perf_arm_enables_only_the_production_correction_flags(self) -> None:
        lock = Path("/tmp/perf-contract.lock")
        off = RUNNER_MODULE.performance_environment_for("off", lock)
        corrected = RUNNER_MODULE.performance_environment_for("corrected", lock)
        self.assertNotIn("DS4_GLM_PREFETCH", off)
        self.assertNotIn("DS4_GLM_PREFETCH_SHARED_CORRECTION", off)
        self.assertNotIn("DS4_GLM_PREDACC_SHARED", corrected)
        self.assertEqual(corrected["DS4_GLM_PREFETCH"], "1")
        self.assertEqual(corrected["DS4_GLM_PREFETCH_SHARED_CORRECTION"], "1")
        self.assertEqual(corrected["DS4_GLM_PREFETCH_THREADS"], "8")

    def test_actual_server_command_is_single_slot(self) -> None:
        command = RUNNER_MODULE.server_command(Path("/tmp/ds4-server"), 8040)
        self.assertNotIn("--batched-sessions", command)
        self.assertEqual(command.count("--port"), 1)

    def test_containment_forwards_the_correction_flag(self) -> None:
        source = CGROUP.read_text(encoding="utf-8")
        self.assertIn("DS4_GLM_PREFETCH_SHARED_CORRECTION", source)

    def test_campaign_configuration_identity_ignores_only_arm_lock_path(self) -> None:
        off_a = RUNNER_MODULE.campaign_configuration_sha256("off", Path("/tmp/a.lock"))
        off_b = RUNNER_MODULE.campaign_configuration_sha256("off", Path("/tmp/b.lock"))
        corrected = RUNNER_MODULE.campaign_configuration_sha256(
            "corrected", Path("/tmp/c.lock")
        )
        self.assertEqual(off_a, off_b)
        self.assertNotEqual(off_a, corrected)

    def test_balanced_campaign_requires_positive_decode_lower_bound(self) -> None:
        rows = []
        for block, sequence, mode in RUNNER_MODULE.campaign_schedule(False):
            rows.append({
                "block": block,
                "sequence": sequence,
                "mode": mode,
                "decode_tokens_per_second": 2.1 if mode == "corrected" else 2.0,
                "ttft_seconds": 1.0,
                "completion_tokens": 128,
                "response_signature": valid_response_signature(),
                "fixture_sha256": "a" * 64,
                "server_boot_id": f"boot-{block}-{sequence}",
                "binary_sha256": "b" * 64,
                "configuration_sha256": ("c" if mode == "off" else "d") * 64,
            })
        passed = RUNNER_MODULE.campaign_verdict(rows, False)
        self.assertEqual(len(rows), 20)
        self.assertEqual(passed["verdict"], "PASS")
        self.assertGreater(passed["decode_ratio_lower_95"], 1.0)
        for row in rows:
            if row["mode"] == "corrected":
                row["decode_tokens_per_second"] = 1.99
        failed = RUNNER_MODULE.campaign_verdict(rows, False)
        self.assertEqual(failed["verdict"], "FAIL")

    def test_balanced_campaign_rejects_wrong_order_or_output(self) -> None:
        rows = []
        for block, sequence, mode in RUNNER_MODULE.campaign_schedule(True):
            rows.append({
                "block": block,
                "sequence": sequence,
                "mode": mode,
                "decode_tokens_per_second": 2.1 if mode == "corrected" else 2.0,
                "ttft_seconds": 1.0,
                "completion_tokens": 128,
                "response_signature": valid_response_signature(),
                "fixture_sha256": "a" * 64,
                "server_boot_id": f"boot-{block}-{sequence}",
                "binary_sha256": "b" * 64,
                "configuration_sha256": ("c" if mode == "off" else "d") * 64,
            })
        with self.assertRaises(ValueError):
            RUNNER_MODULE.campaign_verdict(rows, False)
        rows[-1]["response_signature"]["token_ids"][-1] = 999
        with self.assertRaises(ValueError):
            RUNNER_MODULE.campaign_verdict(rows, True)

    def test_balanced_campaign_rejects_malformed_response_signatures(self) -> None:
        def rows() -> list[dict[str, object]]:
            return [{
                "block": block,
                "sequence": sequence,
                "mode": mode,
                "decode_tokens_per_second": 2.1 if mode == "corrected" else 2.0,
                "ttft_seconds": 1.0,
                "completion_tokens": 128,
                "response_signature": valid_response_signature(),
                "fixture_sha256": "a" * 64,
                "server_boot_id": f"boot-{block}-{sequence}",
                "binary_sha256": "b" * 64,
                "configuration_sha256": ("c" if mode == "off" else "d") * 64,
            } for block, sequence, mode in RUNNER_MODULE.campaign_schedule(False)]

        mutations = (
            lambda row: row.update(response_signature={}),
            lambda row: row.update(response_signature="invalid"),
            lambda row: row["response_signature"].pop("request_sha256"),
            lambda row: row["response_signature"].update(request_sha256="b" * 64),
            lambda row: row["response_signature"].update(token_ids=list(range(127))),
            lambda row: row["response_signature"]["token_ids"].__setitem__(0, True),
            lambda row: row["response_signature"].update(generated_content_sha256="BAD"),
            lambda row: row["response_signature"].update(generated_content_bytes=-1),
            lambda row: row["response_signature"].update(completion_tokens=127),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(rows())
            for row in candidate:
                mutate(row)
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError):
                    RUNNER_MODULE.campaign_verdict(candidate, False)


if __name__ == "__main__":
    unittest.main()
