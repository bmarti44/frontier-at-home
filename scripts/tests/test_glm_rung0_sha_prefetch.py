#!/usr/bin/env python3
"""Executable RED contracts for collision-resistant slab prefetch."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "results/glm52-gates/harness/ds4_slab_prefetch_state.h"
CPP_TEST = ROOT / "scripts/tests/cpp/test_ds4_slab_prefetch_state.cpp"
CAMPAIGN_PATH = ROOT / "scripts/70_glm_rung0_slab_campaign.py"


class GlmRung0ShaPrefetchTests(unittest.TestCase):
    @staticmethod
    def load_campaign():
        spec = importlib.util.spec_from_file_location("slab_campaign", CAMPAIGN_PATH)
        assert spec and spec.loader
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        return campaign

    @staticmethod
    def passing_campaign(campaign):
        binary = "a" * 64
        quality_binary = "7" * 64
        fixture = "b" * 64
        access = "c" * 64
        commit = "1" * 40
        environments = {
            arm: campaign.canonical_sha_prefetch_environment(mode)
            for arm, mode in {
                "A": "off", "B": "demand_sha", "C": "prefetch_sha"
            }.items()
        }
        configs = {
            arm: campaign.canonical_environment_sha256(environment)
            for arm, environment in environments.items()
        }
        for marker in range(256):
            signature = bytes([marker]) * 96
            randomness_hex = hashlib.sha256(signature).hexdigest()
            seed = campaign.confirmation_seed(
                randomness_hex, commit, binary, quality_binary
            )
            if not bool(int(seed[:2], 16) & 1):
                break
        randomness = {
            "round": 1,
            "randomness": randomness_hex,
            "signature": signature.hex(),
            "chain_hash": "8" * 64,
            "published_epoch_s": 200,
            "seed_sha256": seed,
            "flip": False,
        }
        schedule = campaign.sha_prefetch_schedule(False)
        start_ns = 2_000_000_000_000
        records = []
        for ordinal, (block, sequence, arm) in enumerate(schedule):
            mode = {"A": "off", "B": "demand_sha", "C": "prefetch_sha"}[arm]
            step_ns = {"A": 100_000_000, "B": 90_000_000, "C": 70_000_000}[arm]
            reps = []
            for rep, phase in enumerate(("cold", "warm")):
                request = hashlib.sha256(f"request-{block}-{rep}".encode()).hexdigest()
                output = hashlib.sha256(f"output-{block}-{rep}".encode()).hexdigest()
                request_started = start_ns + ordinal * 10**12 + rep * 10**11
                ttft_ns = int(({"cold": 2.0, "warm": 1.0}[phase]) *
                              (1.01 if arm == "C" else 1.0) * 1e9)
                first = request_started + ttft_ns
                raw = [first + index * step_ns for index in range(128)]
                client = [first + index * step_ns for index in range(128)]
                reps.append({
                    "phase": phase,
                    "request_ordinal": rep,
                    "request_sha256": request,
                    "generated_bytes_sha256": output,
                    "token_ids": list(range(128)),
                    "completion_tokens": 128,
                    "raw_token_timestamps_ns": raw,
                    "client_token_timestamps_ns": client,
                    "client_request_started_ns": request_started,
                    "client_first_token_ns": first,
                    "client_last_token_ns": client[-1],
                    "ttft_seconds": ttft_ns / 1e9,
                })
            attempts = 0 if arm == "A" else 100
            sha_success = 0 if arm == "A" else 100
            ready = 100 if arm == "C" else 0
            late = 20 if arm == "C" else 0
            fallback = 20 if arm == "C" else 0
            copies = 70 if arm == "C" else 100 if arm == "B" else 0
            validated = sha_success * campaign.EXPERT_RECORD_PAYLOAD_BYTES
            copied = copies * campaign.EXPERT_RECORD_PAYLOAD_BYTES
            records.append({
                "schema_version": 1,
                "block": block,
                "sequence": sequence,
                "arm": arm,
                "mode": mode,
                "server_instance_id": f"server-{block}-{sequence}",
                "candidate_commit": commit,
                "binary_sha256": binary,
                "configuration_sha256": configs[arm],
                "fixture_sha256": fixture,
                "access_stream_sha256": access,
                "recorded_monotonic_ns": start_ns + ordinal * 10**12,
                "reps": reps,
                "engine": {
                    "mode": mode,
                    "model_generation": 9,
                    "slab_reads": attempts,
                    "slab_peak_qd": 0 if arm == "A" else 4,
                    "completed_fetch_ms": [] if arm == "A" else
                        ([10.0, 10.2, 9.8] if arm == "B" else [8.5, 8.6, 8.4]),
                    "telemetry": {
                        "attempts": attempts,
                        "sha_successes": sha_success,
                        "sha_failures": 0,
                        "ready": ready,
                        "late": late,
                        "stale": 0,
                        "fallback": fallback,
                        "copies": copies,
                        "validated_bytes": validated,
                        "copied_bytes": copied,
                        "publications": copies,
                        "current_ready": 30 if arm == "C" else 0,
                        "read_ns": attempts * 100,
                        "sha_ns": attempts * 50,
                        "wait_ns": attempts * 10 if arm == "C" else 0,
                        "copy_ns": copies * 20,
                    },
                },
                "safety": {
                    "minimum_available_gib": 24.0,
                    "swap_bytes": 0,
                    "oom_events": 0,
                    "xid": False,
                    "survivors": [],
                },
                "artifact_sha256": {
                    name: hashlib.sha256(name.encode()).hexdigest()
                    for name in (
                        "server.log", "result.json", "nvme-inflight.log",
                        "safety.main.log", "safety.samples.log",
                        "safety.kernel.log", "safety.cmd.log",
                    )
                },
            })
        source_hash = campaign._canonical_object_sha256({
            "patch_sha256": campaign.sha256_file(
                ROOT / "results/glm52-gates/harness/"
                "ds4-expert-slab-prefetch-sha-pipeline.patch"
            ),
            "state_header_sha256": campaign.sha256_file(HEADER),
        })
        freeze = {
            "schema_version": 1,
            "candidate_commit": commit,
            "binary_sha256": binary,
            "quality_binary_sha256": quality_binary,
            "source_sha256": source_hash,
            "scorer_sha256": campaign.sha256_file(CAMPAIGN_PATH),
            "tests_sha256_by_path": campaign.sha_prefetch_test_hashes(),
            "frozen_epoch_s": 100,
        }
        manifest = {
            "schema_version": 2,
            "candidate_commit": commit,
            "binary_sha256": binary,
            "quality_binary_sha256": quality_binary,
            "model_generation": 9,
            "configuration_by_arm": environments,
            "configuration_sha256_by_arm": configs,
            "fixture_sha256": fixture,
            "access_stream_sha256": access,
            "campaign_started_monotonic_ns": start_ns - 1,
            "campaign_finished_monotonic_ns": start_ns + 15 * 10**12,
            "freeze_sha256": campaign._canonical_object_sha256(freeze),
            "randomness": randomness,
        }
        rows = [
            {
                "case_id": f"case-{index:03d}",
                "target_tokens": 1,
                "total_nll": 0.4515,
                "top1_matches": int(index < 83),
            }
            for index in range(100)
        ]
        quality_attempts = [
            {
                "schema_version": 1,
                "arm": arm,
                "mode": {"A": "off", "B": "demand_sha", "C": "prefetch_sha"}[arm],
                "candidate_commit": commit,
                "binary_sha256": binary,
                "configuration_sha256": configs[arm],
                "fixture_sha256": fixture,
                "quality_fixture_content_sha256":
                    campaign.QUALITY_FIXTURE_CONTENT_SHA256,
                "ordered_case_ids": [row["case_id"] for row in rows],
                "output_sha256": hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()
                ).hexdigest(),
                "rows": copy.deepcopy(rows),
            }
            for arm in "ABC"
        ]
        return records, manifest, freeze, quality_attempts

    @staticmethod
    def score(campaign, evidence):
        records, manifest, freeze, quality_attempts = evidence
        return campaign.score_sha_prefetch_campaign(
            records, manifest, freeze=freeze, quality_attempts=quality_attempts
        )

    def test_executable_slot_state_and_corruption_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "state-test"
            compiled = subprocess.run(
                ["g++", "-std=c++17", "-pthread", "-O1", "-g",
                 "-I", str(HEADER.parent), str(CPP_TEST), "-o", str(binary)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run(
                [str(binary)], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_engine_uses_state_machine_as_sole_transition_authority(self):
        engine = Path("/tmp/glm52-rung0-prefetch-candidate/ds4_cuda.cu")
        self.assertTrue(engine.is_file(), "frozen compiled engine source is absent")
        source = engine.read_text(encoding="utf-8")
        self.assertTrue(
            '#include "ds4_slab_prefetch_state.h"' in source,
            "engine does not include the shared prefetch state authority",
        )
        required_calls = (
            "->issue(", "->complete_read(", "->claim(", "->copy_sync(",
            "->discard_ready(", "ds4_pf_cleanup();",
        )
        for call in required_calls:
            self.assertIn(call, source)
        # Manual slot-state writes would allow the engine to bypass SHA and
        # lease ordering even when the shared machine exists.
        self.assertNotRegex(source, r"g_pf\.slots\[[^]]+\]\.state\s*=")

    def test_three_arm_scorer_is_frozen_before_async_implementation(self):
        campaign = self.load_campaign()
        self.assertTrue(callable(campaign.score_sha_prefetch_campaign))
        self.assertEqual(
            campaign.sha_prefetch_schedule(False),
            ((0, 0, "A"), (0, 1, "B"), (0, 2, "C"),
             (1, 0, "B"), (1, 1, "C"), (1, 2, "A"),
             (2, 0, "C"), (2, 1, "A"), (2, 2, "B"),
             (3, 0, "C"), (3, 1, "B"), (3, 2, "A"),
             (4, 0, "A"), (4, 1, "C"), (4, 2, "B")),
        )
        evidence = self.passing_campaign(campaign)
        scored = self.score(campaign, evidence)
        self.assertEqual(scored["verdict"], "PASS")
        self.assertLessEqual(scored["probe_completed_fetch_ratio"], 0.90)
        self.assertEqual(
            set(scored["decode_ratio_lower_95_by_comparator_and_clock"]),
            {"C/A:client_wall", "C/A:raw_token", "C/B:client_wall", "C/B:raw_token"},
        )

    def test_prefetch_flags_are_part_of_the_observed_environment(self):
        campaign = self.load_campaign()
        self.assertTrue(
            {
                "DS4_GLM_PREFETCH",
                "DS4_CUDA_EXPERT_SLAB_PREFETCH_SHA",
                "DS4_GLM_PREFETCH_THREADS",
            }.issubset(campaign.PROVENANCE_NAMES)
        )

    def test_scorer_does_not_accept_caller_selected_schedule_or_manifest_quality(self):
        campaign = self.load_campaign()
        parameters = inspect.signature(
            campaign.score_sha_prefetch_campaign
        ).parameters
        self.assertNotIn("schedule_flip", parameters)
        self.assertIn("freeze", parameters)
        self.assertIn("quality_attempts", parameters)

    def test_production_campaign_uses_existing_containment_and_raw_producers(self):
        campaign = self.load_campaign()
        for name in (
            "run_sha_prefetch_campaign",
            "run_sha_prefetch_quality_campaign",
            "score_sha_prefetch_directory",
        ):
            self.assertTrue(callable(getattr(campaign, name, None)), name)
        run_source = inspect.getsource(campaign.run_sha_prefetch_campaign)
        quality_source = inspect.getsource(
            campaign.run_sha_prefetch_quality_campaign
        )
        self.assertIn("CGROUP_RUNNER", run_source)
        self.assertIn('"sha-prefetch-arm"', run_source)
        self.assertIn("_copy_and_parse_arm_safety", run_source)
        self.assertIn(
            "parse_safety_logs",
            inspect.getsource(campaign._copy_and_parse_arm_safety),
        )
        self.assertIn('"raw.jsonl"', run_source)
        self.assertIn("CGROUP_RUNNER", quality_source)
        self.assertIn('"sha-prefetch-quality-arm"', quality_source)
        self.assertIn('"quality-raw.jsonl"', quality_source)

    def test_sha_prefetch_cli_exposes_run_quality_and_score_without_flip(self):
        campaign = self.load_campaign()
        parser_source = inspect.getsource(campaign.parse_cli)
        for command in (
            'add_parser("sha-prefetch-run")',
            'add_parser("sha-prefetch-quality")',
            'add_parser("sha-prefetch-score")',
        ):
            self.assertIn(command, parser_source)
        self.assertNotIn("--schedule-flip", parser_source)

    def test_first_retained_request_is_the_true_cold_observation(self):
        campaign = self.load_campaign()
        arm_source = inspect.getsource(campaign.execute_arm)
        normalize_source = inspect.getsource(campaign.normalize_sha_prefetch_reps)
        self.assertIn('"--warmup", "0"', arm_source)
        self.assertIn('"request_ordinal": ordinal', normalize_source)
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        records[0]["reps"][0]["request_ordinal"] = 1
        with self.assertRaises(ValueError):
            campaign.score_sha_prefetch_campaign(
                records, manifest, freeze=freeze, quality_attempts=quality
            )

    def test_quality_path_binds_complete_official_fixture_before_and_after(self):
        campaign = self.load_campaign()
        source = inspect.getsource(campaign.run_sha_prefetch_quality_campaign)
        scorer = inspect.getsource(campaign.score_sha_prefetch_directory)
        self.assertGreaterEqual(source.count("content_complete_fixture_sha256"), 2)
        self.assertIn("QUALITY_FIXTURE_CONTENT_SHA256", source)
        self.assertIn("ordered_case_ids", source)
        self.assertIn("fixture_content_sha256", scorer)
        self.assertIn("parse_quality_tsv", scorer)

    def test_bounded_probe_must_pass_before_full_campaign(self):
        campaign = self.load_campaign()
        self.assertTrue(callable(getattr(campaign, "run_sha_prefetch_probe", None)))
        run_source = inspect.getsource(campaign.run_sha_prefetch_campaign)
        cli_source = inspect.getsource(campaign.parse_cli)
        self.assertIn('add_parser("sha-prefetch-probe")', cli_source)
        self.assertIn("verified_sha_prefetch_probe_receipt", run_source)
        self.assertLess(
            run_source.index("verified_sha_prefetch_probe_receipt"),
            run_source.index("out.mkdir"),
        )

    def test_probe_receipt_rejects_stale_or_rewritten_evidence(self):
        campaign = self.load_campaign()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            probe_rows = [
                {"arm": "B", "access_stream_sha256": "4" * 64,
                 "engine": {"completed_fetch_ms": [10.0, 10.0, 10.0]}},
                {"arm": "C", "access_stream_sha256": "4" * 64,
                 "engine": {"completed_fetch_ms": [8.0, 8.0, 8.0]}},
            ]
            raw.write_text("\n".join(json.dumps(row) for row in probe_rows) + "\n",
                           encoding="utf-8")
            randomness = {"round": 1}
            receipt = {
                "schema_version": 1, "verdict": "PASS",
                "candidate_commit": "1" * 40, "binary_sha256": "2" * 64,
                "freeze_sha256": "3" * 64, "randomness": randomness,
                "raw_sha256": campaign.sha256_file(raw), "arm_count": 2,
                "access_stream_sha256": "4" * 64,
                "completed_fetch_ratio": 0.8,
                "model_generation": 9,
            }
            path = root / "probe-receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(
                campaign, "derive_sha_prefetch_record_from_artifacts"
            ):
                campaign.verified_sha_prefetch_probe_receipt(
                    path, "1" * 40, "2" * 64, "3" * 64, randomness, 9
                )
            for field, value in (
                ("verdict", "FAIL"), ("binary_sha256", "5" * 64),
                ("completed_fetch_ratio", 0.91),
            ):
                changed = copy.deepcopy(receipt)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError), mock.patch.object(
                    campaign, "derive_sha_prefetch_record_from_artifacts"
                ):
                    campaign.verified_sha_prefetch_probe_receipt(
                        path, "1" * 40, "2" * 64, "3" * 64, randomness, 9
                    )
            path.write_text(json.dumps(receipt), encoding="utf-8")
            raw.write_text('{"fabricated":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError), mock.patch.object(
                campaign, "derive_sha_prefetch_record_from_artifacts"
            ):
                campaign.verified_sha_prefetch_probe_receipt(
                    path, "1" * 40, "2" * 64, "3" * 64, randomness, 9
                )

    def test_arm_artifact_map_changes_when_any_witness_changes(self):
        campaign = self.load_campaign()
        names = (
            "server.log", "result.json", "nvme-inflight.log",
            "safety.main.log", "safety.samples.log", "safety.kernel.log",
            "safety.cmd.log",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in names:
                (root / name).write_text(name, encoding="utf-8")
            before = campaign.sha_prefetch_artifact_hashes(root)
            for name in names:
                with self.subTest(name=name):
                    path = root / name
                    original = path.read_text(encoding="utf-8")
                    path.write_text(original + "-changed", encoding="utf-8")
                    self.assertNotEqual(
                        campaign.sha_prefetch_artifact_hashes(root), before
                    )
                    path.write_text(original, encoding="utf-8")

    def test_scorer_consumes_the_actual_safety_parser_schema(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        canonical = {
            "minimum_available_gib": 24.0,
            "cgroup_high_events": 0,
            "cgroup_max_events": 0,
            "cgroup_oom_events": 0,
            "cgroup_swap_bytes": 0,
            "xid": False,
            "survivors": [],
            "failures": [],
        }
        for record in records:
            record["safety"] = copy.deepcopy(canonical)
        self.assertEqual(
            campaign.score_sha_prefetch_campaign(
                records, manifest, freeze=freeze, quality_attempts=quality
            )["verdict"],
            "PASS",
        )

    def test_duplicate_authentic_artifact_bundle_is_rejected(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        records[1]["artifact_sha256"] = copy.deepcopy(
            records[0]["artifact_sha256"]
        )
        with self.assertRaises(ValueError):
            campaign.score_sha_prefetch_campaign(
                records, manifest, freeze=freeze, quality_attempts=quality
            )

    def test_external_nvme_trace_rejects_qd1_qd9_and_reordering(self):
        campaign = self.load_campaign()
        good = [
            "META read_before=10 read_after=1000010 start_ns=1000 end_ns=5000",
            "1000 0", "2000 4", "3000 8", "4000 2", "5000 0",
        ]
        parsed = campaign.parse_nvme_inflight_log(
            "\n".join(good), require_ring_qd=True
        )
        self.assertEqual(parsed["peak_read_qd"], 8)
        for mutation in (
            [line.replace("2000 4", "2000 1").replace("3000 8", "3000 1")
             for line in good],
            [line.replace("3000 8", "3000 9") for line in good],
            [good[0], good[2], good[1], *good[3:]],
        ):
            with self.assertRaises(ValueError):
                campaign.parse_nvme_inflight_log(
                    "\n".join(mutation), require_ring_qd=True
                )

    def test_score_reauthenticates_randomness_and_rederives_raw_artifacts(self):
        campaign = self.load_campaign()
        score_source = inspect.getsource(campaign.score_sha_prefetch_directory)
        self.assertIn("authenticate_confirmation", score_source)
        self.assertIn("derive_sha_prefetch_record_from_artifacts", score_source)
        self.assertIn(
            "artifact_sha256",
            inspect.getsource(campaign.derive_sha_prefetch_record_from_artifacts),
        )

    def test_freeze_binds_every_declared_acceptance_test(self):
        campaign = self.load_campaign()
        expected = {
            "scripts/tests/test_glm_rung0_sha_prefetch.py",
            "scripts/tests/test_glm_expert_slab_source.py",
            "scripts/tests/cpp/test_ds4_slab_prefetch_state.cpp",
        }
        self.assertEqual(set(campaign.sha_prefetch_test_hashes()), expected)
        source = inspect.getsource(campaign.validate_sha_prefetch_freeze)
        self.assertIn("tests_sha256_by_path", source)

    def test_scorer_rejects_arbitrary_configuration_digests(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        # These are syntactically valid and distinct, but are not hashes of the
        # exact A=off/B=demand-SHA/C=prefetch-SHA environment maps.
        manifest["configuration_sha256_by_arm"] = {
            "A": "d" * 64, "B": "e" * 64, "C": "f" * 64
        }
        with self.assertRaises((TypeError, ValueError)):
            campaign.score_sha_prefetch_campaign(
                records, manifest, freeze=freeze, quality_attempts=quality
            )

    def test_scorer_rejects_zero_sha_coverage_and_count_byte_fabrication(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        zero = copy.deepcopy(records)
        for row in zero:
            if row["arm"] in {"B", "C"}:
                row["engine"]["telemetry"] = {
                    name: 0 for name in row["engine"]["telemetry"]
                }
        with self.assertRaises((TypeError, ValueError)):
            campaign.score_sha_prefetch_campaign(
                zero, manifest, freeze=freeze, quality_attempts=quality
            )

        mismatch = copy.deepcopy(records)
        target = next(row for row in mismatch if row["arm"] == "C")
        target["engine"]["telemetry"]["copied_bytes"] = 0
        with self.assertRaises((TypeError, ValueError)):
            campaign.score_sha_prefetch_campaign(
                mismatch, manifest, freeze=freeze, quality_attempts=quality
            )

    def test_raw_log_parser_requires_unique_per_attempt_auth_records(self):
        campaign = self.load_campaign()
        payload = campaign.EXPERT_RECORD_PAYLOAD_BYTES
        lines = [
            (
                f"SLABAUTH mode=demand_sha generation=9 attempt={index} "
                f"key={index} submit_ns={1000 + index * 100} "
                f"complete_ns={1050 + index * 100} "
                f"payload_bytes={payload} ok=1"
            )
            for index in range(1, 5)
        ]
        lines.append(
            "LOADPROF L3 uniq=8 hits=4 miss=4 hit_ms=1.00 fetch_ms=8.00 "
            "fill_ms=1.00 total_ms=10.00 slab_mode=on slab_reads=4 "
            f"slab_bytes={4 * payload} slab_actual_bytes={4 * payload} "
            "slab_peak_qd=4 slab_io_ms=7.000 slab_validation_ms=1.000 "
            "slab_copy_ms=1.000"
        )
        parsed = campaign.parse_sha_prefetch_engine_log(
            "\n".join(lines), "demand_sha", model_generation=9
        )
        self.assertEqual(parsed["completed_fetch_ms"], [0.00005] * 4)
        self.assertEqual(parsed["telemetry"]["attempts"], 4)
        for mutation in (
            lines[:-1],
            [*lines, lines[0]],
            [line.replace("ok=1", "ok=0") if index == 0 else line
             for index, line in enumerate(lines)],
        ):
            with self.assertRaises(ValueError):
                campaign.parse_sha_prefetch_engine_log(
                    "\n".join(mutation), "demand_sha", model_generation=9
                )

    def test_prefetch_log_parser_binds_raw_auth_to_terminal_telemetry(self):
        campaign = self.load_campaign()
        payload = campaign.EXPERT_RECORD_PAYLOAD_BYTES
        auth = [
            (
                f"SLABAUTH mode=prefetch_sha generation=9 attempt={index} "
                f"key={100 + index} submit_ns={1000 + index * 100} "
                f"complete_ns={1080 + index * 100} "
                f"payload_bytes={payload} ok=1"
            )
            for index in range(1, 5)
        ]
        load = (
            "LOADPROF L3 uniq=8 hits=4 miss=4 hit_ms=1.00 fetch_ms=8.00 "
            "fill_ms=1.00 total_ms=10.00 slab_mode=on slab_reads=2 "
            f"slab_bytes={2 * payload} slab_actual_bytes={2 * payload} "
            "slab_peak_qd=2 slab_io_ms=7.000 slab_validation_ms=0.500 "
            "slab_copy_ms=0.400"
        )
        marker = (
            "PREFETCHSHA mode=prefetch_sha generation=9 attempts=4 "
            "sha_successes=4 sha_failures=0 ready=3 late=1 stale=0 "
            f"fallback=1 copies=2 validated_bytes={4 * payload} "
            f"copied_bytes={2 * payload} publications=2 read_ns=400 "
            "sha_ns=200 wait_ns=100 copy_ns=80 current_ready=1 peak_qd=4"
        )
        parsed = campaign.parse_sha_prefetch_engine_log(
            "\n".join([*auth, load, marker]),
            "prefetch_sha",
            model_generation=9,
        )
        self.assertEqual(parsed["telemetry"]["attempts"], 4)
        self.assertEqual(parsed["telemetry"]["validated_bytes"], 4 * payload)
        self.assertEqual(parsed["slab_peak_qd"], 4)
        for bad_marker in (
            marker.replace("attempts=4", "attempts=3"),
            marker.replace("sha_successes=4", "sha_successes=3"),
            marker.replace("sha_failures=0", "sha_failures=1"),
        ):
            with self.subTest(marker=bad_marker):
                with self.assertRaises(ValueError):
                    campaign.parse_sha_prefetch_engine_log(
                        "\n".join([*auth, load, bad_marker]),
                        "prefetch_sha",
                        model_generation=9,
                    )

    def test_three_arm_scorer_rejects_malformed_or_partial_evidence(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        mutations = {
            "missing arm": lambda r, m: r.pop(),
            "duplicate arm": lambda r, m: r.__setitem__(1, copy.deepcopy(r[0])),
            "reordered arm": lambda r, m: r.__setitem__(slice(0, 2), [r[1], r[0]]),
            "stale binary": lambda r, m: r[3].__setitem__("binary_sha256", "9" * 64),
            "fixture drift": lambda r, m: r[4].__setitem__("fixture_sha256", "8" * 64),
            "access drift": lambda r, m: r[5].__setitem__("access_stream_sha256", "7" * 64),
            "short output": lambda r, m: (
                r[6]["reps"][0].__setitem__("completion_tokens", 127),
                r[6]["reps"][0]["token_ids"].pop(),
                r[6]["reps"][0]["raw_token_timestamps_ns"].pop(),
                r[6]["reps"][0]["client_token_timestamps_ns"].pop(),
            ),
            "nan fetch": lambda r, m: r[1]["engine"]["completed_fetch_ms"].__setitem__(0, math.nan),
            "telemetry mismatch": lambda r, m: r[2]["engine"]["telemetry"].__setitem__("copies", 101),
            "historical row": lambda r, m: r[7].__setitem__(
                "recorded_monotonic_ns", m["campaign_started_monotonic_ns"] - 1
            ),
            "output mismatch": lambda r, m: r[8]["reps"][1].__setitem__(
                "generated_bytes_sha256", "6" * 64
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed_records = copy.deepcopy(records)
                changed_manifest = copy.deepcopy(manifest)
                mutate(changed_records, changed_manifest)
                with self.assertRaises((TypeError, ValueError)):
                    campaign.score_sha_prefetch_campaign(
                        changed_records, changed_manifest,
                        freeze=freeze, quality_attempts=quality,
                    )

    def test_three_arm_scorer_requires_every_comparator_clock_and_ttft_state(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        cases = {}
        # C loses only to B on the independent client clock.
        client_loss = copy.deepcopy(records)
        for row in client_loss:
            if row["arm"] == "C":
                for rep in row["reps"]:
                    first = rep["client_first_token_ns"]
                    rep["client_token_timestamps_ns"] = [
                        first + index * 92_000_000 for index in range(128)
                    ]
                    rep["client_last_token_ns"] = rep["client_token_timestamps_ns"][-1]
        cases["one clock"] = client_loss
        for phase in ("cold", "warm"):
            changed = copy.deepcopy(records)
            for row in changed:
                if row["arm"] == "C":
                    for rep in row["reps"]:
                        if rep["phase"] == phase:
                            first = rep["client_request_started_ns"] + 3_000_000_000
                            delta = first - rep["client_first_token_ns"]
                            rep["client_first_token_ns"] = first
                            rep["client_last_token_ns"] += delta
                            rep["client_token_timestamps_ns"] = [x + delta for x in rep["client_token_timestamps_ns"]]
                            rep["ttft_seconds"] = 3.0
            cases[f"{phase} TTFT"] = changed
        for label, changed in cases.items():
            with self.subTest(label=label):
                scored = campaign.score_sha_prefetch_campaign(
                    changed, manifest, freeze=freeze, quality_attempts=quality
                )
                self.assertEqual(scored["verdict"], "FAIL")

    def test_three_arm_scorer_reconciles_every_telemetry_class(self):
        campaign = self.load_campaign()
        records, manifest, freeze, quality = self.passing_campaign(campaign)
        mutations = {
            "attempts": ("attempts", 101),
            "sha successes": ("sha_successes", 99),
            "sha failures": ("sha_failures", 1),
            "ready": ("ready", 99),
            "late": ("late", 21),
            "stale": ("stale", 1),
            "fallback": ("fallback", 21),
            "copies": ("copies", 101),
            "validated bytes": ("validated_bytes", 1),
            "copied bytes": ("copied_bytes", 25_601),
            "publications": ("publications", 71),
            "current ready": ("current_ready", 31),
            "read timer": ("read_ns", -1),
            "sha timer": ("sha_ns", -1),
            "wait timer": ("wait_ns", -1),
            "copy timer": ("copy_ns", -1),
        }
        c_index = next(i for i, row in enumerate(records) if row["arm"] == "C")
        for label, (field, value) in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(records)
                changed[c_index]["engine"]["telemetry"][field] = value
                with self.assertRaises(ValueError):
                    campaign.score_sha_prefetch_campaign(
                        changed, manifest, freeze=freeze, quality_attempts=quality
                    )


if __name__ == "__main__":
    unittest.main()
