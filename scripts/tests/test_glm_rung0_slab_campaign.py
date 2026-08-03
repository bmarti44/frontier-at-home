#!/usr/bin/env python3
"""Contracts for the minimal GLM Rung 0.1 slab campaign."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/70_glm_rung0_slab_campaign.py"
SPEC = importlib.util.spec_from_file_location("glm_rung0_slab_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class Rung0SlabCampaignTests(unittest.TestCase):
    @staticmethod
    def passing_confirmation(
        candidate_commit: str, binary_sha256: str, quality_binary_sha256: str
    ):
        signature = "22" * 96
        randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        seed = CAMPAIGN.confirmation_seed(
            randomness, candidate_commit, binary_sha256, quality_binary_sha256
        )
        return {
            "round": 11,
            "randomness": randomness,
            "signature": signature,
            "chain_hash": "a" * 64,
            "published_epoch_s": 1300,
            "seed_sha256": seed,
            "flip": bool(int(seed[:2], 16) & 1),
        }

    def passing_records(self, *, flip: bool = False):
        records = []
        for block, sequence, arm in CAMPAIGN.arm_schedule(flip=flip):
            mode = "off" if arm == "A" else "on"
            step = 100_000_000 if mode == "off" else 80_000_000
            reps = []
            for rep in range(2):
                ttft_s = 1.0 if mode == "off" else 1.02
                first_client_ns = 2_000_000_000
                last_client_ns = first_client_ns + 127 * step
                raw_step = step + 1_000_000
                raw_elapsed_ns = 127 * raw_step
                reps.append(
                    {
                        "valid": True,
                        "request_sha256": "d" * 64,
                        "generated_reasoning_sha256": "e" * 64,
                        "generated_content_sha256": "f" * 64,
                        # SSE content fragments and a re-tokenized pair of client
                        # channels are not one-to-one with raw generated tokens.
                        "client_completion_tokens": 127,
                        "completion_tokens": 128,
                        "server_completion_tokens": 128,
                        "event_completion_tokens": 125,
                        "timing_source": "server_raw_token_log",
                        "sse_token_timestamps_ns": [
                            first_client_ns
                            + index * (last_client_ns - first_client_ns) // 124
                            for index in range(125)
                        ],
                        "token_timestamps_ns": [
                            900_000_000 + index * raw_step for index in range(128)
                        ],
                        "token_ids": list(range(128)),
                        "client_request_started_ns": first_client_ns
                        - int(ttft_s * 1_000_000_000),
                        "client_first_content_ns": first_client_ns,
                        "client_last_content_ns": last_client_ns,
                        "raw_client_timing_ratio": raw_elapsed_ns
                        / (last_client_ns - first_client_ns)
                        + 2e-12,
                        "ttft_s": ttft_s,
                        "client_prompt_tokens": 256,
                    }
                )
            records.append(
                {
                    "schema_version": 1,
                    "block": block,
                    "sequence": sequence,
                    "arm": arm,
                    "mode": mode,
                    "server_instance_id": f"server-{block}-{sequence}",
                    "binary_sha256": "a" * 64,
                    "configuration_sha256": CAMPAIGN.canonical_environment_sha256(
                        CAMPAIGN.canonical_engine_environment(mode)
                    ),
                    "fixture_sha256": "d" * 64,
                    "suite_valid": True,
                    "reps": reps,
                    "engine": {
                        "slab_mode": mode,
                        "slab_reads": 0 if mode == "off" else 20,
                        "slab_peak_qd": 0 if mode == "off" else 8,
                        "access_stream_sha256": "1" * 64,
                        "arena_pin_ok": True,
                        "trace_lines": 0,
                    },
                    "external_io": {
                        "read_bytes_delta": (
                            1000 if mode == "off" else 401_104_554_048
                        ),
                        "elapsed_seconds": 1.0,
                        "peak_read_qd": 1 if mode == "off" else 8,
                        "sample_count": 20,
                    },
                    "safety": {
                        "minimum_available_gib": 18.0,
                        "cgroup_high_events": 0,
                        "cgroup_max_events": 0,
                        "cgroup_oom_events": 0,
                        "cgroup_swap_bytes": 0,
                        "xid": False,
                        "survivors": [],
                        "failures": [],
                    },
                }
            )
        return records

    @staticmethod
    def passing_nll():
        return {
            "case_count": 100,
            "token_weighted_delta_nll": 0.0,
            "top1_loss_pp": 0.0,
            "deterministic": True,
        }

    def test_schedule_is_five_fresh_abba_baab_blocks(self):
        schedule = CAMPAIGN.arm_schedule()
        self.assertEqual(len(schedule), 20)
        for block in range(5):
            group = [row for row in schedule if row[0] == block]
            self.assertEqual([row[1] for row in group], list(range(4)))
            self.assertEqual(
                "".join(row[2] for row in group),
                "ABBA" if block % 2 == 0 else "BAAB",
            )
        flipped = CAMPAIGN.arm_schedule(flip=True)
        self.assertEqual("".join(row[2] for row in flipped[:4]), "BAAB")
        self.assertNotEqual(schedule, flipped)

    def test_public_randomness_seed_is_bound_to_both_frozen_binaries(self):
        first = CAMPAIGN.confirmation_seed(
            "1" * 64, "2" * 40, "3" * 64, "4" * 64
        )
        self.assertEqual(first, CAMPAIGN.confirmation_seed(
            "1" * 64, "2" * 40, "3" * 64, "4" * 64
        ))
        self.assertNotEqual(
            first,
            CAMPAIGN.confirmation_seed("1" * 64, "2" * 40, "3" * 64, "5" * 64),
        )

    def test_public_randomness_must_be_published_after_freeze(self):
        record = {
            "round": 11,
            "randomness": "1" * 64,
            "signature": "2" * 192,
            "obtained_at": "2026-08-01T00:00:00+00:00",
        }
        relay = CAMPAIGN.subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=(
                b'{"genesis_time":1000,"period":30,"hash":"'
                + b"a" * 64 + b'"}'
            ),
            stderr=b"",
        )
        with (
            mock.patch.object(CAMPAIGN, "_drand_record", return_value=record),
            mock.patch.object(
                CAMPAIGN,
                "_authenticate_drand",
                side_effect=[
                    record,
                    {**record, "obtained_at": "2026-08-01T00:00:01+00:00"},
                    {**record, "obtained_at": "2026-08-01T00:00:02+00:00"},
                ],
            ),
            mock.patch.object(CAMPAIGN.subprocess, "run", return_value=relay),
        ):
            accepted = CAMPAIGN.authenticate_confirmation(
                Path("/unused"), "2" * 40, "3" * 64, "4" * 64, 1299,
            )
            self.assertEqual(accepted["published_epoch_s"], 1300)
            repeated = CAMPAIGN.authenticate_confirmation(
                Path("/unused"), "2" * 40, "3" * 64, "4" * 64, 1299,
            )
            self.assertEqual(accepted, repeated)
            self.assertNotIn("obtained_at", accepted)
            with self.assertRaisesRegex(ValueError, "before the frozen candidate"):
                CAMPAIGN.authenticate_confirmation(
                    Path("/unused"), "2" * 40, "3" * 64, "4" * 64, 1300,
                )

    def test_timed_arms_differ_only_by_slab_identity(self):
        off = CAMPAIGN.canonical_engine_environment("off")
        on = CAMPAIGN.canonical_engine_environment("on")
        slab = {
            "DS4_CUDA_EXPERT_SLAB_PATH",
            "DS4_CUDA_EXPERT_SLAB_SHA256",
            "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256",
        }
        self.assertEqual(set(on) - set(off), slab)
        self.assertEqual(
            {key: value for key, value in on.items() if key not in slab}, off
        )
        self.assertEqual(off["DS4_CUDA_FETCH_THREADS"], "8")
        self.assertEqual(off["DS4_LOCK_FILE"], CAMPAIGN.INSTANCE_LOCK)
        self.assertTrue(CAMPAIGN.INSTANCE_LOCK.startswith("/run/user/1000/"))
        self.assertNotIn("DS4_CUDA_EXPERT_SLAB_TRACE", on)
        self.assertNotEqual(
            CAMPAIGN.canonical_environment_sha256(off),
            CAMPAIGN.canonical_environment_sha256(on),
        )

    def test_engine_log_parser_requires_resolved_positive_slab_mode(self):
        log = "\n".join(
            (
                "ds4: expert slab full-model identity verified via O_DIRECT "
                "bytes=211075856448",
                "ds4: expert slab full-sidecar identity verified via O_DIRECT "
                "bytes=190028697600",
                "ds4: CUDA contiguous expert slab enabled records=19456 path=/f ",
                "LOADPROF L3 uniq=8 hits=1 miss=7 hit_ms=1 fetch_ms=2 "
                "fill_ms=1 total_ms=4 slab_mode=on slab_reads=7 "
                "slab_bytes=70 slab_actual_bytes=72 slab_peak_qd=7 slab_io_ms=2",
                "ds4: expert-cache arena pin: ok (1.0 s)",
                "ds4: expert-cache window tag=models-get lookup_bytes=70 "
                "hit_bytes=10 stream_sha256=" + "1" * 64,
            )
        )
        parsed = CAMPAIGN.parse_engine_log(log, "on")
        self.assertEqual(parsed["slab_reads"], 7)
        self.assertEqual(parsed["slab_peak_qd"], 7)
        for broken in (
            log.replace("full-model identity verified", "model identity pending"),
            log.replace("full-sidecar identity verified", "sidecar identity pending"),
            log.replace("slab_reads=7", "slab_reads=0"),
            log.replace("slab_mode=on", "slab_mode=off"),
            log.replace("arena pin: ok", "arena pin: failed"),
            log + "\nSLABIO worker=1",
        ):
            with self.assertRaises(ValueError):
                CAMPAIGN.parse_engine_log(broken, "on")

    def test_off_arm_needs_resolved_zero_counters_not_unreachable_marker(self):
        log = "\n".join(
            (
                "LOADPROF L3 uniq=8 hits=1 miss=7 hit_ms=1 fetch_ms=2 "
                "fill_ms=1 total_ms=4 slab_mode=off slab_reads=0 "
                "slab_bytes=0 slab_actual_bytes=0 slab_peak_qd=0 slab_io_ms=0",
                "ds4: expert-cache arena pin: ok (1.0 s)",
                "ds4: expert-cache window tag=models-get lookup_bytes=70 "
                "hit_bytes=10 stream_sha256=" + "1" * 64,
            )
        )
        self.assertEqual(CAMPAIGN.parse_engine_log(log, "off")["slab_reads"], 0)
        with self.assertRaises(ValueError):
            CAMPAIGN.parse_engine_log(
                log + "\nds4: CUDA contiguous expert slab enabled records=19456", "off"
            )

    def test_safety_log_parser_requires_clean_external_evidence(self):
        main = "\n".join(
            (
                "candidate_binary_sha256=" + "a" * 64,
                "cgroup_final current_bytes=1 peak_bytes=2 swap_current_bytes=0 "
                "events=low 0,high 0,max 0,oom 0,oom_kill 0,",
                "executed candidate was verified alive at least once; no identity "
                "contradiction observed by the periodic sampler; actual cadence "
                "is recorded in samples.log; wrapper and descendant checks clean",
                "SAFE_RUN end rc=0 killed=no",
            )
        )
        samples = "\n".join(
            f"t mem_avail_kb={20 * 1048576} eng_rss_kb=1 read_bytes={index} "
            "cgroup_current_bytes=1 cgroup_peak_bytes=2 cgroup_swap_current_bytes=0"
            for index in range(3)
        )
        parsed = CAMPAIGN.parse_safety_logs(main, samples, "kernel clean")
        self.assertEqual(parsed["cgroup_high_events"], 0)
        self.assertGreaterEqual(parsed["minimum_available_gib"], 10)
        with self.assertRaises(ValueError):
            CAMPAIGN.parse_safety_logs(
                main.replace("high 0", "high 1"), samples, "kernel clean"
            )

    def test_external_io_summary_uses_completed_proc_bytes_and_block_qd(self):
        result = CAMPAIGN.summarize_external_io(
            [(1, 0), (2, 3), (3, 1)], 100, 1100, 2.0
        )
        self.assertEqual(result["read_bytes_delta"], 1000)
        self.assertEqual(result["peak_read_qd"], 3)
        self.assertEqual(result["sample_count"], 3)
        with self.assertRaises(ValueError):
            CAMPAIGN.summarize_external_io([(1, 0)], 100, 50, 2.0)

    def test_memory_envelope_is_derived_from_measured_non_arena_peak(self):
        envelope = CAMPAIGN.derive_memory_envelope(
            non_arena_peak_bytes=30 * 1024**3,
            non_arena_cgroup_peak_bytes=1 * 1024**3,
            host_total_bytes=120 * 1024**3,
        )
        self.assertEqual(envelope["arena_bytes"], 68_000_000_000)
        self.assertEqual(envelope["margin_bytes"], 4 * 1024**3)
        self.assertEqual(envelope["memory_high_gib"], 69)
        self.assertEqual(envelope["memory_max_gib"], 71)
        self.assertEqual(envelope["minimum_start_available_gib"], 116)
        self.assertGreaterEqual(
            envelope["memory_high_bytes"],
            envelope["non_arena_cgroup_peak_bytes"]
            + envelope["arena_bytes"]
            + envelope["margin_bytes"],
        )
        self.assertEqual(
            envelope["projected_physical_peak_bytes"],
            envelope["non_arena_peak_bytes"]
            + envelope["arena_bytes"]
            + envelope["margin_bytes"],
        )
        for runner in (
            CAMPAIGN.run_sha_prefetch_campaign,
            CAMPAIGN.run_sha_prefetch_quality_campaign,
        ):
            source = inspect.getsource(runner)
            self.assertIn("minimum_start_available_gib", source)
            self.assertNotIn('"GLM_SAFE_MIN_START_GIB": "110"', source)
        with self.assertRaises(ValueError):
            CAMPAIGN.derive_memory_envelope(
                non_arena_peak_bytes=35 * 1024**3,
                non_arena_cgroup_peak_bytes=1 * 1024**3,
                host_total_bytes=120 * 1024**3,
            )

    def test_memory_envelope_rederives_bound_raw_probe(self):
        binary_sha256 = "a" * 64
        candidate_commit = "b" * 40
        environment_sha256 = CAMPAIGN.observed_environment_sha256(
            CAMPAIGN.memory_probe_environment()
        )
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory)
            main = "\n".join((
                f"candidate_binary_sha256={binary_sha256}",
                "cmd: memory-probe-arm "
                f"--candidate-commit {candidate_commit}",
                "MemTotal:       125829120 kB",
                "MemAvailable:   120586240 kB",
                f"executed_environment_sha256={environment_sha256}",
                "executed_candidate_verified pid=1 start_ticks=1 path=/x "
                f"executed_binary_sha256={binary_sha256} device_inode=1:2",
                "cgroup_final current_bytes=1 peak_bytes=1073741824 "
                "swap_current_bytes=0 events=low 0,high 0,max 0,oom 0,oom_kill 0,",
                "executed candidate was verified alive at least once; no identity "
                "contradiction observed by the periodic sampler; actual cadence is "
                "recorded in samples.log; wrapper and descendant checks clean",
                "SAFE_RUN end rc=0 killed=no",
            ))
            samples = "\n".join(
                f"t{index} mem_avail_kb={110 * 1048576} "
                f"eng_rss_kb={9 * 1048576} read_bytes={index + 1} "
                "cgroup_current_bytes=1 cgroup_peak_bytes=1073741824 "
                "cgroup_swap_current_bytes=0"
                for index in range(3)
            )
            (probe / "safety.main.log").write_text(main, encoding="utf-8")
            (probe / "safety.samples.log").write_text(samples, encoding="utf-8")
            (probe / "safety.kernel.log").write_text("kernel clean\n", encoding="utf-8")
            (probe / "partial.json").write_text(
                '{"binary_sha256":"' + binary_sha256
                + '","candidate_commit":"' + candidate_commit
                + '","mode":"off","probe_environment_sha256":"'
                + environment_sha256 + '","schema_version":1}\n',
                encoding="utf-8",
            )
            partial = probe / "partial.json"
            details = partial.stat()
            main += (
                "\nfinal_artifact_verified path=" + str(partial)
                + " sha256=" + hashlib.sha256(partial.read_bytes()).hexdigest()
                + f" device_inode={details.st_dev}:{details.st_ino}:{details.st_size}"
            )
            for copied_name, receipt_name in (
                ("safety.samples.log", "samples.log"),
                ("safety.kernel.log", "kernel.log"),
            ):
                evidence_path = probe / copied_name
                main += (
                    f"\nsafety_artifact_verified name={receipt_name} sha256="
                    + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                    + f" size={evidence_path.stat().st_size}"
                )
            (probe / "safety.main.log").write_text(main, encoding="utf-8")
            evidence = CAMPAIGN.derive_memory_probe_evidence(
                probe, binary_sha256, candidate_commit
            )
            CAMPAIGN.require_memory_probe_evidence(
                probe, evidence, binary_sha256, candidate_commit
            )
            with self.assertRaises(ValueError):
                CAMPAIGN.require_memory_probe_evidence(
                    probe, evidence, binary_sha256, "c" * 40
                )
            partial.write_text(
                partial.read_text(encoding="utf-8").replace(
                    candidate_commit, "c" * 40
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                CAMPAIGN.derive_memory_probe_evidence(
                    probe, binary_sha256, "c" * 40
                )
            partial.write_text(
                partial.read_text(encoding="utf-8").replace(
                    "c" * 40, candidate_commit
                ),
                encoding="utf-8",
            )
            original_samples = (probe / "safety.samples.log").read_text(
                encoding="utf-8"
            )
            (probe / "safety.samples.log").write_text(
                samples.replace(str(110 * 1048576), str(100 * 1048576)),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                CAMPAIGN.derive_memory_probe_evidence(
                    probe, binary_sha256, candidate_commit
                )
            (probe / "safety.samples.log").write_text(
                original_samples, encoding="utf-8"
            )
            (probe / "safety.kernel.log").write_text(
                "different but still clean kernel log\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                CAMPAIGN.derive_memory_probe_evidence(
                    probe, binary_sha256, candidate_commit
                )
            with self.assertRaises(ValueError):
                CAMPAIGN.require_memory_probe_evidence(
                    probe, evidence, binary_sha256, candidate_commit
                )

    def test_full_quality_parser_and_comparator_require_exact_100_case_identity(self):
        header = "id\ttarget_tokens\tnll\ttarget_top1_correct\n"
        rows = "".join(
            f"case-{index:03d}\t4\t{index + 0.25}\t3\n" for index in range(100)
        )
        with tempfile.TemporaryDirectory() as directory:
            off = Path(directory) / "off.tsv"
            on = Path(directory) / "on.tsv"
            off.write_text(header + rows, encoding="utf-8")
            on.write_text(header + rows, encoding="utf-8")
            baseline = CAMPAIGN.parse_quality_tsv(off)
            candidate = CAMPAIGN.parse_quality_tsv(on)
            result = CAMPAIGN.compare_quality_rows(baseline, candidate)
            self.assertEqual(result["case_count"], 100)
            self.assertEqual(result["token_weighted_delta_nll"], 0.0)
            self.assertEqual(result["top1_loss_pp"], 0.0)
            self.assertTrue(result["deterministic"])
            on.write_text(
                (header + rows).replace("case-050\t4\t50.25", "case-050\t4\t50.2500001"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                CAMPAIGN.compare_quality_rows(
                    baseline, CAMPAIGN.parse_quality_tsv(on)
                )

    def test_official_fixture_manifest_uses_hash_prefixed_id_header(self):
        rows = "".join(
            f"case_{index:03d}\tp/{index}\tc/{index}\tr/{index}\n"
            for index in range(100)
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.tsv"
            manifest.write_text(
                "# id\tprompt_file\tcontinuation_file\tresponse_file\n" + rows,
                encoding="utf-8",
            )
            identifiers = CAMPAIGN.fixture_manifest_case_ids(manifest)
            self.assertEqual(len(identifiers), 100)
            self.assertEqual(identifiers[0], "case_000")

    def test_campaign_requires_a_bound_measured_memory_envelope(self):
        parsed = CAMPAIGN.parse_cli(
            [
                "run",
                "--tag", "test",
                "--candidate", "/home/bmarti44/.cache/glm52-test",
                "--quality-candidate", "/home/bmarti44/.cache/glm52-test-quality",
                "--candidate-commit", "a" * 40,
                "--binary-sha256", "b" * 64,
                "--quality-binary-sha256", "d" * 64,
                "--drand-json", "/tmp/drand.json",
                "--memory-envelope", "/tmp/envelope.json",
            ]
        )
        self.assertEqual(parsed.memory_envelope, Path("/tmp/envelope.json"))
        self.assertFalse(hasattr(parsed, "memory_high_gib"))

    def test_probe_is_cache_off_and_peak_rss_comes_from_external_samples(self):
        environment = CAMPAIGN.memory_probe_environment()
        self.assertEqual(environment["DS4_CUDA_EXPERT_CACHE_GB"], "0")
        self.assertEqual(environment["DS4_LOCK_FILE"], CAMPAIGN.INSTANCE_LOCK)
        self.assertNotIn("DS4_CUDA_EXPERT_SLAB_PATH", environment)
        samples = "\n".join(
            (
                "t mem_avail_kb=100 eng_rss_kb=10485760 read_bytes=0",
                "t mem_avail_kb=90 eng_rss_kb=31457280 read_bytes=1",
                "t mem_avail_kb=80 eng_rss_kb=20971520 read_bytes=2",
            )
        )
        self.assertEqual(CAMPAIGN.peak_engine_rss_bytes(samples), 30 * 1024**3)
        with self.assertRaises(ValueError):
            CAMPAIGN.peak_engine_rss_bytes("t mem_avail_kb=100 eng_rss_kb=0")
        charged = samples.replace(
            "read_bytes=1",
            f"read_bytes=1 cgroup_peak_bytes={35 * 1024**3}",
        )
        self.assertEqual(
            CAMPAIGN.measured_non_arena_peak_bytes(
                charged, "MemAvailable:       100 kB\n"
            ),
            35 * 1024**3,
        )
        unified = (
            samples.replace("mem_avail_kb=100", "mem_avail_kb=41943040")
            .replace("mem_avail_kb=90", "mem_avail_kb=20971520")
            .replace("mem_avail_kb=80", "mem_avail_kb=10485760")
        )
        self.assertEqual(
            CAMPAIGN.measured_non_arena_peak_bytes(
                unified, "MemAvailable:       41943040 kB\n"
            ),
            30 * 1024**3,
        )

    def test_single_request_probe_flushes_nonempty_access_window(self):
        source = inspect.getsource(CAMPAIGN.execute_memory_probe_arm)
        self.assertIn("flush_expert_cache_window(args.port)", source)

    def test_authoritative_plan_binds_current_frozen_candidate(self):
        plan = (ROOT / "results/glm52-gates/RUNG-PLAN.md").read_text(
            encoding="utf-8"
        )
        current = plan.split("Current Rung 0.1 candidate reconciliation:", 1)[1]
        current = current.split("The retired W8 branch", 1)[0]
        for expected in (
            "e637b6f1eaaf9fbc5f08874d5f2a28e5ac618004",
            "5a7caa3e7fded039797e6a0158dd4687b932d3b3c5f225c05ac7a656021fbd1a",
            "3f4f6d197a37369ec20413e7ee77b87508803511106d0250ecc45671ac01e349",
            "/tmp/glm52-score-official",
            "glm52-rung0-e637-mem-20260802a",
            "R0-slab-canary-attempts-2026-08-02.json",
        ):
            self.assertIn(expected, current)
        self.assertNotIn("afdf7dcf04a9c46710eeb61d1e7df623a33051cd", current)

    def test_quality_command_uses_frozen_scorer_and_full_manifest(self):
        command = CAMPAIGN.quality_command(
            Path("/home/bmarti44/.cache/glm52-test-quality/ds4-server"),
            CAMPAIGN.MODEL_PATH,
            Path("/home/bmarti44/.cache/glm52-test/manifest.tsv"),
            Path("/home/bmarti44/.local/state/glm52-rung0-test/quality-off.tsv"),
        )
        self.assertEqual(
            command[-3:],
            ["--ssd-streaming", "--ssd-streaming-cache-experts", "40GB"],
        )
        self.assertEqual(command[4], "8192")
        self.assertEqual(command[1], CAMPAIGN.MODEL_PATH)
        self.assertEqual(command[2].name, "manifest.tsv")
        self.assertEqual(command[3].name, "quality-off.tsv")

    def test_quality_schedule_proves_each_arm_deterministic_with_itself(self):
        self.assertEqual(CAMPAIGN.quality_schedule(), ("A", "B", "B", "A"))
        attempts = []
        for arm in CAMPAIGN.quality_schedule():
            attempts.append(
                {
                    "arm": arm,
                    "mode": "off" if arm == "A" else "on",
                    "rows": [
                        {
                            "case_id": f"case-{index:03d}",
                            "tokens": 4,
                            "nll_sum": index + 0.25,
                            "top1_correct": 3,
                        }
                        for index in range(100)
                    ],
                    "output_sha256": "8" * 64,
                    "configuration_sha256": CAMPAIGN.canonical_environment_sha256(
                        CAMPAIGN.canonical_engine_environment(
                            "off" if arm == "A" else "on"
                        )
                    ),
                    "engine": {
                        "slab_mode": "off" if arm == "A" else "on",
                        "slab_reads": 0 if arm == "A" else 20,
                        "slab_peak_qd": 0 if arm == "A" else 8,
                    },
                    "safety": {
                        "minimum_available_gib": 18.0,
                        "cgroup_high_events": 0,
                        "cgroup_max_events": 0,
                        "cgroup_oom_events": 0,
                        "cgroup_swap_bytes": 0,
                        "xid": False,
                        "survivors": [],
                        "failures": [],
                    },
                }
            )
        expected_ids = [f"case-{index:03d}" for index in range(100)]
        result = CAMPAIGN.validate_quality_attempts(attempts, expected_ids)
        self.assertEqual(result, self.passing_nll())
        attempts[1]["rows"][50]["nll_sum"] += 1e-7
        with self.assertRaises(ValueError):
            CAMPAIGN.validate_quality_attempts(attempts, expected_ids)
        attempts[1]["rows"][50]["nll_sum"] -= 1e-7
        attempts[1]["rows"][50]["case_id"] = expected_ids[50]
        attempts[1].pop("engine")
        with self.assertRaises(ValueError):
            CAMPAIGN.validate_quality_attempts(attempts, expected_ids)
        attempts[1]["engine"] = {
            "slab_mode": "on", "slab_reads": 20, "slab_peak_qd": 8
        }
        attempts[1]["rows"][50]["case_id"] = "substituted-case"
        with self.assertRaises(ValueError):
            CAMPAIGN.validate_quality_attempts(attempts, expected_ids)

    def test_final_scorer_derives_and_binds_quality_raw_evidence(self):
        performance = {
            "candidate_commit": "a" * 40,
            "binary_sha256": "b" * 64,
            "quality_binary_sha256": "d" * 64,
            "model_sha256": CAMPAIGN.MODEL_SHA256,
            "memory_envelope_sha256": "c" * 64,
            "randomness": self.passing_confirmation(
                "a" * 40, "b" * 64, "d" * 64
            ),
        }
        attempts = []
        expected_ids = [f"case-{index:03d}" for index in range(100)]
        for arm in CAMPAIGN.quality_schedule(
            flip=performance["randomness"]["flip"]
        ):
            attempts.append(
                {
                    "arm": arm,
                    "mode": "off" if arm == "A" else "on",
                    "rows": [
                        {
                            "case_id": case_id,
                            "tokens": 4,
                            "nll_sum": index + 0.25,
                            "top1_correct": 3,
                        }
                        for index, case_id in enumerate(expected_ids)
                    ],
                    "output_sha256": "8" * 64,
                    "configuration_sha256": CAMPAIGN.canonical_environment_sha256(
                        CAMPAIGN.canonical_engine_environment(
                            "off" if arm == "A" else "on"
                        )
                    ),
                    "engine": {
                        "slab_mode": "off" if arm == "A" else "on",
                        "slab_reads": 0 if arm == "A" else 20,
                        "slab_peak_qd": 0 if arm == "A" else 8,
                    },
                    "safety": {
                        "minimum_available_gib": 18.0,
                        "cgroup_high_events": 0,
                        "cgroup_max_events": 0,
                        "cgroup_oom_events": 0,
                        "cgroup_swap_bytes": 0,
                        "xid": False,
                        "survivors": [],
                        "failures": [],
                    },
                }
            )
        manifest = {
            **performance,
            "schema_version": 1,
            "quality_binary_sha256": "d" * 64,
            "fixture_content_sha256": CAMPAIGN.QUALITY_FIXTURE_CONTENT_SHA256,
            "fixture_content_sha256_after": CAMPAIGN.QUALITY_FIXTURE_CONTENT_SHA256,
            "model_stat_before": {"size": 1},
            "model_stat_after": {"size": 1},
            "ordered_case_ids": expected_ids,
            "schedule": list(CAMPAIGN.quality_schedule(
                flip=performance["randomness"]["flip"]
            )),
            "fixture_sha256": "5" * 64,
            "quality_raw_sha256": "6" * 64,
            "nll_sha256": "7" * 64,
            "randomness": performance["randomness"],
        }
        result = CAMPAIGN.validate_bound_quality_evidence(
            performance, manifest, attempts
        )
        self.assertEqual(result, self.passing_nll())
        for key, value in (
            ("candidate_commit", "e" * 40),
            ("binary_sha256", "e" * 64),
            ("memory_envelope_sha256", "e" * 64),
            ("fixture_content_sha256", "e" * 64),
        ):
            broken = copy.deepcopy(manifest)
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CAMPAIGN.validate_bound_quality_evidence(
                    performance, broken, attempts
                )

    def test_performance_raw_is_bound_to_manifest_schedule_and_identities(self):
        records = self.passing_records()
        manifest = {
            "schema_version": 1,
            "gate": "glm-rung0-slab",
            "candidate_source": "/home/bmarti44/.cache/glm52-test",
            "candidate_commit": "f" * 40,
            "binary_sha256": "a" * 64,
            "quality_binary_sha256": "d" * 64,
            "model_sha256": CAMPAIGN.MODEL_SHA256,
            "sidecar_sha256": CAMPAIGN.SLAB_SHA256,
            "tokenizer_sha256": CAMPAIGN.TOKENIZER_SHA256,
            "fixture_sha256": "d" * 64,
            "randomness": self.passing_confirmation(
                "f" * 40, "a" * 64, "d" * 64
            ),
            "seed_sha256": "",
            "memory_envelope_sha256": "2" * 64,
            "memory_high_gib": 98,
            "memory_max_gib": 100,
            "kill_floor_gib": 18,
            "artifact_sha256": {},
            "sidecar_stat_before": {},
        }
        manifest["seed_sha256"] = manifest["randomness"]["seed_sha256"]
        manifest["schedule"] = [
            list(row) for row in CAMPAIGN.arm_schedule(
                flip=manifest["randomness"]["flip"]
            )
        ]
        CAMPAIGN.validate_performance_binding(manifest, records)
        broken_seed = copy.deepcopy(manifest)
        broken_seed["seed_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            CAMPAIGN.validate_performance_binding(broken_seed, records)
        records[0]["binary_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            CAMPAIGN.validate_performance_binding(manifest, records)

    def test_runtime_is_a_thin_wrapper_around_existing_measurement(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "scripts/30_bench_speed.py",
            '"--warmup", "0"',
            '"--request-timeout", "2700"',
            "/sys/class/block/nvme0n1/inflight",
            'f"/proc/{pid}/io"',
            "start_new_session=False",
            "arm_schedule()",
        ):
            self.assertIn(marker, source)
        self.assertNotIn('"DS4_CUDA_EXPERT_SLAB_TRACE": "1"', source)

    def test_slab_on_arms_keep_full_verification_and_have_90_minute_budget(self):
        self.assertEqual(CAMPAIGN.safe_timeout_seconds("off"), 3600)
        self.assertEqual(CAMPAIGN.safe_timeout_seconds("on"), 5400)

    def test_full_quality_arms_have_a_separate_150_minute_budget(self):
        self.assertEqual(CAMPAIGN.quality_timeout_seconds("off"), 9000)
        self.assertEqual(CAMPAIGN.quality_timeout_seconds("on"), 9000)

    def test_single_request_slab_canary_has_fixed_safety_acceptance(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "def parse_slab_staging_telemetry",
            "expert slab pinned staging ready",
            "expert slab pinned staging allocation failed",
            "NV_ERR_NO_MEMORY",
            'probe_arm.add_argument("--mode", choices=("off", "on")',
            'request_timeout = 2700 if mode == "on" else 300',
            '"--request-timeout", str(request_timeout)',
        ):
            self.assertIn(marker, source)

        line = (
            "ds4: expert slab pinned staging ready count=8 "
            "buffer_bytes=9736192 total_bytes=77889536 "
            "cuda_free_before=25000000000 cuda_free_after=24922110464 "
            "cuda_total=128495218688\n"
        )
        pools = CAMPAIGN.parse_slab_staging_telemetry(line)
        self.assertEqual(pools[0]["total_bytes"], 77_889_536)
        for mutation in (
            "",
            line + "ds4: expert slab pinned staging allocation failed index=0 count=8\n",
            line.replace("total_bytes=77889536", "total_bytes=1"),
            line.replace("count=8", "count=33", 1),
            line.replace("cuda_free_after=24922110464", "cuda_free_after=0"),
        ):
            with self.subTest(mutation=mutation[-80:]):
                with self.assertRaises(ValueError):
                    CAMPAIGN.parse_slab_staging_telemetry(mutation)

    def test_fixed_scorer_accepts_complete_lossless_campaign(self):
        with self.assertRaises(ValueError):
            CAMPAIGN.score_campaign(self.passing_records(), self.passing_nll())
        result = CAMPAIGN.score_campaign(
            self.passing_records(), self.passing_nll(), quality_bound=True
        )
        self.assertEqual(result["verdict"], "PASS")
        flipped = CAMPAIGN.score_campaign(
            self.passing_records(flip=True), self.passing_nll(),
            quality_bound=True, schedule_flip=True,
        )
        self.assertEqual(flipped["verdict"], "PASS")
        with self.assertRaises(ValueError):
            CAMPAIGN.score_campaign(
                self.passing_records(flip=True), self.passing_nll(),
                quality_bound=True, schedule_flip=False,
            )
        self.assertGreater(result["decode_ratio_lower_95"], 1.0)
        self.assertLessEqual(result["warm_ttft_ratio_upper_95"], 1.05)

    def test_fixed_scorer_accepts_fragmented_sse_with_complete_raw_tokens(self):
        records = self.passing_records()
        rep = records[0]["reps"][0]
        self.assertEqual(len(rep["token_ids"]), 128)
        self.assertEqual(len(rep["token_timestamps_ns"]), 128)
        self.assertEqual(len(rep["sse_token_timestamps_ns"]), 125)
        self.assertEqual(rep["client_completion_tokens"], 127)
        result = CAMPAIGN.score_campaign(
            records, self.passing_nll(), quality_bound=True
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_fixed_scorer_rejects_arm_differential_client_envelopes(self):
        records = self.passing_records()
        for record in records:
            ratio = 0.75 if record["mode"] == "off" else 1.25
            for rep in record["reps"]:
                raw = rep["token_timestamps_ns"]
                raw_step = 100_000_000
                rep["token_timestamps_ns"] = [
                    raw[0] + index * raw_step for index in range(128)
                ]
                raw_elapsed = 127 * raw_step
                client_elapsed = round(raw_elapsed / ratio)
                first = rep["client_first_content_ns"]
                last = first + client_elapsed
                rep["client_last_content_ns"] = last
                rep["sse_token_timestamps_ns"] = [
                    first + index * client_elapsed // 124 for index in range(125)
                ]
                rep["raw_client_timing_ratio"] = raw_elapsed / client_elapsed
        result = CAMPAIGN.score_campaign(
            records, self.passing_nll(), quality_bound=True
        )
        self.assertGreater(result["decode_ratio_lower_95_by_clock"]["client_wall"], 1.0)
        self.assertLessEqual(
            result["decode_ratio_lower_95_by_clock"]["raw_token"], 1.0
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_slab_arm_requires_full_identity_physical_reads(self):
        records = self.passing_records()
        slab = next(record for record in records if record["mode"] == "on")
        slab["external_io"]["read_bytes_delta"] = 401_104_554_047
        with self.assertRaisesRegex(ValueError, "full identity read coverage"):
            CAMPAIGN.score_campaign(
                records,
                self.passing_nll(),
                quality_bound=True,
            )

    def test_fixed_scorer_rejects_false_success_mutations(self):
        mutations = {
            "identical arms": lambda rows: rows[1].update(
                configuration_sha256="b" * 64
            ),
            "zero slab reads": lambda rows: rows[1]["engine"].update(
                slab_reads=0
            ),
            "output mismatch": lambda rows: rows[1]["reps"][0].update(
                generated_content_sha256="0" * 64
            ),
            "short output": lambda rows: rows[1]["reps"][0].update(
                token_ids=[1, 2], token_timestamps_ns=[1, 2]
            ),
            "server token mismatch": lambda rows: rows[1]["reps"][0].update(
                server_completion_tokens=127
            ),
            "nonmonotonic raw timing": lambda rows: rows[1]["reps"][0][
                "token_timestamps_ns"
            ].__setitem__(64, 1),
            "nonmonotonic client events": lambda rows: rows[1]["reps"][0][
                "sse_token_timestamps_ns"
            ].__setitem__(64, 1),
            "client endpoint mismatch": lambda rows: rows[1]["reps"][0].update(
                client_last_content_ns=99_000_000_000
            ),
            "event count mismatch": lambda rows: rows[1]["reps"][0].update(
                event_completion_tokens=124
            ),
            "raw client ratio mismatch": lambda rows: rows[1]["reps"][0].update(
                raw_client_timing_ratio=1.0
            ),
            "missing io": lambda rows: rows[1]["external_io"].update(
                sample_count=0
            ),
            "stale server": lambda rows: rows[1].update(
                server_instance_id=rows[0]["server_instance_id"]
            ),
            "safety event": lambda rows: rows[1]["safety"].update(
                cgroup_high_events=1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                rows = copy.deepcopy(self.passing_records())
                mutate(rows)
                with self.assertRaises(ValueError):
                    CAMPAIGN.score_campaign(
                        rows, self.passing_nll(), quality_bound=True
                    )

    def test_fixed_scorer_requires_exact_zero_nll_for_lossless_transport(self):
        nll = self.passing_nll()
        nll["token_weighted_delta_nll"] = 1e-9
        with self.assertRaises(ValueError):
            CAMPAIGN.score_campaign(
                self.passing_records(), nll, quality_bound=True
            )


if __name__ == "__main__":
    unittest.main()
