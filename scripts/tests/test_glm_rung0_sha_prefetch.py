#!/usr/bin/env python3
"""Executable RED contracts for collision-resistant slab prefetch."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


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
        schedule = campaign.sha_prefetch_schedule(False)
        binary = "a" * 64
        fixture = "b" * 64
        access = "c" * 64
        configs = {"A": "d" * 64, "B": "e" * 64, "C": "f" * 64}
        commit = "1" * 40
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
            ready = 80 if arm == "C" else 0
            late = 20 if arm == "C" else 0
            fallback = 20 if arm == "C" else 0
            copies = 70 if arm == "C" else 100 if arm == "B" else 0
            validated = sha_success * 256
            copied = copies * 256
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
                        "read_ns": attempts * 100,
                        "sha_ns": attempts * 50,
                        "wait_ns": attempts * 10,
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
            })
        manifest = {
            "schema_version": 1,
            "candidate_commit": commit,
            "binary_sha256": binary,
            "configuration_sha256_by_arm": configs,
            "fixture_sha256": fixture,
            "access_stream_sha256": access,
            "campaign_started_monotonic_ns": start_ns - 1,
            "campaign_finished_monotonic_ns": start_ns + 15 * 10**12,
            "quality": {
                "case_count": 100,
                "token_weighted_delta_nll": 0.0,
                "top1_loss_pp": 0.0,
                "deterministic": True,
            },
        }
        return records, manifest

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
        engine = Path("/tmp/glm52-score-official/ds4_cuda.cu")
        self.assertTrue(engine.is_file(), "frozen compiled engine source is absent")
        source = engine.read_text(encoding="utf-8")
        self.assertTrue(
            '#include "ds4_slab_prefetch_state.h"' in source,
            "engine does not include the shared prefetch state authority",
        )
        required_calls = (
            "->issue(", "->complete_read(", "->claim(", "->copy_sync(",
            "->discard_ready(", "->reload(",
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
        records, manifest = self.passing_campaign(campaign)
        scored = campaign.score_sha_prefetch_campaign(records, manifest)
        self.assertEqual(scored["verdict"], "PASS")
        self.assertLessEqual(scored["probe_completed_fetch_ratio"], 0.90)
        self.assertEqual(
            set(scored["decode_ratio_lower_95_by_comparator_and_clock"]),
            {"C/A:client_wall", "C/A:raw_token", "C/B:client_wall", "C/B:raw_token"},
        )

    def test_three_arm_scorer_rejects_malformed_or_partial_evidence(self):
        campaign = self.load_campaign()
        records, manifest = self.passing_campaign(campaign)
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
                        changed_records, changed_manifest
                    )

    def test_three_arm_scorer_requires_every_comparator_clock_and_ttft_state(self):
        campaign = self.load_campaign()
        records, manifest = self.passing_campaign(campaign)
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
                scored = campaign.score_sha_prefetch_campaign(changed, manifest)
                self.assertEqual(scored["verdict"], "FAIL")

    def test_three_arm_scorer_reconciles_every_telemetry_class(self):
        campaign = self.load_campaign()
        records, manifest = self.passing_campaign(campaign)
        mutations = {
            "attempts": ("attempts", 101),
            "sha successes": ("sha_successes", 99),
            "sha failures": ("sha_failures", 1),
            "ready": ("ready", 81),
            "late": ("late", 21),
            "stale": ("stale", 1),
            "fallback": ("fallback", 21),
            "copies": ("copies", 101),
            "validated bytes": ("validated_bytes", 1),
            "copied bytes": ("copied_bytes", 25_601),
            "publications": ("publications", 71),
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
                    campaign.score_sha_prefetch_campaign(changed, manifest)


if __name__ == "__main__":
    unittest.main()
