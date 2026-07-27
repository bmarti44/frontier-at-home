#!/usr/bin/env python3
"""Fail-closed conversion of matched campaign artifacts into controller raw data."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "56_collect_matched_evidence.py"


def load_module():
    spec = importlib.util.spec_from_file_location("matched_evidence", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MatchedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = load_module()

    def make_campaign(self, root: Path) -> tuple[Path, Path, Path]:
        campaign = root / "campaign"
        campaign.mkdir()
        fixture = root / "fixture.txt"
        fixture.write_text("fixed matched fixture\n", encoding="utf-8")
        dsv4_profile = root / "dsv4-profile.json"
        dsv4_profile.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "profile": "dsv4",
                    "binary_sha256": "c" * 64,
                    "configuration_sha256": "e" * 64,
                }
            ),
            encoding="utf-8",
        )
        for block in range(5):
            order = "ABBA" if block % 2 == 0 else "BAAB"
            for sequence, arm in enumerate(order):
                glm = arm == "A"
                label = f"block{block}-seq{sequence}-arm{arm}"
                directory = campaign / label
                directory.mkdir()
                decode = 1.0 if glm else 10.0
                reps = []
                for rep_index in range(2):
                    timestamps = [
                        int((rep_index * 1000 + index / decode) * 1_000_000_000)
                        for index in range(128)
                    ]
                    reps.append(
                        {
                            "valid": True,
                            "ttft_s": (2.0 if glm else 1.0)
                            + rep_index / 10,
                            "decode_tok_s": decode,
                            "prefill_tok_s": 50.0 if glm else 100.0,
                            "completion_tokens": 128,
                            "server_completion_tokens": 128,
                            "prompt_tokens": 100,
                            "timing_source": (
                                "server_raw_token_log"
                                if glm
                                else "sse_content_events"
                            ),
                            "token_timestamps_ns": timestamps,
                            "request_sha256": f"{block}{sequence}{rep_index}".ljust(
                                64, "0"
                            ),
                        }
                    )
                result = {
                    "suite_valid": True,
                    "metadata": {
                        "model": "glm-5.2" if glm else "deepseek-v4-flash",
                        "seed": 1234,
                        "reps": 2,
                        "fixture_path": str(fixture),
                    },
                    "cells": [{"ctx_tokens": 0, "valid": True, "reps": reps}],
                }
                (directory / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                (directory / "kernel.log").write_text("", encoding="utf-8")
                if glm:
                    (directory / "runtime.config").write_text(
                        "expert_cache_gib=32\niq2_reference=0\n"
                        "no_expert_tiles=1\n",
                        encoding="utf-8",
                    )
                    (directory / "process.identity").write_text(
                        f"{1000 + block * 4 + sequence} "
                        f"{2000 + block * 4 + sequence} {'b' * 64}\n",
                        encoding="utf-8",
                    )
                    (directory / "host.boot_id").write_text(
                        "11111111-2222-3333-4444-555555555555\n",
                        encoding="ascii",
                    )
                    (directory / "samples.log").write_text(
                        "2026-07-27T00:00:00+00:00 "
                        "mem_avail_kb=62914560 eng_rss_kb=1 read_bytes=1\n",
                        encoding="utf-8",
                    )
                    (directory / "safety.main.log").write_text(
                        "SAFE_RUN_DONE rc=0\n", encoding="utf-8"
                    )
                else:
                    (directory / "process.identity.json").write_text(
                        json.dumps(
                            {
                                "boot_id": "11111111-2222-3333-4444-555555555555",
                                "server_pid": 3000 + block * 4 + sequence,
                                "server_start_ticks": 4000 + block * 4 + sequence,
                                "server_alive": True,
                                "memwatch_alive": True,
                                "watchdog_armed": True,
                                "healthy": True,
                            }
                        ),
                        encoding="utf-8",
                    )
                    (directory / "memwatch.segment.log").write_text(
                        "ts=2026-07-27T00:00:00Z mem_available_gib=20.00\n",
                        encoding="utf-8",
                    )
        return campaign, fixture, dsv4_profile

    def test_collects_exact_twenty_safe_matched_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile = self.make_campaign(Path(tmp))
            records = self.collector.collect_records(
                campaign, fixture, profile
            )
            self.assertEqual(len(records), 20)
            self.assertEqual(
                [(row["block"], row["sequence"]) for row in records],
                [(block, sequence) for block in range(5) for sequence in range(4)],
            )
            self.assertEqual(
                {row["profile"] for row in records}, {"glm52", "dsv4"}
            )
            self.assertEqual(
                min(row["available_memory_gib"] for row in records), 20.0
            )
            self.assertTrue(
                all(len(row["token_timestamps"]) == 128 for row in records)
            )

    def test_rejects_missing_memory_and_duplicate_server_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile = self.make_campaign(Path(tmp))
            first = campaign / "block0-seq0-armA"
            (first / "samples.log").unlink()
            with self.assertRaisesRegex(ValueError, "samples"):
                self.collector.collect_records(campaign, fixture, profile)

            (first / "samples.log").write_text(
                "mem_avail_kb=62914560\n", encoding="utf-8"
            )
            source = campaign / "block0-seq0-armA" / "process.identity"
            target = campaign / "block0-seq3-armA" / "process.identity"
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "fresh servers|server boot"):
                self.collector.collect_records(campaign, fixture, profile)


if __name__ == "__main__":
    unittest.main()
