#!/usr/bin/env python3
"""Fail-closed conversion of matched campaign artifacts into controller raw data."""

from __future__ import annotations

import importlib.util
import hashlib
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

    def make_campaign(self, root: Path):
        campaign = root / "campaign"
        campaign.mkdir()
        fixture = root / "fixture.txt"
        fixture.write_text("fixed matched fixture\n", encoding="utf-8")
        # The serving weights manifest is a real file whose digest the profile
        # must match: the collector hashes it rather than trusting the profile,
        # so that editing the profile alone cannot relabel the baseline.
        serving_manifest = root / "serving-manifest.json"
        serving_manifest.write_text(
            json.dumps({"repo": "unsloth/test-GGUF", "files": []}),
            encoding="utf-8",
        )
        serving_digest = hashlib.sha256(serving_manifest.read_bytes()).hexdigest()
        dsv4_profile = root / "dsv4-profile.json"
        dsv4_profile.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "profile": "dsv4",
                    "binary_sha256": "c" * 64,
                    "configuration_sha256": "e" * 64,
                    "serving_weights_manifest_sha256": serving_digest,
                    "serving_weights_release": {
                        "repo": "unsloth/test-GGUF",
                        "revision": "f" * 40,
                    },
                }
            ),
            encoding="utf-8",
        )
        glm_profile = root / "glm52-profile.json"
        glm_profile.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "profile": "glm52",
                    "binary_sha256": "b" * 64,
                    "model_sha256": "a" * 64,
                    "context_cap": 1_048_576,
                    "runtime": {
                        "engine_environment": {
                            "DS4_CUDA_EXPERT_CACHE_GB": "0",
                            "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                            "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                            "DS4_CUDA_FETCH_THREADS": "6",
                            "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                            "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                            "DS4_CUDA_STABLE_MODEL_REMAP": "1",
                            "DS4_TOKEN_TIMING_LOG": "1",
                        }
                    },
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
                long_reps = []
                for rep_index in range(2):
                    long_reps.append(
                        {
                            "valid": True,
                            "ttft_s": (20.0 if glm else 10.0) + rep_index,
                            "decode_tok_s": decode,
                            "prefill_tok_s": 1_440.0 if glm else 2_880.0,
                            "completion_tokens": 128,
                            "server_completion_tokens": 128,
                            "prompt_tokens": 28_800,
                            "timing_source": "server_raw_token_log" if glm else "sse_content_events",
                            "token_timestamps_ns": [
                                int((3000 + rep_index * 1000 + index / decode) * 1_000_000_000)
                                for index in range(128)
                            ],
                            "request_sha256": f"long{rep_index}".ljust(64, "0"),
                            "prompt_sha256": f"prompt{rep_index}".ljust(64, "0"),
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
                    "cells": [
                        {"ctx_tokens": 0, "valid": True, "reps": reps},
                        {"ctx_tokens": 28672, "valid": True, "reps": long_reps},
                    ],
                }
                (directory / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                (directory / "kernel.log").write_text("", encoding="utf-8")
                if glm:
                    (directory / "runtime.config").write_text(
                        "context_cap=32768\nexpert_cache_gib=0\n"
                        "iq2_reference=1\nno_expert_tiles=0\n"
                        "stable_model_remap=1\nmodel_sha256=" + "a" * 64 + "\n",
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
        return campaign, fixture, dsv4_profile, serving_manifest, glm_profile

    def test_collects_exact_twenty_safe_matched_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            records = self.collector.collect_records(
                campaign, fixture, profile, serving, glm_profile
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

    def test_rejects_a_swapped_deepseek_weight_generation(self):
        """A GGUF generation change must invalidate the matched baseline.

        Before the serving identity was bound, swapping the served weights from
        the pre-0731 release to 0731 changed neither binary_sha256 nor
        configuration_sha256 -- the engine and its unit are the same -- so two GLM
        candidates measured against different DeepSeek models produced evidence
        claiming the same baseline. The collector must hash the manifest on disk,
        not trust the profile's copy of the digest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            # Sanity: unmutated campaign collects.
            self.assertEqual(
                len(self.collector.collect_records(
                    campaign, fixture, profile, serving, glm_profile
                )),
                20,
            )
            # Now the served weights change underneath an unchanged profile.
            serving.write_text(
                json.dumps({"repo": "unsloth/test-GGUF", "files": [{"n": 1}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "served GGUF generation"):
                self.collector.collect_records(
                    campaign, fixture, profile, serving, glm_profile
                )

    def test_rejects_missing_memory_and_duplicate_server_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            first = campaign / "block0-seq0-armA"
            (first / "samples.log").unlink()
            with self.assertRaisesRegex(ValueError, "samples"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

            (first / "samples.log").write_text(
                "mem_avail_kb=62914560\n", encoding="utf-8"
            )
            source = campaign / "block0-seq0-armA" / "process.identity"
            target = campaign / "block0-seq3-armA" / "process.identity"
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "fresh servers|server boot"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_short_geometry_and_wrong_glm_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            first = campaign / "block0-seq0-armA"
            result = json.loads((first / "result.json").read_text())
            result["cells"] = [result["cells"][0]]
            (first / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "32K-class"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

            second = Path(tmp) / "second"
            second.mkdir()
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(second)
            value = json.loads(glm_profile.read_text())
            value["binary_sha256"] = "9" * 64
            glm_profile.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "GLM binary"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)


if __name__ == "__main__":
    unittest.main()
