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
                    "measured_server_context_cap": 32_768,
                    "matched_model_first_shard_bytes": 5_257_664,
                    "model_shards": [
                        {"name": "dsv4-00001-of-00003.gguf", "bytes": 5_257_664, "sha256": "1" * 64},
                        {"name": "dsv4-00002-of-00003.gguf", "bytes": 49_437_013_568, "sha256": "2" * 64},
                        {"name": "dsv4-00003-of-00003.gguf", "bytes": 47_390_237_120, "sha256": "3" * 64},
                    ],
                    "model_path": "/models/dsv4-00001-of-00003.gguf",
                    "launch_arguments": [
                        "--model", "{model}", "-c", "32768", "--port", "{port}",
                        "--no-cache-prompt",
                    ],
                    "safety": {
                        "kill_floor_gib": 8,
                        "minimum_start_gib": 110,
                        "memory_high_gib": 105,
                        "memory_max_gib": 107,
                        "sample_hz": 4,
                        "swap_max_bytes": 0,
                        "timeout_seconds": 5400,
                    },
                    "runtime_closure_sha256": {
                        str((ROOT / "scripts/30_bench_speed.py").resolve()): hashlib.sha256(
                            (ROOT / "scripts/30_bench_speed.py").read_bytes()
                        ).hexdigest()
                    },
                    "artifact_sha256": {
                        "scripts/30_bench_speed.py": hashlib.sha256(
                            (ROOT / "scripts/30_bench_speed.py").read_bytes()
                        ).hexdigest()
                    },
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
                    "model_supported_context_cap": 1_048_576,
                    "measured_server_context_cap": 32_768,
                    # Candidate 2's legacy field keeps the previously-green
                    # collector fixtures valid until the bounded candidate-3
                    # implementation replaces it with the two explicit caps.
                    "model_path": "/models/glm52.gguf",
                    "model_bytes": 211_075_856_448,
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
                        },
                        "launch_arguments": [
                            "--cuda", "-m", "{model}", "-c", "32768",
                            "--host", "127.0.0.1", "--port", "{port}",
                            "--ssd-streaming", "--ssd-streaming-cache-experts",
                            "40GB",
                        ],
                        "benchmark": {
                            "fixture_context_tokens": [0, 28672],
                            "max_completion_tokens": 160,
                            "minimum_completion_tokens": 128,
                            "raw_token_timing_required": True,
                            "request_timeout_seconds": 2700,
                            "prefill_timing": "external_request_to_first_token_wall",
                        },
                        "safety": {
                            "kill_floor_gib": 40,
                            "minimum_start_gib": 110,
                            "sample_hz": 4,
                            "swap_max_bytes": 0,
                            "timeout_seconds": 5400,
                            "virtual_memory_limit_kib": 419430400,
                        },
                    },
                    "artifact_sha256": {
                        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                        for path in (
                            "results/glm52-goal/harness/decisive_matched.sh",
                            "results/glm52-goal/harness/glm_decisive_arm.sh",
                            "scripts/30_bench_speed.py",
                            "scripts/56_collect_matched_evidence.py",
                        )
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
                            "production_prompt_tokens": 100,
                            "timing_source": (
                                "server_raw_token_log"
                                if glm
                                else "sse_content_events"
                            ),
                            "token_timestamps_ns": timestamps,
                            "request_sha256": f"{block}{sequence}{rep_index}".ljust(
                                64, "0"
                            ),
                            "prompt_sha256": hashlib.sha256(
                                f"short-{rep_index}".encode()
                            ).hexdigest(),
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
                            "production_prompt_tokens": 28_800,
                            "timing_source": "server_raw_token_log" if glm else "sse_content_events",
                            "token_timestamps_ns": [
                                int((3000 + rep_index * 1000 + index / decode) * 1_000_000_000)
                                for index in range(128)
                            ],
                            "request_sha256": f"long{rep_index}".ljust(64, "0"),
                            "prompt_sha256": hashlib.sha256(
                                f"long-{rep_index}".encode()
                            ).hexdigest(),
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
                counts = (100, 100, 28_800, 28_800)
                if glm:
                    server_lines = [
                        f"ds4-server: chat ctx=0..1:{count} prompt start"
                        for count in counts
                    ]
                else:
                    server_lines = [
                        f"slot print_timing: prompt eval time = 1.0 ms / {count} tokens (x)"
                        for count in counts
                    ]
                (directory / "server.log").write_text(
                    "\n".join(server_lines) + "\n", encoding="utf-8"
                )
                if glm:
                    (directory / "runtime.config").write_text(
                        "context_cap=32768\nexpert_cache_gib=0\n"
                        "iq2_reference=1\nno_expert_tiles=0\n"
                        "stable_model_remap=1\nmodel_sha256=" + "a" * 64 + "\n",
                        encoding="utf-8",
                    )
                    environment = {
                        "environment": {
                            "DS4_CUDA_EXPERT_CACHE_GB": "0",
                            "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                            "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                            "DS4_CUDA_FETCH_THREADS": "6",
                            "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                            "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                            "DS4_CUDA_STABLE_MODEL_REMAP": "1",
                            "DS4_TOKEN_TIMING_LOG": "1",
                        }
                    }
                    canonical = "".join(
                        f"{key}={value}\n"
                        for key, value in sorted(environment["environment"].items())
                    )
                    environment["sha256"] = hashlib.sha256(
                        canonical.encode("ascii")
                    ).hexdigest()
                    (directory / "process.environment").write_text(
                        json.dumps(environment) + "\n", encoding="ascii"
                    )
                    (directory / "process.command").write_text(
                        json.dumps(
                            {
                                "argv": [
                                    "/candidate/ds4-server", "--cuda", "-m",
                                    "/models/glm52.gguf", "-c", "32768",
                                    "--host", "127.0.0.1", "--port", "8021",
                                    "--ssd-streaming", "--ssd-streaming-cache-experts",
                                    "40GB",
                                ],
                                "context_cap": 32768,
                                "model_device_inode_size": "66306:679227:211075856448",
                                "stable_model_remap": True,
                            }
                        ) + "\n",
                        encoding="ascii",
                    )
                    (directory / "model.device-inode-size").write_text(
                        "66306:679227:211075856448\n", encoding="ascii"
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
                    safety_prefix = (
                        "executed_environment_allowlist=bound "
                        "executed_environment_sha256=" + environment["sha256"] + "\n"
                    )
                else:
                    (directory / "process.identity.json").write_text(
                        json.dumps(
                            {
                                "boot_id": "11111111-2222-3333-4444-555555555555",
                                "server_pid": 3000 + block * 4 + sequence,
                                "server_start_ticks": 4000 + block * 4 + sequence,
                            }
                        ),
                        encoding="utf-8",
                    )
                    (directory / "process.command").write_text(
                        json.dumps(
                            {
                                "argv": [
                                    "/candidate/llama-server", "--model",
                                    "/models/dsv4-00001-of-00003.gguf", "-c", "32768",
                                    "--port", "8021", "--no-cache-prompt",
                                ],
                                "binary_sha256": "c" * 64,
                                "context_cap": 32768,
                                "model_device_inode_size": "66306:779227:5257664",
                            }
                        ) + "\n",
                        encoding="ascii",
                    )
                    (directory / "model.device-inode-size").write_text(
                        "66306:779227:5257664\n", encoding="ascii"
                    )
                    shard_values = (
                        ("dsv4-00001-of-00003.gguf", 779227, 5_257_664),
                        ("dsv4-00002-of-00003.gguf", 779228, 49_437_013_568),
                        ("dsv4-00003-of-00003.gguf", 779229, 47_390_237_120),
                    )
                    checkpoints = []
                    for checkpoint_index, checkpoint in enumerate(
                        ("prelaunch", "ready", "post_requests")
                    ):
                        checkpoints.append(
                            json.dumps(
                                {
                                    "checkpoint": checkpoint,
                                    "monotonic_ns": 10 + checkpoint_index,
                                    "shards": [
                                        {
                                            "path": f"/models/{name}",
                                            "device": 66306,
                                            "inode": inode,
                                            "bytes": size,
                                        }
                                        for name, inode, size in shard_values
                                    ],
                                },
                                separators=(",", ":"),
                            )
                        )
                    (directory / "model.shards.jsonl").write_text(
                        "\n".join(checkpoints) + "\n", encoding="ascii"
                    )
                    (directory / "process.observations.json").write_text(
                        json.dumps(
                            {
                                "readiness_http_status": 200,
                                "post_requests_http_status": 200,
                                "server_pid": 3000 + block * 4 + sequence,
                                "server_start_ticks": 4000 + block * 4 + sequence,
                                "recorded_monotonic_ns": 20,
                            }
                        ),
                        encoding="ascii",
                    )
                    (directory / "process.runtime-closure.json").write_text(
                        json.dumps(
                            {
                                str((ROOT / "scripts/30_bench_speed.py").resolve()): hashlib.sha256(
                                    (ROOT / "scripts/30_bench_speed.py").read_bytes()
                                ).hexdigest()
                            }
                        ) + "\n",
                        encoding="ascii",
                    )
                    (directory / "samples.log").write_text(
                        "2026-07-27T00:00:00+00:00 "
                        "mem_avail_kb=20971520 eng_rss_kb=1 read_bytes=1\n",
                        encoding="utf-8",
                    )
                    safety_prefix = ""
                safety_main = (
                    safety_prefix
                    + "cgroup_final current_bytes=1 peak_bytes=2 swap_current_bytes=0 "
                    "events=high=0,high_delta=0,max=0,max_delta=0,oom=0,oom_delta=0,"
                    "oom_kill=0,oom_kill_delta=0,oom_group_kill=0,oom_group_kill_delta=0\n"
                    "SAFE_RUN end rc=0 killed=no (124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n"
                )
                (directory / "safety.main.log").write_text(safety_main, encoding="utf-8")
                (directory / "safety.kernel.log").write_text("", encoding="utf-8")
                main_hash = hashlib.sha256((directory / "safety.main.log").read_bytes()).hexdigest()
                samples_hash = hashlib.sha256((directory / "samples.log").read_bytes()).hexdigest()
                kernel_hash = hashlib.sha256((directory / "safety.kernel.log").read_bytes()).hexdigest()
                (directory / "safety.wrapper.out").write_text(
                    "SAFE_RUN_DONE rc=0 killed=no dir=/tmp/safe "
                    f"main_sha256={main_hash} samples_sha256={samples_hash} "
                    f"kernel_sha256={kernel_hash}\n",
                    encoding="ascii",
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
            with self.assertRaisesRegex(ValueError, "samples|canonical safety"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
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

    def test_rejects_unequal_prompts_and_wrong_runtime_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq1-armB" / "result.json"
            value = json.loads(target.read_text())
            value["cells"][1]["reps"][0]["prompt_sha256"] = "8" * 64
            target.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "unequal prompt bytes"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq0-armA" / "runtime.config"
            target.write_text(target.read_text().replace("a" * 64, "7" * 64))
            with self.assertRaisesRegex(ValueError, "runtime configuration"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_inflated_usage_and_unbound_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq0-armA" / "result.json"
            value = json.loads(target.read_text())
            value["cells"][1]["reps"][0]["prompt_tokens"] = 999_999_999
            target.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "production prompt"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            value = json.loads(glm_profile.read_text())
            value["artifact_sha256"]["scripts/30_bench_speed.py"] = "9" * 64
            glm_profile.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "campaign artifact"):
                self.collector.collect_records(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_coherently_inflated_prompt_fields_without_raw_log_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            for directory in campaign.iterdir():
                result_path = directory / "result.json"
                value = json.loads(result_path.read_text())
                for rep in value["cells"][1]["reps"]:
                    rep["prompt_tokens"] = 32_640
                    rep["production_prompt_tokens"] = 32_640
                result_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "raw production prompt"):
                self.collector.collect_records(
                    campaign, fixture, profile, serving, glm_profile
                )

    def test_rejects_prefixed_or_killed_safety_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            for directory in campaign.iterdir():
                prefix = ""
                environment_path = directory / "process.environment"
                if environment_path.exists():
                    environment = json.loads(environment_path.read_text())
                    prefix = (
                        "executed_environment_allowlist=bound "
                        f"executed_environment_sha256={environment['sha256']}\n"
                    )
                (directory / "safety.main.log").write_text(
                    prefix + "NOT_SAFE_RUN_DONE rc=0 killed=floor\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "canonical safety"):
                self.collector.collect_records(
                    campaign, fixture, profile, serving, glm_profile
                )

    def test_rejects_live_environment_command_and_model_identity_drift(self):
        mutations = (
            (
                "process.environment",
                lambda value: value["environment"].__setitem__(
                    "DS4_CUDA_FETCH_THREADS", "7"
                ),
                "executed environment",
            ),
            (
                "process.command",
                lambda value: value.__setitem__("context_cap", 8192),
                "executed command",
            ),
        )
        for artifact, mutate, message in mutations:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
                path = campaign / "block0-seq0-armA" / artifact
                value = json.loads(path.read_text())
                mutate(value)
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, message):
                    self.collector.collect_records(
                        campaign, fixture, profile, serving, glm_profile
                    )

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            path = campaign / "block0-seq0-armA" / "model.device-inode-size"
            path.write_text("66306:679228:211075856448\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "executed command"):
                self.collector.collect_records(
                    campaign, fixture, profile, serving, glm_profile
                )


if __name__ == "__main__":
    unittest.main()
