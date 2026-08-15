#!/usr/bin/env python3
"""Fail-closed conversion of matched campaign artifacts into controller raw data."""

from __future__ import annotations

import datetime as dt
import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "56_collect_matched_evidence.py"
DRAND_VERIFIER = ROOT / "scripts" / "89_verify_drand_receipt.mjs"
HISTORICAL_RANDOMNESS = (
    ROOT
    / "results"
    / "glm52-gates"
    / "lossless-plateau-candidate10-first-round-test-receipt.json"
)
HISTORICAL_RANDOMNESS_VALUE = json.loads(
    HISTORICAL_RANDOMNESS.read_text(encoding="ascii")
)
HISTORICAL_MATCHED_SEED = HISTORICAL_RANDOMNESS_VALUE["seed_derivation"][
    "matched_seed"
]
CANDIDATE10_PREAUDIT = (
    ROOT / "results/glm52-gates/lossless-plateau-candidate10-preaudit.json"
)
CANDIDATE13_PREAUDIT = (
    ROOT / "results/glm52-gates/lossless-plateau-candidate13-preaudit.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("matched_evidence", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MatchedEvidenceTests(unittest.TestCase):
    DSV4_SAFETY_ENVELOPE = {
        "kill_floor_gib": 8,
        "minimum_start_gib": 110,
        "memory_high_gib": 100,
        "memory_max_gib": 102,
        "sample_hz": 4,
        "swap_max_bytes": 0,
        "timeout_seconds": 5400,
    }

    @classmethod
    def setUpClass(cls):
        cls.collector = load_module()

    def make_campaign(self, root: Path, *, seed: int = HISTORICAL_MATCHED_SEED):
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
                        "memory_high_gib": 100,
                        "memory_max_gib": 102,
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
                        "seed": seed,
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
                    "events=low=0,low_delta=0,high=0,high_delta=0,max=0,max_delta=0,oom=0,oom_delta=0,"
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

    def write_randomness(self, campaign: Path, value: dict[str, object]) -> Path:
        path, _, _ = self.write_retained_authority(
            campaign, value, CANDIDATE10_PREAUDIT
        )
        return path

    @staticmethod
    def git_bytes(commit: str, relative: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout

    def write_retained_authority(
        self,
        campaign: Path,
        randomness: dict[str, object],
        freeze_receipt_path: Path,
    ) -> tuple[Path, Path, dict[str, object]]:
        retained = campaign / "retained"
        retained.mkdir(exist_ok=True)
        randomness_path = retained / "randomness-receipt.json"
        randomness_raw = (
            json.dumps(randomness, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        randomness_path.write_bytes(randomness_raw)

        relative_receipt = freeze_receipt_path.relative_to(ROOT).as_posix()
        freeze_raw = freeze_receipt_path.read_bytes()
        freeze_value = json.loads(freeze_raw)
        candidate = freeze_value["candidate_commit"]
        freeze = subprocess.run(
            [
                "git", "-C", str(ROOT), "log", "-1", "--diff-filter=A",
                "--format=%H", "--", relative_receipt,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        freeze_path = retained / "freeze-receipt.json"
        freeze_path.write_bytes(freeze_raw)

        profile_paths = (
            "configs/glm52-lossless-plateau-profile.json",
            "configs/dsv4-matched-32k-profile.json",
        )
        profiles = [
            json.loads(self.git_bytes(candidate, relative))
            for relative in profile_paths
        ]
        paths = set(profile_paths)
        for profile in profiles:
            paths.update(profile["artifact_sha256"])
        digests = {
            relative: hashlib.sha256(self.git_bytes(candidate, relative)).hexdigest()
            for relative in sorted(paths)
        }
        digests["runtime/tokenizers.abi3.so"] = profiles[0]["python_runtime"][
            "tokenizer_native_sha256"
        ]
        digests["freeze-receipt.json"] = hashlib.sha256(freeze_raw).hexdigest()
        digests["randomness-receipt.json"] = hashlib.sha256(
            randomness_raw
        ).hexdigest()
        git_head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        manifest = {
            "schema": "matched-retained-closure-v1",
            "git_head": git_head,
            "reviewed_runtime_commit": candidate,
            "freeze_commit": freeze,
            "freeze_receipt_sha256": hashlib.sha256(freeze_raw).hexdigest(),
            "randomness_receipt_sha256": hashlib.sha256(randomness_raw).hexdigest(),
            "python_runtime": profiles[0]["python_runtime"],
            "sha256": digests,
        }
        manifest_path = campaign / "retained-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        return randomness_path, freeze_path, manifest

    def first_round_receipt(self, source: dict, preaudit_path: Path) -> dict:
        preaudit = json.loads(preaudit_path.read_text(encoding="ascii"))
        relative = preaudit_path.relative_to(ROOT).as_posix()
        freeze_commit = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--diff-filter=A", "-1",
             "--format=%H", "--", relative],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        freeze_epoch = int(subprocess.run(
            ["git", "-C", str(ROOT), "show", "-s", "--format=%ct", freeze_commit],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip())
        freeze_iso = subprocess.run(
            ["git", "-C", str(ROOT), "show", "-s", "--format=%cI", freeze_commit],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        target = ((freeze_epoch - self.collector.DRAND_GENESIS_UNIX)
                  // self.collector.DRAND_PERIOD_SECONDS + 2)
        publication = (self.collector.DRAND_GENESIS_UNIX
                       + (target - 1) * self.collector.DRAND_PERIOD_SECONDS)
        candidate = preaudit["candidate_commit"]
        result = json.loads(json.dumps(source))
        result["candidate_hash"] = candidate
        result["freeze_commit"] = freeze_commit
        result["freeze_committed_at"] = freeze_iso
        result["receipt"]["round"] = target
        result["receipt"]["published_at_utc"] = dt.datetime.fromtimestamp(
            publication, tz=dt.timezone.utc
        ).isoformat()
        result["obtained_at_utc"] = dt.datetime.fromtimestamp(
            publication + 1, tz=dt.timezone.utc
        ).isoformat()
        seed_sha256 = hashlib.sha256(
            self.collector.DRAND_DOMAIN + candidate.encode("ascii") + b"\0"
            + result["receipt"]["randomness"].encode("ascii")
        ).hexdigest()
        result["seed_derivation"]["seed_sha256"] = seed_sha256
        result["seed_derivation"]["matched_seed"] = int(seed_sha256[:15], 16)
        return result

    def collect_with_mocked_bls(self, *args, **kwargs):
        real_run = subprocess.run

        def verified(arguments, *run_args, **run_kwargs):
            if arguments[0] == str(self.collector.DRAND_NODE):
                return subprocess.CompletedProcess(
                    arguments, 0, stdout="DRAND_BLS_RECEIPT_OK\n", stderr=""
                )
            return real_run(arguments, *run_args, **run_kwargs)

        with mock.patch.object(self.collector.subprocess, "run", side_effect=verified):
            return self.collect_with_randomness(*args, **kwargs)

    def collect_with_randomness(
        self,
        campaign: Path,
        fixture: Path,
        profile: Path,
        serving: Path,
        glm_profile: Path,
        receipt: Path,
        *,
        candidate_hash: str,
        freeze_commit: str,
    ):
        return self.collector.collect_records(
            campaign,
            fixture,
            profile,
            serving,
            glm_profile,
            randomness_receipt=receipt,
            candidate_hash=candidate_hash,
            freeze_commit=freeze_commit,
            drand_verifier=DRAND_VERIFIER,
        )

    def collect_current(
        self,
        campaign: Path,
        fixture: Path,
        profile: Path,
        serving: Path,
        glm_profile: Path,
    ):
        receipt = campaign / "retained" / "randomness-receipt.json"
        if not receipt.exists():
            receipt = self.write_randomness(campaign, HISTORICAL_RANDOMNESS_VALUE)
        return self.collect_with_randomness(
            campaign,
            fixture,
            profile,
            serving,
            glm_profile,
            receipt,
            candidate_hash=HISTORICAL_RANDOMNESS_VALUE["candidate_hash"],
            freeze_commit=HISTORICAL_RANDOMNESS_VALUE["freeze_commit"],
        )

    def test_collector_independently_verifies_committed_randomness_and_arm_seed(self):
        committed = HISTORICAL_RANDOMNESS.read_bytes()
        source = json.loads(committed)
        candidate_hash = source["candidate_hash"]
        freeze_commit = source["freeze_commit"]
        matched_seed = source["seed_derivation"]["matched_seed"]
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=matched_seed
            )
            receipt = self.write_randomness(campaign, source)
            records = self.collect_with_randomness(
                campaign,
                fixture,
                profile,
                serving,
                glm_profile,
                receipt,
                candidate_hash=candidate_hash,
                freeze_commit=freeze_commit,
            )
            self.assertEqual(len(records), 20)

    def test_collector_rejects_randomness_and_seed_mutations(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        candidate_hash = source["candidate_hash"]
        freeze_commit = source["freeze_commit"]
        matched_seed = source["seed_derivation"]["matched_seed"]

        mutations = {
            "absent": (None, candidate_hash, freeze_commit, matched_seed, "receipt"),
            "stale_publication": (
                {**source, "freeze_commit": "8bb660dde1ed18e3d1c93e0e2830453af83f7bc6"},
                candidate_hash,
                "8bb660dde1ed18e3d1c93e0e2830453af83f7bc6",
                matched_seed,
                "post-freeze|publication|freeze|candidate",
            ),
            "wrong_candidate": (
                {**source, "candidate_hash": "0" * 40},
                candidate_hash,
                freeze_commit,
                matched_seed,
                "candidate",
            ),
            "wrong_freeze": (
                {**source, "freeze_commit": "0" * 40},
                candidate_hash,
                freeze_commit,
                matched_seed,
                "freeze",
            ),
            "bad_signature": (
                {
                    **source,
                    "receipt": {
                        **source["receipt"],
                        "signature": "0" + source["receipt"]["signature"][1:],
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "signature|BLS|randomness",
            ),
            "bad_previous_signature": (
                {
                    **source,
                    "receipt": {
                        **source["receipt"],
                        "previous_signature": "0"
                        + source["receipt"]["previous_signature"][1:],
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "signature|BLS|randomness",
            ),
            "changed_round": (
                {
                    **source,
                    "receipt": {
                        **source["receipt"],
                        "round": source["receipt"]["round"] + 1,
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "round|signature|BLS|publication",
            ),
            "changed_randomness": (
                {
                    **source,
                    "receipt": {
                        **source["receipt"],
                        "randomness": "0" * 64,
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "randomness|signature",
            ),
            "self_authored_publication_time": (
                {
                    **source,
                    "receipt": {
                        **source["receipt"],
                        "published_at_utc": "2099-01-01T00:00:00+00:00",
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "publication|published",
            ),
            "altered_derivation": (
                {
                    **source,
                    "seed_derivation": {
                        **source["seed_derivation"],
                        "formula": "sha256('ALTERED-DOMAIN' || randomness)",
                        "seed_sha256": "0" * 64,
                        "matched_seed": 0,
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "derivation|seed",
            ),
            "altered_seed_digest_only": (
                {
                    **source,
                    "seed_derivation": {
                        **source["seed_derivation"],
                        "seed_sha256": "0" * 64,
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "derivation|seed",
            ),
            "altered_matched_seed_only": (
                {
                    **source,
                    "seed_derivation": {
                        **source["seed_derivation"],
                        "matched_seed": 0,
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "derived seed|seed mismatch|derivation",
            ),
            "changed_relay_list": (
                {**source, "relay_agreement": ["api.drand.sh"]},
                candidate_hash,
                freeze_commit,
                matched_seed,
                "relay|schema",
            ),
            "forged_verification_result": (
                {
                    **source,
                    "verification": {
                        **source["verification"],
                        "result": "NOT_VERIFIED",
                    },
                },
                candidate_hash,
                freeze_commit,
                matched_seed,
                "verification|BLS",
            ),
            "uniformly_wrong_arm_seed": (
                source,
                candidate_hash,
                freeze_commit,
                999_999_999_999_999_999,
                "arm seed|derived seed|seed mismatch",
            ),
        }

        for label, (value, expected_candidate, expected_freeze, arm_seed, error) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                    Path(tmp), seed=arm_seed
                )
                receipt = campaign / "retained" / "randomness-receipt.json"
                if value is not None:
                    receipt = self.write_randomness(campaign, value)
                with self.assertRaisesRegex((OSError, ValueError), error):
                    self.collect_with_randomness(
                        campaign,
                        fixture,
                        profile,
                        serving,
                        glm_profile,
                        receipt,
                        candidate_hash=expected_candidate,
                        freeze_commit=expected_freeze,
                    )

    def test_collector_rejects_randomness_receipt_digest_and_path_replacement(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        candidate_hash = source["candidate_hash"]
        freeze_commit = source["freeze_commit"]
        matched_seed = source["seed_derivation"]["matched_seed"]
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=matched_seed
            )
            receipt = self.write_randomness(campaign, source)
            manifest = campaign / "retained-manifest.json"
            manifest.write_text(
                json.dumps({"randomness_receipt_sha256": "0" * 64}) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "receipt|digest|manifest"):
                self.collect_with_randomness(
                    campaign,
                    fixture,
                    profile,
                    serving,
                    glm_profile,
                    receipt,
                    candidate_hash=candidate_hash,
                    freeze_commit=freeze_commit,
                )

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=matched_seed
            )
            receipt = self.write_randomness(campaign, source)
            replacement = campaign / "replacement.json"
            replacement.write_bytes(receipt.read_bytes())
            receipt.unlink()
            receipt.symlink_to(replacement)
            with self.assertRaisesRegex((OSError, ValueError), "receipt|symlink|regular"):
                self.collect_with_randomness(
                    campaign,
                    fixture,
                    profile,
                    serving,
                    glm_profile,
                    receipt,
                    candidate_hash=candidate_hash,
                    freeze_commit=freeze_commit,
                )

    def test_collector_accepts_exact_retained_authority_and_first_round(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        source = self.first_round_receipt(source, CANDIDATE13_PREAUDIT)
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=source["seed_derivation"]["matched_seed"]
            )
            receipt, _, _ = self.write_retained_authority(
                campaign, source, CANDIDATE13_PREAUDIT
            )
            records = self.collect_with_mocked_bls(
                campaign,
                fixture,
                profile,
                serving,
                glm_profile,
                receipt,
                candidate_hash=source["candidate_hash"],
                freeze_commit=source["freeze_commit"],
            )
            self.assertEqual(len(records), 20)

    def test_collector_derives_lineage_and_rejects_cross_candidate_receipt(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=HISTORICAL_MATCHED_SEED
            )
            receipt, _, _ = self.write_retained_authority(
                campaign, source, CANDIDATE13_PREAUDIT
            )
            with self.assertRaisesRegex(
                ValueError, "retained|preaudit|candidate|cross-candidate"
            ):
                self.collect_with_randomness(
                    campaign,
                    fixture,
                    profile,
                    serving,
                    glm_profile,
                    receipt,
                    # These historical caller values are deliberately coherent.
                    # The collector must ignore/reject them and derive authority
                    # from the candidate-13 retained manifest and freeze receipt.
                    candidate_hash=source["candidate_hash"],
                    freeze_commit=source["freeze_commit"],
                )

    def test_collector_rejects_retained_manifest_and_preaudit_mutations(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        mutations = (
            "manifest_extra_key",
            "manifest_missing_schema",
            "manifest_reviewed_candidate",
            "manifest_freeze_commit",
            "manifest_freeze_receipt_digest",
            "manifest_artifact_digest",
            "preaudit_candidate_tree",
            "preaudit_artifact_binding",
        )
        for label in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                    Path(tmp), seed=HISTORICAL_MATCHED_SEED
                )
                receipt, freeze_receipt, manifest = self.write_retained_authority(
                    campaign, source, CANDIDATE10_PREAUDIT
                )
                if label == "manifest_extra_key":
                    manifest["unreviewed"] = True
                elif label == "manifest_missing_schema":
                    del manifest["schema"]
                elif label == "manifest_reviewed_candidate":
                    manifest["reviewed_runtime_commit"] = "0" * 40
                elif label == "manifest_freeze_commit":
                    manifest["freeze_commit"] = "0" * 40
                elif label == "manifest_freeze_receipt_digest":
                    manifest["freeze_receipt_sha256"] = "0" * 64
                elif label == "manifest_artifact_digest":
                    manifest["sha256"][
                        "results/glm52-goal/harness/decisive_matched.sh"
                    ] = "0" * 64
                else:
                    preaudit = json.loads(freeze_receipt.read_text(encoding="ascii"))
                    if label == "preaudit_candidate_tree":
                        preaudit["candidate_tree"] = "0" * 40
                    else:
                        first = next(iter(preaudit["artifact_sha256"]))
                        preaudit["artifact_sha256"][first] = "0" * 64
                    freeze_raw = (
                        json.dumps(preaudit, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("ascii")
                    freeze_receipt.write_bytes(freeze_raw)
                    freeze_digest = hashlib.sha256(freeze_raw).hexdigest()
                    manifest["freeze_receipt_sha256"] = freeze_digest
                    manifest["sha256"]["freeze-receipt.json"] = freeze_digest
                (campaign / "retained-manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(
                    ValueError, "manifest|retained|preaudit|candidate|artifact|tree"
                ):
                    self.collect_with_randomness(
                        campaign,
                        fixture,
                        profile,
                        serving,
                        glm_profile,
                        receipt,
                        candidate_hash=source["candidate_hash"],
                        freeze_commit=source["freeze_commit"],
                    )

    def test_collector_rejects_target_plus_one_before_bls_verification(self):
        source = json.loads(HISTORICAL_RANDOMNESS.read_text(encoding="ascii"))
        source = self.first_round_receipt(source, CANDIDATE13_PREAUDIT)
        later = json.loads(json.dumps(source))
        later["receipt"]["round"] = source["receipt"]["round"] + 1
        publication = (
            self.collector.DRAND_GENESIS_UNIX
            + (later["receipt"]["round"] - 1)
            * self.collector.DRAND_PERIOD_SECONDS
        )
        later["receipt"]["published_at_utc"] = dt.datetime.fromtimestamp(
            publication, tz=dt.timezone.utc
        ).isoformat()
        later["obtained_at_utc"] = dt.datetime.fromtimestamp(
            publication + 1, tz=dt.timezone.utc
        ).isoformat()

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp), seed=later["seed_derivation"]["matched_seed"]
            )
            receipt, _, _ = self.write_retained_authority(
                campaign, later, CANDIDATE13_PREAUDIT
            )
            real_run = subprocess.run

            def reject_verifier_execution(arguments, *args, **kwargs):
                if arguments[0] == str(self.collector.DRAND_NODE):
                    raise AssertionError(
                        "target+1 reached BLS verification instead of failing round selection"
                    )
                return real_run(arguments, *args, **kwargs)

            with mock.patch.object(
                self.collector.subprocess, "run", side_effect=reject_verifier_execution
            ), self.assertRaisesRegex(ValueError, "unique|first.*round|target round"):
                self.collect_with_randomness(
                    campaign,
                    fixture,
                    profile,
                    serving,
                    glm_profile,
                    receipt,
                    candidate_hash=source["candidate_hash"],
                    freeze_commit=source["freeze_commit"],
                )


    @staticmethod
    def set_dsv4_safety(profile: Path, safety: dict[str, object]) -> None:
        value = json.loads(profile.read_text(encoding="utf-8"))
        value["safety"] = safety
        profile.write_text(
            json.dumps(value, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    def test_collects_exact_twenty_safe_matched_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            records = self.collect_current(
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

    def test_collector_accepts_exact_owner_dsv4_safety_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp)
            )
            self.set_dsv4_safety(profile, dict(self.DSV4_SAFETY_ENVELOPE))
            records = self.collect_current(
                campaign, fixture, profile, serving, glm_profile
            )
            self.assertEqual(len(records), 20)

    def test_collector_rejects_stale_or_malformed_dsv4_safety_envelope(self):
        malformed = {
            "stale_105_107": {
                **self.DSV4_SAFETY_ENVELOPE,
                "memory_high_gib": 105,
                "memory_max_gib": 107,
            },
            "float_minimum_start": {
                **self.DSV4_SAFETY_ENVELOPE,
                "minimum_start_gib": 110.0,
            },
            "float_memory_high": {
                **self.DSV4_SAFETY_ENVELOPE,
                "memory_high_gib": 100.0,
            },
            "float_memory_max": {
                **self.DSV4_SAFETY_ENVELOPE,
                "memory_max_gib": 102.0,
            },
            "string_memory_max": {
                **self.DSV4_SAFETY_ENVELOPE,
                "memory_max_gib": "102",
            },
            "high_not_below_max": {
                **self.DSV4_SAFETY_ENVELOPE,
                "memory_high_gib": 102,
            },
            "wrong_minimum_start": {
                **self.DSV4_SAFETY_ENVELOPE,
                "minimum_start_gib": 109,
            },
            "extra_key": {
                **self.DSV4_SAFETY_ENVELOPE,
                "diagnostic_override": 1,
            },
            "missing_memory_high": {
                key: value
                for key, value in self.DSV4_SAFETY_ENVELOPE.items()
                if key != "memory_high_gib"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(
                Path(tmp)
            )
            for label, safety in malformed.items():
                with self.subTest(label=label):
                    self.set_dsv4_safety(profile, safety)
                    with self.assertRaisesRegex(
                        ValueError, "approved DeepSeek profile is invalid"
                    ):
                        self.collect_current(
                            campaign, fixture, profile, serving, glm_profile
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
                len(self.collect_current(
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
                self.collect_current(
                    campaign, fixture, profile, serving, glm_profile
                )

    def test_rejects_missing_memory_and_duplicate_server_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            first = campaign / "block0-seq0-armA"
            (first / "samples.log").unlink()
            with self.assertRaisesRegex(ValueError, "samples|canonical safety"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            source = campaign / "block0-seq0-armA" / "process.identity"
            target = campaign / "block0-seq3-armA" / "process.identity"
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "fresh servers|server boot"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_short_geometry_and_wrong_glm_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            first = campaign / "block0-seq0-armA"
            result = json.loads((first / "result.json").read_text())
            result["cells"] = [result["cells"][0]]
            (first / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "32K-class"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

            second = Path(tmp) / "second"
            second.mkdir()
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(second)
            value = json.loads(glm_profile.read_text())
            value["binary_sha256"] = "9" * 64
            glm_profile.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "GLM binary"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_unequal_prompts_and_wrong_runtime_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq1-armB" / "result.json"
            value = json.loads(target.read_text())
            value["cells"][1]["reps"][0]["prompt_sha256"] = "8" * 64
            target.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "unequal prompt bytes"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq0-armA" / "runtime.config"
            target.write_text(target.read_text().replace("a" * 64, "7" * 64))
            with self.assertRaisesRegex(ValueError, "runtime configuration"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

    def test_rejects_inflated_usage_and_unbound_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            target = campaign / "block0-seq0-armA" / "result.json"
            value = json.loads(target.read_text())
            value["cells"][1]["reps"][0]["prompt_tokens"] = 999_999_999
            target.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "production prompt"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            value = json.loads(glm_profile.read_text())
            value["artifact_sha256"]["scripts/30_bench_speed.py"] = "9" * 64
            glm_profile.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "campaign artifact"):
                self.collect_current(campaign, fixture, profile, serving, glm_profile)

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
                self.collect_current(
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
                self.collect_current(
                    campaign, fixture, profile, serving, glm_profile
                )

    def test_authentic_kill_floor_configuration_is_not_a_breach_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "samples.log").write_text("sample\n", encoding="ascii")
            (directory / "safety.kernel.log").write_text("", encoding="ascii")

            def write_main(extra: str = ""):
                main = (
                    "2026-08-15T00:00:00+00:00 SAFE_RUN start tag=matched "
                    "vlimit_kb=419430400 kill_floor_gib=40 min_start_gib=110 "
                    "timeout_s=5400 allow_cgroup_high=0\n"
                    "2026-08-15T00:00:01+00:00 cgroup_final current_bytes=1 "
                    "peak_bytes=2 swap_current_bytes=0 events=low 0,high 0,max 0,"
                    "oom 0,oom_kill 0,oom_group_kill 0,\n"
                    + extra
                    + "2026-08-15T00:00:02+00:00 SAFE_RUN end rc=0 killed=no "
                    "(124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n"
                )
                (directory / "safety.main.log").write_text(main, encoding="ascii")
                digests = [
                    hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("safety.main.log", "samples.log", "safety.kernel.log")
                ]
                (directory / "safety.wrapper.out").write_text(
                    "SAFE_RUN_DONE rc=0 killed=no dir=/tmp/safe "
                    f"main_sha256={digests[0]} samples_sha256={digests[1]} "
                    f"kernel_sha256={digests[2]}\n",
                    encoding="ascii",
                )

            write_main()
            self.collector._parse_canonical_safety(directory)
            write_main("2026-08-15T00:00:01+00:00 KILL_FLOOR breached: 7 GiB available\n")
            with self.assertRaisesRegex(ValueError, "records a failure"):
                self.collector._parse_canonical_safety(directory)

    def test_rejects_noncanonical_or_hidden_nonzero_cgroup_events(self):
        valid_items = [
            "low=0", "low_delta=0", "high=0", "high_delta=0",
            "max=0", "max_delta=0",
            "oom=0", "oom_delta=0", "oom_kill=0", "oom_kill_delta=0",
            "oom_group_kill=0", "oom_group_kill_delta=0",
        ]
        mutations = {
            "duplicate_high_delta": valid_items[:4] + ["high_delta=9", "high_delta=0"] + valid_items[4:],
            "duplicate_max_delta": valid_items[:6] + ["max_delta=9", "max_delta=0"] + valid_items[6:],
            "duplicate_oom_delta": valid_items[:8] + ["oom_delta=9", "oom_delta=0"] + valid_items[8:],
            "duplicate_oom_kill_delta": valid_items[:10] + ["oom_kill_delta=9", "oom_kill_delta=0"] + valid_items[10:],
            "duplicate_oom_group_kill_delta": valid_items + ["oom_group_kill_delta=9", "oom_group_kill_delta=0"],
            "unknown_key": valid_items + ["pressure_delta=0"],
            "malformed_value": [*valid_items[:-1], "oom_group_kill_delta=zero"],
            "missing_key": valid_items[:-1],
            "missing_low_pair": valid_items[2:],
            "interior_empty_item": [*valid_items[:4], "", *valid_items[4:]],
            "mixed_separator": ["low 0", *valid_items[1:]],
            "nonzero_absolute": [*valid_items[:2], "high=1", *valid_items[3:]],
            "nonzero_delta": [*valid_items[:3], "high_delta=1", *valid_items[4:]],
        }
        mutations.update(
            {
                f"missing_{item.split('=', 1)[0]}": valid_items[:index] + valid_items[index + 1:]
                for index, item in enumerate(valid_items)
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "samples.log").write_text("sample\n", encoding="ascii")
            (directory / "safety.kernel.log").write_text("", encoding="ascii")

            def write_main(items):
                (directory / "safety.main.log").write_text(
                    "2026-08-15T00:00:01+00:00 cgroup_final current_bytes=1 "
                    "peak_bytes=2 swap_current_bytes=0 events="
                    + ",".join(items)
                    + "\n2026-08-15T00:00:02+00:00 SAFE_RUN end rc=0 killed=no "
                    "(124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n",
                    encoding="ascii",
                )
                digests = [
                    hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("safety.main.log", "samples.log", "safety.kernel.log")
                ]
                (directory / "safety.wrapper.out").write_text(
                    "SAFE_RUN_DONE rc=0 killed=no dir=/tmp/safe "
                    f"main_sha256={digests[0]} samples_sha256={digests[1]} "
                    f"kernel_sha256={digests[2]}\n",
                    encoding="ascii",
                )

            write_main(valid_items)
            self.collector._parse_canonical_safety(directory)
            write_main([
                "low 0", "high 0", "max 0", "oom 0", "oom_kill 0",
                "oom_group_kill 0", "",
            ])
            self.collector._parse_canonical_safety(directory)
            for name, items in mutations.items():
                with self.subTest(name=name):
                    write_main(items)
                    with self.assertRaisesRegex(ValueError, "canonical safety"):
                        self.collector._parse_canonical_safety(directory)

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
                    self.collect_current(
                        campaign, fixture, profile, serving, glm_profile
                    )

        with tempfile.TemporaryDirectory() as tmp:
            campaign, fixture, profile, serving, glm_profile = self.make_campaign(Path(tmp))
            path = campaign / "block0-seq0-armA" / "model.device-inode-size"
            path.write_text("66306:679228:211075856448\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "executed command"):
                self.collect_current(
                    campaign, fixture, profile, serving, glm_profile
                )


if __name__ == "__main__":
    unittest.main()
