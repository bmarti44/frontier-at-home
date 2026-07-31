#!/usr/bin/env python3
"""Production-path contracts for the sudo-free matched foundation runtime."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "69_foundation_user_runtime.py"


def load_runtime():
    spec = importlib.util.spec_from_file_location("foundation_user_runtime", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FoundationRuntimeTests(unittest.TestCase):
    def test_cli_requires_exact_arm_identity_arguments(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--binary-sha256", completed.stderr)
        self.assertIn("--configuration-sha256", completed.stderr)
        self.assertIn("--fixture-sha256", completed.stderr)

    def test_deepseek_uses_largest_proven_context_and_bounded_batches(self):
        runtime = load_runtime()
        command, environment = runtime.server_invocation(
            "dsv4", Path("/candidate/llama-server"), Path("/models/dsv4.gguf"), 8013
        )
        self.assertEqual(environment, {})
        self.assertEqual(command[0], "/candidate/llama-server")
        self.assertIn("1048576", command)
        # Fresh 1M diagnostics loaded at 512/256 and 256/128, but request-time
        # allocation crossed the fixed 15 GiB watchdog floor (14.89 and 15.00
        # GiB respectively). Keep full context and reduce only transient
        # request buffers enough to leave measurable safety headroom.
        self.assertEqual(command[command.index("-b") + 1], "128")
        self.assertEqual(command[command.index("-ub") + 1], "64")
        self.assertIn("--no-mmap", command)
        self.assertEqual(command[command.index("--cache-ram") + 1], "0")

    def test_glm_uses_largest_currently_capable_context_and_packed_format(self):
        runtime = load_runtime()
        command, environment = runtime.server_invocation(
            "glm52", Path("/candidate/ds4-server"), Path("/models/glm.gguf"), 8014
        )
        self.assertEqual(command[0], "/candidate/ds4-server")
        self.assertEqual(command[command.index("-c") + 1], "8192")
        self.assertEqual(environment["DS4_GLM_COMPACT_CACHE_AFFINE_INT8"], "1")
        self.assertEqual(environment["DS4_CUDA_EXPERT_CACHE_GB"], "40")

    def test_cgroup_contract_rejects_swap_and_weak_limits(self):
        runtime = load_runtime()
        runtime.validate_cgroup("dsv4", 105 * 2**30, 110 * 2**30, 0)
        runtime.validate_cgroup("glm52", 68 * 2**30, 72 * 2**30, 0)
        for values in (
            ("dsv4", 105 * 2**30, 110 * 2**30, 1),
            ("dsv4", 100 * 2**30, 110 * 2**30, 0),
            ("glm52", 68 * 2**30, 80 * 2**30, 0),
        ):
            with self.assertRaises(ValueError):
                runtime.validate_cgroup(*values)

    def test_cgroup_limits_are_read_from_current_unit_files(self):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            cgroup = Path(tmp)
            (cgroup / "memory.high").write_text(str(105 * 2**30))
            (cgroup / "memory.max").write_text(str(110 * 2**30))
            (cgroup / "memory.swap.max").write_text("0")
            self.assertEqual(
                runtime.read_cgroup_limits(cgroup),
                (105 * 2**30, 110 * 2**30, 0),
            )
            (cgroup / "memory.max").write_text("max")
            with self.assertRaises(ValueError):
                runtime.read_cgroup_limits(cgroup)

    def test_baseline_uses_raw_timestamps_and_cold_then_warm_reps(self):
        runtime = load_runtime()
        timestamps = [10_000_000_000 + index * 500_000_000 for index in range(128)]
        result = {
            "suite_valid": True,
            "metadata": {"reps": 2, "model": "deepseek-v4-flash"},
            "cells": [{
                "ctx_tokens": 0,
                "valid": True,
                "reps": [
                    {"valid": True, "ttft_s": 9.0},
                    {
                        "valid": True,
                        "ttft_s": 2.0,
                        "prompt_tokens": 64,
                        "prefill_tok_s": 32.0,
                        "completion_tokens": 128,
                        "token_timestamps_ns": timestamps,
                    },
                ],
            }],
        }
        baseline = runtime.baseline_from_result(
            result,
            profile="dsv4",
            server_instance_id="a" * 64,
            fixture_sha256="b" * 64,
            binary_sha256="c" * 64,
            configuration_sha256="d" * 64,
            available_memory_gib=15.5,
        )
        self.assertEqual(baseline["cold_ttft_seconds"], 9.0)
        self.assertEqual(baseline["warm_ttft_seconds"], 2.0)
        self.assertEqual(baseline["prefill_seconds"], 2.0)
        self.assertEqual(len(baseline["token_timestamps"]), 128)
        self.assertEqual(baseline["failures"], [])

    def test_probe_command_is_fixed_and_glm_tokenizer_bound(self):
        runtime = load_runtime()
        dsv4 = runtime.benchmark_invocation(
            "dsv4", Path("/evidence/dsv4.json"), 8013, 123, None, None
        )
        self.assertIn("scripts/30_bench_speed.py", " ".join(dsv4))
        self.assertEqual(dsv4[dsv4.index("--reps") + 1], "2")
        self.assertEqual(dsv4[dsv4.index("--max-tokens") + 1], "160")
        self.assertEqual(dsv4[dsv4.index("--min-completion-tokens") + 1], "128")
        self.assertEqual(dsv4[dsv4.index("--model-id") + 1], "deepseek-v4-flash")

        glm = runtime.benchmark_invocation(
            "glm52",
            Path("/evidence/glm.json"),
            8014,
            123,
            Path("/tokenizer.json"),
            "e" * 64,
        )
        self.assertEqual(glm[glm.index("--model-id") + 1], "glm-5.2")
        self.assertEqual(
            glm[glm.index("--output-tokenizer-path") + 1], "/tokenizer.json"
        )
        self.assertEqual(
            glm[glm.index("--output-tokenizer-sha256") + 1], "e" * 64
        )

    def test_artifact_hash_verification_fails_closed(self):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.write_bytes(b"verified")
            expected = hashlib.sha256(b"verified").hexdigest()
            runtime.verify_artifact(artifact, expected)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                runtime.verify_artifact(artifact, expected)

    def test_deepseek_shards_are_bound_to_manifest_and_evicted(self):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "model-00001-of-00001.gguf"
            shard.write_bytes(b"shard")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "name": shard.name,
                                "bytes": 5,
                                "sha256": hashlib.sha256(b"shard").hexdigest(),
                            }
                        ]
                    }
                )
            )
            runtime.verify_dsv4_shards(manifest)
            shard.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                runtime.verify_dsv4_shards(manifest)

    def test_supervisor_arms_watchdog_runs_probe_and_cleans_process_group(self):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "fake-server.py"
            server.write_text(
                "#!/usr/bin/env python3\n"
                "import http.server,sys\n"
                "port=int(sys.argv[1])\n"
                "class H(http.server.BaseHTTPRequestHandler):\n"
                " def do_GET(self):\n"
                "  self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
                " def log_message(self,*args): pass\n"
                "http.server.ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()\n",
                encoding="utf-8",
            )
            server.chmod(0o700)
            port = 18000 + os.getpid() % 1000
            result = runtime.supervise_process(
                [str(server), str(port)],
                {},
                root / "arm",
                port=port,
                watchdog_floor_gib=1,
                startup_timeout_seconds=10,
                probe_command=["/bin/sh", "-c", "printf passed >\"$1\"", "probe", str(root / "probe")],
            )
            self.assertTrue((root / "probe").is_file())
            self.assertRegex(result["server_instance_id"], r"^[0-9a-f]{64}$")
            self.assertGreater(result["available_memory_gib"], 10)
            self.assertIn("DISARMED", (root / "arm" / "memwatch.log").read_text())
            pid = result["server_pid"]
            self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_execute_arm_writes_fixed_baseline_from_production_probe(self):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-server"
            model = root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            binary.chmod(0o700)
            cgroup = root / "cgroup"
            cgroup.mkdir()
            for name, value in (
                ("memory.high", 105 * 2**30),
                ("memory.max", 110 * 2**30),
                ("memory.swap.max", 0),
            ):
                (cgroup / name).write_text(str(value))
            out = root / "arm"
            timestamps = [1_000_000_000 + index * 100_000_000 for index in range(128)]

            def fake_supervise(command, environment, output, **kwargs):
                output.mkdir()
                result_path = output / "result.json"
                result_path.write_text(
                    json.dumps(
                        {
                            "suite_valid": True,
                            "metadata": {"reps": 2, "model": "deepseek-v4-flash"},
                            "cells": [{
                                "ctx_tokens": 0,
                                "valid": True,
                                "reps": [
                                    {"valid": True, "ttft_s": 8.0},
                                    {
                                        "valid": True,
                                        "ttft_s": 1.0,
                                        "prompt_tokens": 32,
                                        "prefill_tok_s": 32.0,
                                        "completion_tokens": 128,
                                        "token_timestamps_ns": timestamps,
                                    },
                                ],
                            }],
                        }
                    )
                )
                return {
                    "server_instance_id": "a" * 64,
                    "server_pid": 123,
                    "available_memory_gib": 15.0,
                }

            with (
                mock.patch.object(runtime, "current_cgroup_path", return_value=cgroup),
                mock.patch.object(runtime, "supervise_process", side_effect=fake_supervise),
                mock.patch.object(runtime, "_mem_available_gib", return_value=115.0),
            ):
                baseline = runtime.execute_arm(
                    profile="dsv4",
                    out=out,
                    binary=binary,
                    binary_sha256=hashlib.sha256(b"binary").hexdigest(),
                    model=model,
                    model_sha256=hashlib.sha256(b"model").hexdigest(),
                    configuration_sha256="b" * 64,
                    fixture_sha256="c" * 64,
                    port=8013,
                    seed=123,
                    tokenizer=None,
                    tokenizer_sha256=None,
                )
            self.assertEqual(baseline["profile"], "dsv4")
            self.assertEqual(baseline["binary_sha256"], hashlib.sha256(b"binary").hexdigest())
            self.assertEqual(json.loads((out / "baseline.json").read_text()), baseline)


if __name__ == "__main__":
    unittest.main()
