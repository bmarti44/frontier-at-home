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
    def test_deepseek_uses_largest_proven_context_and_bounded_batches(self):
        runtime = load_runtime()
        command, environment = runtime.server_invocation(
            "dsv4", Path("/candidate/llama-server"), Path("/models/dsv4.gguf"), 8013
        )
        self.assertEqual(environment, {})
        self.assertEqual(command[0], "/candidate/llama-server")
        self.assertIn("1048576", command)
        self.assertEqual(command[command.index("-b") + 1], "512")
        self.assertEqual(command[command.index("-ub") + 1], "256")
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


if __name__ == "__main__":
    unittest.main()
