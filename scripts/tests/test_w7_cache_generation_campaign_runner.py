#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
SPEC = importlib.util.spec_from_file_location("w7_cache_campaign_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7CacheGenerationCampaignRunnerTest(unittest.TestCase):
    def make_arm(self, arm: str = "off") -> tuple[tempfile.TemporaryDirectory, Path, str]:
        temporary = tempfile.TemporaryDirectory(prefix="w7-runner-test-")
        out = Path(temporary.name)
        response_id = "cmpl-test"
        lines = ["ds4-server: listening on http://127.0.0.1:8097"]
        if arm == "off":
            lines.append(MODULE.FALSE_FLUSH)
        for index in range(1, 129):
            lines.append(
                f"DS4_TOKEN_TIMING request={response_id} index={index} "
                f"monotonic_ns={2_000_000_000 + index * 500_000_000} token={index}"
            )
        lines.append("ds4-server: shutdown requested")
        (out / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        client = {
            "request_start_ns": 1_000_000_000,
            "response_id": response_id,
            "generated_text": "test",
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 5066,
                "completion_tokens": 128,
                "total_tokens": 5194,
                "prompt_tokens_details": {"cached_tokens": 5044, "cache_write_tokens": 22},
            },
            "done": True,
        }
        (out / "primary-client.json").write_text(json.dumps(client), encoding="utf-8")
        (out / "live-response.json").write_text("{}\n", encoding="utf-8")
        (out / "child-exit.json").write_text(
            '{"exit_status":0,"forced_kill":false,"shutdown_requested":true}\n',
            encoding="utf-8",
        )
        for index in range(1, 130):
            (out / f"logits.sync{index}.start5044.prompt5066.suffix22").write_bytes(
                index.to_bytes(4, "little")
            )
        crash_root = Path("/home/bmarti44/.local/state/glm52-crashlog")
        crash = crash_root / f"w7-runner-test-{Path(temporary.name).name}"
        crash.mkdir(mode=0o700)
        self.addCleanup(lambda: [path.unlink() for path in crash.iterdir()] and crash.rmdir())
        final_lines = []
        for name in ("server.log", "live-response.json", "primary-client.json", "child-exit.json"):
            path = out / name
            metadata = path.stat()
            final_lines.append(
                f"final_artifact_verified path={path} "
                f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()} "
                f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}"
            )
        (crash / "main.log").write_text(
            "cgroup_final current_bytes=1 peak_bytes=2 swap_current_bytes=0 "
            "events=low 0,high 1,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n"
            + "\n".join(final_lines) + "\n",
            encoding="utf-8",
        )
        (crash / "samples.log").write_text(
            "x mem_avail_kb=49000000 cgroup_swap_current_bytes=0\n", encoding="utf-8"
        )
        (crash / "kernel.log").write_text("-- No entries --\n", encoding="utf-8")
        receipt = (
            f"SAFE_RUN_DONE rc=0 killed=no dir={crash} "
            f"main_sha256={hashlib.sha256((crash / 'main.log').read_bytes()).hexdigest()} "
            f"samples_sha256={hashlib.sha256((crash / 'samples.log').read_bytes()).hexdigest()} "
            f"kernel_sha256={hashlib.sha256((crash / 'kernel.log').read_bytes()).hexdigest()}\n"
        )
        return temporary, out, receipt

    def test_self_test_validates_dependencies_without_starting_engine(self) -> None:
        before = subprocess.run(
            ["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True,
            check=False,
        ).stdout
        completed = subprocess.run(
            ["/usr/bin/python3", str(RUNNER), "--self-test"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        after = subprocess.run(
            ["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True,
            check=False,
        ).stdout
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "W7_CACHE_GENERATION_CAMPAIGN_SELFTEST_OK\n")
        self.assertEqual(after, before)

    def test_schedule_is_deterministic_and_uses_only_abba_baab(self) -> None:
        seed = "a" * 64
        first = MODULE.derive_schedules(seed)
        self.assertEqual(first, MODULE.derive_schedules(seed))
        self.assertEqual(len(first), 5)
        self.assertTrue(all(value in {"ABBA", "BAAB"} for value in first))
        self.assertNotEqual(first, MODULE.derive_schedules("b" * 64))
        for invalid in ("", "a" * 63, "A" * 64, "z" * 64):
            with self.assertRaises(ValueError):
                MODULE.derive_schedules(invalid)

    def test_runner_declares_fixed_containment_and_measurement_surface(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required = (
            "glm_cgroup_run.sh", "glm_safe_run.sh", "DS4_TOKEN_TIMING_LOG",
            "DS4_GLM_LOGIT_DUMP_ALL", "DS4_CUDA_STABLE_MODEL_REMAP",
            "GLM_SAFE_MEMORY_HIGH_GIB", "GLM_SAFE_KILL_FLOOR_GIB",
            "GLM_SAFE_MIN_START_GIB", "GLM_SAFE_TIMEOUT_S",
            "MemorySwapMax", "false_generation_flushes", "server_fresh",
            "manifest.json", "raw.jsonl", "summary.json",
            "score_campaign_rows", "pgrep", "ds4-server",
        )
        for value in required:
            self.assertIn(value, source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)
        for value in (
            "os.O_NOFOLLOW", "pass_fds", "/proc/self/fd/", "fcntl.flock",
            "CAMPAIGN_LOCK", "DRAND_FREEZE_FLOOR_ROUND", "frozen_scorer_bytes",
            "finalize_failure_triplet",
        ):
            self.assertIn(value, source)

    def test_raw_arm_aggregation_accepts_bound_evidence_and_rejects_mutations(self) -> None:
        temporary, out, receipt = self.make_arm()
        self.addCleanup(temporary.cleanup)
        with mock.patch.object(MODULE, "server_pids", return_value=[]), mock.patch.object(
            MODULE, "LOGIT_BYTES", 4
        ):
            row = MODULE.parse_arm(
                "off", 0, 0, out, 0, receipt, "4" * 64, "3" * 64
            )
        self.assertEqual(len(row["token_timestamps_ns"]), 128)
        self.assertEqual(row["safety"]["false_generation_flushes"], 1)
        self.assertEqual(row["safety"]["minimum_mem_available_kb"], 49_000_000)
        self.assertTrue((out / "safety/main.log").is_file())
        for copied in (out / "safety").iterdir():
            copied.unlink()
        (out / "safety").rmdir()

        client_path = out / "primary-client.json"
        client = json.loads(client_path.read_text(encoding="utf-8"))
        client["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        client_path.write_text(json.dumps(client), encoding="utf-8")
        with mock.patch.object(MODULE, "server_pids", return_value=[]), mock.patch.object(
            MODULE, "LOGIT_BYTES", 4
        ):
            with self.assertRaises(MODULE.CampaignError):
                MODULE.parse_arm("off", 0, 0, out, 0, receipt, "4" * 64, "3" * 64)

        for copied in (out / "safety").iterdir():
            copied.unlink()
        (out / "safety").rmdir()
        client["usage"]["prompt_tokens_details"]["cached_tokens"] = 5044
        client_path.write_text(json.dumps(client), encoding="utf-8")
        next(out.glob("logits.sync129.*")).unlink()
        with mock.patch.object(MODULE, "server_pids", return_value=[]), mock.patch.object(
            MODULE, "LOGIT_BYTES", 4
        ):
            with self.assertRaises(MODULE.CampaignError):
                MODULE.parse_arm("off", 0, 0, out, 0, receipt, "4" * 64, "3" * 64)

    def test_ambient_ds4_configuration_is_not_forwarded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-env-test-") as raw:
            out = Path(raw)
            with mock.patch.dict(os.environ, {"DS4_GLM_PREFETCH": "1"}, clear=True):
                environment, _ = MODULE.environment_for_arm("off", out, "4" * 64)
        self.assertNotIn("DS4_GLM_PREFETCH", environment)
        self.assertNotIn("GLM_SAFE_ALLOW_CGROUP_HIGH", environment)
        self.assertNotIn("GLM_SAFE_MEMORY_HIGH_GIB", environment)

    def test_memory_guard_uses_retained_descriptor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-env-test-") as raw:
            environment, _ = MODULE.environment_for_arm(
                "off", Path(raw), "4" * 64,
                memory_guard_path="/proc/123/fd/4",
            )
        self.assertEqual(environment["GLM_SAFE_MEMORY_GUARD_PATH"], "/proc/123/fd/4")

    def test_parent_lock_validation_distinguishes_owner_from_third_party(self) -> None:
        lock = MODULE.CAMPAIGN_LOCK
        descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        metadata = os.fstat(descriptor)
        base = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_KILL_FLOOR_GIB": "invalid-after-lock-check",
            "GLM_SAFE_PARENT_LOCK_PID": str(os.getpid()),
            "GLM_SAFE_PARENT_LOCK_START_TICKS": str(MODULE.process_start_ticks(os.getpid())),
            "GLM_SAFE_PARENT_LOCK_FD": str(descriptor),
            "GLM_SAFE_PARENT_LOCK_DEV_INO": f"{metadata.st_dev}:{metadata.st_ino}",
            "GLM_SAFE_PARENT_LOCK_KERNEL_KEY": MODULE.lock_kernel_key(metadata),
        }
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            accepted = subprocess.run(
                [str(MODULE.CGROUP), "--tag", "w7-lock-owner", "--", "/usr/bin/true"],
                env=base, capture_output=True, text=True, check=False,
            )
            self.assertIn("invalid cgroup resource configuration", accepted.stderr)
            self.assertNotIn("parent inference-lock ownership mismatch", accepted.stderr)
            fcntl.flock(descriptor, fcntl.LOCK_UN)

            holder = subprocess.Popen(
                ["/usr/bin/flock", "-x", str(lock), "/usr/bin/sleep", "10"]
            )
            try:
                for _ in range(100):
                    probe = subprocess.run(
                        ["/usr/bin/flock", "-n", str(lock), "/usr/bin/true"], check=False
                    )
                    if probe.returncode != 0:
                        break
                    time.sleep(0.01)
                rejected = subprocess.run(
                    [str(MODULE.CGROUP), "--tag", "w7-lock-foreign", "--", "/usr/bin/true"],
                    env=base, capture_output=True, text=True, check=False,
                )
                self.assertIn("parent inference-lock ownership mismatch", rejected.stderr)
            finally:
                holder.terminate()
                holder.wait(timeout=5)
        finally:
            os.close(descriptor)

    def test_failure_triplet_cannot_preserve_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-failure-test-") as raw:
            attempt = Path(raw)
            (attempt / "raw.jsonl").write_bytes(b"")
            (attempt / "summary.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")
            with mock.patch.object(MODULE, "_ACTIVE_ATTEMPT", attempt), mock.patch.object(
                MODULE, "_ACTIVE_CANDIDATE", "a" * 40
            ):
                MODULE.finalize_failure_triplet(RuntimeError("injected"))
            summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(manifest["verdict"], "FAIL")

    def test_safety_receipt_digest_mutation_is_rejected(self) -> None:
        temporary, out, receipt = self.make_arm()
        self.addCleanup(temporary.cleanup)
        marker = "main_sha256="
        start = receipt.index(marker) + len(marker)
        bad_receipt = receipt[:start] + "0" * 64 + receipt[start + 64:]
        with mock.patch.object(MODULE, "server_pids", return_value=[]), mock.patch.object(
            MODULE, "LOGIT_BYTES", 4
        ):
            with self.assertRaises(MODULE.CampaignError):
                MODULE.parse_arm("off", 0, 0, out, 0, bad_receipt, "4" * 64, "3" * 64)

    def test_scorer_executes_retained_verified_bytes(self) -> None:
        scorer_bytes = (ROOT / "scripts/90_score_w7_cache_generation_campaign.py").read_bytes()
        digest = hashlib.sha256(scorer_bytes).hexdigest()
        module = MODULE.load_scorer(scorer_bytes, digest)
        self.assertTrue(callable(module.score_campaign_rows))
        with self.assertRaises(MODULE.CampaignError):
            MODULE.load_scorer(scorer_bytes + b"\n# mutation\n", digest)

if __name__ == "__main__":
    unittest.main()
