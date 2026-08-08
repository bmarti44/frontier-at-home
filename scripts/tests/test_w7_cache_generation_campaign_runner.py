#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
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
            "GLM_SAFE_KILL_FLOOR_GIB",
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

            ready_read, ready_write = os.pipe()
            stop_read, stop_write = os.pipe()
            holder_pid = os.fork()
            if holder_pid == 0:
                try:
                    os.close(ready_read); os.close(stop_write)
                    holder_fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                    fcntl.flock(holder_fd, fcntl.LOCK_EX)
                    os.write(ready_write, b"1")
                    os.read(stop_read, 1)
                    os.close(holder_fd)
                finally:
                    os._exit(0)
            os.close(ready_write); os.close(stop_read)
            try:
                self.assertEqual(os.read(ready_read, 1), b"1")
                rejected = subprocess.run(
                    [str(MODULE.CGROUP), "--tag", "w7-lock-foreign", "--", "/usr/bin/true"],
                    env=base, capture_output=True, text=True, check=False,
                )
                self.assertIn("parent inference-lock ownership mismatch", rejected.stderr)
            finally:
                os.write(stop_write, b"1")
                os.close(stop_write); os.close(ready_read)
                os.waitpid(holder_pid, 0)

            locked_fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            fcntl.flock(locked_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stop_read, stop_write = os.pipe()
            inheritor_pid = os.fork()
            if inheritor_pid == 0:
                try:
                    os.close(stop_write)
                    os.read(stop_read, 1)
                finally:
                    os._exit(0)
            os.close(stop_read)
            os.close(locked_fd)
            try:
                inherited_bypass = subprocess.run(
                    [str(MODULE.CGROUP), "--tag", "w7-lock-inherited", "--", "/usr/bin/true"],
                    env=base, capture_output=True, text=True, check=False,
                )
                self.assertIn("parent inference-lock ownership mismatch", inherited_bypass.stderr)
            finally:
                os.write(stop_write, b"1"); os.close(stop_write)
                os.waitpid(inheritor_pid, 0)

            post = subprocess.run(
                ["/usr/bin/flock", "-n", str(lock), "/usr/bin/true"], check=False
            )
            self.assertEqual(post.returncode, 0, "mutation leaked the global inference lock")
        finally:
            os.close(descriptor)

    def test_failure_triplet_cannot_preserve_pass_summary(self) -> None:
        prior = b'{"verdict":"PASS"}\n'
        with tempfile.TemporaryDirectory(prefix="w7-failure-test-") as raw:
            attempt = Path(raw)
            (attempt / "raw.jsonl").write_bytes(b"")
            (attempt / "summary.json").write_bytes(prior)
            with mock.patch.object(MODULE, "_ACTIVE_ATTEMPT", attempt), mock.patch.object(
                MODULE, "_ACTIVE_CANDIDATE", "a" * 40
            ):
                MODULE.finalize_failure_triplet(RuntimeError("injected"))
            summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
            displaced = (attempt / "summary.pre-finalization.json").read_bytes()
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(manifest["verdict"], "FAIL")
        self.assertEqual(displaced, prior)
        self.assertEqual(
            manifest["artifacts"]["summary.pre-finalization.json"],
            hashlib.sha256(prior).hexdigest(),
        )

    def test_attempt_is_active_before_pending_interrupt_is_delivered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-activation-test-") as raw:
            parent = Path(raw)
            original = Path.mkdir
            previous = signal.getsignal(signal.SIGINT)

            def interrupt_after_mkdir(path: Path, *args: object, **kwargs: object) -> None:
                original(path, *args, **kwargs)
                os.kill(os.getpid(), signal.SIGINT)

            def raise_interrupt(_signal: int, _frame: object) -> None:
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, raise_interrupt)
            try:
                with mock.patch.object(Path, "mkdir", interrupt_after_mkdir):
                    with self.assertRaises(KeyboardInterrupt):
                        MODULE.create_and_activate_attempt(parent, "a" * 40, "fixed")
                self.assertEqual(MODULE._ACTIVE_ATTEMPT, parent / "attempt-fixed")
                MODULE.finalize_failure_triplet(KeyboardInterrupt())
                self.assertTrue((parent / "attempt-fixed/manifest.json").is_file())
            finally:
                signal.signal(signal.SIGINT, previous)
                MODULE._ACTIVE_ATTEMPT = None
                MODULE._ACTIVE_CANDIDATE = None

    def test_termination_signals_raise_for_failure_finalization(self) -> None:
        previous = MODULE.install_campaign_signal_handlers()
        try:
            for selected in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                with self.assertRaises(MODULE.CampaignInterrupted) as raised:
                    os.kill(os.getpid(), selected)
                self.assertEqual(raised.exception.signum, selected)
        finally:
            MODULE.restore_campaign_signal_handlers(previous)

    def test_interrupted_containment_is_terminated_and_reaped(self) -> None:
        events: list[str] = []

        class FakeProcess:
            pid = 4242
            returncode = None
            terminated = False
            waited = False
            killed = False

            def communicate(self) -> tuple[str, str]:
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("signal handler did not interrupt communicate")

            def terminate(self) -> None:
                events.append("terminate")
                self.terminated = True

            def wait(self, timeout: int | None = None) -> int:
                events.append("wait")
                self.waited = True
                self.returncode = -15
                return self.returncode

            def kill(self) -> None:
                self.killed = True

        process = FakeProcess()
        previous = MODULE.install_campaign_signal_handlers()
        try:
            with mock.patch.object(MODULE.subprocess, "Popen", return_value=process), mock.patch.object(
                MODULE, "stop_exact_containment_unit",
                side_effect=lambda unit: events.append(f"stop:{unit}"),
            ) as stop_unit, mock.patch.object(
                MODULE, "_unit_is_stopped", return_value=True,
            ), mock.patch.object(
                MODULE.os, "killpg", side_effect=lambda _pid, _sig: events.append("terminate"),
            ), mock.patch.object(MODULE, "server_pids", return_value=[]), mock.patch.object(
                MODULE, "_listener_is_active", return_value=False,
            ):
                with self.assertRaises(MODULE.CampaignInterrupted):
                    MODULE.run_contained_command(["/usr/bin/true"], {}, "w7-test")
        finally:
            MODULE.restore_campaign_signal_handlers(previous)
        stop_unit.assert_called_once_with("glm52-w7-test-4242.service")
        self.assertEqual(events[:3], ["terminate", "wait", "stop:glm52-w7-test-4242.service"])
        self.assertTrue(process.waited)
        self.assertFalse(process.killed)
        self.assertIsNone(MODULE._ACTIVE_CONTAINMENT)

    def test_containment_unit_name_rejects_untrusted_tag(self) -> None:
        for value in ("", "../escape", "has space", "a" * 41):
            with self.assertRaises(MODULE.CampaignError):
                MODULE.containment_unit_name(value, 123)
        self.assertEqual(
            MODULE.containment_unit_name("w7p-b0p0-a.b", 123),
            "glm52-w7p-b0p0-a-b-123.service",
        )

    def test_unit_state_does_not_treat_control_failure_as_not_found(self) -> None:
        bus_failure = subprocess.CompletedProcess(
            ["systemctl"], 1, "", "Failed to connect to bus: No medium found\n"
        )
        with mock.patch.object(MODULE.subprocess, "run", return_value=bus_failure):
            with self.assertRaises(MODULE.CampaignError):
                MODULE._unit_state("glm52-w7-test-123.service")
        missing = subprocess.CompletedProcess(
            ["systemctl"], 1, "", "Unit glm52-w7-test-123.service could not be found.\n"
        )
        with mock.patch.object(MODULE.subprocess, "run", return_value=missing):
            self.assertTrue(MODULE._unit_is_stopped("glm52-w7-test-123.service"))

    def test_unit_state_requires_complete_unique_fail_closed_schema(self) -> None:
        unit = "glm52-w7-test-123.service"
        masked_active = subprocess.CompletedProcess(
            ["systemctl"], 0,
            "LoadState=masked\nActiveState=active\nSubState=running\nMainPID=9\nControlPID=0\n", "",
        )
        with mock.patch.object(MODULE.subprocess, "run", return_value=masked_active):
            self.assertFalse(MODULE._unit_is_stopped(unit))
        malformed_cases = (
            "LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\n",
            "LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\nControlPID=0\nControlPID=0\n",
            "LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=zero\nControlPID=0\n",
            "LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\nControlPID=0\nUnexpected=value\n",
        )
        for stdout in malformed_cases:
            completed = subprocess.CompletedProcess(["systemctl"], 0, stdout, "")
            with self.subTest(stdout=stdout), mock.patch.object(
                MODULE.subprocess, "run", return_value=completed,
            ):
                with self.assertRaises(MODULE.CampaignError):
                    MODULE._unit_is_stopped(unit)
        masked_stopped = subprocess.CompletedProcess(
            ["systemctl"], 0,
            "LoadState=masked\nActiveState=inactive\nSubState=dead\nMainPID=0\nControlPID=0\n", "",
        )
        with mock.patch.object(MODULE.subprocess, "run", return_value=masked_stopped):
            self.assertTrue(MODULE._unit_is_stopped(unit))

    def test_cleanup_reaps_before_stop_and_rechecks_unit_last(self) -> None:
        events: list[str] = []
        process = mock.Mock(pid=4242, returncode=None)
        with mock.patch.object(
            MODULE, "_terminate_and_reap", side_effect=lambda _: events.append("reap")
        ), mock.patch.object(
            MODULE, "stop_exact_containment_unit", side_effect=lambda _: events.append("stop")
        ), mock.patch.object(
            MODULE, "_unit_is_stopped", side_effect=lambda _: events.append("final-unit") or True
        ), mock.patch.object(
            MODULE, "server_pids", side_effect=lambda: events.append("servers") or []
        ), mock.patch.object(
            MODULE, "_listener_is_active", side_effect=lambda: events.append("listener") or False
        ):
            MODULE._cleanup_interrupted_containment(process, "glm52-w7-test-4242.service")
        self.assertEqual(events, ["reap", "stop", "final-unit", "servers", "listener"])

    def test_cleanup_control_failure_still_proves_no_survivors_and_fails(self) -> None:
        events: list[str] = []
        process = mock.Mock(pid=4242, returncode=None)
        control_error = MODULE.CampaignError("bus unavailable")
        with mock.patch.object(
            MODULE, "_terminate_and_reap", side_effect=lambda _: events.append("reap")
        ), mock.patch.object(
            MODULE, "stop_exact_containment_unit", side_effect=control_error
        ), mock.patch.object(
            MODULE, "_kill_and_verify_containment_cgroup",
            side_effect=lambda _: events.append("cgroup-empty"), create=True,
        ), mock.patch.object(
            MODULE, "server_pids", side_effect=lambda: events.append("servers") or []
        ), mock.patch.object(
            MODULE, "_listener_is_active", side_effect=lambda: events.append("listener") or False
        ):
            with self.assertRaisesRegex(MODULE.CampaignError, "bus unavailable"):
                MODULE._cleanup_interrupted_containment(
                    process, "glm52-w7-test-4242.service"
                )
        self.assertEqual(events, ["reap", "cgroup-empty", "servers", "listener"])

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
