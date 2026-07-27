#!/usr/bin/env python3
"""Memory-safety contract for matched engine measurements."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "03_memory_guard.py"
HARNESS = ROOT / "results" / "glm52-goal" / "harness" / "decisive_matched.sh"
GLM_SAFE = ROOT / "results" / "glm52-gates" / "harness" / "glm_safe_run.sh"
GLM_CGROUP = ROOT / "results" / "glm52-gates" / "harness" / "glm_cgroup_run.sh"
DSV4_LAUNCHER = ROOT / "scripts" / "21_serve_llamacpp.sh"
GLM_ARM = ROOT / "results" / "glm52-goal" / "harness" / "glm_decisive_arm.sh"


class MemoryGuardTests(unittest.TestCase):
    def test_requires_stable_consecutive_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemAvailable: 120000000 kB\n", encoding="ascii")
            result = subprocess.run(
                [
                    "python3",
                    str(GUARD),
                    "--required-gib",
                    "110",
                    "--stable-samples",
                    "3",
                    "--interval-seconds",
                    "0",
                    "--timeout-seconds",
                    "0",
                    "--meminfo",
                    str(meminfo),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"stable_samples_observed":3', result.stdout)

    def test_rejects_insufficient_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemAvailable: 80000000 kB\n", encoding="ascii")
            result = subprocess.run(
                [
                    "python3",
                    str(GUARD),
                    "--required-gib",
                    "110",
                    "--stable-samples",
                    "1",
                    "--interval-seconds",
                    "0",
                    "--timeout-seconds",
                    "0",
                    "--meminfo",
                    str(meminfo),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 1)


class MatchedHarnessContractTests(unittest.TestCase):
    def test_harness_waits_for_full_release_and_serializes_engines(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("03_memory_guard.py", source)
        self.assertIn("--required-gib 110", source)
        self.assertIn("--stable-samples 3", source)
        self.assertIn(
            "/run/dsv4/inference.lock",
            source + GLM_CGROUP.read_text(encoding="utf-8"),
        )
        self.assertIn("MATCHED_BLOCKS:-5", source)
        self.assertNotIn("21_serve_llamacpp.sh\" start >/dev/null 2>&1 || true", source)
        self.assertIn("watchdog_armed", source)
        self.assertIn("matched campaign finished without verified DeepSeek restoration", source)
        self.assertIn("glm_cgroup_run.sh", source)
        self.assertNotIn('bash "$SAFE" --tag "$label"', source)

    def test_harness_does_not_lower_emergency_floor(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertNotIn("GLM_SAFE_KILL_FLOOR_GIB=10", source)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=18", source)
        safe_source = GLM_SAFE.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_MIN_START_GIB=110", source)
        self.assertIn("setsid timeout", safe_source)
        self.assertIn("trap 'forward_signal TERM 143' TERM", safe_source)
        self.assertIn('kill -TERM -- "-$PG"', safe_source)
        self.assertIn('kill -0 -- "-$PG"', safe_source)
        self.assertIn("survived signal escalation", safe_source)

    def test_glm_wrapper_rejects_unsafe_safety_overrides(self):
        source = GLM_SAFE.read_text(encoding="utf-8")
        for variable in (
            "GLM_SAFE_VLIMIT_KB",
            "GLM_SAFE_KILL_FLOOR_GIB",
            "GLM_SAFE_MIN_START_GIB",
            "GLM_SAFE_TIMEOUT_S",
        ):
            self.assertIn(f'"{variable}:', source)
            self.assertIn(f'config_error "{variable}"', source)
        self.assertIn("KILL_FLOOR_GIB < 18", source)
        self.assertIn("MIN_START_GIB < 110", source)
        self.assertIn("TIMEOUT_S > 3600", source)
        self.assertIn("VLIMIT_KB > 419430400", source)
        self.assertIn("sleep 0.25", source)
        for variable, value in (
            ("GLM_SAFE_VLIMIT_KB", "999999999"),
            ("GLM_SAFE_VLIMIT_KB", "9" * 100),
            ("GLM_SAFE_KILL_FLOOR_GIB", "0"),
            ("GLM_SAFE_KILL_FLOOR_GIB", "09"),
            ("GLM_SAFE_MIN_START_GIB", "0"),
            ("GLM_SAFE_TIMEOUT_S", "3601"),
        ):
            environment = os.environ.copy()
            environment[variable] = value
            result = subprocess.run(
                ["bash", str(GLM_SAFE), "--", "/bin/true"],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result)
            self.assertIn(f"invalid {variable}", result.stderr)

    def test_glm_wrapper_hashes_the_selected_candidate_binary(self):
        source = GLM_SAFE.read_text(encoding="utf-8")
        self.assertIn("GLM_CANDIDATE_SRC", source)
        self.assertIn("candidate_binary_sha256=", source)
        self.assertIn('sha256sum -- "$CANDIDATE_BINARY"', source)
        self.assertIn('readlink -f -- "/proc/$SPID2/exe"', source)
        self.assertIn("executed candidate binary was not observed", source)
        self.assertIn("executed_binary_sha256=", source)
        self.assertIn("EXECUTED_START_TICKS", source)
        self.assertIn("executed candidate identity changed", source)
        self.assertIn("isolated process group survived command completion", source)
        self.assertNotIn(
            "SRC=/home/dsv4/ds4-project/src/ds4-upstream-master",
            source,
        )

    def test_cgroup_launcher_contains_setsid_descendants_and_memory(self):
        source = GLM_CGROUP.read_text(encoding="utf-8")
        self.assertIn("systemd-run --user --wait --collect", source)
        self.assertIn("KillMode=control-group", source)
        self.assertIn("MemoryHigh=", source)
        self.assertIn("MemoryMax=", source)
        self.assertIn("MemorySwapMax=0", source)
        self.assertIn("OOMPolicy=kill", source)
        self.assertIn("systemctl --user stop", source)
        self.assertIn("GLM_SAFE_REQUIRE_CGROUP=1", source)

    def test_production_watchdog_reserves_18_gib(self):
        source = DSV4_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_WATCHDOG_FLOOR_GIB:-18", source)
        self.assertIn('--threshold-gib "$watchdog_floor_gib"', source)
        self.assertIn("llamacpp.start-failed", source)
        self.assertIn("DSV4_ALLOW_RETRY_AFTER_FAILED_START", source)
        self.assertIn("reboot before retrying", source)
        self.assertIn("--alias deepseek-v4-flash", source)
        marker_index = source.index("failed_at=%s exit_status=%s")
        wait_index = source.index('wait "$flock_pid"')
        self.assertLess(marker_index, wait_index)
        self.assertIn("start group remains alive after SIGKILL", source)
        self.assertIn("watchdog_floor_gib < 18", source)
        self.assertIn("watchdog_floor_gib > 64", source)
        self.assertIn("mem_floor_gib < 18", source)
        self.assertIn("--interval-sec 0.25", source)
        self.assertIn("10#$watchdog_floor_gib", source)
        self.assertIn("10#$mem_floor_gib", source)

    def test_persistent_maintenance_hold_blocks_boot_start_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            hold = Path(tmp) / ".dsv4-start-hold"
            hold.touch()
            environment = os.environ.copy()
            environment["HOME"] = tmp
            environment["DSV4_START_HOLD_FILE"] = str(hold)
            result = subprocess.run(
                ["bash", str(DSV4_LAUNCHER), "start"],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("persistent maintenance hold", result.stderr)

        guard = (ROOT / "scripts/03_guard.sh").read_text(encoding="utf-8")
        hold_index = guard.index("dsv4-start-hold")
        restart_check_index = guard.index("/usr/sbin/runuser")
        self.assertLess(hold_index, restart_check_index)

    def test_launcher_port_override_is_bounded_and_default_off(self):
        source = DSV4_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_PORT:-8011", source)
        self.assertIn("DSV4_PORT must be an integer from 1024 through 65535", source)
        self.assertIn("port < 1024 || port > 65535", source)

    def test_candidate_source_override_is_explicit_and_default_off(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertIn("GLM_CANDIDATE_SRC:-", source)
        self.assertIn("ds4-goal-clean-0a7ad776", source)
        self.assertIn("GLM_EXPERT_CACHE_GB:-0", source)
        self.assertIn("DS4_TOKEN_TIMING_LOG=1", source)
        self.assertIn('--token-timing-log "$OUT/server.log"', source)
        self.assertNotIn("DS4_GLM_TP_DEBUG=0", source)

    def test_glm_arm_port_override_is_bounded_and_default_off(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertIn("GLM_PORT:-8011", source)
        self.assertIn("GLM_PORT must be an integer from 1024 through 65535", source)
        self.assertIn("port < 1024 || port > 65535", source)

    def test_glm_arm_streaming_timing_fallback_is_explicit_and_default_off(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertIn("GLM_REQUIRE_TOKEN_TIMING_LOG:-1", source)
        self.assertIn("GLM_REQUIRE_TOKEN_TIMING_LOG must be 0 or 1", source)
        self.assertIn('timing_args=(--token-timing-log "$OUT/server.log")', source)
        self.assertIn('"${timing_args[@]}"', source)


if __name__ == "__main__":
    unittest.main()
