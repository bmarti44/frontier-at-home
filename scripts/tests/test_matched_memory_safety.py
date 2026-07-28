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
DSV4_SERVICE = ROOT / "configs" / "systemd" / "deepseek-v4-flash-llamacpp.service"
GLM_ARM = ROOT / "results" / "glm52-goal" / "harness" / "glm_decisive_arm.sh"
GLM_LOGIT_ARM = (
    ROOT / "results" / "glm52-goal" / "harness" / "glm_logit_arm.sh"
)
HEADLESS_SCHEDULER = ROOT / "scripts" / "54_schedule_headless_foundation.sh"
HEADLESS_WORKER = ROOT / "scripts" / "55_headless_foundation_worker.sh"


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
    def test_glm_logit_arm_is_one_token_and_path_bound(self):
        source = GLM_LOGIT_ARM.read_text(encoding="utf-8")
        self.assertIn('DS4_GLM_LOGIT_DUMP="$OUT/prefill.logits"', source)
        self.assertIn('"max_tokens": 1', source)
        self.assertIn("completion_tokens == 1", source)
        self.assertIn("DS4_CUDA_IQ2_DOWN_REFERENCE:-1", source)
        self.assertIn("IQ2_REFERENCE must be 0 or 1", source)
        self.assertIn("glm52-b4734de4/tokenizer.json", source)
        self.assertNotIn("DS4_CUDA_EXPERT_CACHE_GB=72", source)
        self.assertIn("DS4_CUDA_MOE_NO_EXPERT_TILES:-0", source)
        self.assertIn("NO_EXPERT_TILES must be 0 or 1", source)

        cgroup = GLM_CGROUP.read_text(encoding="utf-8")
        self.assertIn("DS4_CUDA_MOE_NO_EXPERT_TILES", cgroup)

    def test_headless_foundation_runner_restores_display_and_fails_closed(self):
        scheduler = HEADLESS_SCHEDULER.read_text(encoding="utf-8")
        worker = HEADLESS_WORKER.read_text(encoding="utf-8")
        self.assertIn("must run as root", scheduler)
        self.assertIn("/home/dsv4/.dsv4-start-hold", scheduler)
        self.assertIn("systemd-run", scheduler)
        self.assertIn("--no-block", scheduler)
        self.assertIn("dsv4-headless-smoke.service", scheduler)
        self.assertNotIn("glm52-headless-foundation.service", scheduler)
        self.assertIn("OnFailure=display-manager.service", scheduler)
        self.assertIn("trap cleanup EXIT", worker)
        self.assertIn("systemctl stop display-manager.service", worker)
        self.assertIn("systemctl start display-manager.service", worker)
        self.assertIn("--required-gib 116", worker)
        self.assertIn("--required-gib 110", worker)
        self.assertIn("DSV4_MEM_FLOOR_GIB=18", worker)
        self.assertIn("DSV4_WATCHDOG_FLOOR_GIB=18", worker)
        self.assertIn("DSV4_UBATCH=512", worker)
        self.assertIn("CTX=8192", worker)
        self.assertIn("DSV4_PARALLEL=1", worker)
        self.assertIn("DSV4_PORT=8013", worker)
        self.assertIn("journalctl -k -n 0 --show-cursor", worker)
        self.assertIn("journalctl -k --after-cursor", worker)
        self.assertIn("NV_ERR_NO_MEMORY", worker)
        self.assertIn("candidate hash changed", worker)
        self.assertIn("repository is not clean", worker)
        self.assertIn("--reps 2", worker)
        for evidence_file in ("manifest.json", "raw.jsonl", "summary.json"):
            self.assertIn(evidence_file, worker)
        display_stop = worker.index("systemctl stop display-manager.service")
        admission = worker.index("--required-gib 116", display_stop)
        swap_baseline = worker.index("SWAP_START_KIB=$(", admission)
        model_start = worker.index("dsv4_launcher start", swap_baseline)
        self.assertLess(display_stop, admission)
        self.assertLess(admission, swap_baseline)
        self.assertLess(swap_baseline, model_start)
        self.assertIn('"post_admission_pre_model"', worker)
        self.assertNotIn("rm -f -- /home/dsv4/.dsv4-start-hold", worker)

    def test_production_service_uses_launcher_accepted_memory_floor(self):
        source = DSV4_SERVICE.read_text(encoding="utf-8")
        self.assertIn(
            "Environment=DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=14",
            source,
        )
        self.assertIn("Environment=DSV4_MEM_FLOOR_GIB=14", source)
        self.assertIn("Environment=DSV4_WATCHDOG_FLOOR_GIB=14", source)
        self.assertIn(
            "Environment=DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=3",
            source,
        )

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

    def test_harness_uses_the_frozen_production_dsv4_profile(self):
        source = HARNESS.read_text(encoding="utf-8")
        expected = (
            "DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/"
            "llama.cpp-fusion/build/bin/llama-server",
            "DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json",
            "DSV4_MEM_FLOOR_GIB=18",
            "DSV4_WATCHDOG_FLOOR_GIB=18",
            "DSV4_UBATCH=512",
            "DSV4_BATCH=2048",
            "DSV4_UBATCH_LARGE=0",
            "CTX=8192",
            "DSV4_PARALLEL=1",
            "DSV4_NO_MMAP=1",
            "DSV4_SPEC_TYPE=ngram-map-k4v",
        )
        for setting in expected:
            self.assertIn(setting, source)
        self.assertIn("MATCHED_PORT:-8021", source)
        self.assertIn('DSV4_PORT="$PORT"', source)
        self.assertIn('GLM_PORT="$PORT"', source)
        self.assertEqual(source.count("--reps 2"), 1)
        self.assertEqual(GLM_ARM.read_text(encoding="utf-8").count("--reps 2"), 1)
        self.assertNotIn("--reps 1", source)
        self.assertNotIn("--reps 1", GLM_ARM.read_text(encoding="utf-8"))
        self.assertIn("process.identity.json", source)
        self.assertIn("memwatch.segment.log", source)
        self.assertIn("samples.log", source)
        self.assertIn("kernel.log", source)
        self.assertIn("host.boot_id", GLM_ARM.read_text(encoding="utf-8"))

    def test_harness_rejects_kernel_gpu_and_oom_faults_from_each_arm(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("journalctl -k -n 0 --show-cursor", source)
        self.assertIn("journalctl -k --after-cursor", source)
        self.assertIn("NV_ERR_NO_MEMORY", source)
        self.assertIn("NVRM.*Xid", source)
        self.assertIn("oom-kill", source)
        self.assertGreaterEqual(source.count("assert_no_kernel_faults_since"), 3)

    def test_harness_does_not_lower_emergency_floor(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertNotIn("GLM_SAFE_KILL_FLOOR_GIB=10", source)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=40", source)
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
        self.assertIn("executed candidate exited during wrapper shutdown", source)
        self.assertIn("CANDIDATE_EXIT_GRACE_TICKS=8", source)
        self.assertIn("replacement candidate appeared during shutdown", source)
        self.assertIn("EXECUTED_CANDIDATE_CLEAN_EXIT=1", source)
        self.assertIn("wrapper command failed after candidate exit", source)
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

    def test_cgroup_launcher_exports_exact_evidence_without_masking_failure(self):
        source = GLM_CGROUP.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_EVIDENCE_DIR", source)
        self.assertIn("/home/dsv4/ds4-project/glm52-confirm-", source)
        self.assertIn('"$EVIDENCE_EXPORT" "$EVIDENCE_DIR"', source)
        self.assertIn("command_rc=$?", source)
        self.assertIn("evidence_export_rc=", source)
        self.assertIn("exit \"$command_rc\"", source)

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

    def test_launcher_stop_tolerates_watchdog_exit_after_engine_shutdown(self):
        source = DSV4_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'if "$memwatch_verified" && pid_alive "$memwatch_pid"; then',
            source,
        )
        self.assertIn(
            'verify_aux_identity "$memwatch_pid" "$memwatch_start_ticks"',
            source,
        )

    def test_candidate_source_and_bounded_cache_overrides_are_default_off(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertIn("GLM_CANDIDATE_SRC:-", source)
        self.assertIn("ds4-goal-clean-0a7ad776", source)
        self.assertIn("GLM_EXPERT_CACHE_GB:-0", source)
        self.assertIn("CACHE_GB must be an integer from 0 through 40", source)
        self.assertIn("cache_gb < 0 || cache_gb > 40", source)
        self.assertIn('DS4_CUDA_EXPERT_CACHE_GB="$CACHE_GB"', source)
        self.assertIn("DS4_CUDA_IQ2_DOWN_REFERENCE:-1", source)
        self.assertIn("IQ2_REFERENCE must be 0 or 1", source)
        self.assertIn('IQ2_ENV+=(DS4_CUDA_IQ2_DOWN_REFERENCE=1)', source)
        self.assertIn("DS4_CUDA_MOE_NO_EXPERT_TILES:-0", source)
        self.assertIn("NO_EXPERT_TILES must be 0 or 1", source)
        self.assertIn("DS4_TOKEN_TIMING_LOG=1", source)
        self.assertIn('--token-timing-log "$OUT/server.log"', source)
        self.assertIn("glm52-b4734de4/tokenizer.json", source)
        self.assertIn(
            "19e773648cb4e65de8660ea6365e10ac"
            + "ca112d42a854923df93db4a6f333a82d",
            source.replace("\\\n", ""),
        )
        self.assertIn('--output-tokenizer-path "$TOKENIZER"', source)
        self.assertIn('--output-tokenizer-sha256 "$TOKENIZER_SHA256"', source)
        self.assertNotIn('    --tokenizer-path "$TOKENIZER"', source)
        self.assertNotIn("DS4_GLM_TP_DEBUG=0", source)

    def test_glm_arm_port_override_is_bounded_and_default_off(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertIn("GLM_PORT:-8011", source)
        self.assertIn("GLM_PORT must be an integer from 1024 through 65535", source)
        self.assertIn("port < 1024 || port > 65535", source)

    def test_glm_arm_requires_raw_token_timing(self):
        source = GLM_ARM.read_text(encoding="utf-8")
        self.assertNotIn("GLM_REQUIRE_TOKEN_TIMING_LOG", source)
        self.assertIn('--token-timing-log "$OUT/server.log"', source)

    def test_matched_glm_arm_uses_profile_safety_and_exports_evidence(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=40", source)
        self.assertNotIn("GLM_SAFE_KILL_FLOOR_GIB=18", source)
        self.assertIn('GLM_SAFE_EVIDENCE_DIR="$arm_out"', source)
        cgroup = GLM_CGROUP.read_text(encoding="utf-8")
        self.assertIn("glm52-decisive-", cgroup)


if __name__ == "__main__":
    unittest.main()
