#!/usr/bin/env python3
"""Safety contract for the turnkey engine switch."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "52_engine_switch.sh"
GLM_PROFILE = ROOT / "configs" / "glm52-profile.json"
DSV4_PROFILE = ROOT / "configs" / "dsv4-profile.json"
DSV4_SERVICE = ROOT / "configs/systemd/deepseek-v4-flash-llamacpp.service"
DSV4_BUILD = ROOT / "configs/build-manifests/llamacpp-fusion.json"
GLM_BUILD = ROOT / "configs/build-manifests/glm52-ds4-repro.json"
GLM_BUILD_SCRIPT = ROOT / "scripts/11_build_glm52_repro.sh"
INSTALLER = ROOT / "scripts/41_install_service.sh"
RESTORE_SERVICE = ROOT / "configs/systemd/dsv4-engine-restore.service"
CONTROL_INSTALLER = ROOT / "scripts/53_install_switch_control.sh"


class EngineSwitchTests(unittest.TestCase):
    def run_switch(self, root: Path, *args: str):
        env = {
            "PATH": os.environ["PATH"],
            "ENGINE_SWITCH_TESTING": "1",
            "ENGINE_SWITCH_TEST_ROOT": str(root),
        }
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )

    def test_status_json_is_machine_readable_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_switch(root, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual(status["active_profile"], None)
            self.assertEqual(status["state"], "inactive")
            self.assertFalse((root / "actions.log").exists())

    def test_implementation_has_lock_and_no_broad_pkill(self):
        source = SCRIPT.read_text()
        self.assertIn("flock", source)
        self.assertNotIn("pkill", source)
        self.assertIn("start_ticks", source)
        self.assertIn("rollback", source)
        self.assertIn("release-check", source)
        self.assertNotIn("DS4_CUDA_EXPERT_CACHE_GB=72", source)
        self.assertIn("memwatch_start_ticks", source)
        self.assertIn("DISARM %s %s %s", source)
        self.assertIn("GLM switching remains disabled", source)
        self.assertIn("wait_model_ready", source)
        self.assertIn("Waiting for %s load", source)
        self.assertIn("deadline=$((SECONDS + 1800))", source)
        self.assertIn("trap 'rollback \"$command\"' EXIT", source)
        self.assertIn("expected=deepseek-v4-flash", source)
        self.assertIn('expected == item["id"].lower()', source)
        self.assertIn("DSV4_ALLOW_RETRY_AFTER_FAILED_START || true", source)
        self.assertIn("/v1/chat/completions", source)
        self.assertIn('"content":"Calculate 2+2.', source)
        self.assertIn('value["choices"][0]["message"]', source)
        self.assertIn('finish_reason not in {"stop", "length"}', source)
        self.assertNotIn('"ready" not in text', source)

    def test_switch_uses_a_dedicated_internal_port_without_changing_auth_endpoint(self):
        source = SCRIPT.read_text()
        service = DSV4_SERVICE.read_text()
        installer = INSTALLER.read_text()
        self.assertIn("readonly PORT=8013", source)
        self.assertIn("readonly AUTH_PORT=8010", source)
        self.assertIn('DSV4_PORT="$PORT"', source)
        self.assertIn("Environment=DSV4_PORT=8013", service)
        self.assertIn("upstream_port=8013", installer)
        self.assertNotIn("readonly PORT=8011", source)

    def test_deepseek_readiness_requires_the_exact_1m_context(self):
        source = SCRIPT.read_text()
        self.assertIn('"/slots"', source)
        self.assertIn('slot["n_ctx"] != 1048576', source)

    def test_switch_runs_deepseek_as_engine_user_with_frozen_1m_profile(self):
        source = SCRIPT.read_text()
        self.assertIn(
            "install -d -o dsv4 -g dsv4 -m 0700 /run/dsv4", source
        )
        self.assertIn("/usr/sbin/runuser -u dsv4 --", source)
        for setting in (
            "DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/"
            "llama.cpp-fusion/build/bin/llama-server",
            "DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json",
            "DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=14",
            "DSV4_MEM_FLOOR_GIB=14",
            "DSV4_WATCHDOG_FLOOR_GIB=14",
            "DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=3",
            "DSV4_ALLOW_RETRY_AFTER_FAILED_START=1",
            "DSV4_UBATCH=256",
            "DSV4_BATCH=512",
            "DSV4_UBATCH_LARGE=0",
            "CTX=1048576",
            "DSV4_PARALLEL=1",
            "DSV4_NO_MMAP=1",
            "DSV4_SPEC_TYPE=none",
        ):
            self.assertIn(setting, source)
        self.assertIn(
            "DSV4_API_KEY_FILE:-/etc/deepseek-v4-flash/api-key", source
        )

    def test_installed_deepseek_service_uses_the_qualified_1m_profile(self):
        service = DSV4_SERVICE.read_text()
        for setting in (
            "Environment=DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=14",
            "Environment=DSV4_MEM_FLOOR_GIB=14",
            "Environment=DSV4_WATCHDOG_FLOOR_GIB=14",
            "Environment=DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=3",
            "Environment=DSV4_UBATCH=256",
            "Environment=DSV4_BATCH=512",
            "Environment=DSV4_UBATCH_LARGE=0",
            "Environment=CTX=1048576",
            "Environment=DSV4_PARALLEL=1",
            "Environment=DSV4_SPEC_TYPE=none",
        ):
            self.assertIn(setting, service)

    def test_selected_profile_has_a_boot_restore_unit(self):
        source = SCRIPT.read_text()
        installer = INSTALLER.read_text()
        unit = RESTORE_SERVICE.read_text()
        self.assertIn("status [--json]|restore|dsv4|glm52", source)
        self.assertIn('if [[ $command == restore ]]', source)
        self.assertIn("dsv4-engine-restore.service", installer)
        self.assertIn("52_engine_switch.sh restore", unit)
        self.assertIn("WantedBy=multi-user.target", unit)
        restore = source[source.index('if [[ $command == restore ]]') :]
        self.assertIn('stop_profile "$command"', restore)
        self.assertIn("After=dsv4-authhelper.service", unit)

    def test_control_plane_installer_cannot_start_a_model(self):
        source = CONTROL_INSTALLER.read_text()
        self.assertIn("must be run as root", source)
        self.assertIn("/home/dsv4/.dsv4-start-hold", source)
        self.assertIn("UPSTREAM_PORT=8013", source)
        self.assertIn("systemctl daemon-reload", source)
        self.assertIn("systemctl disable deepseek-v4-flash-llamacpp.service", source)
        self.assertIn("systemctl enable dsv4-engine-restore.service", source)
        self.assertIn("systemctl restart dsv4-authhelper.service", source)
        self.assertIn(
            "systemctl reset-failed deepseek-v4-flash-llamacpp.service "
            "2>/dev/null || true",
            source,
        )
        self.assertIn("deadline=$((SECONDS + 30))", source)
        self.assertIn("while (( SECONDS < deadline ))", source)
        self.assertIn("sleep 1", source)
        self.assertNotIn(
            "systemctl restart deepseek-v4-flash-llamacpp.service", source
        )
        self.assertNotIn("52_engine_switch.sh dsv4", source)

    def test_frozen_profiles_pin_the_verified_production_candidates(self):
        glm = json.loads(GLM_PROFILE.read_text())
        dsv4 = json.loads(DSV4_PROFILE.read_text())
        glm_build = json.loads(GLM_BUILD.read_text())
        dsv4_build = json.loads(DSV4_BUILD.read_text())
        weights = json.loads(
            (ROOT / "configs/build-manifests/ds4-weights.json").read_text()
        )
        self.assertEqual(glm["schema_version"], 2)
        self.assertEqual(glm["profile"], "glm52")
        self.assertEqual(glm["binary_sha256"], glm_build["binary_sha256"])
        self.assertEqual(glm["model_sha256"], glm_build["model_sha256"])
        self.assertEqual(glm["context_cap"], 1_048_576)
        self.assertNotIn("evidence", glm_build)
        self.assertEqual(len(glm_build["evidence_archive"]), 14)
        for path, digest in glm_build["evidence_archive"].items():
            self.assertTrue(
                path.startswith("results/glm52-goal/evidence/build-repro/"),
                path,
            )
            artifact = ROOT / path
            self.assertTrue(artifact.is_file(), path)
            self.assertEqual(
                digest,
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                path,
            )
        self.assertEqual(
            glm["build_manifest_sha256"],
            hashlib.sha256(GLM_BUILD.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            glm["tokenizer_sha256"],
            "19e773648cb4e65de8660ea6365e10ac"
            "ca112d42a854923df93db4a6f333a82d",
        )
        self.assertEqual(
            glm["runtime"]["engine_environment"],
            {
                "DS4_CUDA_EXPERT_CACHE_GB": "0",
                "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                "DS4_CUDA_FETCH_THREADS": "6",
                "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                "DS4_TOKEN_TIMING_LOG": "1",
            },
        )
        self.assertEqual(
            glm["runtime"]["launch_arguments"],
            [
                "--cuda", "-m", "{model}", "-c", "8192",
                "--host", "127.0.0.1", "--port", "{port}",
                "--ssd-streaming",
                "--ssd-streaming-cache-experts", "40GB",
            ],
        )
        self.assertEqual(
            glm["runtime"]["benchmark"],
            {
                "fixture_context_tokens": 0,
                "max_completion_tokens": 160,
                "minimum_completion_tokens": 128,
                "raw_token_timing_required": True,
            },
        )
        self.assertEqual(
            glm["runtime"]["safety"],
            {
                "kill_floor_gib": 40,
                "minimum_start_gib": 110,
                "sample_hz": 4,
                "swap_max_bytes": 0,
                "timeout_seconds": 2400,
                "virtual_memory_limit_kib": 419_430_400,
            },
        )
        self.assertEqual(
            set(glm["artifact_sha256"]),
            {
                "scripts/11_build_glm52_repro.sh",
                "results/glm52-goal/harness/decisive_matched.sh",
                "results/glm52-goal/harness/glm_decisive_arm.sh",
                "results/glm52-gates/harness/glm_safe_run.sh",
                "results/glm52-gates/harness/glm_cgroup_run.sh",
                "results/glm52-gates/harness/glm_evidence_export.py",
                "scripts/30_bench_speed.py",
            },
        )
        for path, digest in glm["artifact_sha256"].items():
            self.assertEqual(
                digest,
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                path,
            )

        self.assertEqual(dsv4["schema_version"], 2)
        self.assertEqual(dsv4["profile"], "dsv4")
        self.assertEqual(
            dsv4["binary_sha256"],
            dsv4_build["binaries"]["llama-server"]["sha256"],
        )
        self.assertEqual(
            dsv4["configuration_sha256"],
            hashlib.sha256(DSV4_SERVICE.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            dsv4["build_manifest_sha256"],
            hashlib.sha256(DSV4_BUILD.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            dsv4["weights_manifest_sha256"],
            hashlib.sha256(
                (ROOT / "configs/build-manifests/ds4-weights.json").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            dsv4["shared_libraries"],
            {
                name: record["sha256"]
                for name, record in dsv4_build["shared_libraries"].items()
            },
        )
        self.assertEqual(
            dsv4["model_files"],
            {
                record["role"]: record["sha256"]
                for record in weights["files"]
            },
        )

    def test_glm_repro_build_fixes_all_nondeterministic_inputs(self):
        source = GLM_BUILD_SCRIPT.read_text()
        self.assertIn("SOURCE_DATE_EPOCH", source)
        self.assertIn("--frandom-seed=", source)
        self.assertIn("--keep-dir=", source)
        self.assertIn("-ffile-prefix-map=", source)
        self.assertIn("git ls-files -z", source)
        self.assertIn("touch --date=", source)
        self.assertIn("-j2", source)
        self.assertIn("cmp -s", source)
        self.assertIn(
            "CANONICAL_WORK_ROOT=/home/bmarti44/.cache/glm52-ds4-repro-v1",
            source,
        )
        self.assertIn('[[ $WORK_ROOT == "$CANONICAL_WORK_ROOT" ]]', source)

    def test_glm_repro_build_ignores_caller_toolchain_environment(self):
        source = GLM_BUILD_SCRIPT.read_text()
        self.assertIn("readonly CC_PATH=/usr/bin/cc", source)
        self.assertIn("readonly MAKE_PATH=/usr/bin/make", source)
        self.assertIn("readonly CUDA_HOME_PATH=/usr/local/cuda", source)
        self.assertIn(
            "readonly NVCC_PATH=/usr/local/cuda/bin/nvcc",
            source,
        )
        self.assertIn("env -i", source)
        self.assertIn('CC="$CC_PATH"', source)
        self.assertIn('NVCC="$NVCC_PATH"', source)
        self.assertIn('CUDA_HOME="$CUDA_HOME_PATH"', source)
        self.assertIn('DS4_LINK="$NVCC_PATH $nvccflags"', source)
        self.assertIn("clean_git()", source)
        self.assertIn("/usr/bin/git", source)
        self.assertIn("GIT_CONFIG_NOSYSTEM=1", source)
        self.assertIn("GIT_CONFIG_GLOBAL=/dev/null", source)
        self.assertNotIn("\ngit -C", source)
        self.assertIn("os.lstat", source)
        self.assertIn("stat.S_ISLNK", source)
        self.assertIn('exec 8<"$WORK_ROOT"', source)
        self.assertIn("WORK_ROOT_IDENTITY", source)
        self.assertIn("verify_work_root", source)

    def test_authenticated_probe_keeps_bearer_secret_out_of_argv(self):
        source = SCRIPT.read_text()
        self.assertNotIn('-H "Authorization: Bearer $key"', source)
        self.assertIn("clean_curl --config -", source)
        self.assertIn("printf 'header = \"Authorization: Bearer %s\"", source)

    def test_switch_subprocesses_use_a_frozen_environment_allowlist(self):
        source = SCRIPT.read_text()
        self.assertIn("clean_python()", source)
        self.assertIn("clean_curl()", source)
        self.assertIn("dsv4_launcher()", source)
        self.assertIn("env -i", source)
        self.assertIn('/usr/bin/curl --disable "$@"', source)
        self.assertNotIn("PYTHONOPTIMIZE", source)

    def test_unqualified_glm_is_rejected_before_active_profile_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            result = self.run_switch(root, "glm52")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not qualified", result.stderr.lower())
            self.assertFalse((root / "actions.log").exists())
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )

    def test_concurrent_status_calls_return_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processes = [
                subprocess.Popen(
                    ["bash", str(SCRIPT), "status", "--json"],
                    cwd=ROOT,
                    env={
                        "PATH": os.environ["PATH"],
                        "ENGINE_SWITCH_TESTING": "1",
                        "ENGINE_SWITCH_TEST_ROOT": str(root),
                    },
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(8)
            ]
            for process in processes:
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                json.loads(stdout)


if __name__ == "__main__":
    unittest.main()
