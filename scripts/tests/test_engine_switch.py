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
GLM_PRODUCTION_PROFILE = (
    ROOT / "configs" / "glm52-fullq4-production-profile.json"
)
QWEN_PRODUCTION_PROFILE = ROOT / "configs" / "qwen38-production-profile.json"
QWEN_1M_PRODUCTION_PROFILE = (
    ROOT / "configs" / "qwen38-1m-production-profile.json"
)
QWEN_1M_EVIDENCE = (
    ROOT / "results" / "qwen38-gates" / "trackc-1m-np4-2026-08-19"
)
QWEN_BUILD = (
    ROOT / "configs" / "build-manifests" / "llamacpp-qwen38-9d77fa17.json"
)
QWEN_WEIGHTS = ROOT / "weights" / "qwen3.8-27b" / "manifest.json"
LAGUNA_PRODUCTION_PROFILE = ROOT / "configs" / "laguna-production-profile.json"
LAGUNA_BUILD = (
    ROOT / "configs" / "build-manifests" / "llamacpp-laguna-06f8cebd.json"
)
LAGUNA_WEIGHTS = ROOT / "weights" / "laguna-s-2.1" / "manifest.json"
DSV4_PROFILE = ROOT / "configs" / "dsv4-profile.json"
DSV4_SERVICE = ROOT / "configs/systemd/deepseek-v4-flash-llamacpp.service"
DSV4_BUILD = ROOT / "configs/build-manifests/llamacpp-fusion.json"
GLM_BUILD = ROOT / "configs/build-manifests/glm52-ds4-repro.json"
W7_ADOPTION = ROOT / "results/glm52-gates/W7-cache-generation-W7.1a-owner-adoption.json"
W7_ADOPTION_REVIEW = ROOT / "results/glm52-gates/W7-cache-generation-W7.1a-review-r295.json"
W7_BINARY_FREEZE = ROOT / "results/glm52-gates/W7-cache-generation-freeze-v9.json"
W7_BINARY_SHA256 = "eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2"
GLM_BUILD_SCRIPT = ROOT / "scripts/11_build_glm52_repro.sh"
INSTALLER = ROOT / "scripts/41_install_service.sh"
RESTORE_SERVICE = ROOT / "configs/systemd/dsv4-engine-restore.service"
CONTROL_INSTALLER = ROOT / "scripts/53_install_switch_control.sh"



# ---------------------------------------------------------------------------
# Launch truth renders from configs/profiles/ (docs/PROFILE-SCHEMA.md). The
# helpers below assemble one alias's launch snapshot through the switch's
# test-only `render` verb and map the fixture harness's test-root paths back
# to the production paths captured in
# scripts/tests/fixtures/profile-conformance/.
FIXTURES = ROOT / "scripts/tests/fixtures/profile-conformance"
SWITCH_ALIASES = ("dsv4", "glm52", "qwen38", "qwen38-1m", "laguna")


def switch_production_map(test_root: str) -> dict[str, str]:
    state = "/home/dsv4/ds4-project/engine-switch"
    return {
        f"{test_root}/source/ds4-server":
            "/home/bmarti44/.cache/glm52-dynexp2-patched/ds4-server",
        f"{test_root}/source/llama-server":
            "/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17/src/build/bin/llama-server",
        f"{test_root}/source/laguna-server":
            "/home/bmarti44/.cache/llamacpp-laguna-06f8cebd/src/build/bin/llama-server",
        f"{test_root}/model.gguf":
            "/home/bmarti44/models/glm52-full-denseq40.gguf",
        f"{test_root}/qwen-model.gguf":
            "/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
        f"{test_root}/qwen-mmproj.gguf":
            "/home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf",
        f"{test_root}/laguna-model-00001-of-00003.gguf":
            "/home/bmarti44/models/laguna-s-2.1/unsloth/UD-Q4_K_XL/"
            "Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf",
        f"{test_root}/laguna-dflash.gguf":
            "/home/bmarti44/models/laguna-s-2.1/poolside/laguna-s-2.1-DFlash-BF16.gguf",
        test_root: state,
    }


def _map_to_production(value, mapping):
    if isinstance(value, str):
        for test_path in sorted(mapping, key=len, reverse=True):
            value = value.replace(test_path, mapping[test_path])
        return value
    if isinstance(value, list):
        return [_map_to_production(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _map_to_production(item, mapping)
                for key, item in value.items()}
    return value


def render_switch_snapshot(alias: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["bash", str(SCRIPT), "render", alias],
            cwd=ROOT,
            env={
                "PATH": os.environ["PATH"],
                "ENGINE_SWITCH_TESTING": "1",
                "ENGINE_SWITCH_TEST_ROOT": tmp,
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(f"render {alias} failed: {result.stderr}")
        return _map_to_production(
            json.loads(result.stdout), switch_production_map(tmp)
        )


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
        self.assertNotIn("GLM switching remains disabled", source)
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

    def test_glm_production_launcher_matches_the_accepted_fullq4_profile(self):
        snapshot = render_switch_snapshot("glm52")
        self.assertEqual(snapshot["argv"][:5], [
            "--cuda", "-m",
            "/home/bmarti44/models/glm52-full-denseq40.gguf",
            "-c", "32768",
        ])
        self.assertIn("--ssd-streaming", snapshot["argv"])
        self.assertEqual(snapshot["env"]["DS4_CUDA_EXPERT_CACHE_GB"], "94")
        source = SCRIPT.read_text()
        self.assertIn("configs/glm52-fullq4-production-profile.json", source)
        self.assertIn("provisional", source)
        self.assertIn("GLM process record already exists", source)

    def test_qwen_production_launcher_matches_the_pinned_profile(self):
        snapshot = render_switch_snapshot("qwen38")
        argv = snapshot["argv"]
        self.assertEqual(argv[:2], [
            "--model", "/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
        ])
        for flag in ("-ngl", "--no-mmap", "--spec-type", "--cache-reuse"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("-c") + 1], "32768")
        self.assertEqual(argv[argv.index("--parallel") + 1], "1")
        properties = snapshot["systemd"]["properties"]
        self.assertEqual(properties["MemoryHigh"], "45G")
        self.assertEqual(properties["MemoryMax"], "50G")
        source = SCRIPT.read_text()
        self.assertIn("Qwen transient unit executed an unapproved binary", source)
        self.assertIn("stop_qwen_verified", source)
        self.assertIn("start_qwen38", source)
        self.assertIn("expected=qwen3.8-27b", source)
        self.assertIn('systemctl show "$QWEN_UNIT"', source)
        self.assertIn("--threshold-gib 18 --interval-sec 1", source)
        self.assertIn("ARMED $pid $pgid $ticks provisional", source)
        self.assertIn("ARMED $pid $pgid $ticks engine", source)

    def test_qwen_readiness_is_bound_to_unit_executable_and_listener(self):
        source = SCRIPT.read_text()
        readiness = source[
            source.index("verify_qwen_process_ready() {") :
            source.index("wait_model_ready() {")
        ]
        for contract in (
            'systemctl show "$QWEN_UNIT" --property=MainPID',
            'proc_identity "$pid"',
            'readlink -f "/proc/$pid/exe"',
            'readlink -f "$QWEN_BINARY"',
            'ss -H -ltnp "sport = :$PORT"',
            'sockets == *"pid=$pid,"*',
        ):
            self.assertIn(contract, readiness)
        wait = source[
            source.index("wait_model_ready() {") : source.index("verify_serving() {")
        ]
        self.assertIn("verify_qwen_process_ready", wait)

    def test_qwen_1m_production_launcher_matches_the_pinned_profile(self):
        snapshot = render_switch_snapshot("qwen38-1m")
        argv = snapshot["argv"]
        self.assertEqual(argv[argv.index("-c") + 1], "1048576")
        self.assertEqual(argv[argv.index("--parallel") + 1], "4")
        self.assertNotIn("-ctk", argv)
        self.assertNotIn("-ctv", argv)
        self.assertFalse(any("rope" in token.lower() for token in argv))
        properties = snapshot["systemd"]["properties"]
        self.assertEqual(properties["MemoryHigh"], "88G")
        self.assertEqual(properties["MemoryMax"], "95G")
        source = SCRIPT.read_text()
        self.assertIn("configs/qwen38-1m-production-profile.json", source)
        self.assertIn("verify_qwen_1m_hashes", source)
        self.assertIn("start_qwen38-1m", source)
        context = source[
            source.index("verify_qwen_1m_context() {"):
            source.index("verify_qwen_process_ready() {")
        ]
        self.assertIn("len(value) != 4", context)
        self.assertIn('slot["n_ctx"] != 262144', context)

    def test_qwen_uses_the_shared_authenticated_semantic_verifier(self):
        source = SCRIPT.read_text()
        verify = source[
            source.index("verify_serving() {") : source.index("commit_active() {")
        ]
        self.assertIn('"http://127.0.0.1:$PORT/v1/models"', verify)
        self.assertIn('"http://127.0.0.1:$AUTH_PORT/health"', verify)
        self.assertIn('"http://127.0.0.1:$AUTH_PORT/v1/chat/completions"', verify)
        qwen_block = verify[
            verify.index("if [[ $profile == qwen38 || $profile == qwen38-1m ]]") :
        ]
        self.assertNotIn("return 0", qwen_block.split("unauth=", 1)[0])
        self.assertIn("verify_qwen_1m_context", qwen_block)

    def test_laguna_production_launcher_matches_the_pinned_profile(self):
        snapshot = render_switch_snapshot("laguna")
        argv = snapshot["argv"]
        self.assertEqual(argv[argv.index("-c") + 1], "393216")
        self.assertEqual(argv[argv.index("--parallel") + 1], "4")
        self.assertIn("draft-dflash", argv)
        self.assertIn("--jinja", argv)
        properties = snapshot["systemd"]["properties"]
        self.assertEqual(properties["MemoryHigh"], "88G")
        self.assertEqual(properties["NoNewPrivileges"], "yes")
        source = SCRIPT.read_text()
        verify = source[
            source.index("verify_laguna_profile_hashes() {"):
            source.index("revalidate_laguna_identities() {")
        ]
        self.assertIn("os.O_NOFOLLOW", verify)
        self.assertIn("before.st_size != expected_bytes", verify)
        self.assertIn("len(shards) != 3", verify)

    def test_laguna_and_qwen_launch_safety_guards_are_explicit(self):
        source = SCRIPT.read_text()
        for start_name, end_name in (
            ("start_laguna_profile() {", "start_laguna() {"),
            ("start_qwen_profile() {", "start_qwen38() {"),
        ):
            starter = source[source.index(start_name):source.index(end_name)]
            self.assertRegex(
                starter,
                r"(?s)03_memory_guard\.py.*?--timeout-seconds 180\s+\|\|\s+die",
            )
        self.assertEqual(
            render_switch_snapshot("laguna")["systemd"]["properties"]
            .get("NoNewPrivileges"),
            "yes",
        )

    def test_rollback_waits_for_every_restored_profile_before_verification(self):
        source = SCRIPT.read_text()
        rollback = source[source.index("rollback() {") : source.index("command=${1")]
        start = rollback.index('"start_$previous_profile"')
        wait = rollback.index('wait_model_ready "$previous_profile"')
        verify = rollback.index('verify_serving "$previous_profile"')
        self.assertLess(start, wait)
        self.assertLess(wait, verify)

    def test_status_and_restore_accept_qwen38(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "qwen38"})
            )
            result = self.run_switch(root, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["active_profile"], "qwen38")
        source = SCRIPT.read_text()
        self.assertIn(
            "status [--json]|stop|restore|dsv4|glm52|qwen38|qwen38-1m", source
        )

    def test_qwen38_1m_successful_launch_cleans_killed_unit_and_omits_q8_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            (root / "dsv4-running").touch()
            (root / "qwen-hashes-valid").touch()
            (root / "qwen-unit-killed").touch()
            result = self.run_switch(root, "qwen38-1m")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"],
                "qwen38-1m",
            )
            actions = (root / "actions.log").read_text()
            self.assertIn("SYSTEMCTL reset-failed qwen38-engine.service", actions)
            self.assertIn("SYSTEMD_RUN", actions)
            self.assertIn("-c 1048576", actions)
            self.assertIn("--parallel 4", actions)
            self.assertNotIn("-ctk", actions)
            self.assertNotIn("-ctv", actions)
            self.assertFalse((root / "qwen-unit-killed").exists())

    def test_restore_qwen38_1m_start_failure_falls_back_to_dsv4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "qwen38-1m"})
            )
            (root / "qwen-hashes-valid").touch()
            (root / "fail-qwen-start").touch()
            result = self.run_switch(root, "restore")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )
            self.assertIn(
                "RESTORE FAILED for recorded profile qwen38-1m; "
                "falling back to dsv4",
                result.stderr,
            )
            self.assertIn(
                "RESTORE FALLBACK committed dsv4 in active.json", result.stderr
            )
            actions = (root / "actions.log").read_text()
            self.assertIn("SYSTEMD_RUN", actions)
            self.assertIn("DSV4 start", actions)
            self.assertIn("WAIT dsv4", actions)
            self.assertIn("VERIFY dsv4", actions)

    def test_laguna_hash_failure_is_rejected_before_active_profile_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            (root / "dsv4-running").touch()
            result = self.run_switch(root, "laguna")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("laguna", result.stderr.lower())
            self.assertTrue((root / "dsv4-running").exists())
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )

    def test_laguna_successful_launch_cleans_killed_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            (root / "dsv4-running").touch()
            (root / "laguna-hashes-valid").touch()
            (root / "laguna-unit-killed").touch()
            result = self.run_switch(root, "laguna")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "laguna"
            )
            actions = (root / "actions.log").read_text()
            self.assertIn("SYSTEMCTL reset-failed laguna-engine.service", actions)
            self.assertIn("SYSTEMD_RUN --unit=laguna-engine", actions)
            self.assertIn("-c 393216", actions)
            self.assertIn("--parallel 4", actions)
            self.assertFalse((root / "laguna-unit-killed").exists())

    def test_status_accepts_laguna(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "laguna"})
            )
            result = self.run_switch(root, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["active_profile"], "laguna")

    def test_stop_halts_laguna_without_touching_active_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "laguna"})
            )
            (root / "laguna-running").touch()
            result = self.run_switch(root, "stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STOPPED laguna", result.stdout)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "laguna"
            )
            self.assertFalse((root / "laguna-running").exists())

    def test_restore_laguna_start_failure_falls_back_to_dsv4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "laguna"})
            )
            (root / "laguna-hashes-valid").touch()
            (root / "fail-laguna-start").touch()
            result = self.run_switch(root, "restore")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )
            self.assertIn(
                "RESTORE FAILED for recorded profile laguna; falling back to dsv4",
                result.stderr,
            )

    def test_background_spawns_close_the_switch_lock_fd(self):
        # Regression for the 2026-08-21 deadlock: a memwatch spawned without
        # `9>&-` inherited the switch.lock open-file description, so the lock
        # outlived the switch and every later invocation hung in flock.
        # Every backgrounded command in this script must close fd 9.
        source = SCRIPT.read_text()
        offenders = []
        for number, line in enumerate(source.splitlines(), start=1):
            stripped = line.rstrip()
            if not stripped.endswith("&") or stripped.endswith("&&"):
                continue
            # Find the full command by walking back over continuation lines.
            command_lines = [stripped]
            index = number - 2
            lines = source.splitlines()
            while index >= 0 and lines[index].rstrip().endswith("\\"):
                command_lines.insert(0, lines[index].rstrip())
                index -= 1
            command = " ".join(command_lines)
            if "9>&-" not in command:
                offenders.append(f"line {number}: {stripped.strip()}")
        self.assertEqual(
            offenders, [],
            "backgrounded commands must close the switch-lock fd with 9>&- "
            "so children cannot inherit the lock's open-file description:\n"
            + "\n".join(offenders),
        )

    def test_lock_acquisition_is_bounded_and_names_the_holder(self):
        source = SCRIPT.read_text()
        self.assertNotIn(
            "\n    flock -x 9\n", source,
            "raw unbounded flock reintroduced; use acquire_switch_lock",
        )
        self.assertNotIn(
            "\nflock -x 9\n", source,
            "raw unbounded flock reintroduced; use acquire_switch_lock",
        )
        self.assertIn("acquire_switch_lock() {", source)
        self.assertIn('flock -w "$SWITCH_LOCK_TIMEOUT_SECONDS" -x 9', source)
        self.assertIn("/proc/locks", source)
        # All three entry points (stop, restore, profile switch) go through
        # the helper: one definition plus three call sites.
        self.assertEqual(source.count("acquire_switch_lock"), 4)

    def test_stop_halts_active_profile_without_touching_active_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "qwen38-1m"})
            )
            (root / "qwen-running").touch()
            result = self.run_switch(root, "stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STOPPED qwen38-1m", result.stdout)
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"],
                "qwen38-1m",
            )
            self.assertFalse((root / "qwen-running").exists())
            self.assertIn("STOP qwen", (root / "actions.log").read_text())

    def test_stop_with_no_recorded_profile_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_switch(root, "stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "actions.log").exists())

    def test_qwen_hash_failure_is_rejected_before_active_profile_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            result = self.run_switch(root, "qwen38")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("qwen", result.stderr.lower())
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )

    def test_qwen_1m_hash_failure_is_rejected_before_active_profile_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.json").write_text(
                json.dumps({"schema_version": 1, "profile": "dsv4"})
            )
            result = self.run_switch(root, "qwen38-1m")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("qwen", result.stderr.lower())
            self.assertEqual(
                json.loads((root / "active.json").read_text())["profile"], "dsv4"
            )

    def test_switch_uses_a_dedicated_internal_port_without_changing_auth_endpoint(self):
        source = SCRIPT.read_text()
        service = DSV4_SERVICE.read_text()
        installer = INSTALLER.read_text()
        self.assertIn("readonly PORT=8013", source)
        self.assertIn("readonly AUTH_PORT=8010", source)
        # DSV4_PORT renders from the dsv4 profile's {port} placeholder.
        self.assertEqual(
            render_switch_snapshot("dsv4")["env"].get("DSV4_PORT"), "8013"
        )
        self.assertIn("Environment=DSV4_PORT=8013", service)
        self.assertIn("upstream_port=8013", installer)
        self.assertIn("configs/tmpfiles/frontier-at-home.conf", installer)
        self.assertIn("systemd-tmpfiles --create", installer)
        self.assertIn("68_provision_runtime_locks.py", installer)
        self.assertNotIn("readonly PORT=8011", source)

    def test_deepseek_readiness_requires_the_exact_1m_context(self):
        source = SCRIPT.read_text()
        self.assertIn('"/slots"', source)
        self.assertIn('slot["n_ctx"] != 393216', source)

    def test_switch_runs_deepseek_as_engine_user_with_frozen_1m_profile(self):
        source = SCRIPT.read_text()
        self.assertIn("install -d -o root -g dsv4 -m 1770 /run/dsv4", source)
        self.assertIn("/usr/sbin/runuser -u dsv4 --", source)
        env = render_switch_snapshot("dsv4")["env"]
        for key, value in (
            ("DSV4_SERVER_BINARY",
             "/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server"),
            ("DSV4_BUILD_MANIFEST",
             f"{ROOT}/configs/build-manifests/llamacpp-fusion.json"),
            ("DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB", "8"),
            ("DSV4_MEM_FLOOR_GIB", "8"),
            ("DSV4_WATCHDOG_FLOOR_GIB", "8"),
            ("DSV4_MEASURED_HEADLESS_OVERHEAD_GIB", "12"),
            ("DSV4_ALLOW_RETRY_AFTER_FAILED_START", "1"),
            ("DSV4_UBATCH", "2048"),
            ("DSV4_BATCH", "2048"),
            ("DSV4_UBATCH_LARGE", "1"),
            ("CTX", "1048576"),
            ("DSV4_PARALLEL", "2"),
            ("DSV4_NO_MMAP", "1"),
            ("DSV4_SPEC_TYPE", "none"),
        ):
            self.assertEqual(env.get(key), value, key)
        self.assertIn("DSV4_API_KEY_FILE:-/etc/deepseek-v4-flash/api-key", source)

    def test_installed_deepseek_service_uses_the_qualified_1m_profile(self):
        service = DSV4_SERVICE.read_text()
        for setting in (
            "Environment=DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=8",
            "Environment=DSV4_MEM_FLOOR_GIB=8",
            "Environment=DSV4_WATCHDOG_FLOOR_GIB=8",
            "Environment=DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=12",
            "Environment=DSV4_ALLOW_RETRY_AFTER_FAILED_START=1",
            "Environment=DSV4_UBATCH=2048",
            "Environment=DSV4_BATCH=2048",
            "Environment=DSV4_UBATCH_LARGE=1",
            "Environment=CTX=1048576",
            "Environment=DSV4_PARALLEL=2",
            "Environment=DSV4_SPEC_TYPE=none",
        ):
            self.assertIn(setting, service)

    def test_selected_profile_has_a_boot_restore_unit(self):
        source = SCRIPT.read_text()
        installer = INSTALLER.read_text()
        unit = RESTORE_SERVICE.read_text()
        self.assertIn("status [--json]|stop|restore|dsv4|glm52", source)
        self.assertIn('if [[ $command == restore ]]', source)
        self.assertIn("dsv4-engine-restore.service", installer)
        self.assertIn("52_engine_switch.sh restore", unit)
        self.assertIn("WantedBy=multi-user.target", unit)
        restore = source[source.index('if [[ $command == restore ]]') :]
        self.assertIn('stop_profile "$command"', restore)
        self.assertIn("After=dsv4-authhelper.service", unit)

    def test_restore_tolerates_launcher_removing_an_exact_stale_state(self):
        source = SCRIPT.read_text()
        stop = source[source.index("stop_profile() {") : source.index("start_dsv4() {")]
        self.assertIn("if dsv4_launcher stop", stop)
        self.assertIn("[[ ! -e /run/dsv4/llamacpp.state.json ]]", stop)
        self.assertIn('"http://127.0.0.1:$PORT/health"', stop)

    def test_control_plane_installer_cannot_start_a_model(self):
        source = CONTROL_INSTALLER.read_text()
        self.assertIn("must be run as root", source)
        self.assertIn("/home/dsv4/.dsv4-start-hold", source)
        self.assertIn("UPSTREAM_PORT=8013", source)
        self.assertIn("systemctl daemon-reload", source)
        self.assertIn("systemctl disable deepseek-v4-flash-llamacpp.service", source)
        self.assertIn("systemctl enable dsv4-engine-restore.service", source)
        self.assertIn("systemctl restart dsv4-authhelper.service", source)
        self.assertIn("configs/tmpfiles/frontier-at-home.conf", source)
        self.assertIn("systemd-tmpfiles --create", source)
        self.assertIn("68_provision_runtime_locks.py", source)
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
        self.assertEqual(glm["schema_version"], 3)
        self.assertEqual(glm["profile"], "glm52")
        self.assertEqual(
            glm["binary_sha256"],
            W7_BINARY_SHA256,
        )
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
            glm["promotion"],
            {
                "gate": "W7.1a-stable-model-cache-generation-owner-adoption",
                "engine_commit": "bccf0b6dd769854fe9e1cb8b5b3af966b161c071",
                "binary_freeze_sha256": hashlib.sha256(
                    W7_BINARY_FREEZE.read_bytes()
                ).hexdigest(),
                "owner_decision_sha256": hashlib.sha256(
                    W7_ADOPTION.read_bytes()
                ).hexdigest(),
                "review_sha256": hashlib.sha256(
                    W7_ADOPTION_REVIEW.read_bytes()
                ).hexdigest(),
            },
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
                "DS4_CUDA_STABLE_MODEL_REMAP": "1",
                "DS4_TOKEN_TIMING_LOG": "1",
            },
        )
        self.assertNotIn(
            "DS4_KV_SKIP_PRELOAD_EVICT_STORE_DIAGNOSTIC",
            glm["runtime"]["engine_environment"],
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
            # This superseded W7 manifest preserves the hashes reviewed at its
            # promotion. Several harness paths legitimately evolved later; do
            # not silently rebind that historical evidence to their live bytes.
            self.assertTrue((ROOT / path).is_file(), path)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", path)

        self.assertEqual(dsv4["schema_version"], 3)
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
