#!/usr/bin/env python3
"""Execute engine-switch safety checks through their production definitions."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.tests.switch_safety_fixtures import (
    SCRIPT,
    SwitchSafetyFixture,
    proc_identity,
)


ROOT = Path(__file__).resolve().parents[2]

SHOW_MAINPID = {
    "argv_prefix": [
        "show",
        "qwen38-engine.service",
        "--property=MainPID",
        "--value",
    ],
    "stdout": "0\n",
    "returncode": 0,
}


class SwitchProductionSafetyPathTests(unittest.TestCase):
    def test_source_only_fixture_seam_returns_before_main_dispatch(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                '[[ -z ${ENGINE_SWITCH_TESTING+x} ]]\n'
                'printf "SOURCED:%s\\n" "$STATE"'
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, f"SOURCED:{fixture.state}\n")
            self.assertEqual(fixture.systemctl_calls(), [])

        source = SCRIPT.read_text()
        seam = source.index("ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT")
        dispatch = source.index("command=${1:-status}")
        self.assertLess(seam, dispatch)

    def test_stop_qwen_dead_stale_record_is_removed_without_unit_stop(self):
        with SwitchSafetyFixture() as fixture:
            dead = fixture.spawn_sleep()
            pgid, ticks = proc_identity(dead.pid)
            fixture.stop_child(dead)
            record = fixture.write_qwen_record(
                dead.pid,
                pgid,
                ticks,
                memwatch_pid=dead.pid,
                memwatch_ticks=ticks,
            )
            fixture.set_systemctl_responses(SHOW_MAINPID)

            result = fixture.run_function("stop_qwen_verified")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(record.exists())
            calls = fixture.systemctl_calls()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "show")
            self.assertNotIn("stop", [call[0] for call in calls])

    def test_stop_qwen_live_start_ticks_mismatch_fails_closed(self):
        with SwitchSafetyFixture() as fixture:
            engine = fixture.spawn_sleep()
            pgid, ticks = proc_identity(engine.pid)
            record = fixture.write_qwen_record(
                engine.pid,
                pgid,
                ticks + 1,
                memwatch_pid=engine.pid,
                memwatch_ticks=ticks,
                exe_sha256=fixture.process_exe_sha256(engine.pid),
            )
            response = dict(SHOW_MAINPID, stdout=f"{engine.pid}\n")
            fixture.set_systemctl_responses(response)

            result = fixture.run_function("stop_qwen_verified")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale Qwen PID identity", result.stderr)
            self.assertTrue(record.exists())
            self.assertIsNone(engine.poll())
            self.assertNotIn(
                "stop", [call[0] for call in fixture.systemctl_calls()]
            )

    def test_memwatch_disarm_rejects_wrong_command_line(self):
        with SwitchSafetyFixture() as fixture:
            dead = fixture.spawn_sleep()
            engine_pgid, engine_ticks = proc_identity(dead.pid)
            fixture.stop_child(dead)
            impostor = fixture.spawn_sleep()
            _, impostor_ticks = proc_identity(impostor.pid)
            record = fixture.write_qwen_record(
                dead.pid,
                engine_pgid,
                engine_ticks,
                memwatch_pid=impostor.pid,
                memwatch_ticks=impostor_ticks,
            )
            fixture.set_systemctl_responses(SHOW_MAINPID)

            result = fixture.run_function("stop_qwen_verified")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Qwen memwatch identity changed; refusing to disarm",
                result.stderr,
            )
            self.assertTrue(record.exists())
            self.assertIsNone(impostor.poll())

    def test_memwatch_authenticated_disarm_acknowledgement_succeeds(self):
        with SwitchSafetyFixture() as fixture:
            dead = fixture.spawn_sleep()
            engine_pgid, engine_ticks = proc_identity(dead.pid)
            fixture.stop_child(dead)
            memwatch = fixture.spawn_memwatch("ack")
            _, memwatch_ticks = proc_identity(memwatch.pid)
            record = fixture.write_qwen_record(
                dead.pid,
                engine_pgid,
                engine_ticks,
                memwatch_pid=memwatch.pid,
                memwatch_ticks=memwatch_ticks,
            )
            fixture.set_systemctl_responses(SHOW_MAINPID)

            result = fixture.run_function("stop_qwen_verified")

            self.assertEqual(result.returncode, 0, result.stderr)
            fixture.wait_for_exit(memwatch)
            self.assertFalse(record.exists())
            self.assertFalse((fixture.state / "qwen38.memwatch.target").exists())
            self.assertFalse((fixture.state / "qwen38.memwatch.ready").exists())

    def test_memwatch_exit_without_disarm_acknowledgement_fails(self):
        with SwitchSafetyFixture() as fixture:
            dead = fixture.spawn_sleep()
            engine_pgid, engine_ticks = proc_identity(dead.pid)
            fixture.stop_child(dead)
            memwatch = fixture.spawn_memwatch("no-ack")
            _, memwatch_ticks = proc_identity(memwatch.pid)
            record = fixture.write_qwen_record(
                dead.pid,
                engine_pgid,
                engine_ticks,
                memwatch_pid=memwatch.pid,
                memwatch_ticks=memwatch_ticks,
            )
            fixture.set_systemctl_responses(SHOW_MAINPID)

            result = fixture.run_function("stop_qwen_verified")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Qwen memwatch disarm acknowledgement is missing",
                result.stderr,
            )
            fixture.wait_for_exit(memwatch)
            self.assertTrue(record.exists())

    def test_systemctl_show_failure_without_record_fails_closed(self):
        with SwitchSafetyFixture() as fixture:
            fixture.set_systemctl_responses(
                dict(SHOW_MAINPID, stdout="", returncode=1)
            )

            result = fixture.run_function("stop_qwen_verified")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to assume it is stopped", result.stderr)
            self.assertEqual(len(fixture.systemctl_calls()), 1)

    def test_laguna_guard_failure_under_if_caller_cannot_proceed(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                "laguna_hashes_verified=true\n"
                "if start_laguna_profile; then\n"
                '    printf "STARTER_PROCEEDED\\n"\n'
                "    exit 90\n"
                "fi\n"
                'printf "CALLER_CONTINUED\\n"'
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-load memory release gate failed", result.stderr)
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertNotIn("CALLER_CONTINUED", result.stdout)
            guard_args = (fixture.root / "guard.called").read_text()
            self.assertIn("--required-gib 100", guard_args)
            self.assertEqual(fixture.systemctl_calls(), [])

    def test_glm53_guard_failure_under_if_caller_cannot_proceed(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                "glm53_hashes_verified=true\n"
                "if start_glm53_profile; then\n"
                '    printf "STARTER_PROCEEDED\\n"\n'
                "    exit 90\n"
                "fi\n"
                'printf "CALLER_CONTINUED\\n"'
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-load memory release gate failed", result.stderr)
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertNotIn("CALLER_CONTINUED", result.stdout)
            guard_args = (fixture.root / "guard.called").read_text()
            self.assertIn("--required-gib 110", guard_args)
            self.assertEqual(fixture.systemctl_calls(), [])

    def test_stop_glm53_live_start_ticks_mismatch_fails_closed(self):
        with SwitchSafetyFixture() as fixture:
            engine = fixture.spawn_sleep()
            pgid, ticks = proc_identity(engine.pid)
            record = fixture.state / "glm53.process.json"
            record.write_text(json.dumps({
                "pid": engine.pid, "pgid": pgid, "start_ticks": ticks + 1,
                "exe_sha256": fixture.process_exe_sha256(engine.pid),
                "unit": "glm53-engine.service",
                "memwatch_pid": engine.pid, "memwatch_start_ticks": ticks,
            }))
            fixture.set_systemctl_responses({
                "argv_prefix": [
                    "show", "glm53-engine.service", "--property=MainPID", "--value",
                ],
                "stdout": f"{engine.pid}\n",
                "returncode": 0,
            })

            result = fixture.run_function("stop_glm53_verified")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale GLM-5.3 PID identity", result.stderr)
            self.assertTrue(record.exists())
            self.assertIsNone(engine.poll())
            self.assertNotIn(
                "stop", [call[0] for call in fixture.systemctl_calls()]
            )

    # -- Generic profile aliases -------------------------------------------
    # The source-only fixture has no profile tree, so each test pre-loads the
    # resolved plan exactly as generic_load caches it (generic_plan[alias:key]).

    @staticmethod
    def generic_plan_lines(fixture, **overrides) -> str:
        plan = {
            "alias": "dsv41flash",
            "relpath": "configs/profiles/deepseek-v4.1-flash/cuda-spark-128g-1m.json",
            "status": "qualified",
            "binary": str(fixture.artifacts / "bin" / "ds4-server"),
            "binary_sha256": "0" * 64,
            "model": str(fixture.artifacts / "models" / "dsv41.gguf"),
            "mmproj": "",
            "draft_model": "",
            "unit": "dsv41-engine.service",
            "unit_name": "dsv41-engine",
            "user": "bmarti44",
            "log_name": "dsv41",
            "required_gib": "110",
            "floor_gib": "10",
            "timeout": "1800",
            "served_model_id": "deepseek-v4.1-flash",
            "health_path": "",
            "context_check": "",
            "context_path": "",
        }
        plan.update(overrides)
        return "".join(
            f"generic_plan[{plan['alias']}:{key}]='{value}'\n"
            for key, value in plan.items()
        )

    @classmethod
    def qwen38fn_plan_lines(cls, fixture, **overrides) -> str:
        # The committed qwen38fn plan (test_switch_generic pins these values):
        # a vLLM /health route and the served id's model card at 262,144.
        plan = {
            "alias": "qwen38fn",
            "relpath": "configs/profiles/qwen3.8-flash-next/cuda-spark-128g-1m.json",
            "binary": str(fixture.artifacts / "bin" / "python3.12"),
            "model": str(fixture.artifacts / "models" / "qwen3.8-flash-next-nvfp4"),
            "unit": "qwen38fn-engine.service",
            "unit_name": "qwen38fn-engine",
            "log_name": "qwen38fn",
            "served_model_id": "qwen3.8-flash-next",
            "health_path": "/health",
            "context_check": json.dumps(
                {"kind": "model_card", "max_model_len": 262144,
                 "served_model_id": "qwen3.8-flash-next"},
                separators=(",", ":"), sort_keys=True),
            "context_path": "/v1/models",
        }
        plan.update(overrides)
        return cls.generic_plan_lines(fixture, **plan)

    def test_generic_guard_failure_under_if_caller_cannot_proceed(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                self.generic_plan_lines(fixture)
                + "generic_identities[dsv41flash]=verified\n"
                "if start_generic_profile dsv41flash; then\n"
                '    printf "STARTER_PROCEEDED\\n"\n'
                "    exit 90\n"
                "fi\n"
                'printf "CALLER_CONTINUED\\n"'
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-load memory release gate failed", result.stderr)
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertNotIn("CALLER_CONTINUED", result.stdout)
            guard_args = (fixture.root / "guard.called").read_text()
            self.assertIn("--required-gib 110", guard_args)
            self.assertEqual(fixture.systemctl_calls(), [])

    def test_generic_start_aborts_before_launch_when_memwatch_never_arms(self):
        with SwitchSafetyFixture() as fixture:
            fixture.set_systemctl_responses({
                "argv_prefix": [
                    "show", "dsv41-engine.service", "--property=LoadState",
                    "--value",
                ],
                "stdout": "not-found\n",
                "returncode": 0,
            })
            result = fixture.run_function(
                self.generic_plan_lines(fixture)
                + "generic_identities[dsv41flash]=verified\n"
                "export SWITCH_GUARD_EXIT=0\n"
                "start_generic_profile dsv41flash\n"
                'printf "STARTER_PROCEEDED\\n"',
                timeout=30,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "dsv41flash memory watchdog failed to initialize", result.stderr
            )
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertIn("--required-gib 110",
                          (fixture.root / "guard.called").read_text())
            calls = fixture.systemctl_calls()
            self.assertEqual(len(calls), 1)
            self.assertNotIn("stop", [call[0] for call in calls])
            self.assertFalse((fixture.state / "dsv41.process.json").exists())

    def test_generic_start_refuses_an_estimated_profile_before_the_guard(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                self.generic_plan_lines(fixture, status="estimated")
                + "generic_identities[dsv41flash]=verified\n"
                "start_generic_profile dsv41flash\n"
                'printf "STARTER_PROCEEDED\\n"'
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("estimated", result.stderr)
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertFalse((fixture.root / "guard.called").exists())
            self.assertEqual(fixture.systemctl_calls(), [])

    def test_stop_generic_live_start_ticks_mismatch_fails_closed(self):
        with SwitchSafetyFixture() as fixture:
            engine = fixture.spawn_sleep()
            pgid, ticks = proc_identity(engine.pid)
            record = fixture.state / "dsv41.process.json"
            record.write_text(json.dumps({
                "pid": engine.pid, "pgid": pgid, "start_ticks": ticks + 1,
                "exe_sha256": fixture.process_exe_sha256(engine.pid),
                "unit": "dsv41-engine.service",
                "memwatch_pid": engine.pid, "memwatch_start_ticks": ticks,
            }))
            fixture.set_systemctl_responses({
                "argv_prefix": [
                    "show", "dsv41-engine.service", "--property=MainPID", "--value",
                ],
                "stdout": f"{engine.pid}\n",
                "returncode": 0,
            })

            result = fixture.run_function(
                self.generic_plan_lines(fixture)
                + "stop_generic_verified dsv41flash"
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale dsv41flash PID identity", result.stderr)
            self.assertTrue(record.exists())
            self.assertIsNone(engine.poll())
            self.assertNotIn(
                "stop", [call[0] for call in fixture.systemctl_calls()]
            )

    def test_stop_generic_dead_record_is_removed_without_unit_stop(self):
        with SwitchSafetyFixture() as fixture:
            dead = fixture.spawn_sleep()
            pgid, ticks = proc_identity(dead.pid)
            fixture.stop_child(dead)
            record = fixture.state / "dsv41.process.json"
            record.write_text(json.dumps({
                "pid": dead.pid, "pgid": pgid, "start_ticks": ticks,
                "exe_sha256": "0" * 64, "unit": "dsv41-engine.service",
                "memwatch_pid": dead.pid, "memwatch_start_ticks": ticks,
            }))
            fixture.set_systemctl_responses({
                "argv_prefix": [
                    "show", "dsv41-engine.service", "--property=MainPID", "--value",
                ],
                "stdout": "0\n",
                "returncode": 0,
            })

            result = fixture.run_function(
                self.generic_plan_lines(fixture)
                + "stop_generic_verified dsv41flash"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(record.exists())
            self.assertEqual(len(fixture.systemctl_calls()), 1)

    def test_stop_generic_unresolvable_alias_fails_closed(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function("stop_generic_verified dsv41flash")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dsv41flash", result.stderr)
            self.assertEqual(fixture.systemctl_calls(), [])

    def test_legacy_verify_serving_stops_at_a_wrong_served_model(self):
        # verify_serving runs as a condition (restore, rollback, "already
        # active"), where set -e is off: a wrong /v1/models id must still end
        # the check before any further probe.
        fake_curl = (
            "clean_curl() {\n"
            '    printf "CURL %s\\n" "${*: -1}" >>"$SWITCH_FIXTURE_ROOT/curl.log"\n'
            '    printf \'{"data":[{"id":"%s"}]}\' "$FAKE_MODEL_ID"\n'
            "}\n"
        )
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                fake_curl
                + "FAKE_MODEL_ID=deepseek-v4-flash\n"
                "if verify_serving glm52; then echo VERIFY_OK; else echo VERIFY_FAILED; fi\n",
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("VERIFY_FAILED", result.stdout)
            calls = (fixture.root / "curl.log").read_text().splitlines()
            self.assertEqual(len(calls), 1, calls)
            self.assertIn("/v1/models", calls[0])

    def test_generic_readiness_requires_the_profile_served_model_id(self):
        # A DeepSeek V4 server answering on the port must never satisfy the
        # dsv41flash readiness or serving checks.
        fake_curl = (
            "clean_curl() {\n"
            '    printf "CURL %s\\n" "${*: -1}" >>"$SWITCH_FIXTURE_ROOT/curl.log"\n'
            '    printf \'{"data":[{"id":"%s"}]}\' "$FAKE_MODEL_ID"\n'
            "}\n"
            "verify_generic_process_ready() { return 0; }\n"
        )
        with SwitchSafetyFixture() as fixture:
            library = fixture.repo / "scripts" / "lib"
            library.mkdir(parents=True)
            shutil.copy(ROOT / "scripts/lib/switch_generic.py", library)
            result = fixture.run_function(
                self.generic_plan_lines(fixture, timeout="1")
                + fake_curl
                + "FAKE_MODEL_ID=deepseek-v4-flash\n"
                "if wait_model_ready dsv41flash; then echo WAIT_OK; else echo WAIT_FAILED; fi\n"
                "if verify_serving dsv41flash; then echo VERIFY_OK; else echo VERIFY_FAILED; fi\n"
                "FAKE_MODEL_ID=deepseek-v4.1-flash\n"
                "if wait_model_ready dsv41flash; then echo WAIT2_OK; else echo WAIT2_FAILED; fi\n",
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("WAIT_FAILED", result.stdout)
            self.assertIn("VERIFY_FAILED", result.stdout)
            self.assertIn("WAIT2_OK", result.stdout)
            calls = (fixture.root / "curl.log").read_text()
            self.assertIn("/v1/models", calls)
            self.assertNotIn("/v1/chat/completions", calls)
            self.assertNotIn("/slots", calls)
            self.assertNotIn("/health", calls)

    def test_qwen38fn_low_memory_guard_blocks_the_launch(self):
        with SwitchSafetyFixture() as fixture:
            result = fixture.run_function(
                self.qwen38fn_plan_lines(fixture)
                + "generic_identities[qwen38fn]=verified\n"
                "if start_generic_profile qwen38fn; then\n"
                '    printf "STARTER_PROCEEDED\\n"\n'
                "    exit 90\n"
                "fi\n"
                'printf "CALLER_CONTINUED\\n"'
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-load memory release gate failed", result.stderr)
            self.assertNotIn("STARTER_PROCEEDED", result.stdout)
            self.assertNotIn("CALLER_CONTINUED", result.stdout)
            self.assertIn("--required-gib 110",
                          (fixture.root / "guard.called").read_text())
            self.assertEqual(fixture.systemctl_calls(), [])
            self.assertFalse((fixture.state / "qwen38fn.process.json").exists())

    def test_qwen38fn_readiness_requires_health_and_the_262144_model_card(self):
        # vLLM answers /v1/models before it is healthy; a wrong
        # --max-model-len must never pass the readiness check either.
        fake_curl = (
            "clean_curl() {\n"
            '    printf "CURL %s\\n" "${*: -1}" >>"$SWITCH_FIXTURE_ROOT/curl.log"\n'
            '    if [[ ${*: -1} == */health ]]; then return "$FAKE_HEALTH"; fi\n'
            '    printf \'{"data":[{"id":"qwen3.8-flash-next","max_model_len":%s},\' "$FAKE_MAX_LEN"\n'
            '    printf \'{"id":"default","max_model_len":%s}]}\' "$FAKE_MAX_LEN"\n'
            "}\n"
            "verify_generic_process_ready() { return 0; }\n"
        )
        with SwitchSafetyFixture() as fixture:
            library = fixture.repo / "scripts" / "lib"
            library.mkdir(parents=True)
            for name in ("switch_generic.py", "profile_resolver.py"):
                shutil.copy(ROOT / "scripts/lib" / name, library)
            result = fixture.run_function(
                self.qwen38fn_plan_lines(fixture, timeout="1")
                + fake_curl
                + "FAKE_HEALTH=0 FAKE_MAX_LEN=131072\n"
                "if wait_model_ready qwen38fn; then echo SHORT_OK; else echo SHORT_FAILED; fi\n"
                "FAKE_HEALTH=22 FAKE_MAX_LEN=262144\n"
                "if wait_model_ready qwen38fn; then echo SICK_OK; else echo SICK_FAILED; fi\n"
                "FAKE_HEALTH=0 FAKE_MAX_LEN=262144\n"
                "if wait_model_ready qwen38fn; then echo READY_OK; else echo READY_FAILED; fi\n",
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SHORT_FAILED", result.stdout)
            self.assertIn("SICK_FAILED", result.stdout)
            self.assertIn("READY_OK", result.stdout)
            calls = (fixture.root / "curl.log").read_text()
            self.assertIn("/health", calls)
            self.assertIn("/v1/models", calls)
            self.assertNotIn("/slots", calls)

    def test_laguna_hash_verification_rejects_tampered_second_shard(self):
        with SwitchSafetyFixture() as fixture:
            artifacts = fixture.install_laguna_artifacts()
            fixture.tamper_same_size(artifacts["shard2"])

            result = fixture.run_function("verify_laguna_profile_hashes")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Laguna shard 2 hash is not approved", result.stderr)
            self.assertIn("Laguna artifact verification failed", result.stderr)

    def test_laguna_hash_verification_rejects_tampered_shared_library(self):
        with SwitchSafetyFixture() as fixture:
            artifacts = fixture.install_laguna_artifacts()
            fixture.tamper_same_size(artifacts["library"])

            result = fixture.run_function("verify_laguna_profile_hashes")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Laguna shared-library hash is not approved", result.stderr
            )
            self.assertIn("Laguna artifact verification failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
