#!/usr/bin/env python3
"""Execute engine-switch safety checks through their production definitions."""

from __future__ import annotations

import unittest

from scripts.tests.switch_safety_fixtures import (
    SCRIPT,
    SwitchSafetyFixture,
    proc_identity,
)


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
