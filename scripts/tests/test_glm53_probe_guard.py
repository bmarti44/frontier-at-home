"""Real process controls for the evidence-only Python probe supervisor."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

GUARD = Path(__file__).resolve().parents[1] / "38_guard_glm53_probe.py"


class ProbeGuardTests(unittest.TestCase):
    def run_probe(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text(source)
            output = root / "evidence"
            result = subprocess.run([sys.executable, "-I", "-B", str(GUARD), "--output", str(output),
                                     "--", str(target)], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                                    capture_output=True, text=True, timeout=15)
            self.assertTrue((output / "summary.json").is_file(), result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            rows = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
            return result, summary, rows

    def test_pinned_child_gets_continuous_identity_samples(self):
        result, summary, rows = self.run_probe("import time\ntime.sleep(0.65)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["verdict"], "PASS")
        samples = [row for row in rows if row["event"] == "identity"]
        self.assertGreaterEqual(len(samples), 3)
        self.assertEqual(len({row["pid"] for row in samples}), 1)
        self.assertEqual(len({row["start_ticks"] for row in samples}), 1)
        self.assertTrue(all(row["executable_verified"] and row["argv_verified"] and row["environment_verified"] for row in samples))

    def test_same_python_exec_with_changed_command_rejects(self):
        result, summary, _ = self.run_probe("import os,sys\nos.execv(sys.executable,[sys.executable,'-c','import time;time.sleep(5)'])\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("argv", summary["failure"])

    def test_replaced_executable_rejects(self):
        result, summary, _ = self.run_probe("import os\nos.execv('/bin/sleep',['sleep','5'])\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("executable", summary["failure"])

    def test_terminal_exec_cannot_substitute_for_verified_completion(self):
        result, summary, _ = self.run_probe("import os,time\ntime.sleep(0.35)\nos.execv('/bin/true',['true'])\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completion", summary["failure"])

    def test_normal_zero_system_exit_gets_verified_completion(self):
        result, summary, rows = self.run_probe("import time\ntime.sleep(0.35)\nraise SystemExit(0)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(sum(row.get("completion_verified", False) for row in rows), 1)

    def test_atexit_replacement_cannot_bypass_terminal_identity(self):
        result, summary, _ = self.run_probe("import atexit,os,time\natexit.register(lambda: os.execv('/bin/true',['true']))\ntime.sleep(0.35)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["verdict"], "FAIL")

    def test_normal_atexit_cleanup_is_preserved(self):
        result, summary, _ = self.run_probe("import atexit,time\natexit.register(lambda: print('NORMAL_FINALIZER_RAN'))\ntime.sleep(0.35)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertIn("NORMAL_FINALIZER_RAN", result.stdout)

    def test_immediate_exit_cannot_substitute_for_verified_completion(self):
        result, summary, _ = self.run_probe("import os,time\ntime.sleep(0.35)\nos._exit(0)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completion", summary["failure"])

    def test_reused_group_anchor_is_never_signaled(self):
        spec = importlib.util.spec_from_file_location("probe_guard", GUARD)
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with mock.patch.object(guard, "process_stat", return_value={"start_ticks": 43, "pgid": 123, "ppid": os.getpid(), "state": "Z"}), \
             mock.patch.object(guard, "live_group") as group, mock.patch.object(guard.os, "pidfd_open") as open_pid:
            with self.assertRaisesRegex(ValueError, "anchor"):
                guard.terminate_group(123, 42)
            group.assert_not_called()
            open_pid.assert_not_called()

    def test_surviving_descendant_fails_and_is_killed(self):
        result, summary, _ = self.run_probe("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])\ntime.sleep(0.35)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("descendant", summary["failure"])
        self.assertEqual(summary["live_process_group_after"], [])


if __name__ == "__main__":
    unittest.main()
