"""Real process controls for the evidence-only Python probe supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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

    def test_surviving_descendant_fails_and_is_killed(self):
        result, summary, _ = self.run_probe("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])\ntime.sleep(0.35)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("descendant", summary["failure"])
        self.assertEqual(summary["live_process_group_after"], [])


if __name__ == "__main__":
    unittest.main()
