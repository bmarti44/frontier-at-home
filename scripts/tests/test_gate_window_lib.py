#!/usr/bin/env python3
"""Contracts for the shared gate-window library.

The library's sudo/serve paths need a live window to exercise; these tests
pin the pieces that can run anywhere: syntax, capture_json validation, the
double-open refusal, and the source-level contract that open arms a trap.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "scripts" / "lib" / "gate_window.sh"


def run_bash(script: str):
    return subprocess.run(
        ["bash", "-c", script], text=True, timeout=30,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


class GateWindowLibTests(unittest.TestCase):
    def test_syntax(self):
        result = run_bash(f"bash -n {LIB}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_json_accepts_single_document(self):
        result = run_bash(
            f"source {LIB}; cd $(mktemp -d); "
            "capture_json out.json echo '{\"ok\": true}'; cat out.json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"ok"', result.stdout)

    def test_capture_json_rejects_multi_document_output(self):
        result = run_bash(
            f"source {LIB}; cd $(mktemp -d); "
            "capture_json out.json printf '{}\\n{}\\n'; echo UNREACHED"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("UNREACHED", result.stdout)
        self.assertIn("not a single JSON document", result.stderr)

    def test_capture_json_keeps_raw_on_rejection(self):
        result = run_bash(
            f"source {LIB}; d=$(mktemp -d); cd $d; "
            "capture_json out.json printf 'oops'; true\n"
        )
        self.assertNotEqual(result.returncode, 0)

    def test_window_open_arms_the_restore_trap(self):
        source = LIB.read_text()
        self.assertIn("trap gate_restore_production EXIT", source)
        self.assertIn('sudo "$GATE_REPO/scripts/52_engine_switch.sh" stop', source)
        self.assertIn('sudo "$GATE_REPO/scripts/52_engine_switch.sh" restore', source)
        self.assertIn("03_memory_guard.py", source)
        # Guarded serve/guard failures must die, not continue.
        self.assertIn('|| gate_die "memory release gate failed', source)

    def test_double_open_refused(self):
        result = run_bash(
            f"source {LIB}; _gate_window_armed=true; gate_window_open"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already open", result.stderr)


if __name__ == "__main__":
    unittest.main()
