#!/usr/bin/env python3
"""Off-Spark (macOS) portability of the portable serve and build paths.

The apple-silicon profiles were authored without ever running on a Mac.
These tests pin the pieces that must agree for a Mac host to build the
pinned engine where its profile looks for it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN38_BUILD = ROOT / "scripts/12_build_qwen38_llamacpp.sh"
PROFILE_SERVE = ROOT / "scripts/93_profile_serve.sh"
MEMWATCH_MACOS = ROOT / "scripts/06_memwatch_macos.sh"
SETSID_SHIM = ROOT / "scripts/lib/setsid.py"
DARWIN = sys.platform == "darwin"


def _shell_function(script: Path, name: str) -> str:
    """One top-level `name() {` ... `}` definition from a bash script."""
    body, inside = [], False
    for line in script.read_text().splitlines():
        if line.startswith(f"{name}() {{"):
            inside = True
        if inside:
            body.append(line)
            if line == "}":
                return "\n".join(body)
    raise AssertionError(f"{name} not found in {script}")


def _readonly_block(script: Path) -> str:
    """The script's leading readonly path constants, as shell source."""
    lines = [line for line in script.read_text().splitlines()
             if re.match(r"^readonly (CACHE_ROOT|SOURCE_DIR|BUILD_DIR)=", line)]
    return "\n".join(lines)


class Qwen38BuildCacheRootTests(unittest.TestCase):
    def _cache_root(self, home: str) -> str:
        source = _readonly_block(QWEN38_BUILD) + '\nprintf "%s" "$CACHE_ROOT"\n'
        return subprocess.run(
            ["bash", "-c", source], env={"HOME": home, "PATH": os.environ["PATH"]},
            capture_output=True, text=True, check=True).stdout

    def test_cache_root_follows_home(self):
        # The profile's binary_path is {cache_root}/llamacpp-qwen38-9d77fa17/...;
        # a hardcoded Linux home makes the Mac build unwritable.
        self.assertEqual(self._cache_root("/Users/someone"),
                         "/Users/someone/.cache/llamacpp-qwen38-9d77fa17")

    def test_spark_cache_root_is_unchanged(self):
        self.assertEqual(self._cache_root("/home/bmarti44"),
                         "/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17")


@unittest.skipUnless(DARWIN, "macOS process-identity primitives")
class WatchdogIdentityTests(unittest.TestCase):
    """93 publishes the target the macOS watchdog verifies before arming."""

    def test_serve_start_ticks_match_memwatch_identity(self):
        # 06 arms only when the published start_ticks is numeric and equal
        # to its own epoch; anything else fails closed and leaves the server
        # unguarded.
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            source = "\n".join([
                "set -Eeuo pipefail",
                _shell_function(PROFILE_SERVE, "proc_start_ticks"),
                'printf "%s\\n" "$(proc_start_ticks "$1")"',
            ])
            ticks = subprocess.run(["bash", "-c", source, "_", str(sleeper.pid)],
                                   capture_output=True, text=True,
                                   check=True).stdout.strip()
            source = "\n".join([
                "set -Eeuo pipefail", "export LC_ALL=C",
                _shell_function(MEMWATCH_MACOS, "proc_identity"),
                'proc_identity "$1"',
            ])
            identity = subprocess.run(["bash", "-c", source, "_", str(sleeper.pid)],
                                      capture_output=True, text=True,
                                      check=True).stdout.split()
        finally:
            sleeper.kill()
            sleeper.wait()
        self.assertRegex(ticks, r"^[0-9]+$")
        self.assertEqual(ticks, identity[1])


class SetsidLauncherTests(unittest.TestCase):
    def test_serve_never_calls_bare_setsid(self):
        # macOS has no setsid(1); launch sites go through the launcher
        # resolved once at startup.
        launches = [line for line in PROFILE_SERVE.read_text().splitlines()
                    if re.match(r"^\s*setsid\s", line)]
        self.assertEqual(launches, [])

    def test_shim_makes_the_child_its_own_group_leader(self):
        # Stop and the watchdog signal -PID, so PGID must equal the PID that
        # bash recorded in $!.
        source = (f'python3 "{SETSID_SHIM}" sleep 30 & child=$!; sleep 0.5; '
                  'printf "%s %s\\n" "$child" "$(ps -o pgid= -p "$child" | tr -d " ")"; '
                  'kill "$child"')
        pid, pgid = subprocess.run(["bash", "-c", source], capture_output=True,
                                   text=True, check=True).stdout.split()
        self.assertEqual(pid, pgid)


if __name__ == "__main__":
    unittest.main()
