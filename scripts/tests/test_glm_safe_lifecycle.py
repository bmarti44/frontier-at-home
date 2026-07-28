#!/usr/bin/env python3
"""No-model integration mutations for candidate provenance shutdown."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
FIXTURE = ROOT / "scripts/tests/fixtures/candidate_lifecycle.c"


class CandidateLifecycleSourceTests(unittest.TestCase):
    def test_exit_race_enters_shutdown_grace_before_identity_failure(self):
        source = SAFE.read_text(encoding="utf-8")
        shutdown = source.index("if [[ -z $CURRENT_STATE")
        identity = source.index(
            'plog "FATAL executed candidate identity changed pid=$EXECUTED_PID',
            shutdown,
        )
        guarded = source[shutdown:identity]
        self.assertIn("-z $CURRENT_HASH", guarded)
        self.assertIn("-z $CURRENT_DEVICE_INODE", guarded)


class CandidateLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is unavailable")
        probe = subprocess.run(
            ["sudo", "-n", "-u", "dsv4", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode:
            raise unittest.SkipTest("passwordless dsv4 test account is unavailable")
        available_gib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        ) / 1048576
        if available_gib < 110:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

        cls.local_tmp = Path(tempfile.mkdtemp(prefix="glm-lifecycle-"))
        os.chmod(cls.local_tmp, 0o755)
        cls.runner = cls.local_tmp / "runner"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-o", cls.runner, FIXTURE],
            check=True,
        )
        os.chmod(cls.runner, 0o755)
        created = subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "mktemp", "-d",
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle.XXXXXX",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        cls.candidate_src = Path(created.stdout.strip())
        subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "cp", "--",
                "/usr/bin/sleep", cls.candidate_src / "ds4-server",
            ],
            check=True,
        )
        cls.digest = hashlib.sha256(
            Path("/usr/bin/sleep").read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "candidate_src"):
            target = str(cls.candidate_src)
            if target.startswith(
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle."
            ):
                subprocess.run(
                    ["sudo", "-n", "-u", "dsv4", "rm", "-rf", "--", target],
                    check=False,
                )
        if hasattr(cls, "local_tmp"):
            shutil.rmtree(cls.local_tmp)

    def run_mutation(self, mode: str) -> subprocess.CompletedProcess[str]:
        environment = {
            "HOME": "/home/dsv4",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GLM_CANDIDATE_SRC": str(self.candidate_src),
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "30",
        }
        return subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "env",
                *[f"{key}={value}" for key, value in environment.items()],
                "bash", str(SAFE), "--tag", f"lifecycle-{mode}", "--",
                str(self.runner), mode, str(self.candidate_src / "ds4-server"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )

    def test_clean_candidate_zombie_during_wrapper_exit_is_accepted(self):
        result = self.run_mutation("clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_replacement_candidate_during_exit_is_rejected(self):
        result = self.run_mutation("replace")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonzero_wrapper_after_candidate_exit_is_rejected(self):
        result = self.run_mutation("fail")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrapper_exceeding_shutdown_grace_is_rejected(self):
        result = self.run_mutation("linger")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
