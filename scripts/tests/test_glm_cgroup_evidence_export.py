#!/usr/bin/env python3
"""No-model integration checks for fail-safe evidence readability."""

from __future__ import annotations

import os
import subprocess
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"


class CgroupEvidenceExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if subprocess.run(
            ["systemctl", "--user", "show-environment"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode:
            raise unittest.SkipTest("user systemd manager is unavailable")
        if subprocess.run(
            ["sudo", "-n", "-u", "dsv4", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode:
            raise unittest.SkipTest("passwordless dsv4 test account is unavailable")
        available_kib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        if available_kib < 110 * 1048576:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

    def run_case(self, requested_rc: int) -> tuple[subprocess.CompletedProcess[str], Path]:
        suffix = uuid.uuid4().hex[:10]
        evidence = Path(
            f"/home/dsv4/ds4-project/glm52-confirm-export-{suffix}"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "GLM_SAFE_EVIDENCE_DIR": str(evidence),
                "GLM_SAFE_KILL_FLOOR_GIB": "40",
                "GLM_SAFE_MIN_START_GIB": "110",
                "GLM_SAFE_TIMEOUT_S": "30",
            }
        )
        result = subprocess.run(
            [
                str(LAUNCHER), "--tag", f"export-{suffix}", "--",
                "/usr/bin/bash", "-c",
                'mkdir -p -- "$1"; umask 077; printf evidence >"$1/raw"; '
                'sleep 1; exit "$2"',
                "sh", str(evidence), str(requested_rc),
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
        return result, evidence

    def cleanup(self, evidence: Path) -> None:
        target = str(evidence)
        if target.startswith(
            "/home/dsv4/ds4-project/glm52-confirm-export-"
        ):
            subprocess.run(
                ["sudo", "-n", "-u", "dsv4", "rm", "-rf", "--", target],
                check=False,
            )

    def test_symlink_root_fails_without_changing_target_permissions(self):
        suffix = uuid.uuid4().hex[:10]
        evidence = Path(
            f"/home/dsv4/ds4-project/glm52-confirm-export-{suffix}"
        )
        protected = Path(f"/home/dsv4/ds4-project/export-protected-{suffix}")
        environment = os.environ.copy()
        environment.update(
            {
                "GLM_SAFE_EVIDENCE_DIR": str(evidence),
                "GLM_SAFE_KILL_FLOOR_GIB": "40",
                "GLM_SAFE_MIN_START_GIB": "110",
                "GLM_SAFE_TIMEOUT_S": "30",
            }
        )
        result = subprocess.run(
            [
                str(LAUNCHER), "--tag", f"export-link-{suffix}", "--",
                "/usr/bin/bash", "-c",
                'umask 077; mkdir -- "$1"; printf protected >"$1/raw"; '
                'ln -s -- "$1" "$2"; sleep 1',
                "sh", str(protected), str(evidence),
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
        try:
            self.assertEqual(result.returncode, 16, result.stdout + result.stderr)
            self.assertTrue(evidence.is_symlink())
            self.assertEqual(protected.stat().st_mode & 0o777, 0o700)
            self.assertEqual((protected / "raw").stat().st_mode & 0o777, 0o600)
        finally:
            self.cleanup(evidence)
            if str(protected).startswith(
                "/home/dsv4/ds4-project/export-protected-"
            ):
                subprocess.run(
                    ["sudo", "-n", "-u", "dsv4", "rm", "-rf", "--",
                     str(protected)],
                    check=False,
                )

    def test_success_exports_reviewer_readable_evidence(self):
        result, evidence = self.run_case(0)
        try:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((evidence / "raw").read_text(), "evidence")
        finally:
            self.cleanup(evidence)

    def test_failure_exports_evidence_and_preserves_status(self):
        result, evidence = self.run_case(7)
        try:
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertEqual((evidence / "raw").read_text(), "evidence")
        finally:
            self.cleanup(evidence)


if __name__ == "__main__":
    unittest.main()
