#!/usr/bin/env python3
"""Regression test for post-build Git metadata isolation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "scripts/glm52_w1_affine_campaign.py"


def load_campaign():
    spec = importlib.util.spec_from_file_location(
        "glm52_w1_affine_campaign", CAMPAIGN
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootGitIsolationTests(unittest.TestCase):
    def test_rewritten_worktree_git_file_cannot_run_configured_helper(self):
        campaign = load_campaign()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            trusted = base / "trusted"
            worktree = base / "worktree"
            malicious = base / "malicious"
            marker = base / "payload-ran"
            payload = base / "payload.sh"
            payload.write_text(
                f"#!/bin/sh\n/usr/bin/touch {marker}\nexit 0\n",
                encoding="utf-8",
            )
            payload.chmod(0o700)
            environment = {
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            }

            subprocess.run(
                ["/usr/bin/git", "init", "-q", str(trusted)],
                check=True,
                env=environment,
            )
            (trusted / "tracked").write_text("trusted\n", encoding="utf-8")
            subprocess.run(
                ["/usr/bin/git", "-C", str(trusted), "add", "tracked"],
                check=True,
                env=environment,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(trusted), "commit", "-qm", "base"],
                check=True,
                env=environment,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(trusted),
                    "worktree",
                    "add",
                    "-q",
                    str(worktree),
                    "HEAD",
                ],
                check=True,
                env=environment,
            )
            trusted_git_dir = Path(
                subprocess.run(
                    [
                        "/usr/bin/git",
                        "-C",
                        str(worktree),
                        "rev-parse",
                        "--absolute-git-dir",
                    ],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    env=environment,
                ).stdout.strip()
            )

            subprocess.run(
                ["/usr/bin/git", "init", "-q", str(malicious)],
                check=True,
                env=environment,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(malicious),
                    "config",
                    "core.fsmonitor",
                    str(payload),
                ],
                check=True,
                env=environment,
            )
            (worktree / ".git").write_text(
                f"gitdir: {malicious / '.git'}\n",
                encoding="utf-8",
            )

            subprocess.run(
                ["/usr/bin/git", "-C", str(worktree), "status", "--porcelain"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            self.assertTrue(marker.exists(), "attack control did not execute")
            marker.unlink()

            completed = subprocess.run(
                campaign._trusted_git(
                    worktree,
                    "status",
                    "--porcelain",
                    git_dir=trusted_git_dir,
                ),
                cwd=worktree,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=campaign._git_environment(),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
