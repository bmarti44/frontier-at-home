#!/usr/bin/env python3
"""Security contract for the one-time, root-owned GLM W1 authority."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = ROOT / "scripts" / "65_glm52_w1_submit.py"
INSTALLER = ROOT / "scripts" / "66_install_glm52_w1_attestor.sh"


def load_submitter():
    spec = importlib.util.spec_from_file_location("glm52_w1_submit", SUBMITTER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load GLM W1 submitter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootAttestorContractTests(unittest.TestCase):
    def test_submitter_accepts_hashes_only(self):
        submitter = load_submitter()
        sha1 = "1" * 40
        sha256 = "2" * 64
        self.assertEqual(
            submitter.parse_request(["run", sha1, sha1, sha256]),
            ("run", sha1, sha1, sha256),
        )
        self.assertEqual(
            submitter.parse_request(["status", sha256]),
            ("status", sha256),
        )
        for malformed in (
            [],
            ["run", sha1, sha1],
            ["run", "../repo", sha1, sha256],
            ["run", sha1, sha1, sha256, "--command=id"],
            ["status", sha256, "extra"],
            ["shell", sha256],
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    submitter.parse_request(malformed)

    def test_submitter_has_fixed_trust_roots_and_no_shell_escape(self):
        submitter = load_submitter()
        self.assertEqual(
            submitter.REPOSITORY,
            Path("/home/bmarti44/spark-deepseek-v4-flash"),
        )
        self.assertEqual(
            submitter.ENGINE_REPOSITORY,
            Path("/home/bmarti44/.cache/glm52-w1-real-capture-a37"),
        )
        self.assertEqual(submitter.STATE_ROOT, Path("/var/lib/glm52-w1"))
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("eval(", source)

    def test_installer_closes_docker_root_equivalence_and_is_hash_pinned(self):
        source = INSTALLER.read_text(encoding="utf-8")
        self.assertIn(
            "/usr/sbin/gpasswd -d bmarti44 docker",
            source,
        )
        self.assertRegex(
            source,
            r"systemctl disable --now docker\\.socket docker\\.service",
        )
        self.assertIn("/usr/sbin/visudo -cf", source)
        self.assertIn("NOPASSWD: /usr/local/sbin/glm52-w1-submit *", source)
        self.assertNotRegex(source, r"NOPASSWD:\\s*ALL")
        expected = hashlib.sha256(SUBMITTER.read_bytes()).hexdigest()
        match = re.search(
            r"^readonly SUBMITTER_SHA256='([0-9a-f]{64})'$",
            source,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), expected)

    def test_installer_requires_clean_exact_head(self):
        source = INSTALLER.read_text(encoding="utf-8")
        for required in (
            "must run as root",
            "candidate is not HEAD",
            "repository is not clean",
            "reviewed submitter digest differs",
            "install -o root -g root -m 0755",
            "install -d -o root -g root -m 0700",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
