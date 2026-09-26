#!/usr/bin/env python3
"""Mutation tests for bundle_attempt.py (review round 2, findings 1-3).

Builds synthetic attempt directories from the real, clean diag-cache42gb run
(its containment logs and timestamps) and checks that every missing, stale or
failing piece of evidence yields FAIL, that the text arm stays PENDING_REVIEW
without an explicit review, and that exit codes are 0/1/2.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLER = os.path.join(HERE, "bundle_attempt.py")
SOURCE = os.path.join(HERE, "..", "smoke-2026-09-25", "diag-cache42gb")
KEEP = ["command.txt", "unit.txt", "started_at.txt", "finished_at.txt", "exit_code.txt", "containment"]


def steps(n):
    word = list(b" token")
    return {"source": "ds4", "prompt_tokens": 20, "ctx": 8192, "top_k": 5,
            "steps": [{"step": i, "selected": {"id": 1000 + i, "text": " token", "bytes": word},
                       "top_logprobs": []} for i in range(n)]}


class BundleMutations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.join(self.tmp.name, "attempt")
        os.makedirs(self.d)
        for name in KEEP:
            src = os.path.join(SOURCE, name)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(self.d, name))
            else:
                shutil.copy2(src, os.path.join(self.d, name))
        # Fresh timestamps for everything except started_at.txt's content.
        for root, _, files in os.walk(self.d):
            for f in files:
                os.utime(os.path.join(root, f))
        self.write("identity.json", {"verified": True})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, obj):
        with open(os.path.join(self.d, name), "w") as f:
            if isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f)

    def bundle(self, arm):
        proc = subprocess.run([sys.executable, BUNDLER, self.d, arm], capture_output=True, text=True)
        with open(os.path.join(self.d, "summary.json")) as f:
            return proc.returncode, json.load(f)["verdict"]

    def test_clean_diag_passes(self):
        self.assertEqual(self.bundle("diag"), (0, "PASS"))

    def test_missing_kernel_log_fails(self):
        os.remove(os.path.join(self.d, "containment", "kernel.log"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_samples_fails(self):
        os.remove(os.path.join(self.d, "containment", "samples.log"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_identity_fails(self):
        os.remove(os.path.join(self.d, "identity.json"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_unverified_identity_fails(self):
        self.write("identity.json", {"verified": False})
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_stale_artifact_fails(self):
        p = os.path.join(self.d, "containment", "cmd.log")
        os.utime(p, (1_000_000_000, 1_000_000_000))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_kernel_fault_fails(self):
        with open(os.path.join(self.d, "containment", "kernel.log"), "a") as f:
            f.write("kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=1\n")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_nonzero_exit_fails(self):
        self.write("exit_code.txt", "16\n")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_text_without_steps_fails(self):
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_short_fails(self):
        self.write("steps.json", steps(100))
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_without_review_is_pending(self):
        self.write("steps.json", steps(130))
        self.assertEqual(self.bundle("text"), (2, "PENDING_REVIEW"))
        with open(os.path.join(self.d, "output.txt")) as f:
            self.assertTrue(f.read().startswith(" token token"))

    def test_text_negative_review_fails(self):
        self.write("steps.json", steps(130))
        self.write("text-review.json", {"coherent_on_topic": False})
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_positive_review_passes(self):
        self.write("steps.json", steps(130))
        self.write("text-review.json", {"coherent_on_topic": True})
        self.assertEqual(self.bundle("text"), (0, "PASS"))

    def test_fidelity_summary_without_tsv_fails(self):
        self.write("fidelity-summary.json", {"verdict": "PASS"})
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_failed_score_fails(self):
        self.write("cuda.tsv", "id\tnll\ncase_000\t1.0\n")
        self.write("fidelity-summary.json", {"verdict": "FAIL"})
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
