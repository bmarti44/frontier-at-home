#!/usr/bin/env python3
"""Mutation tests for score_fidelity.py (review finding H3).

Uses the upstream Metal reference TSV and fixture manifest from the pinned ds4
worktree. Every malformed or adversarial input must be rejected (non-zero exit),
and the unmodified self-comparison must PASS.
"""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "score_fidelity.py")
FIX = ("/home/bmarti44/.cache/ds4-v41-0aaea5a2/gguf-tools/quality-testing/"
       "deepseek-v4.1-flash-20260919-router")
METAL = os.path.join(FIX, "results", "base-default.tsv")
MANIFEST = os.path.join(FIX, "manifest.tsv")


def read_rows():
    with open(METAL, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class ScoreFidelityMutations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fields, self.rows = read_rows()

    def tearDown(self):
        self.tmp.cleanup()

    def run_scorer(self, cuda_rows, metal_rows=None):
        cuda = os.path.join(self.tmp.name, "cuda.tsv")
        metal = os.path.join(self.tmp.name, "metal.tsv")
        out = os.path.join(self.tmp.name, "summary.json")
        write_rows(cuda, self.fields, cuda_rows)
        write_rows(metal, self.fields, metal_rows if metal_rows is not None else self.rows)
        proc = subprocess.run([sys.executable, SCORER, cuda, metal, MANIFEST, out],
                              capture_output=True, text=True)
        verdict = None
        if os.path.exists(out):
            with open(out) as f:
                verdict = json.load(f).get("verdict")
        return proc.returncode, verdict

    def mutate(self, case_id, **fields):
        rows = [dict(r) for r in self.rows]
        for r in rows:
            if r["id"] == case_id:
                r.update({k: str(v) for k, v in fields.items()})
        return rows

    def test_self_comparison_passes(self):
        rc, verdict = self.run_scorer(self.rows)
        self.assertEqual((rc, verdict), (0, "PASS"))

    def test_missing_case_rejected(self):
        rc, _ = self.run_scorer(self.rows[:-1])
        self.assertNotEqual(rc, 0)

    def test_duplicate_case_rejected(self):
        rc, _ = self.run_scorer(self.rows + [dict(self.rows[0])])
        self.assertNotEqual(rc, 0)

    def test_nan_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", nll="nan"))
        self.assertNotEqual(rc, 0)

    def test_empty_field_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", api_top1_count=""))
        self.assertNotEqual(rc, 0)

    def test_negative_nll_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", nll=-5.0))
        self.assertNotEqual(rc, 0)

    def test_match_above_count_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", api_top1_match=65, api_top1_count=64))
        self.assertNotEqual(rc, 0)

    def test_non_integer_tokens_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", target_tokens=63.5))
        self.assertNotEqual(rc, 0)

    def test_zero_tokens_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", target_tokens=0))
        self.assertNotEqual(rc, 0)

    def test_target_token_mismatch_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", target_tokens=63))
        self.assertNotEqual(rc, 0)

    def test_api_count_mismatch_rejected(self):
        rc, _ = self.run_scorer(self.mutate("case_005", api_top1_count=60, api_top1_match=50))
        self.assertNotEqual(rc, 0)

    def test_invented_ids_in_both_arms_rejected(self):
        invented = [dict(r, id=f"fake_{i:03d}") for i, r in enumerate(self.rows)]
        rc, verdict = self.run_scorer(invented, invented)
        self.assertNotEqual(rc, 0)
        self.assertEqual(verdict, "FAIL")

    def test_degraded_nll_fails(self):
        worse = [dict(r, nll=str(float(r["nll"]) * 1.2)) for r in self.rows]
        rc, verdict = self.run_scorer(worse)
        self.assertEqual((rc, verdict), (1, "FAIL"))

    def test_degraded_top1_fails(self):
        worse = [dict(r, api_top1_match=str(max(0, int(float(r["api_top1_match"])) - 3)))
                 for r in self.rows]
        rc, verdict = self.run_scorer(worse)
        self.assertEqual((rc, verdict), (1, "FAIL"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
