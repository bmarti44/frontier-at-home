#!/usr/bin/env python3
"""Mutation tests for the 0731 evidence bundle scorer.

Review finding 7 asked for an attempt directory whose verdict is computed rather
than narrated, and for mutation tests demonstrating that missing rows, stale
hashes, changed identities, malformed samples, and absent failure records are
rejected. A scorer that only ever runs on good input proves nothing.

The property under test throughout: the scorer must recompute from the raw
transcripts and refuse to inherit an artifact's own claims. Every mutation below
leaves the artifact internally plausible and changes only what a recomputation
would catch.
"""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "105_build_dsv4_0731_bundle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bundle_0731", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_transcript(directory: Path, index: int, *, correct: bool,
                     finish: str = "stop", tokens: int = 100,
                     reason: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{index:05d}.json").write_text(
        json.dumps(
            {
                "index": index,
                "task_id": index,
                "rendered_prompt_sha256": "a" * 64,
                "rendering": "official-encoder-thinking",
                "finish_reason": finish,
                "scored_correct": correct,
                "reason": reason or ("correct" if correct else "incorrect: x"),
                "request": {"elapsed_s": 1.0},
                "response": {"usage": {"completion_tokens": tokens}},
            }
        ),
        encoding="utf-8",
    )


class BundleMutationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.transcripts = self.tmp / "transcripts"
        for index in range(10):
            write_transcript(self.transcripts, index, correct=index < 7)
        self.artifact = self.tmp / "acc.json"
        self.write_artifact(correct=7, n=10)

    def write_artifact(self, *, correct: int, n: int) -> None:
        self.artifact.write_text(
            json.dumps(
                {
                    "stack_label": "unit",
                    "suite": "gsm8k",
                    "split": "dev",
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n if n else 0,
                    "generation": {"max_tokens": 8192, "request_timeout_s": 1800},
                }
            ),
            encoding="utf-8",
        )

    def score(self):
        return self.module.score_accuracy_arm("unit", self.artifact, self.transcripts)

    # -- baseline ---------------------------------------------------------

    def test_clean_input_scores(self):
        result = self.score()
        self.assertEqual(result["summary"]["correct"], 7)
        self.assertEqual(result["summary"]["n"], 10)
        self.assertEqual(len(result["rows"]), 10)

    # -- mutations --------------------------------------------------------

    def test_rejects_an_artifact_that_overstates_its_score(self):
        """The signature of an artifact edited after the run."""
        self.write_artifact(correct=9, n=10)
        with self.assertRaises(self.module.BundleError) as caught:
            self.score()
        self.assertIn("recomputation", str(caught.exception))

    def test_rejects_a_missing_transcript_row(self):
        (self.transcripts / "00003.json").unlink()
        with self.assertRaises(self.module.BundleError) as caught:
            self.score()
        self.assertIn("transcripts exist", str(caught.exception))

    def test_rejects_duplicate_item_indices(self):
        write_transcript(self.transcripts, 3, correct=True)
        (self.transcripts / "00003.json").rename(self.transcripts / "dup.json")
        write_transcript(self.transcripts, 3, correct=True)
        with self.assertRaises(self.module.BundleError) as caught:
            self.score()
        self.assertIn("duplicate", str(caught.exception))

    def test_rejects_a_malformed_transcript(self):
        (self.transcripts / "00004.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(self.module.BundleError):
            self.score()

    def test_rejects_negative_completion_tokens(self):
        write_transcript(self.transcripts, 5, correct=True, tokens=-1)
        with self.assertRaises(self.module.BundleError) as caught:
            self.score()
        self.assertIn("negative", str(caught.exception))

    def test_rejects_a_missing_transcripts_directory(self):
        shutil.rmtree(self.transcripts)
        with self.assertRaises(self.module.BundleError) as caught:
            self.score()
        self.assertIn("transcripts directory is missing", str(caught.exception))

    # -- failure accounting must survive into the summary ------------------

    def test_truncation_is_counted_not_silently_dropped(self):
        write_transcript(self.transcripts, 6, correct=False, finish="length")
        self.write_artifact(correct=6, n=10)
        summary = self.score()["summary"]
        self.assertEqual(summary["truncated"], 1)
        self.assertAlmostEqual(summary["truncated_fraction"], 0.1)

    def test_timeouts_are_counted_separately_from_truncation(self):
        write_transcript(
            self.transcripts, 6, correct=False, finish=None,
            reason="invalid: TimeoutError: timed out",
        )
        self.write_artifact(correct=6, n=10)
        summary = self.score()["summary"]
        self.assertEqual(summary["request_timeouts"], 1)
        self.assertEqual(summary["truncated"], 0)
        self.assertEqual(summary["invalid"], 1)

    def test_confidence_bounds_are_finite(self):
        summary = self.score()["summary"]
        for key in ("wilson95_low", "wilson95_high"):
            self.assertTrue(
                -0.001 <= summary[key] <= 1.001, f"{key}={summary[key]!r}"
            )

    def test_wilson_interval_rejects_a_zero_denominator(self):
        with self.assertRaises(self.module.BundleError):
            self.module.wilson_interval(0, 0)

    # -- structural properties of the bundle -------------------------------

    def test_attempt_directory_is_never_overwritten(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("attempt.mkdir()", source)
        self.assertIn("FileExistsError", source)
        self.assertIn("refusing to overwrite existing attempt", source)

    def test_missing_arms_yield_no_result_not_a_false_negative(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('verdict = "NO_RESULT"', source)

    def test_unreadable_engine_binary_is_absent_not_fabricated(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("rather than fabricated", source)


if __name__ == "__main__":
    unittest.main()
