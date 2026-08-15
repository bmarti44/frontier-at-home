import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "w6_scorer", ROOT / "scripts/106_score_w6_indexer_tile_reuse.py")
SCORER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCORER)


def valid_raw(schedule):
    counters = {1: 0, 2: 0, 4: 0}
    rows = []
    values = {1: [1.00, 1.01, 0.99, 1.00, 1.00],
              2: [0.90, 0.91, 0.89, 0.90, 0.90],
              4: [0.80, 0.81, 0.79, 0.80, 0.80]}
    for sequence, width in enumerate(schedule):
        index = counters[width]
        counters[width] += 1
        rows.append({
            "kind": "timing", "sequence": sequence, "width": width,
            "elapsed_ms": values[width][index],
            "logical_k_bytes": SCORER.EXPECTED_LOGICAL_BYTES[width],
            "complete_write": True, "exact_scores": True,
            "exact_ids": True, "canaries_intact": True,
        })
    rows.append({
        "kind": "result", "verdict": "PASS", "correctness_cases": 12,
        "causal_cases": 5, "ragged_row_cases": 1,
        "invalid_values_rejected": 6, "quality_rejected": True,
    })
    return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode()


class W6ScorerTests(unittest.TestCase):
    def setUp(self):
        self.schedule = SCORER.schedule_from_randomness("42" * 32)
        self.raw = valid_raw(self.schedule)

    def test_valid_rows_pass(self):
        self.assertEqual(SCORER.validate_and_score_rows(self.schedule, self.raw)["verdict"], "PASS")

    def test_wrong_order_rejected(self):
        rows = self.raw.decode().splitlines()
        first = json.loads(rows[0]); first["width"] = 4 if first["width"] != 4 else 2
        rows[0] = json.dumps(first)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, ("\n".join(rows) + "\n").encode())

    def test_missing_row_rejected(self):
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, b"\n".join(self.raw.splitlines()[:-1]) + b"\n")

    def test_nonfinite_rejected(self):
        mutated = self.raw.replace(b'"elapsed_ms": 1.0', b'"elapsed_ms": NaN', 1)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, mutated)

    def test_incomplete_write_rejected(self):
        mutated = self.raw.replace(b'"complete_write": true', b'"complete_write": false', 1)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, mutated)

    def test_cuda_or_correctness_failure_rejected(self):
        mutated = self.raw.replace(b'"exact_scores": true', b'"exact_scores": false', 1)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, mutated)

    def test_logical_bytes_are_not_self_reported_freely(self):
        mutated = self.raw.replace(b'"logical_k_bytes": 2147483648', b'"logical_k_bytes": 1', 1)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, mutated)

    def test_duplicate_key_rejected(self):
        mutated = self.raw.replace(b'{', b'{"kind":"timing",', 1)
        with self.assertRaises(SCORER.ScoreError):
            SCORER.validate_and_score_rows(self.schedule, mutated)

    def test_expected_stderr_is_schedule_bound(self):
        transcript = SCORER.expected_stderr(self.schedule)
        self.assertIn(b"CUDA backend initialized", transcript)
        self.assertEqual(transcript.count(b"must be 1, 2, or 4"), 6)
        changed = list(self.schedule)
        changed[0], changed[1] = changed[1], changed[0]
        if changed != self.schedule:
            self.assertNotEqual(transcript, SCORER.expected_stderr(changed))


if __name__ == "__main__":
    unittest.main()
