import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "w3_score", ROOT / "scripts/83_score_w3_slot_microgate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def confirmation(run):
    return {
        "kind": "confirmation", "run": run, "exit_code": 0,
        "compact_samples_ms": [164.0] * 6,
        "direct_samples_ms": [138.0] * 6,
        "output_mismatches": 0,
        "finite_nonzero_reference_values": 36864,
        "expert_evaluations_per_arm": 600,
        "samples": 6, "cuda_event_synchronized": True,
    }


class ScoreTest(unittest.TestCase):
    def write(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return Path(handle.name)

    def valid_rows(self):
        rows = [confirmation(run) for run in (1, 2, 3)]
        rows.extend({"kind": "mutation", "name": name,
                     "exit_code": 1, "output_mismatches": 36864}
                    for name in sorted(MODULE.MUTATIONS))
        return rows

    def test_accepts_complete_raw_samples(self):
        self.assertEqual(MODULE.score(self.write(self.valid_rows()))["status"], "PASS")

    def test_rejects_candidate_owned_aggregate_without_samples(self):
        rows = self.valid_rows()
        del rows[0]["compact_samples_ms"]
        with self.assertRaisesRegex(ValueError, "raw timing arrays"):
            MODULE.score(self.write(rows))

    def test_rejects_missing_mutation(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            MODULE.score(self.write(self.valid_rows()[:-1]))

    def test_rejects_false_success_mutation(self):
        rows = self.valid_rows()
        rows[-1]["exit_code"] = 0
        with self.assertRaisesRegex(ValueError, "did not fail closed"):
            MODULE.score(self.write(rows))


if __name__ == "__main__":
    unittest.main()
