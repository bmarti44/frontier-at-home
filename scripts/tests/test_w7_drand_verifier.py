#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
FIXTURE = ROOT / "results/glm52-gates/W7-resume-production-drand-verifier-fixture.json"


class W7DrandVerifierTest(unittest.TestCase):
    def _run(self, record: dict) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(NODE), str(VERIFIER), str(record["round"]),
                record["randomness"], record["signature"],
                record["previous_signature"],
            ],
            cwd=ROOT, text=True, capture_output=True, timeout=30,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
        )

    def test_pinned_default_chain_beacon_verifies(self) -> None:
        record = json.loads(FIXTURE.read_text())
        result = self._run(record)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "DRAND_BLS_RECEIPT_OK\n")

    def test_signature_round_previous_and_randomness_mutations_fail(self) -> None:
        original = json.loads(FIXTURE.read_text())
        mutations = (
            {**original, "round": original["round"] + 1},
            {**original, "randomness": "0" * 64},
            {**original, "signature": "0" * 192},
            {**original, "previous_signature": "0" * 192},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertNotEqual(self._run(mutation).returncode, 0)


if __name__ == "__main__":
    unittest.main()
