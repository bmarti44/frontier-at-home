#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/97_verify_w7_cache_freeze.py"
SPEC = importlib.util.spec_from_file_location("w7_cache_freeze", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7CacheFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "ds4-server"
        self.binary.write_bytes(b"candidate-binary")
        self.binary.chmod(0o755)
        self.freeze = self.root / "freeze.json"
        self.record = {
            "binary": {
                "path": str(self.binary),
                "bytes": self.binary.stat().st_size,
                "mode": 0o755,
                "sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            }
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def score(self, record: dict[str, object] | None = None) -> str:
        self.freeze.write_text(json.dumps(record or self.record), encoding="utf-8")
        return MODULE.verify(self.freeze)["verdict"]

    def test_accepts_exact_binary(self) -> None:
        self.assertEqual(self.score(), "PASS")

    def test_rejects_size_digest_and_mode_mutations(self) -> None:
        for field, value in (
            ("bytes", self.binary.stat().st_size + 1),
            ("sha256", "0" * 64),
            ("mode", 0o775),
        ):
            with self.subTest(field=field):
                record = json.loads(json.dumps(self.record))
                record["binary"][field] = value
                self.assertEqual(self.score(record), "FAIL")

    def test_rejects_symlink(self) -> None:
        link = self.root / "link"
        link.symlink_to(self.binary)
        record = json.loads(json.dumps(self.record))
        record["binary"]["path"] = str(link)
        self.assertEqual(self.score(record), "FAIL")


if __name__ == "__main__":
    unittest.main()
