#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


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

    def test_rejects_metadata_digest_from_different_path_instances(self) -> None:
        replacement = self.root / "replacement"
        replacement.write_bytes(b"replacement-data")
        replacement.chmod(0o755)
        # Keep the metadata fields compatible while binding the record digest to
        # the replacement.  The old verifier lstat()s the first inode, then
        # reopens the path for hashing and can therefore accept this mixture.
        replacement.write_bytes(b"different-binary")
        self.assertEqual(replacement.stat().st_size, self.binary.stat().st_size)
        record = json.loads(json.dumps(self.record))
        record["binary"]["sha256"] = hashlib.sha256(
            replacement.read_bytes()
        ).hexdigest()
        original_open = Path.open

        def replace_before_artifact_open(path: Path, *args, **kwargs):
            if path == self.binary:
                replacement.replace(path)
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=replace_before_artifact_open):
            self.assertEqual(self.score(record), "FAIL")

    def test_open_descriptor_remains_authoritative_after_path_replacement(self) -> None:
        replacement = self.root / "replacement"
        replacement.write_bytes(b"different-binary")
        replacement.chmod(0o755)
        original_open = MODULE.os.open
        hook_calls = 0

        def open_then_replace(path, flags):
            nonlocal hook_calls
            descriptor = original_open(path, flags)
            if Path(path) == self.binary:
                hook_calls += 1
                replacement.replace(path)
            return descriptor

        with mock.patch.object(MODULE.os, "open", side_effect=open_then_replace):
            self.assertEqual(self.score(), "PASS")
        self.assertEqual(hook_calls, 1)

    def test_rejects_endless_nonregular_source_without_reading_it(self) -> None:
        record = json.loads(json.dumps(self.record))
        record["binary"].update({"path": "/dev/zero", "bytes": 1})
        self.freeze.write_text(json.dumps(record), encoding="utf-8")
        completed = subprocess.run(
            ["python3", str(MODULE_PATH), str(self.freeze)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1,
            check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)

    def test_rejects_fifo_without_waiting_for_a_writer(self) -> None:
        fifo = self.root / "artifact.fifo"
        fifo.touch()
        fifo.unlink()
        import os

        os.mkfifo(fifo, 0o600)
        record = json.loads(json.dumps(self.record))
        record["binary"].update({"path": str(fifo), "bytes": 1})
        self.freeze.write_text(json.dumps(record), encoding="utf-8")
        completed = subprocess.run(
            ["python3", str(MODULE_PATH), str(self.freeze)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1,
            check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
