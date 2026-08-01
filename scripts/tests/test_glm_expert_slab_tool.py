#!/usr/bin/env python3
"""Functional tests for the GLM expert-slab sidecar builder."""

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "results/glm52-gates/harness/glm_expert_slab.py"
SPEC = importlib.util.spec_from_file_location("glm_expert_slab", TOOL_PATH)
assert SPEC and SPEC.loader
slab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slab)


def encoded_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make_gguf(path: Path) -> dict[str, bytes]:
    tensors = [
        ("blk.0.ffn_gate_exps.weight", [2, 3, 2], b"G" * 48),
        ("blk.0.ffn_up_exps.weight", [2, 3, 2], b"U" * 48),
        ("blk.0.ffn_down_exps.weight", [3, 4, 2], b"D" * 96),
    ]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    header.extend(encoded_string("general.alignment"))
    header.extend(struct.pack("<II", 4, 32))
    offset = 0
    payload = bytearray()
    expected = {}
    for name, dims, data in tensors:
        offset = slab.align_up(offset, 32)
        if len(payload) < offset:
            payload.extend(b"\0" * (offset - len(payload)))
        header.extend(encoded_string(name))
        header.extend(struct.pack("<I", len(dims)))
        header.extend(struct.pack("<" + "Q" * len(dims), *dims))
        header.extend(struct.pack("<IQ", 0, offset))
        payload.extend(data)
        expected[name] = data
        offset += len(data)
    header.extend(b"\0" * (slab.align_up(len(header), 32) - len(header)))
    path.write_bytes(header + payload)
    return expected


class ExpertSlabToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "model.gguf"
        self.output = self.root / "experts.slab"
        self.tensors = make_gguf(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_is_byte_identical_and_verifies(self) -> None:
        built = slab.build(self.source, self.output)
        self.assertEqual(built["records"], 2)
        self.assertTrue(slab.verify(self.source, self.output)["verified"])
        _, records = slab.load_index(self.output)
        first = records[0]
        with self.output.open("rb") as stream:
            stream.seek(first[2])
            payload = stream.read(first[3])
        self.assertEqual(payload, b"G" * 24 + b"U" * 24 + b"D" * 48)

    def test_corrupted_record_fails_closed(self) -> None:
        slab.build(self.source, self.output)
        _, records = slab.load_index(self.output)
        with self.output.open("r+b") as stream:
            stream.seek(records[0][2] + 3)
            original = stream.read(1)
            stream.seek(records[0][2] + 3)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "record checksum mismatch"):
            slab.verify(self.source, self.output)

    def test_changed_model_identity_fails_closed(self) -> None:
        slab.build(self.source, self.output)
        with self.source.open("r+b") as stream:
            stream.seek(-1, 2)
            original = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "model identity mismatch"):
            slab.verify(self.source, self.output)

    def test_never_overwrites_existing_artifact(self) -> None:
        slab.build(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            slab.build(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
