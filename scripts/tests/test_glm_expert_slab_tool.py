#!/usr/bin/env python3
"""Functional tests for the GLM expert-slab sidecar builder."""

import importlib.util
import hashlib
import os
from pathlib import Path
import stat
import struct
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "results/glm52-gates/harness/glm_expert_slab.py"
SPEC = importlib.util.spec_from_file_location("glm_expert_slab", TOOL_PATH)
assert SPEC and SPEC.loader
slab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slab)


def encoded_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make_gguf(path: Path, data_type: int = 8) -> dict[str, bytes]:
    block_bytes = 34 if data_type == 8 else 40
    tensors = [
        ("blk.0.ffn_gate_exps.weight", [32, 1, 2], data_type, b"G" * (2 * block_bytes)),
        ("blk.0.ffn_up_exps.weight", [32, 1, 2], data_type, b"U" * (2 * block_bytes)),
        ("blk.0.ffn_down_exps.weight", [32, 2, 2], data_type, b"D" * (4 * block_bytes)),
    ]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    header.extend(encoded_string("general.alignment"))
    header.extend(struct.pack("<II", 4, 32))
    offset = 0
    payload = bytearray()
    expected = {}
    for name, dims, data_type, data in tensors:
        offset = slab.align_up(offset, 32)
        if len(payload) < offset:
            payload.extend(b"\0" * (offset - len(payload)))
        header.extend(encoded_string(name))
        header.extend(struct.pack("<I", len(dims)))
        header.extend(struct.pack("<" + "Q" * len(dims), *dims))
        header.extend(struct.pack("<IQ", data_type, offset))
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
        self.assertEqual(payload, b"G" * 34 + b"U" * 34 + b"D" * 68)

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

    def test_payload_and_embedded_checksum_mutation_is_rejected(self) -> None:
        slab.build(self.source, self.output)
        header, records = slab.load_index(self.output)
        first = list(records[0])
        with self.output.open("r+b") as stream:
            stream.seek(first[2])
            payload = bytearray(stream.read(first[3]))
            payload[5] ^= 1
            stream.seek(first[2])
            stream.write(payload)
            first[6] = hashlib.sha256(payload).digest()
            stream.seek(slab.HEADER.size)
            stream.write(slab.RECORD.pack(*first))
        with self.assertRaisesRegex(ValueError, "payload differs from GGUF"):
            slab.verify(self.source, self.output)

    def test_swapped_record_keys_are_rejected(self) -> None:
        slab.build(self.source, self.output)
        _, records = slab.load_index(self.output)
        first, second = list(records[0]), list(records[1])
        first[:2], second[:2] = second[:2], first[:2]
        with self.output.open("r+b") as stream:
            stream.seek(slab.HEADER.size)
            stream.write(slab.RECORD.pack(*first))
            stream.write(slab.RECORD.pack(*second))
        with self.assertRaisesRegex(ValueError, "index record mismatch"):
            slab.verify(self.source, self.output)

    def test_symlink_model_is_rejected(self) -> None:
        link = self.root / "model-link.gguf"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            slab.plan_records(link)

    def test_large_build_requires_explicit_owner_approval(self) -> None:
        planned = slab.plan_records(self.source)
        oversized = (planned[0], planned[1], planned[2], slab.LARGE_BUILD_BYTES + 1)
        with mock.patch.object(slab, "plan_records_stream", return_value=oversized), \
             mock.patch.object(slab, "file_sha256_stream",
                               side_effect=AssertionError("bulk hash must not run")):
            with self.assertRaisesRegex(ValueError, "owner-approved-large-build"):
                slab.build(self.source, self.output)

    def test_concurrent_builders_publish_once_without_overwrite(self) -> None:
        outcomes = []
        barrier = threading.Barrier(2)

        def run() -> None:
            barrier.wait()
            try:
                slab.build(self.source, self.output)
                outcomes.append("PASS")
            except ValueError as error:
                outcomes.append(str(error))

        with mock.patch.object(
                slab, "file_sha256_stream", wraps=slab.file_sha256_stream
        ) as full_hash:
            workers = [threading.Thread(target=run) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            # One model digest and one retained-descriptor slab digest. The
            # losing builder must not enter either bulk scan.
            self.assertEqual(full_hash.call_count, 2)
        self.assertEqual(outcomes.count("PASS"), 1)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(slab.verify(self.source, self.output)["verified"])

    def test_non_engine_routed_type_is_rejected(self) -> None:
        unsupported = self.root / "q8_1.gguf"
        make_gguf(unsupported, data_type=9)
        with self.assertRaisesRegex(ValueError, "unsupported routed tensor"):
            slab.plan_records(unsupported)

    def test_implausible_dimension_count_is_rejected_before_reads(self) -> None:
        malformed = self.root / "dimension-bomb.gguf"
        malformed.write_bytes(
            b"GGUF" + struct.pack("<IQQ", 3, 1, 0) +
            encoded_string("tensor") + struct.pack("<I", 0xffffffff)
        )
        with self.assertRaisesRegex(ValueError, "tensor dimension count"):
            slab.plan_records(malformed)

    def test_build_remains_bound_to_open_source_descriptor(self) -> None:
        original_name = self.root / "original.gguf"
        replacement = self.root / "replacement.gguf"
        make_gguf(replacement)
        with replacement.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with slab.open_regular(self.source) as source:
            self.source.rename(original_name)
            replacement.rename(self.source)
            slab.build_stream(source, self.output)
        self.assertTrue(slab.verify(original_name, self.output)["verified"])
        with self.assertRaisesRegex(ValueError, "model identity mismatch"):
            slab.verify(self.source, self.output)

    def test_directory_fsync_failure_leaves_no_published_artifact(self) -> None:
        real_fsync = os.fsync

        def fail_directory(descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("injected directory fsync failure")
            real_fsync(descriptor)

        with mock.patch.object(slab.os, "fsync", side_effect=fail_directory):
            with self.assertRaisesRegex(OSError, "directory fsync failure"):
                slab.build(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_replacement_during_publication_cannot_return_pass(self) -> None:
        real_fsync = os.fsync
        replaced = False

        def replace_at_first_directory_sync(descriptor: int) -> None:
            nonlocal replaced
            if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not replaced:
                replaced = True
                self.output.unlink()
                self.output.write_bytes(b"hostile replacement")
            real_fsync(descriptor)

        with mock.patch.object(
                slab.os, "fsync", side_effect=replace_at_first_directory_sync
        ):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                slab.build(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"hostile replacement")

    def test_same_inode_mutation_during_sync_cannot_return_pass(self) -> None:
        real_fsync = os.fsync
        mutated = False

        def mutate_at_first_directory_sync(descriptor: int) -> None:
            nonlocal mutated
            if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not mutated:
                mutated = True
                before = self.output.stat()
                with self.output.open("r+b") as stream:
                    stream.seek(slab.ALIGNMENT + 3)
                    value = stream.read(1)
                    stream.seek(slab.ALIGNMENT + 3)
                    stream.write(bytes([value[0] ^ 1]))
                    stream.flush()
                    real_fsync(stream.fileno())
                os.utime(
                    self.output,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
            real_fsync(descriptor)

        with mock.patch.object(
                slab.os, "fsync", side_effect=mutate_at_first_directory_sync
        ):
            with self.assertRaisesRegex(
                    ValueError, "checksum mismatch|changed during final"
            ):
                slab.build(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
