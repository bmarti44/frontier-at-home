#!/usr/bin/env python3
"""Tests for byte-level GLM GGUF expert pruning."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "58_prune_glm_experts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prune_glm_experts", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_string(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)


def metadata_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<II", 4, value)


def metadata_u32_array(key: str, values: tuple[int, ...]) -> bytes:
    return b"".join(
        (
            gguf_string(key), struct.pack("<IIQ", 9, 4, len(values)),
            struct.pack(f"<{len(values)}I", *values),
        )
    )


def tensor_info(
    name: str, dims: tuple[int, ...], ggml_type: int, offset: int
) -> bytes:
    return b"".join(
        (
            gguf_string(name), struct.pack("<I", len(dims)),
            b"".join(struct.pack("<Q", dim) for dim in dims),
            struct.pack("<IQ", ggml_type, offset),
        )
    )


def tagged_experts(layer: int, kind: int, experts: int, floats_each: int) -> bytes:
    values = []
    for expert in range(experts):
        for item in range(floats_each):
            values.append(float(layer * 1000 + kind * 100 + expert * 10 + item))
    return struct.pack(f"<{len(values)}f", *values)


def iq2_experts(layer: int, kind: int, experts: int) -> bytes:
    return b"".join(
        bytes((layer * 17 + kind * 29 + expert * 43 + item) % 256 for item in range(66))
        for expert in range(experts)
    )


def build_synthetic_gguf(
    path: Path,
    layers: tuple[int, ...] = (3, 4),
    extra_metadata: tuple[bytes, ...] = (),
    iq2_routed: bool = False,
) -> dict[str, bytes]:
    experts = 4
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    for layer in layers:
        # The final GGUF dimension is the outermost/contiguous expert axis.
        routed_dims = (256, 1, experts) if iq2_routed else (2, 3, experts)
        routed_type = 16 if iq2_routed else 0
        tensors.extend(
            (
                (f"blk.{layer}.ffn_gate_exps.weight", routed_dims, routed_type,
                 iq2_experts(layer, 1, experts) if iq2_routed else
                 tagged_experts(layer, 1, experts, 6)),
                (f"blk.{layer}.ffn_up_exps.weight", routed_dims, routed_type,
                 iq2_experts(layer, 2, experts) if iq2_routed else
                 tagged_experts(layer, 2, experts, 6)),
                (f"blk.{layer}.ffn_down_exps.weight", routed_dims, routed_type,
                 iq2_experts(layer, 3, experts) if iq2_routed else
                 tagged_experts(layer, 3, experts, 6)),
                (f"blk.{layer}.ffn_gate_inp.weight", (3, experts), 0,
                 tagged_experts(layer, 4, experts, 3)),
                (f"blk.{layer}.exp_probs_b.bias", (experts,), 0,
                 tagged_experts(layer, 5, experts, 1)),
            )
        )
    unchanged = struct.pack("<4f", 91.0, 92.0, 93.0, 94.0)
    tensors.insert(3, ("token_embd.weight", (2, 2), 0, unchanged))

    metadata_items = (
            metadata_string("general.architecture", "glm-dsa"),
            metadata_u32("general.alignment", 32),
            metadata_u32("glm-dsa.expert_count", experts),
            metadata_string("test.unchanged", "byte-identical metadata"),
        ) + extra_metadata
    metadata = b"".join(metadata_items)
    offsets: list[int] = []
    cursor = 0
    for _, _, _, payload in tensors:
        cursor = (cursor + 31) // 32 * 32
        offsets.append(cursor)
        cursor += len(payload)
    infos = b"".join(
        tensor_info(name, dims, ggml_type, offset)
        for (name, dims, ggml_type, _), offset in zip(tensors, offsets, strict=True)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata_items))
    data_start = (len(header) + len(metadata) + len(infos) + 31) // 32 * 32
    output = bytearray(header + metadata + infos)
    output.extend(b"\0" * (data_start - len(output)))
    expected: dict[str, bytes] = {}
    for (name, _, _, payload), offset in zip(tensors, offsets, strict=True):
        wanted = data_start + offset
        output.extend(b"\0" * (wanted - len(output)))
        output.extend(payload)
        expected[name] = payload
    path.write_bytes(output)
    return expected


class PruneGLMExpertsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pruner = load_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "tiny-source.gguf"
        self.destination = self.root / "tiny-pruned.gguf"
        self.original = build_synthetic_gguf(self.source)
        self.keep = self.root / "keep.json"
        self.write_keep({"layers": {"3": [0, 2], "4": [1, 3]}})

    def write_keep(self, value) -> None:
        self.keep.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, os.fspath(SCRIPT), *extra],
            text=True, capture_output=True, check=False,
        )

    def test_cli_prunes_and_verifies_synthetic_gguf(self):
        built = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        self.assertEqual(json.loads(built.stdout)["verdict"], "PASS")
        self.assertTrue(self.destination.is_file())

        verified = self.run_cli(
            "--verify", os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        result = json.loads(verified.stdout)
        self.assertEqual(result["verdict"], "PASS")
        # Five sliced tensors x two experts x two layers.
        self.assertEqual(result["expert_sha256_checks"], 20)
        self.assertEqual(result["nonexpert_tensors"], 1)

        parsed = self.pruner.parse_gguf(self.destination)
        metadata = {item.key: item.value for item in parsed.metadata}
        self.assertEqual(metadata["glm-dsa.expert_count"], 2)
        self.assertEqual(metadata["test.unchanged"], "byte-identical metadata")
        by_name = {tensor.name: tensor for tensor in parsed.tensors}
        self.assertEqual(by_name["blk.3.ffn_gate_exps.weight"].dims, (2, 3, 2))
        self.assertEqual(by_name["blk.3.ffn_gate_inp.weight"].dims, (3, 2))
        self.assertEqual(by_name["blk.3.exp_probs_b.bias"].dims, (2,))

    def test_dry_run_prints_plan_without_writing(self):
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep), "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["verdict"], "PASS")
        self.assertEqual(plan["source_experts"], 4)
        self.assertEqual(plan["kept_experts"], 2)
        self.assertGreater(plan["layers"]["3"]["bytes_dropped"], 0)
        self.assertFalse(self.destination.exists())

    def test_global_keep_list_applies_to_all_layers(self):
        self.write_keep([0, 3])
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with self.pruner.open_regular(self.source) as source:
            plan = self.pruner.make_plan(source, self.source, self.keep)
            self.assertEqual(plan.keep, {3: (0, 3), 4: (0, 3)})

    def test_refuses_unsorted_ids(self):
        self.write_keep({"layers": {"3": [2, 0], "4": [1, 3]}})
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be sorted", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_refuses_missing_layer(self):
        self.write_keep({"layers": {"3": [0, 2]}})
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("layer mismatch", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_refuses_nonuniform_counts(self):
        self.write_keep({"layers": {"3": [0, 2], "4": [1]}})
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identical", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_refuses_existing_destination(self):
        self.destination.write_bytes(b"do not overwrite")
        before = self.destination.read_bytes()
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stdout)
        self.assertEqual(self.destination.read_bytes(), before)

    def test_verifier_detects_corrupted_surviving_expert(self):
        with self.pruner.open_regular(self.source) as source:
            plan = self.pruner.make_plan(source, self.source, self.keep)
            self.pruner.write_pruned(plan, source, self.destination)
        parsed = self.pruner.parse_gguf(self.destination)
        tensor = next(
            item for item in parsed.tensors
            if item.name == "blk.3.ffn_gate_exps.weight"
        )
        with self.destination.open("r+b") as stream:
            stream.seek(tensor.offset)
            byte = stream.read(1)
            stream.seek(tensor.offset)
            stream.write(bytes([byte[0] ^ 0xFF]))
        result = self.run_cli(
            "--verify", os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["verdict"], "FAIL")
        self.assertIn("expert bytes changed", result.stdout)

    def test_iq2_xxs_expert_slices_have_correct_sha256(self):
        self.original = build_synthetic_gguf(self.source, iq2_routed=True)
        built = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        verified = self.run_cli(
            "--verify", os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        target = self.pruner.parse_gguf(self.destination)
        by_name = {tensor.name: tensor for tensor in target.tensors}
        with self.destination.open("rb") as output:
            for layer, kept in ((3, (0, 2)), (4, (1, 3))):
                for kind in ("gate", "up", "down"):
                    name = f"blk.{layer}.ffn_{kind}_exps.weight"
                    tensor = by_name[name]
                    self.assertEqual(tensor.ggml_type, 16)
                    self.assertEqual(tensor.nbytes, 2 * 66)
                    output.seek(tensor.offset)
                    for output_expert, source_expert in enumerate(kept):
                        actual = output.read(66)
                        expected = self.original[name][source_expert * 66:(source_expert + 1) * 66]
                        self.assertEqual(
                            hashlib.sha256(actual).digest(),
                            hashlib.sha256(expected).digest(),
                            f"{name} output expert {output_expert}",
                        )

    def test_verify_rejects_appended_trailing_bytes(self):
        built = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        with self.destination.open("ab") as output:
            output.write(b"trailing bytes")
        verified = self.run_cli(
            "--verify", os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(verified.returncode, 0)
        self.assertIn("file size differs from plan", verified.stdout)

    def test_refuses_ambiguous_n_experts_metadata(self):
        build_synthetic_gguf(
            self.source,
            extra_metadata=(metadata_u32("provenance.n_experts.limit", 256),),
        )
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ambiguous expert-count metadata key", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_metadata_array_is_copied_raw_byte_identical(self):
        build_synthetic_gguf(
            self.source,
            extra_metadata=(metadata_u32_array("test.array", (0, 1, 17, 0xFFFFFFFF)),),
        )
        source_model = self.pruner.parse_gguf(self.source)
        source_raw = next(
            item.raw for item in source_model.metadata if item.key == "test.array"
        )
        built = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep),
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        target_model = self.pruner.parse_gguf(self.destination)
        target_raw = next(
            item.raw for item in target_model.metadata if item.key == "test.array"
        )
        self.assertEqual(target_raw, source_raw)

    def test_expect_layers_detects_missing_layer(self):
        build_synthetic_gguf(self.source, layers=(3, 5))
        self.write_keep({"layers": {"3": [0, 2], "5": [1, 3]}})
        result = self.run_cli(
            os.fspath(self.source), os.fspath(self.destination),
            "--keep", os.fspath(self.keep), "--expect-layers", "2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing=[4]", result.stdout)
        self.assertIn("extra=[5]", result.stdout)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
