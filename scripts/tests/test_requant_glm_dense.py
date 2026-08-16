#!/usr/bin/env python3
"""Synthetic tests for the offline dense GGUF requantizer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "59_requant_glm_dense.py"
GGUF_PY = ROOT / "results" / "glm52-gates" / "harness" / "gguf-py"
sys.path.insert(0, os.fspath(GGUF_PY))
from gguf import quants
from gguf.constants import GGMLQuantizationType


def load_module():
    spec = importlib.util.spec_from_file_location("requant_glm_dense", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def tensor_info(name: str, dims: tuple[int, ...], qtype: int, offset: int) -> bytes:
    return b"".join((string(name), struct.pack("<I", len(dims)),
                     b"".join(struct.pack("<Q", dim) for dim in dims),
                     struct.pack("<IQ", qtype, offset)))


def build_tiny(path: Path) -> dict[str, bytes]:
    rng = np.random.default_rng(59)
    names = (
        "blk.0.attn_q.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.bias",
        "output.weight",
        "token_embd.weight",
    )
    tensors = []
    payloads: dict[str, bytes] = {}
    for index, name in enumerate(names):
        values = rng.normal(index, 0.5, size=(2, 256)).astype(np.float32)
        if name.endswith(".bias"):
            packed = values.ravel().tobytes()
            qtype = GGMLQuantizationType.F32
        else:
            packed = quants.quantize(values, GGMLQuantizationType.Q8_0).tobytes()
            qtype = GGMLQuantizationType.Q8_0
        tensors.append((name, (256, 2), int(qtype), packed))
        payloads[name] = packed
    metadata = b"".join((
        string("general.architecture"), struct.pack("<I", 8), string("glm-dsa"),
        string("general.alignment"), struct.pack("<II", 4, 32),
    ))
    offsets, cursor = [], 0
    for _, _, _, payload in tensors:
        cursor = (cursor + 31) // 32 * 32
        offsets.append(cursor)
        cursor += len(payload)
    infos = b"".join(tensor_info(name, dims, qtype, offset)
                     for (name, dims, qtype, _), offset in zip(tensors, offsets, strict=True))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    data_start = (len(header) + len(metadata) + len(infos) + 31) // 32 * 32
    result = bytearray(header + metadata + infos)
    result.extend(b"\0" * (data_start - len(result)))
    for (_, _, _, payload), offset in zip(tensors, offsets, strict=True):
        result.extend(b"\0" * (data_start + offset - len(result)))
        result.extend(payload)
    path.write_bytes(result)
    return payloads


class RequantGLMDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "tiny.gguf"
        self.destination = self.root / "tiny-requant.gguf"
        self.payloads = build_tiny(self.source)

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, os.fspath(SCRIPT), *extra],
                              text=True, capture_output=True, check=False)

    def test_default_dry_run_selects_dense_and_never_experts_router_or_norm(self):
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        actions = {item["name"]: item["action"] for item in plan["tensors"]}
        self.assertEqual(actions["blk.0.attn_q.weight"], "convert")
        self.assertEqual(actions["blk.0.ffn_up.weight"], "convert")
        self.assertEqual(actions["output.weight"], "convert")
        self.assertEqual(actions["token_embd.weight"], "convert")
        self.assertEqual(actions["blk.0.ffn_up_exps.weight"], "copy")
        self.assertEqual(actions["blk.0.ffn_gate_inp.weight"], "copy")
        self.assertEqual(actions["blk.0.attn_norm.weight"], "copy")
        self.assertEqual(actions["blk.0.attn_q.bias"], "copy")
        self.assertLess(plan["projected_file_size"], plan["source_bytes"])
        self.assertFalse(self.destination.exists())

    def test_default_q4_k_refuses_and_reports_actual_quantizers(self):
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination))
        self.assertNotEqual(result.returncode, 0)
        failure = json.loads(result.stdout)
        self.assertIn("no Q4_K quantizer", failure["error"])
        self.assertIn("Q4_0", failure["error"])
        self.assertIn("Q5_0", failure["error"])
        self.assertFalse(self.destination.exists())

    def test_reference_q4_0_path_rewrites_and_verifies(self):
        built = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                             "--to", "Q4_0")
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        verified = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                                "--to", "Q4_0", "--verify")
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        report = json.loads(verified.stdout)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(report["converted_mse"]), 4)
        self.assertTrue(all(value >= 0 for value in report["converted_mse"].values()))
        self.assertEqual(report["non_converted_byte_identical"], 4)
        target = self.tool.parse_gguf(self.destination)
        by_name = {item.name: item for item in target.tensors}
        self.assertEqual(by_name["blk.0.attn_q.weight"].ggml_type,
                         int(GGMLQuantizationType.Q4_0))
        expert = by_name["blk.0.ffn_up_exps.weight"]
        with self.destination.open("rb") as stream:
            stream.seek(expert.offset)
            self.assertEqual(stream.read(expert.nbytes),
                             self.payloads[expert.name])

    def test_include_and_exclude_patterns_narrow_plan(self):
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--dry-run", "--include-pattern", "weight$",
                              "--exclude-pattern", "output|ffn_up$")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        converted = [row["name"] for row in json.loads(result.stdout)["tensors"]
                     if row["action"] == "convert"]
        self.assertEqual(converted, ["blk.0.attn_q.weight", "blk.0.ffn_up.weight",
                                     "token_embd.weight"])
        self.assertNotIn("blk.0.ffn_up_exps.weight", converted)

    def test_types_filter_refuses_empty_selection_without_publishing(self):
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--types", "F32", "--to", "Q4_0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no tensors match", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_refuses_existing_destination(self):
        self.destination.write_bytes(b"preserve")
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--to", "Q4_0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stdout)
        self.assertEqual(self.destination.read_bytes(), b"preserve")

    def test_invalid_regex_is_a_clean_refusal(self):
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--dry-run", "--include-pattern", "[")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid --include-pattern regex", result.stdout)

    def test_refuses_non_glm_architecture(self):
        data = self.source.read_bytes().replace(b"glm-dsa", b"not-glm", 1)
        self.source.write_bytes(data)
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be exactly glm-dsa", result.stdout)

    def test_verifier_detects_changed_nonconverted_expert(self):
        built = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                             "--to", "Q4_0")
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        target = self.tool.parse_gguf(self.destination)
        expert = next(item for item in target.tensors if "_exps" in item.name)
        with self.destination.open("r+b") as stream:
            stream.seek(expert.offset)
            byte = stream.read(1)
            stream.seek(expert.offset)
            stream.write(bytes((byte[0] ^ 1,)))
        result = self.run_cli(os.fspath(self.source), os.fspath(self.destination),
                              "--to", "Q4_0", "--verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-converted tensor bytes changed", result.stdout)


if __name__ == "__main__":
    unittest.main()
