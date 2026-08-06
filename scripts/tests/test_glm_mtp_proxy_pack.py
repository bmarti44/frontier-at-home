#!/usr/bin/env python3
"""Bounded mutation tests for the exact GLM MTP proxy pack builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/86_build_glm_mtp_proxy_pack.py"
SPEC = importlib.util.spec_from_file_location("glm_mtp_proxy_pack", SCRIPT)
assert SPEC and SPEC.loader
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class GlmMtpProxyPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="glm-mtp-pack-test-")
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.inventory = self.root / "inventory.json"
        self.output = self.root / "proxy.pack"
        self.receipt = self.root / "proxy.receipt.json"

        model_bytes = bytes((i * 37 + 11) & 0xFF for i in range(480))
        self.model.write_bytes(model_bytes)

        def row(name: str, offset: int) -> dict:
            return {
                "name": name,
                "type": "F32",
                "shape": [1],
                "offset": offset,
                "bytes": 1,
            }

        allowed = [row(f"allowed.{i:03d}", i) for i in range(255)]
        forbidden = [row(f"forbidden.{i:03d}", 255 + i) for i in range(225)]
        payload = {
            "schema_version": 1,
            "model_bytes": len(model_bytes),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "counts": dict(PACK.EXPECTED_COUNTS),
            "global_allowed": allowed[:3],
            "mtp_allowed": allowed[3:30],
            "target_router_allowed": allowed[30:],
            "forbidden_target_experts": forbidden,
        }
        self.inventory.write_text(json.dumps(payload, sort_keys=True) + "\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build(self) -> dict:
        PACK.build(self.model, self.inventory, self.output, self.receipt)
        return PACK.verify(self.output, self.inventory, self.receipt)

    def test_builds_exact_inventory_and_verifies(self) -> None:
        receipt = self.build()
        self.assertEqual(receipt["record_count"], 255)
        self.assertEqual(receipt["forbidden_record_count"], 225)
        self.assertEqual(receipt["pack_sha256"], PACK.sha256_path(self.output))

    def test_payload_mutation_fails(self) -> None:
        self.build()
        with self.output.open("r+b", buffering=0) as handle:
            handle.seek(PACK.align_up(PACK.HEADER.size + 255 * PACK.RECORD.size))
            original = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "payload hash differs"):
            PACK.verify(self.output, self.inventory, self.receipt)

    def test_receipt_hash_mutation_fails(self) -> None:
        receipt = self.build()
        receipt["pack_sha256"] = "0" * 64
        self.receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "whole-pack receipt hash differs"):
            PACK.verify(self.output, self.inventory, self.receipt)

    def test_forbidden_overlap_fails_before_publication(self) -> None:
        payload = json.loads(self.inventory.read_bytes())
        payload["forbidden_target_experts"][0]["offset"] = 0
        self.inventory.write_text(json.dumps(payload, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "intersects forbidden"):
            PACK.build(self.model, self.inventory, self.output, self.receipt)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.receipt.exists())

    def test_model_symlink_is_rejected(self) -> None:
        link = self.root / "model-link.gguf"
        link.symlink_to(self.model)
        with self.assertRaises(OSError):
            PACK.build(link, self.inventory, self.output, self.receipt)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_clobbered(self) -> None:
        self.output.write_bytes(b"owner-data")
        with self.assertRaises(FileExistsError):
            PACK.build(self.model, self.inventory, self.output, self.receipt)
        self.assertEqual(self.output.read_bytes(), b"owner-data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
