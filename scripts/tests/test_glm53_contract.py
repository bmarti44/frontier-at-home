"""GLM-5.3 production admission contract; RED before any implementation.

Tiny synthetic files exercise the verifier, never represent model evidence.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver


class Glm53ProfileContract(unittest.TestCase):
    def profile(self):
        return profile_resolver.load_profile("glm-5.3-flash", "cuda-spark-128g-1m.json")

    def render(self, p):
        return profile_resolver.resolve(p, profile_resolver.load_model("glm-5.3-flash"),
            profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json"))

    def test_profile_is_optional_four_native_slots_totalling_one_million(self):
        p = self.profile()
        self.assertNotEqual(p["status"]["state"], "default")
        self.assertEqual(p["switch_alias"], "glm53-1m")
        self.assertEqual(p["context_cap"], 1048576)
        self.assertEqual(p["serving"]["parallel_slots"] * p["serving"]["request_context_cap"], 1048576)
        argv = p["launch"]["args"]
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "262144")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "4")
        self.assertEqual(p["safety"]["minimum_start_gib"], 110)
        self.assertGreaterEqual(p["safety"]["kill_floor_gib"], 10)

    def test_agentic_serving_semantics_are_explicit(self):
        p = self.profile()
        argv = p["launch"]["args"]
        for flag, value in (("--kv-cache-dtype", "fp8"),
                            ("--tool-call-parser", "glm47"),
                            ("--reasoning-parser", "glm45"),
                            ("--middleware", "glm53_runtime_policy.MediaPolicyMiddleware")):
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--enable-auto-tool-choice", argv)
        index = argv.index("--served-model-name")
        self.assertEqual(argv[index + 1:index + 3], ["glm-5.3-flash", "default"])
        # The middleware module must be importable by the served interpreter.
        self.assertIn("scripts/lib", p["launch"]["env"]["PYTHONPATH"])
        self.assertNotIn("-I", argv[:3])

    def test_render_rejects_mismatched_topology(self):
        p = self.profile()
        p["serving"]["request_context_cap"] = 1048576
        with self.assertRaisesRegex(profile_resolver.ProfileError, "context|topology"):
            self.render(p)

    def test_render_rejects_equals_form_topology_overrides(self):
        for flag in ("--max-model-len=1048576", "--max-num-seqs=1"):
            p = self.profile()
            p["launch"]["args"].append(flag)
            with self.subTest(flag=flag), self.assertRaises(profile_resolver.ProfileError):
                self.render(p)

    def test_render_carries_safety_containment_and_identity_binding(self):
        p = self.profile()
        out = self.render(p)
        self.assertEqual(out["serving"], p["serving"])
        self.assertEqual(out["safety"], p["safety"])
        self.assertEqual(out["systemd"]["flock"], "/run/lock/frontier-at-home/inference.lock")
        props = out["systemd"]["properties"]
        self.assertEqual(props["MemorySwapMax"], "0")
        self.assertEqual(props["KillMode"], "control-group")
        self.assertEqual(props["OOMPolicy"], "kill")
        paths = {Path(c["path"]).name for c in out["digest_checks"]}
        self.assertIn("python3", paths)
        self.assertIn("model.safetensors.index.json", paths)
        self.assertIn("quantization_config.json", paths)

    def test_require_qualified_rejects_estimated_profile(self):
        api = importlib.import_module("glm53_contract")
        with self.assertRaisesRegex(ValueError, "not qualified"):
            api.require_qualified({"status": {"state": "estimated"}})
        with self.assertRaisesRegex(ValueError, "not qualified"):
            api.require_qualified({})
        api.require_qualified({"status": {"state": "qualified"}})
        api.require_qualified(self.profile())  # the shipped profile is qualified (status.evidence set)
        self.assertTrue(self.profile()["status"].get("evidence"))


class ClosedInventoryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = importlib.import_module("glm53_contract")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "engine.so").write_bytes(b"synthetic-extension")
        self.manifest = {"schema_version": 1, "files": [{
            "path": "engine.so", "size_bytes": 19,
            "sha256": hashlib.sha256(b"synthetic-extension").hexdigest(),
        }]}

    def test_complete_inventory_passes(self):
        self.api.verify_inventory(self.root, self.manifest)

    def test_new_payload_during_hashing_is_rejected(self):
        original = self.api.sha256_file
        def add_payload(path):
            result = original(path)
            (self.root / "unlisted.py").write_bytes(b"injected during verification")
            return result
        with mock.patch.object(self.api, "sha256_file", side_effect=add_payload):
            with self.assertRaisesRegex(ValueError, "inventory|coverage|changed"):
                self.api.verify_inventory(self.root, self.manifest)

    def test_json_number_overflow_is_rejected(self):
        for number in ("1e999", "-1e999"):
            path = self.root / "overflow.json"
            path.write_text('{"nested": [{"value": ' + number + '}]}')
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "nonfinite"):
                self.api.strict_json(path)

    def test_same_size_replacement_is_rejected(self):
        (self.root / "engine.so").write_bytes(b"replacement-content")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.api.verify_inventory(self.root, self.manifest)

    def test_empty_duplicate_and_unlisted_payloads_are_rejected(self):
        for entries in ([], self.manifest["files"] * 2):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.api.verify_inventory(self.root, {"schema_version": 1, "files": entries})
        (self.root / "injected.py").write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "inventory|unlisted"):
            self.api.verify_inventory(self.root, self.manifest)

    def test_path_escape_and_symlink_are_rejected(self):
        for path in ("../engine.so", "/engine.so", "a/../engine.so"):
            bad = copy.deepcopy(self.manifest)
            bad["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.api.verify_inventory(self.root, bad)
        (self.root / "engine.so").unlink()
        (self.root / "engine.so").symlink_to("/dev/null")
        with self.assertRaisesRegex(ValueError, "symlink|regular"):
            self.api.verify_inventory(self.root, self.manifest)

    def test_invalid_sizes_and_digest_are_rejected(self):
        for key, value in (("size_bytes", True), ("size_bytes", -1),
                           ("sha256", "not-a-sha256")):
            bad = copy.deepcopy(self.manifest)
            bad["files"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.verify_inventory(self.root, bad)


if __name__ == "__main__":
    unittest.main()
