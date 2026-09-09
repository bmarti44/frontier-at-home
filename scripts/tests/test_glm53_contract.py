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

    def test_profile_is_optional_and_explicitly_four_slots(self):
        p = self.profile()
        self.assertEqual(p["status"]["state"], "estimated")
        self.assertEqual(p["switch_alias"], "glm53-flash")
        self.assertEqual(p["context_cap"], 1048576)
        self.assertEqual(p["serving"]["parallel_slots"], 4)
        self.assertEqual(p["serving"]["request_context_cap"], 262144)
        self.assertEqual(p["serving"]["max_images"], 4)
        self.assertEqual(p["serving"]["max_videos"], 1)
        self.assertEqual(p["serving"]["video_frames"], 16)
        argv = p["launch"]["args"]
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "262144")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "4")
        self.assertNotIn("--speculative-config", argv)
        self.assertNotIn("--skip-mm-profiling", argv)
        self.assertNotIn("--enable-prefix-caching", argv)
        self.assertEqual(p["safety"]["minimum_start_gib"], 110)
        self.assertGreaterEqual(p["safety"]["kill_floor_gib"], 18)

    def test_runtime_semantics_are_explicit_and_media_policy_is_selected(self):
        argv = self.profile()["launch"]["args"]
        for flag, value in (("--kv-cache-dtype", "fp8"),
                            ("--tool-call-parser", "glm47"),
                            ("--reasoning-parser", "glm45"),
                            ("--middleware", "glm53_runtime_policy.MediaPolicyMiddleware")):
            with self.subTest(flag=flag):
                self.assertIn(flag, argv)
                self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertIn("--no-enable-prefix-caching", argv)

    def test_service_cannot_write_builder_owned_runtime_and_cache(self):
        p = self.profile()
        self.assertEqual(p["launch"]["user"], "dsv4")
        self.assertEqual(p["launch"]["args"][:2], ["-I", "-B"])
        properties = p["containment"]["extra_properties"]
        self.assertEqual(properties["Group"], "dsv4")
        self.assertEqual(properties["SupplementaryGroups"], "")
        self.assertEqual(properties["CapabilityBoundingSet"], "")
        self.assertEqual(properties["AmbientCapabilities"], "")
        self.assertEqual(properties["NoNewPrivileges"], "yes")

    def test_default_api_alias_and_no_inductor_baseline_are_explicit(self):
        argv = self.profile()["launch"]["args"]
        index = argv.index("--served-model-name")
        self.assertEqual(argv[index + 1:index + 3], ["glm-5.3-flash", "default"])
        index = argv.index("--compilation-config")
        self.assertEqual(json.loads(argv[index + 1]), {"mode": 0})

    def test_probability_semantics_and_direct_dynamo_disable_are_explicit(self):
        profile = self.profile()
        argv = profile["launch"]["args"]
        self.assertIn("--logprobs-mode", argv)
        self.assertEqual(argv[argv.index("--logprobs-mode") + 1], "raw_logprobs")
        self.assertEqual(profile["launch"]["env"].get("TORCH_COMPILE_DISABLE"), "1")

    def test_render_rejects_mismatched_topology(self):
        p = self.profile()
        p["serving"]["request_context_cap"] = 1048576
        model = profile_resolver.load_model("glm-5.3-flash")
        host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
        with self.assertRaisesRegex(profile_resolver.ProfileError, "context|topology"):
            profile_resolver.resolve(p, model, host)

    def test_render_rejects_equals_form_topology_overrides(self):
        for flag in ("--max-model-len=1048576", "--max-num-seqs=1"):
            p = self.profile()
            p["launch"]["args"].append(flag)
            with self.subTest(flag=flag), self.assertRaises(profile_resolver.ProfileError):
                profile_resolver.resolve(p, profile_resolver.load_model("glm-5.3-flash"),
                    profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json"))

    def test_render_carries_safety_and_environment(self):
        p = self.profile()
        model = profile_resolver.load_model("glm-5.3-flash")
        host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
        out = profile_resolver.resolve(p, model, host)
        self.assertEqual(out["serving"], p["serving"])
        self.assertEqual(out["safety"], p["safety"])
        self.assertEqual(out["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(out["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(out["systemd"]["flock"], "/run/lock/frontier-at-home/inference.lock")
        props = out["systemd"]["properties"]
        self.assertEqual(props["MemorySwapMax"], "0")
        self.assertEqual(props["KillMode"], "control-group")
        self.assertEqual(props["OOMPolicy"], "kill")

    def test_unqualified_switch_rejects_before_mutating_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            before = b'{"schema_version":1,"profile":"qwen38-1m"}\n'
            active.write_bytes(before)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/52_engine_switch.sh"), "glm53-flash"],
                env={"PATH": os.environ["PATH"], "ENGINE_SWITCH_TESTING": "1",
                     "ENGINE_SWITCH_TEST_ROOT": tmp},
                capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GLM-5.3-Flash is not qualified", result.stderr)
            self.assertEqual(active.read_bytes(), before)
            self.assertFalse((root / "actions.log").exists())
            self.assertFalse((root / "switch.lock").exists())

    def test_generic_launcher_rejects_glm_before_any_lifecycle_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/93_profile_serve.sh"), "--profile",
                 "glm-5.3-flash/cuda-spark-128g-1m", "start"],
                env={"PATH": os.environ["PATH"], "HOME": tmp,
                     "FRONTIER_HOST": "spark-aba1", "FAH_RUN_DIR": tmp + "/run"},
                capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("hardened GLM-5.3 lifecycle", result.stderr)
            self.assertFalse((Path(tmp) / "run").exists())


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
