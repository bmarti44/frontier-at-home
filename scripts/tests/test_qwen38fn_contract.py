"""Qwen3.8-Flash-Next production-path profile contract; RED before any implementation.

The profile serves NVIDIA's NVFP4 checkpoint on upstream vLLM with the PLE
n-gram table read in place from NVMe (vLLM PR #58439 checkpoint_mapped), MTP
speculative decoding, and four native 262,144-token slots. It stays
``estimated`` until the qualification bundle exists.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver

SLUG = "qwen3.8-flash-next"
PROFILE = "cuda-spark-128g-1m.json"
# Byte-identical chat template, tokenizer and generation config to Qwen3.8-27B
# (checked 2026-10-10), so the qualified qwen38 encoder is reused unchanged.
QWEN38_TOKENIZER_CONFIG = Path("/home/bmarti44/models/qwen3.8-27b/tokenizer_config.json")
FLASH_CHAT_TEMPLATE = Path("/home/bmarti44/models/qwen3.8-flash-next-nvfp4/chat_template.jinja")
HOST_BOUND = os.environ.get("FRONTIER_HOST_BOUND_TESTS", "1") != "0"


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


class Qwen38FlashNextProfileContract(unittest.TestCase):
    def profile(self):
        return profile_resolver.load_profile(SLUG, PROFILE)

    def render(self, p):
        return profile_resolver.resolve(
            p, profile_resolver.load_model(SLUG),
            profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json"))

    def test_profile_is_estimated_four_native_slots_totalling_one_million(self):
        p = self.profile()
        self.assertEqual(p["status"]["state"], "estimated")
        self.assertNotIn("switch_alias", p)
        self.assertEqual(p["context_cap"], 1048576)
        self.assertEqual(p["serving"]["parallel_slots"], 4)
        self.assertEqual(p["serving"]["request_context_cap"], 262144)
        argv = p["launch"]["args"]
        self.assertEqual(arg(argv, "--max-model-len"), "262144")
        self.assertEqual(arg(argv, "--max-num-seqs"), "4")
        # Native context only: static YaRN degrades short-context quality.
        self.assertFalse(any("yarn" in a for a in argv))
        self.assertNotIn("--hf-overrides", argv)
        self.assertGreaterEqual(p["safety"]["kill_floor_gib"], 10)

    def test_agentic_serving_semantics_are_explicit(self):
        argv = self.profile()["launch"]["args"]
        for flag, value in (("--kv-cache-dtype", "fp8"),
                            ("--tool-call-parser", "qwen3_coder"),
                            ("--reasoning-parser", "qwen3"),
                            ("--quantization", "modelopt"),
                            ("--middleware", "glm53_runtime_policy.SplitTerminalSseMiddleware")):
            with self.subTest(flag=flag):
                self.assertEqual(arg(argv, flag), value)
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertIn("--enable-prefix-caching", argv)
        index = argv.index("--served-model-name")
        self.assertEqual(argv[index + 1:index + 3], [SLUG, "default"])
        self.assertIn('"reasoning_effort":"low"', arg(argv, "--default-chat-template-kwargs"))

    def test_ple_table_is_read_in_place_and_mtp_is_on(self):
        argv = self.profile()["launch"]["args"]
        self.assertEqual(arg(argv, "--engram-config"), '{"checkpoint_mapped":true}')
        self.assertIn('"method":"mtp"', arg(argv, "--speculative-config"))

    def test_text_only_until_media_is_qualified(self):
        argv = self.profile()["launch"]["args"]
        self.assertEqual(arg(argv, "--limit-mm-per-prompt"), '{"image":0,"video":0}')

    def test_render_carries_safety_containment_and_runtime_pin(self):
        p = self.profile()
        out = self.render(p)
        self.assertEqual(out["serving"], p["serving"])
        self.assertEqual(out["safety"], p["safety"])
        self.assertEqual(out["systemd"]["flock"], "/run/lock/frontier-at-home/inference.lock")
        props = out["systemd"]["properties"]
        self.assertEqual(props["MemorySwapMax"], "0")
        self.assertEqual(props["OOMPolicy"], "kill")
        self.assertEqual(props["KillMode"], "control-group")
        env = p["launch"]["env"]
        self.assertIn("qwen38-flash-next/native-runtime-001/runtime/bin", env["PATH"])
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        engine = profile_resolver.load_model(SLUG)["engines"][p["engine"]]
        self.assertRegex(engine["binary_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue((ROOT / engine["build_manifest"]).is_file())

    def test_text_only_serving_still_rejects_bad_media_fields(self):
        for key, value in (("max_images", 0), ("audio_clips", 1)):
            p = self.profile()
            p["serving"][key] = value
            with self.subTest(key=key), self.assertRaises(profile_resolver.ProfileError):
                self.render(p)

    def test_model_pins_the_nvfp4_checkpoint_revision(self):
        model = profile_resolver.load_model(SLUG)
        artifact = model["artifacts"][self.profile()["artifact_roles"]["model"]]
        self.assertEqual(artifact["kind"], "safetensors-dir")
        self.assertIn("fc694b54fb0174e0913e6adf86691ef85a4ead47", artifact["note"])
        names = {Path(s["path"]).name: s["sha256"] for s in artifact["shards"]}
        for required in ("chat_template.jinja", "config.json", "hf_quant_config.json",
                         "model.safetensors.index.json", "tokenizer.json"):
            self.assertRegex(names[required], r"^[0-9a-f]{64}$")

    def test_qualification_kit_reuses_the_qwen38_encoder(self):
        spec = importlib.util.spec_from_file_location(
            "qualify_profile", ROOT / "scripts/94_qualify_profile.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MODEL_ENCODERS[SLUG], "qwen38")

    @unittest.skipUnless(HOST_BOUND, "host-bound: reads local weights")
    def test_local_template_matches_the_qwen38_encoder_source(self):
        if not (FLASH_CHAT_TEMPLATE.is_file() and QWEN38_TOKENIZER_CONFIG.is_file()):
            self.skipTest("weights not present on this host")
        qwen38 = json.loads(QWEN38_TOKENIZER_CONFIG.read_text(encoding="utf-8"))["chat_template"]
        flash = FLASH_CHAT_TEMPLATE.read_bytes()
        self.assertEqual(flash, qwen38.encode("utf-8"))
        artifact = profile_resolver.load_model(SLUG)["artifacts"][
            self.profile()["artifact_roles"]["model"]]
        pinned = {Path(s["path"]).name: s["sha256"] for s in artifact["shards"]}
        self.assertEqual(pinned["chat_template.jinja"], hashlib.sha256(flash).hexdigest())


if __name__ == "__main__":
    unittest.main()
