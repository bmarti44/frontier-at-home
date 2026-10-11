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
                            ("--quantization", "modelopt")):
            with self.subTest(flag=flag):
                self.assertEqual(arg(argv, flag), value)
        # vLLM adds middleware in order and the last is outermost: rename vLLM's
        # ``reasoning`` first so the llama.cpp-shaped split sees reasoning_content.
        middleware = [argv[i + 1] for i, a in enumerate(argv) if a == "--middleware"]
        self.assertEqual(middleware, ["vllm_reasoning_compat.ReasoningContentMiddleware",
                                      "glm53_runtime_policy.SplitTerminalSseMiddleware"])
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertIn("--enable-prefix-caching", argv)
        index = argv.index("--served-model-name")
        self.assertEqual(argv[index + 1:index + 3], [SLUG, "default"])
        self.assertIn('"reasoning_effort":"low"', arg(argv, "--default-chat-template-kwargs"))

    def test_ple_table_is_read_in_place_and_mtp_is_on(self):
        argv = self.profile()["launch"]["args"]
        self.assertEqual(arg(argv, "--engram-config"), '{"checkpoint_mapped":true}')
        self.assertIn('"method":"mtp"', arg(argv, "--speculative-config"))

    def test_kv_arena_holds_four_full_slots_at_the_smoke_measured_cost(self):
        # The smoke's 4,200,000,000-byte arena reported 194,493 tokens at
        # --max-model-len 131,072, so its concurrency was >= 194,493/131,072.
        # vLLM charges a request cdiv(L, 3200) blocks per attention group plus a
        # fixed count per GDN-state and QSA-ring group, and cdiv(262,144, 3200)
        # is exactly twice cdiv(131,072, 3200), so twice the measured
        # per-request cost bounds one 262,144-token slot from above.
        log = ROOT / "results/qwen38fn-gates/smoke-2026-10-11/observe/server-start1.log"
        self.assertIn("GPU KV cache size: 194,493 tokens, Maximum concurrency for "
                      "131,072 tokens per request", log.read_text())
        smoke_arena, smoke_tokens, smoke_len = 4_200_000_000, 194_493, 131_072
        required = -(-4 * 2 * smoke_arena * smoke_len // smoke_tokens)
        p = self.profile()
        arena = int(arg(p["launch"]["args"], "--kv-cache-memory-bytes"))
        self.assertGreaterEqual(arena, required)
        # Every byte past the bound comes out of the 10 GiB whole-host floor.
        self.assertLessEqual(arena, required * 1.01)
        memory_model = p["memory_model"]
        self.assertEqual(memory_model["kv_bytes_per_token"], 0)
        self.assertGreaterEqual(memory_model["extra_gib"], arena / 2**30)

    def test_startup_cannot_compile_kernels_and_tolerates_uma_free_memory(self):
        for name in ("qwen3.8-flash-next/cuda-spark-128g-1m", "qwen3.8-flash-next/cuda-spark-128g-smoke"):
            slug, profile = name.split("/")
            p = profile_resolver.load_profile(slug, profile + ".json")
            env, argv = p["launch"]["env"], p["launch"]["args"]
            with self.subTest(profile=profile):
                # The flashinfer AOT cache ships every kernel; a JIT compile at
                # load would run unbounded nvcc beside the loaded model.
                self.assertEqual(env["FLASHINFER_DISABLE_JIT"], "1")
                self.assertEqual(env["MAX_JOBS"], "2")
                self.assertEqual(env["NVCC_THREADS"], "1")
                self.assertEqual(env["FLASHINFER_NVCC_THREADS"], "1")
                # vLLM requires utilization x total <= free at startup; on UMA
                # free is MemAvailable, below 0.92 x 121.7 GiB with a co-tenant.
                self.assertEqual(arg(argv, "--gpu-memory-utilization"), "0.85")

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
