"""Qwen3.8-Flash-Next production-path profile contract; RED before any implementation.

The profile serves NVIDIA's NVFP4 checkpoint on upstream vLLM with the PLE
n-gram table read in place from NVMe (vLLM PR #58439 checkpoint_mapped), MTP
speculative decoding, and four native 262,144-token slots. It stays
``estimated`` until the qualification bundle exists.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
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

    def test_bench_and_agent_gate_can_see_mtp_blocks_and_cache_reuse(self):
        p = self.profile()
        argv = p["launch"]["args"]
        # One SSE event carries the accepted token plus up to every draft token.
        draft = json.loads(arg(argv, "--speculative-config"))["num_speculative_tokens"]
        self.assertEqual(p["qualification_options"]["speed_max_tokens_per_event"], 1 + draft)
        # The agent gate reads prefix-cache reuse from usage.prompt_tokens_details.
        self.assertIn("--enable-prompt-tokens-details", argv)

    def test_kv_arena_holds_four_full_slots_at_the_smoke_measured_cost(self):
        # Smoke server log: every "GPU KV cache usage" reading is a whole number
        # of blocks over 91, so the 4,200,000,000-byte arena held N = 92 blocks
        # of at most 4.2e9/92 bytes. Its 194,493 tokens at --max-model-len
        # 131,072 is concurrency 92/62: a request reserves 62 blocks. Group
        # block sizes are 3200 or 8 only, and 62 < 2 x cdiv(131,072, 3200), so
        # exactly one group grows by cdiv(L, 3200) (41 blocks); the other 21 are
        # GDN state (4 groups x (2 + 3 draft) blocks) and the QSA ring (1).
        # A 262,144-token slot therefore reserves 82 + 21 = 103 blocks. The
        # pool also gives up one null block (BlockPool), and a prefilling
        # request held 36 blocks against its 32-block reservation (39.6% =
        # 36/91), so allow 4 transient blocks for each of two requests that
        # chunked prefill can split its 4096-token budget across.
        text = (ROOT / "results/qwen38fn-gates/smoke-2026-10-11/observe/server-start1.log").read_text()
        self.assertIn("GPU KV cache size: 194,493 tokens, Maximum concurrency for "
                      "131,072 tokens per request", text)
        self.assertIn("kv cache group sizes (3200, 3200, 3200, 3200, 8, 3200)", text)
        usages = {float(u) for u in re.findall(r"GPU KV cache usage: ([0-9.]+)%", text)} - {0.0}
        self.assertGreaterEqual(len(usages), 5)
        for usage in usages:
            self.assertIn(usage, {round(100 * b / 91, 1) for b in range(92)})
        self.assertEqual(int(92 / 62 * 131_072), 194_493)
        block_bytes = -(-4_200_000_000 // 92)
        required = (4 * (82 + 21) + 1 + 2 * 4) * block_bytes
        p = self.profile()
        arena = int(arg(p["launch"]["args"], "--kv-cache-memory-bytes"))
        self.assertGreaterEqual(arena, required)
        # Every byte past the bound comes out of the 10 GiB whole-host floor.
        self.assertLessEqual(arena, required * 1.01)
        memory_model = p["memory_model"]
        self.assertEqual(memory_model["kv_bytes_per_token"], 0)
        self.assertGreaterEqual(memory_model["extra_gib"], arena / 2**30)

    def test_memory_model_admits_on_a_quiet_host_at_the_measured_footprint(self):
        # Smoke observe/meminfo.log: MemAvailable 112.38 GiB before the load and
        # 25.73 GiB at idle, with 1.44 GiB of other processes pushed to swap,
        # so the engine took ~88.1 GiB with its 3.91 GiB arena: ~84.2 GiB
        # without it. 93_profile_serve admits only when MemAvailable covers
        # weights + overhead + extra + floor; the quiet host read 114.0 GiB
        # (2026-10-11 01:35Z).
        memory_model = self.profile()["memory_model"]
        self.assertGreaterEqual(memory_model["resident_weights_gib"] + memory_model["overhead_gib"], 84.2)
        self.assertEqual(memory_model["floor_gib"], 10)
        self.assertLessEqual(memory_model["resident_weights_gib"] + memory_model["overhead_gib"]
                             + memory_model["extra_gib"] + memory_model["floor_gib"], 114.0)

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

    def test_startup_avoids_the_drafter_requant_and_checkpoint_cache_pressure(self):
        # Phase 2 candidate 3 start 1 died at startup
        # (results/qwen38fn-gates/diag-drafter-2026-10-11). The drafter's
        # DeepGEMM UE8M0 requant grew by more than 16 GiB. Large GPU
        # allocations met MemFree of 1-3 GiB behind checkpoint page cache
        # (NVRM NV_ERR_NO_MEMORY). Diagnostic arm C fixed both.
        p = self.profile()
        env, argv = p["launch"]["env"], p["launch"]["args"]
        self.assertEqual(env["VLLM_MOE_USE_DEEP_GEMM"], "0")
        self.assertEqual(arg(argv, "--worker-cls"), "qwen38fn_worker.CheckpointCacheReleaseWorker")
        tree = ast.parse((ROOT / "scripts/lib/qwen38fn_worker.py").read_text())
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "CheckpointCacheReleaseWorker")
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["ReleaseLoadCacheWorker"])
        methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        # Evict from before the weights load until warmup ends; nothing runs while serving.
        self.assertIn("CheckpointCacheEvictor(self.vllm_config.model_config.model)", methods["load_model"])
        self.assertIn(".start()", methods["load_model"])
        self.assertIn(".stop()", methods["compile_or_warm_up_model"])
        self.assertIn("finally:", methods["compile_or_warm_up_model"])

    def test_text_only_until_media_is_qualified(self):
        # Images at 1M are NO_GO (results/qwen38fn-gates/diag-vision-2026-10-11):
        # after 100 MMMU images the direct 1M fill crossed the 10.5 GiB stop.
        p = self.profile()
        argv = p["launch"]["args"]
        self.assertEqual(arg(argv, "--limit-mm-per-prompt"), '{"image":0,"video":0}')
        self.assertNotIn("max_images", p["serving"])
        self.assertNotIn("vision_min", p["qualification_targets"])

    def test_launch_is_the_configuration_candidate_5_qualified(self):
        # qualify-2026-10-11-c4 (gate PASS) recorded the rendered argv and env.
        # Only the interpreter pin moved, from the python3 symlink to the file
        # it names; the bytes are the same.
        ev = ROOT / "results/qwen38fn-gates/qualify-2026-10-11-c4"
        recorded = json.loads((ev / "manifest.json").read_text())["profile"]
        out = profile_resolver.resolve(self.profile(), profile_resolver.load_model(SLUG),
                                       profile_resolver.load_host(ev / "host-worktree.json"))
        self.assertEqual(out["argv"], recorded["argv"])
        self.assertEqual(out["env"], recorded["env"])
        self.assertEqual(out["binary"], recorded["binary"] + ".12")
        self.assertEqual(profile_resolver.load_model(SLUG)["engines"][self.profile()["engine"]]["binary_sha256"],
                         recorded["binary_sha256"])

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

    def test_serving_rejects_bad_media_fields(self):
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

    @unittest.skipUnless(HOST_BOUND, "host-bound: hashes the local runtime interpreter")
    def test_engine_binary_is_the_pinned_regular_file(self):
        # The generic switch opens binary_path with O_NOFOLLOW and revalidates
        # it with lstat + S_ISREG, so the pin must name the interpreter itself,
        # not the runtime's python3 -> python3.12 symlink.
        engine = profile_resolver.load_model(SLUG)["engines"][self.profile()["engine"]]
        path = Path(engine["binary_path"].replace("{cache_root}", str(Path.home() / ".cache")))
        if not path.parent.is_dir():
            self.skipTest("runtime not present on this host")
        self.assertFalse(path.is_symlink(), path)
        self.assertTrue(path.is_file(), path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), engine["binary_sha256"])

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
