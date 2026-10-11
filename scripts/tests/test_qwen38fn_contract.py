"""Qwen3.8-Flash-Next production-path profile contract; RED before any implementation.

The profile serves NVIDIA's NVFP4 checkpoint on upstream vLLM with the PLE
n-gram table read in place from NVMe (vLLM PR #58439 checkpoint_mapped), MTP
speculative decoding, and four native 262,144-token slots. Candidate 5
qualified it text-only as the optional ``qwen38fn`` switch alias. Candidate 7
turned image input on under the owner's 8 GiB floor (2026-10-11) and failed
on NVRM lines at the KV arena: another engine's model files were still in
page cache. Candidate 8 adds the host page-cache release, so the profile is
``estimated`` until that run passes.
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

    def test_profile_is_a_switch_alias_pending_candidate_8(self):
        # Candidate 5's PASS covers the text-only launch; images on is a new
        # configuration, so the status waits for candidate 8's own gate.
        p = self.profile()
        status = p["status"]
        self.assertEqual(status["state"], "estimated")
        self.assertIn("results/qwen38fn-gates/qualify-2026-10-11-c8/", status["basis"])
        self.assertEqual(p["switch_alias"], "qwen38fn")
        self.assertEqual(p["switch"], {
            "mode": "generic",
            "served_model_id": "qwen3.8-flash-next",
            "health_path": "/health",
            "context_check": {"kind": "model_card", "max_model_len": 262144},
        })
        self.assertEqual(p["context_cap"], 1048576)
        self.assertEqual(p["serving"]["parallel_slots"], 4)
        self.assertEqual(p["serving"]["request_context_cap"], 262144)
        argv = p["launch"]["args"]
        self.assertEqual(arg(argv, "--max-model-len"), "262144")
        self.assertEqual(arg(argv, "--max-num-seqs"), "4")
        # Native context only: static YaRN degrades short-context quality.
        self.assertFalse(any("yarn" in a for a in argv))
        self.assertNotIn("--hf-overrides", argv)

    def test_owner_floor_is_8_gib_and_admission_fits_the_quiet_host(self):
        # Owner decision 2026-10-11: keep 1M with images on and kill at 8 GiB
        # (DeepSeek's 1M-fast profile set the precedent). The switch admits on
        # minimum_start_gib alone; this host reads 115.2-115.6 GiB when quiet.
        p = self.profile()
        self.assertEqual(p["safety"]["kill_floor_gib"], 8)
        self.assertEqual(p["memory_model"]["floor_gib"], 8)
        self.assertEqual(p["safety"]["minimum_start_gib"], 115)

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
        # weights + overhead + extra + floor. Candidate 5 peaked at 102.97 GiB
        # used text-only; images add at least 3.1 GiB
        # (results/qwen38fn-gates/diag-vision-2026-10-11). The quiet host read
        # 115.2-115.6 GiB on 2026-10-11.
        memory_model = self.profile()["memory_model"]
        self.assertGreaterEqual(memory_model["resident_weights_gib"] + memory_model["overhead_gib"], 84.2)
        self.assertGreaterEqual(memory_model["resident_weights_gib"] + memory_model["overhead_gib"]
                                + memory_model["extra_gib"], 102.97 + 3.1)
        self.assertLessEqual(memory_model["resident_weights_gib"] + memory_model["overhead_gib"]
                             + memory_model["extra_gib"] + memory_model["floor_gib"], 115.0)

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
        self.assertEqual(arg(argv, "--worker-cls"), "qwen38fn_worker.HostCacheReleaseWorker")
        tree = ast.parse((ROOT / "scripts/lib/qwen38fn_worker.py").read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        cls = classes["CheckpointCacheReleaseWorker"]
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["ReleaseLoadCacheWorker"])
        methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        # Evict from before the weights load until warmup ends; nothing runs while serving.
        self.assertIn("CheckpointCacheEvictor(self.vllm_config.model_config.model)", methods["load_model"])
        self.assertIn(".start()", methods["load_model"])
        self.assertIn(".stop()", methods["compile_or_warm_up_model"])
        self.assertIn("finally:", methods["compile_or_warm_up_model"])
        # Candidate 7 met the KV arena with another engine's GGUF in page cache
        # (results/qwen38fn-gates/qualify-2026-10-11-c7). Candidate 8 releases
        # the model root before the weights load and again before the arena,
        # then refuses an arena that MemFree cannot cover. Both are startup-only.
        cls = classes["HostCacheReleaseWorker"]
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["CheckpointCacheReleaseWorker"])
        methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(methods), {"load_model", "initialize_from_config"})
        self.assertIn("release_tree(", methods["load_model"])
        self.assertIn("release_tree(", methods["initialize_from_config"])
        self.assertIn("mem_free_gib()", methods["initialize_from_config"])
        self.assertIn("raise RuntimeError", methods["initialize_from_config"])

    def test_images_on_and_video_off_like_the_default(self):
        # Images at 1M were NO_GO above the 10 GiB floor
        # (results/qwen38fn-gates/diag-vision-2026-10-11); the owner moved the
        # floor to 8 GiB. The vision tower adds 897,862,112 BF16 bytes to the
        # 75.65 GiB text-only load (vLLM reported 76.49 GiB with images on).
        p = self.profile()
        argv = p["launch"]["args"]
        self.assertEqual(arg(argv, "--limit-mm-per-prompt"), '{"image":4,"video":0}')
        self.assertEqual(p["serving"]["max_images"], 4)
        self.assertNotIn("max_videos", p["serving"])
        self.assertGreaterEqual(p["memory_model"]["resident_weights_gib"], 76.49)

    def test_processor_cache_is_off_as_in_the_measured_run(self):
        # vLLM's processor cache defaults to 4 GiB, held once in the API server
        # and once in the engine core: up to 8 GiB the memory model does not
        # count. The 3.1-3.3 GiB image cost was measured with it off (diag run
        # 2, 100 images); it only skips re-processing a repeated image.
        argv = self.profile()["launch"]["args"]
        self.assertEqual(argv.count("--mm-processor-cache-gb"), 1)
        self.assertEqual(arg(argv, "--mm-processor-cache-gb"), "0")

    def test_launch_is_candidate_5_plus_image_input(self):
        # qualify-2026-10-11-c4 (gate PASS) recorded the rendered argv and env.
        # Candidates 7 and 8 change only the image limit, the processor cache
        # (off), and the worker class (candidate 8's host page-cache release).
        # The interpreter pin moved from the python3 symlink to the file it
        # names; the bytes are the same.
        ev = ROOT / "results/qwen38fn-gates/qualify-2026-10-11-c4"
        recorded = json.loads((ev / "manifest.json").read_text())["profile"]
        out = profile_resolver.resolve(self.profile(), profile_resolver.load_model(SLUG),
                                       profile_resolver.load_host(ev / "host-worktree.json"))
        expected = list(recorded["argv"])
        index = expected.index("--limit-mm-per-prompt") + 1
        self.assertEqual(expected[index], '{"image":0,"video":0}')
        expected[index] = '{"image":4,"video":0}'
        expected[index + 1:index + 1] = ["--mm-processor-cache-gb", "0"]
        worker = expected.index("--worker-cls") + 1
        self.assertEqual(expected[worker], "qwen38fn_worker.CheckpointCacheReleaseWorker")
        expected[worker] = "qwen38fn_worker.HostCacheReleaseWorker"
        self.assertEqual(out["argv"], expected)
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
        # Media fields follow --limit-mm-per-prompt: an enabled modality declares
        # a positive maximum, and a disabled one declares none.
        for key, value in (("max_images", 0), ("audio_clips", 1), ("max_images", None), ("max_videos", 1)):
            p = self.profile()
            if value is None:
                del p["serving"][key]
            else:
                p["serving"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(profile_resolver.ProfileError):
                self.render(p)
        p = self.profile()
        args = p["launch"]["args"]
        args[args.index("--limit-mm-per-prompt") + 1] = '{"image":0,"video":0}'
        with self.assertRaises(profile_resolver.ProfileError):
            self.render(p)
        del p["serving"]["max_images"]
        self.assertNotIn("max_images", self.render(p)["serving"])

    def test_a_modality_missing_from_the_limit_is_enabled(self):
        # vLLM serves up to 999 items of any modality that --limit-mm-per-prompt
        # leaves out, so {"image":0} alone keeps video on.
        p = self.profile()
        p["serving"].pop("max_images", None)
        args = p["launch"]["args"]
        args[args.index("--limit-mm-per-prompt") + 1] = '{"image":0}'
        with self.assertRaises(profile_resolver.ProfileError):
            self.render(p)
        p["serving"].update(max_videos=1, video_frames=32)
        self.assertEqual(self.render(p)["serving"]["max_videos"], 1)
        # The dict form's count also defaults to 999 (vLLM BaseDummyOptions).
        p = self.profile()
        p["serving"].pop("max_images", None)
        args = p["launch"]["args"]
        args[args.index("--limit-mm-per-prompt") + 1] = '{"image":0,"video":{"num_frames":16}}'
        with self.assertRaises(profile_resolver.ProfileError):
            self.render(p)

    def test_switch_admission_covers_the_measured_peak(self):
        # The generic switch admits on safety.minimum_start_gib alone. Candidate
        # 5's two starts used up to (preload - low) GiB through the 1M fill;
        # admission must leave the kill floor above that peak.
        p = self.profile()
        safety = p["safety"]
        peaks = []
        for start in ("start1", "start2"):
            log = ROOT / "results/qwen38fn-gates/qualify-2026-10-11-c4/observe" / start / "meminfo.log"
            available = [int(re.search(r"MemAvailable:=(\d+)", line).group(1)) / 1048576
                         for line in log.read_text().splitlines()]
            peaks.append(available[0] - min(available))
        self.assertGreater(max(peaks), 100)
        self.assertGreaterEqual(safety["minimum_start_gib"], max(peaks) + safety["kill_floor_gib"])
        budget = p["memory_model"]
        self.assertGreaterEqual(safety["minimum_start_gib"],
                                budget["resident_weights_gib"] + budget["overhead_gib"]
                                + budget["extra_gib"] + budget["floor_gib"])

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
