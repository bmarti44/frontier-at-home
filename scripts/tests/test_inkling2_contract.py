"""Inkling 2 Mini M0 scaffold contract: pins, runtime draft, and the smoke profile.

The smoke profile is a DIAGNOSTIC first load on the vLLM nightly that ships
the Inkling 2 model code: 2 x 131,072-token slots, BF16 KV, no speculative
decoding, and every memory figure an estimate until the first load measures
it. It stays ``estimated`` with no switch alias. The checks here tie the
profile's digests to the weights manifest and the runtime draft's patch
digests to the files in patches/inkling2/, so a changed pin fails here first.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver  # noqa: E402

SLUG = "inkling-2-mini"
SMOKE = "cuda-spark-128g-smoke.json"
ONE_M = "cuda-spark-128g-1m.json"
ARTIFACT = "official-b85c3cfa"
ENGINE = "vllm-inkling2-r001"
RUNTIME = "{cache_root}/inkling-2-mini/native-runtime-001/runtime"
WEIGHTS_MANIFEST = ROOT / "weights/inkling-2-mini/manifest.json"
BUILD_MANIFEST = "configs/build-manifests/inkling-2-mini-runtime.json"
PATCHES = ROOT / "patches/inkling2"
SHARDS = (
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
    "mtp.safetensors",
    "config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "processor_config.json",
)
ADDED_PIN_DIGESTS = {
    # Runtime pins handed over with the M0 task; the install logs show the
    # same files were installed.
    "interpreter": "476b92e8657f3c9d",
    "vllm_wheel": "dfba467982d6d33f",
    "jit_cache": "3965e79b8f7438fd",
    "jit_cache_sm120f": "49f84b2a513da150",
    "jit_cache_sm121a": "166d7f1792c81f52",
}


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inkling2ScaffoldContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = profile_resolver.load_model(SLUG)
        cls.profile = profile_resolver.load_profile(SLUG, SMOKE)
        cls.host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
        cls.weights = {
            entry["name"]: entry
            for entry in json.loads(WEIGHTS_MANIFEST.read_text())["files"]
        }
        cls.build = json.loads((ROOT / BUILD_MANIFEST).read_text())

    def test_dev_port_is_8018(self):
        self.assertEqual(self.host["ports"]["dev"][SLUG], 8018)
        self.assertEqual(len(set(self.host["ports"]["dev"].values())), len(self.host["ports"]["dev"]))

    def test_model_artifact_digests_come_from_the_weights_manifest(self):
        self.assertEqual(self.model["identity_manifest"], "weights/inkling-2-mini/manifest.json")
        artifact = self.model["artifacts"][ARTIFACT]
        self.assertEqual(artifact["kind"], "safetensors-dir")
        self.assertEqual(artifact["path"], "{model_root}/inkling-2-mini")
        shards = {shard["path"]: shard["sha256"] for shard in artifact["shards"]}
        self.assertEqual(set(shards), {f"{{model}}/{name}" for name in SHARDS})
        for name in SHARDS:
            with self.subTest(shard=name):
                self.assertEqual(shards[f"{{model}}/{name}"], self.weights[name]["sha256"])

    def test_engine_is_the_pinned_runtime(self):
        engine = self.model["engines"][ENGINE]
        self.assertEqual(engine["binary_path"], f"{RUNTIME}/bin/python3.12")
        self.assertTrue(engine["binary_sha256"].startswith(ADDED_PIN_DIGESTS["interpreter"]))
        self.assertEqual(engine["build_manifest"], BUILD_MANIFEST)
        self.assertEqual(self.build["runtime_root"], RUNTIME)
        self.assertEqual(self.build["interpreter"]["sha256"], engine["binary_sha256"])

    def test_backend_support_covers_every_registered_backend(self):
        backends = json.loads((ROOT / "configs/backends.json").read_text())["backends"]
        support = self.model["backend_support"]
        self.assertEqual(set(support), set(backends))
        self.assertTrue(support["cuda"]["supported"])
        for backend, entry in support.items():
            if backend != "cuda":
                with self.subTest(backend=backend):
                    self.assertFalse(entry["supported"])
                    self.assertTrue(entry["reason"])

    def test_runtime_draft_pins_the_wheels_and_patch_files(self):
        vllm = self.build["vllm"]
        self.assertEqual(vllm["commit"], "647a76866b4559a74e27d1c6b55fbffe85c07ce4")
        self.assertEqual(vllm["version"], "0.34.0rc1.dev1133+g647a76866")
        self.assertTrue(vllm["wheel"]["url"].startswith(
            "https://wheels.vllm.ai/647a76866b4559a74e27d1c6b55fbffe85c07ce4/"))
        self.assertTrue(vllm["wheel"]["sha256"].startswith(ADDED_PIN_DIGESTS["vllm_wheel"]))
        wheels = {w["name"].split("-0.7.4")[0]: w for w in self.build["flashinfer_jit_cache"]["wheels"]}
        for key, name in (("jit_cache", "flashinfer_jit_cache"),
                          ("jit_cache_sm120f", "flashinfer_jit_cache_sm120f"),
                          ("jit_cache_sm121a", "flashinfer_jit_cache_sm121a")):
            with self.subTest(wheel=name):
                self.assertTrue(wheels[name]["sha256"].startswith(ADDED_PIN_DIGESTS[key]))
                self.assertTrue(wheels[name]["url"].startswith(
                    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.4/"))
        patches = {p["path"]: p["sha256"] for p in self.build["patches"]}
        on_disk = sorted(PATCHES.glob("*.diff"))
        self.assertEqual(len(on_disk), 4)
        self.assertEqual(set(patches), {str(p.relative_to(ROOT)) for p in on_disk})
        for path in on_disk:
            with self.subTest(patch=path.name):
                self.assertEqual(patches[str(path.relative_to(ROOT))], sha256(path))
        # Filled by the install, never guessed.
        for key in ("packages", "freeze_sha256", "vllm_tree", "gpu_selftest"):
            with self.subTest(pending=key):
                self.assertIn(key, self.build)

    def test_smoke_is_an_estimated_diagnostic_without_a_switch_alias(self):
        p = self.profile
        self.assertEqual(p["profile_id"], f"{SLUG}/cuda-spark-128g-smoke")
        self.assertEqual(p["status"]["state"], "estimated")
        self.assertIn("pre-measurement", p["status"]["basis"])
        self.assertIn("DIAGNOSTIC", p["purpose"])
        self.assertNotIn("switch_alias", p)
        self.assertNotIn("switch", p)
        self.assertEqual(p["engine"], ENGINE)
        self.assertEqual(p["artifact_roles"], {"model": ARTIFACT})
        self.assertEqual(p["bench"]["stack_label"], "inkling-2-mini-cuda-spark-128g-smoke")

    def test_smoke_topology_two_slots_of_131072_with_images(self):
        p = self.profile
        argv = p["launch"]["args"]
        self.assertEqual(p["context_cap"], 262144)
        self.assertEqual(p["serving"], {"parallel_slots": 2, "request_context_cap": 131072, "max_images": 4})
        self.assertEqual(arg(argv, "--max-model-len"), "131072")
        self.assertEqual(arg(argv, "--max-num-seqs"), "2")
        self.assertEqual(json.loads(arg(argv, "--limit-mm-per-prompt")), {"image": 4, "video": 0})
        self.assertEqual(arg(argv, "--mm-processor-cache-gb"), "0")

    def test_smoke_serving_flags(self):
        argv = self.profile["launch"]["args"]
        for flag, value in (("--kv-cache-dtype", "auto"),
                            ("--gpu-memory-utilization", "0.80"),
                            # A fixed arena: without it this vLLM's extensible KV
                            # cache commits up to 0.80 x total memory after warmup.
                            ("--kv-cache-memory-bytes", "24000000000"),
                            ("--max-num-batched-tokens", "2048"),
                            ("--reasoning-parser", "inkling"),
                            ("--tool-call-parser", "inkling"),
                            ("--logprobs-mode", "raw_logprobs"),
                            ("--shutdown-timeout", "10"),
                            ("--model", "{model}"),
                            ("--port", "{port}")):
            with self.subTest(flag=flag):
                self.assertEqual(arg(argv, flag), value)
        self.assertEqual(json.loads(arg(argv, "--default-chat-template-kwargs")), {"reasoning_effort": "low"})
        names = argv.index("--served-model-name")
        self.assertEqual(argv[names + 1:names + 3], ["inkling-2-mini", "default"])
        for flag in ("--enable-prefix-caching", "--enable-chunked-prefill",
                     "--enable-auto-tool-choice", "--enable-prompt-tokens-details"):
            with self.subTest(flag=flag):
                self.assertIn(flag, argv)
        # vLLM adds middleware in order and the last is outermost.
        middleware = [argv[i + 1] for i, a in enumerate(argv) if a == "--middleware"]
        self.assertEqual(middleware, ["vllm_reasoning_compat.ReasoningContentMiddleware",
                                      "glm53_runtime_policy.SplitTerminalSseMiddleware"])

    def test_smoke_carries_no_other_models_levers(self):
        launch = self.profile["launch"]
        argv = launch["args"]
        for flag in ("--worker-cls", "--speculative-config", "--engram-config", "--quantization"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, argv)
        self.assertFalse(any("fp8" in a for a in argv))
        self.assertFalse(any("qwen" in key.lower() or "qwen" in value.lower()
                             for key, value in launch["env"].items()))
        # The PR #69288 gate stays at its default (off) in this profile.
        self.assertNotIn("INKLING2_MAX_CTAS_POW2", launch["env"])

    def test_smoke_env_is_the_inkling_runtime_without_jit(self):
        env = self.profile["launch"]["env"]
        self.assertEqual(env["HOME"], "{cache_root}/inkling-2-mini/home")
        self.assertTrue(env["PATH"].startswith(f"{RUNTIME}/bin:"))
        self.assertEqual(env["FLASHINFER_DISABLE_JIT"], "1")
        self.assertEqual(env["TRITON_CACHE_DIR"], "{cache_root}/inkling-2-mini/home/.triton/cache")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        # The wheel's FA2 ships sm_80 SASS + PTX ISA 9.2 only, which the r580
        # driver cannot JIT on SM12x (smoke run1); patch 0003 routes the vision
        # encoder's attention through vLLM's Triton prefill kernel.
        self.assertEqual(env["INKLING2_VISION_ATTN"], "triton")
        # The model card accepts images up to 4096 px a side, but vLLM sizes the
        # encoder budget from a 2048 px profiling image (52 x 52 = 2,704 tokens):
        # smoke run2 rejected two MMMU images of 3,456 and 2,752 tokens with HTTP
        # 400. Patch 0004 lets the profile raise that side to 4096 (103 x 103 =
        # 10,609 tokens per image).
        self.assertEqual(env["INKLING2_MAX_IMAGE_SIDE"], "4096")

    def test_patch_0004_gates_the_profiling_image_side_at_the_upstream_default(self):
        text = (PATCHES / "0004-local-max-image-side.diff").read_text()
        self.assertIn("+++ b/vllm/models/inkling2/processing.py", text)
        self.assertIn('os.environ.get("INKLING2_MAX_IMAGE_SIDE", "2048")', text)
        self.assertIn("-MAX_DUMMY_IMAGE_SIDE = 2048", text)

    def test_smoke_declares_the_vision_answer_budget(self):
        # Inkling reasons even at low effort; at the bench's 256-token default,
        # 5 of 10 smoke run2 MMMU answers ran out of budget before the letter.
        options = self.profile["qualification_options"]
        self.assertEqual(options["vision_max_tokens"], 4096)
        with tempfile.TemporaryDirectory(prefix="inkling2-kit-") as tmp:
            out = Path(tmp) / "bundle"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/94_qualify_profile.py"), "--profile",
                 f"{SLUG}/cuda-spark-128g-smoke", "--host", str(ROOT / "configs/hosts/spark-aba1.json"),
                 "--dry-run", "--out", str(out), "--cells", "vision"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            manifest = json.loads((out / "manifest.json").read_text())
        vision = manifest["cells"]["vision"]["argv"]
        self.assertEqual(arg(vision, "--max-tokens"), "4096")
        self.assertEqual(manifest["profile"]["vision_max_tokens"], 4096)

    def test_smoke_safety_and_memory_are_conservative_estimates(self):
        p = self.profile
        self.assertEqual(p["safety"]["kill_floor_gib"], 15)
        self.assertEqual(p["safety"]["minimum_start_gib"], 110)
        self.assertEqual(p["safety"]["startup_timeout_seconds"], 1800)
        budget = p["memory_model"]
        self.assertEqual((budget["resident_weights_gib"], budget["overhead_gib"],
                          budget["extra_gib"], budget["floor_gib"]), (20, 12, 23, 15))
        self.assertIn("ESTIMATE", budget["basis"])
        containment = p["containment"]
        self.assertEqual((containment["memory_swap_max"], containment["oom_policy"],
                          containment["kill_mode"]), ("0", "kill", "control-group"))

    def test_smoke_renders_with_every_digest_target(self):
        snapshot = profile_resolver.resolve(self.profile, self.model, self.host)
        targets = {check["path"]: check["sha256"] for check in snapshot["digest_checks"]}
        root = self.host["paths"]["model_root"] + "/inkling-2-mini"
        for name in SHARDS:
            with self.subTest(shard=name):
                self.assertEqual(targets[f"{root}/{name}"], self.weights[name]["sha256"])
        binary = self.host["paths"]["cache_root"] + "/inkling-2-mini/native-runtime-001/runtime/bin/python3.12"
        self.assertEqual(snapshot["binary"], binary)
        self.assertIn(binary, targets)
        self.assertEqual(snapshot["status"], "estimated")



class Inkling2OneMillionProfile(unittest.TestCase):
    """The candidate serving profile: four native 262,144-token slots in BF16.

    Sized from smoke run2 (results/inkling2-gates/smoke-2026-11-23/run2): a
    24,000,000,000-byte arena gave vLLM's group-aware capacity 2.86x at
    131,072 tokens, so 4.0x at 262,144 needs about 67.1e9 bytes; 68e9 leaves
    a margin. The arena is the lever screen's carrier (a0) and stays
    ``estimated`` until a load shows >= 4.0x at 262,144.
    """

    @classmethod
    def setUpClass(cls):
        cls.profile = profile_resolver.load_profile(SLUG, ONE_M)
        cls.model = profile_resolver.load_model(SLUG)
        cls.host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
        cls.smoke = profile_resolver.load_profile(SLUG, SMOKE)

    def test_four_native_262144_slots_are_the_full_1m(self):
        p = self.profile
        argv = p["launch"]["args"]
        self.assertEqual(p["profile_id"], f"{SLUG}/cuda-spark-128g-1m")
        self.assertEqual(p["context_cap"], 1048576)
        self.assertEqual(p["serving"], {"parallel_slots": 4, "request_context_cap": 262144, "max_images": 4})
        self.assertEqual(arg(argv, "--max-model-len"), "262144")
        self.assertEqual(arg(argv, "--max-num-seqs"), "4")

    def test_bf16_kv_in_a_fixed_68e9_byte_arena(self):
        argv = self.profile["launch"]["args"]
        self.assertEqual(arg(argv, "--kv-cache-dtype"), "auto")
        self.assertFalse(any("fp8" in a for a in argv))
        self.assertEqual(arg(argv, "--kv-cache-memory-bytes"), "68000000000")
        budget = self.profile["memory_model"]
        self.assertAlmostEqual(budget["extra_gib"], 63.4, places=1)
        self.assertGreaterEqual(budget["extra_gib"] * 2**30, 68e9)
        self.assertEqual(budget["kv_bytes_per_token"], 0)

    def test_carrier_matches_the_smoke_except_topology_and_the_release_worker(self):
        smoke = list(self.smoke["launch"]["args"])
        argv = list(self.profile["launch"]["args"])
        worker = argv.index("--worker-cls")
        self.assertEqual(argv[worker + 1], "qwen38fn_worker.HostCacheReleaseWorker")
        del argv[worker:worker + 2]
        for flag, old, new in (("--max-model-len", "131072", "262144"), ("--max-num-seqs", "2", "4"),
                               ("--kv-cache-memory-bytes", "24000000000", "68000000000")):
            with self.subTest(flag=flag):
                self.assertEqual(arg(smoke, flag), old)
                smoke[smoke.index(flag) + 1] = new
        self.assertEqual(argv, smoke)
        self.assertEqual(arg(argv, "--max-num-batched-tokens"), "2048")
        for flag in ("--speculative-config", "--quantization"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, argv)

    def test_env_adds_only_the_release_roots(self):
        env = dict(self.profile["launch"]["env"])
        self.assertEqual(env.pop("QWEN38FN_HOST_RELEASE_ROOTS"),
                         "{cache_root}/inkling-2-mini:{cache_root}/qwen38-flash-next")
        self.assertEqual(env, self.smoke["launch"]["env"])
        self.assertNotIn("INKLING2_MAX_CTAS_POW2", env)

    def test_estimated_without_a_switch_alias_until_qualified(self):
        p = self.profile
        self.assertEqual(p["status"]["state"], "estimated")
        self.assertNotIn("switch_alias", p)
        self.assertNotIn("switch", p)
        self.assertEqual(p["engine"], ENGINE)
        self.assertEqual(p["artifact_roles"], {"model": ARTIFACT})
        self.assertEqual(p["bench"]["stack_label"], "inkling-2-mini-cuda-spark-128g-1m")
        self.assertEqual(p["containment"]["unit"], "inkling2mini-engine")

    def test_safety_keeps_the_agents_1m_floor(self):
        safety = self.profile["safety"]
        self.assertEqual(safety["kill_floor_gib"], 10)
        self.assertEqual(safety["startup_timeout_seconds"], 1800)
        budget = self.profile["memory_model"]
        self.assertEqual(budget["floor_gib"], 10)
        peak = budget["resident_weights_gib"] + budget["overhead_gib"] + budget["extra_gib"]
        self.assertGreaterEqual(safety["minimum_start_gib"], peak + safety["kill_floor_gib"])
        self.assertLessEqual(safety["minimum_start_gib"], 110)

    def test_targets_are_the_serving_defaults_headline(self):
        # README headline for qwen38fn (candidate 10): the promotion rule's speed half.
        targets = self.profile["qualification_targets"]
        self.assertEqual(targets["decode_tok_s_min"], {"0": 34.07, "28672": 40.92})
        self.assertEqual(targets["prefill_tok_s_min"], {"28672": 2711.5})
        self.assertEqual(targets["toolcall_min"], 19)
        self.assertEqual(targets["vision_min"], 0.64)
        self.assertIs(targets["context_pass"], True)
        self.assertEqual(self.profile["qualification_options"], {"vision_max_tokens": 4096})

    def test_renders_feasible_on_this_host(self):
        snapshot = profile_resolver.resolve(self.profile, self.model, self.host)
        self.assertEqual(snapshot["status"], "estimated")
        binary = self.host["paths"]["cache_root"] + "/inkling-2-mini/native-runtime-001/runtime/bin/python3.12"
        self.assertEqual(snapshot["binary"], binary)


if __name__ == "__main__":
    unittest.main()
