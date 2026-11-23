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
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver  # noqa: E402

SLUG = "inkling-2-mini"
SMOKE = "cuda-spark-128g-smoke.json"
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
        self.assertEqual(len(on_disk), 2)
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

    def test_smoke_safety_and_memory_are_conservative_estimates(self):
        p = self.profile
        self.assertEqual(p["safety"]["kill_floor_gib"], 15)
        self.assertEqual(p["safety"]["minimum_start_gib"], 110)
        self.assertEqual(p["safety"]["startup_timeout_seconds"], 1800)
        budget = p["memory_model"]
        self.assertEqual((budget["resident_weights_gib"], budget["overhead_gib"],
                          budget["extra_gib"], budget["floor_gib"]), (20, 12, 40, 15))
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


if __name__ == "__main__":
    unittest.main()
