#!/usr/bin/env python3
"""MiMo-V2.6-Flash on exllamav3 + TabbyAPI: profile, artifact and runtime contract.

Owner direction 2026-10-12: integrate MiMo-V2.6-Flash (MOPD checkpoint) as the
switchable alias ``mimo26flash``; qwen38fn stays the default. The profile is
``estimated`` until its qualification passes, so the switch refuses it.

* Engine: TabbyAPI 2636ef60 on exllamav3 1.6.1 in native-runtime-001
  (configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json), started
  through scripts/lib/tabby_launch.py.
* Weights: the post-out-scales-fix EXL3 pack MikeRoz 2.52 bpw (h6) with the
  official release's preprocessor_config.json, and benthecarman's 4-bit EXL3
  DFlash drafter. Each served directory is alone in its parent: TabbyAPI lists
  every directory under model_dir as a model.
* Topology: one 262,144-token slot. 1M does not fit beside 94 GiB of weights
  (the 9 global-attention layers need 17.3 GiB of Q8 K/V at 1M).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver  # noqa: E402
import tabby_launch  # noqa: E402

SLUG = "mimo-v2.6-flash"
PROFILE_FILE = "cuda-spark-128g-262k.json"
ALIAS = "mimo26flash"
SERVED_ID = "mimo-v2.6-flash"
SPARK_HOST = ROOT / "configs/hosts/spark-aba1.json"
BUILD = ROOT / "configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json"
WEIGHTS = ROOT / "weights/mimo-v2.6-flash/manifest.json"
LINT_ALLOWLIST = ROOT / "configs/lint-allowlist.json"
REPO = "/home/bmarti44/spark-deepseek-v4-flash"
RUNTIME = "/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001"
MODEL_DIR = "/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash"
DRAFT_DIR = "/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash"
CONTEXT = 262144
HOST_BOUND = os.environ.get("FRONTIER_HOST_BOUND_TESTS", "1") != "0"

ARGV = [
    "-B", "-m", "tabby_launch",
    "--host", "127.0.0.1",
    "--port", "8013",
    "--disable-auth", "true",
    "--disable-fetch-requests", "true",
    "--model-name", MODEL_DIR,
    "--backend", "exllamav3",
    "--inline-model-loading", "false",
    "--max-seq-len", "262144",
    "--cache-size", "262144",
    "--max-batch-size", "1",
    "--cache-mode", "8,8",
    "--chunk-size", "2048",
    # Host-RAM caches bounded below TabbyAPI's defaults (4096 / 1024 MB): sliding-window
    # checkpoints are 49.25 MiB each with the drafter's, and the smoke left 2 GiB above the floor.
    "--sysmem-recurrent-cache", "1024",
    "--sysmem-multimodal-cache", "256",
    "--vision", "true",
    "--reasoning", "true",
    "--tool-format", "qwen3_coder",
    "--draft-model-name", DRAFT_DIR,
    "--draft-mode", "model",
    "--log-prompt", "false",
    "--log-generation-params", "false",
    "--log-requests", "false",
    "--log-chat-completion-requests", "false",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rendered() -> dict:
    profile = profile_resolver.load_profile(SLUG, PROFILE_FILE)
    model = profile_resolver.load_model(SLUG)
    host = profile_resolver.load_host(SPARK_HOST)
    return profile_resolver.resolve(profile, model, host)


class Profile(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = profile_resolver.load_profile(SLUG, PROFILE_FILE)
        cls.model = profile_resolver.load_model(SLUG)
        cls.snapshot = rendered()

    def test_profile_is_an_estimated_generic_alias(self) -> None:
        profile = self.profile
        self.assertEqual(profile["profile_id"], f"{SLUG}/cuda-spark-128g-262k")
        self.assertEqual(profile["switch_alias"], ALIAS)
        self.assertEqual(profile["status"]["state"], "estimated")
        self.assertEqual(profile["port_role"], "production")
        self.assertEqual(profile["context_cap"], CONTEXT)
        self.assertEqual(profile["serving"],
                         {"parallel_slots": 1, "request_context_cap": CONTEXT})
        self.assertEqual(profile["switch"], {
            "mode": "generic", "served_model_id": SERVED_ID,
            "health_path": "/health",
            "context_check": {"kind": "props", "total_slots": 1, "n_ctx": CONTEXT},
        })
        self.assertEqual(profile["engine"], "tabby-exl3-mimo-r001")
        self.assertEqual(sorted(profile["artifact_roles"]), ["draft_model", "model"])
        self.assertEqual(profile["bench"]["stack_label"],
                         "mimo-v2.6-flash-cuda-spark-128g-262k")

    def test_launch_is_the_shim_with_the_profile_topology(self) -> None:
        snapshot = self.snapshot
        self.assertEqual(snapshot["mechanism"], "systemd-run")
        self.assertEqual(snapshot["binary"], f"{RUNTIME}/runtime/bin/python3.12")
        self.assertEqual(snapshot["argv"], ARGV)
        # The shim accepts the rendered argv and derives TabbyAPI's directories.
        tabby = tabby_launch.tabby_argv(snapshot["argv"][3:])
        for flag, value in (("--model-dir", os.path.dirname(MODEL_DIR)),
                            ("--model-name", SERVED_ID),
                            ("--draft-model-dir", os.path.dirname(DRAFT_DIR)),
                            ("--draft-model-name", os.path.basename(DRAFT_DIR))):
            self.assertEqual(tabby[tabby.index(flag) + 1], value, flag)
        self.assertEqual(os.path.basename(MODEL_DIR), SERVED_ID)
        self.assertNotEqual(os.path.dirname(MODEL_DIR), os.path.dirname(DRAFT_DIR))

    def test_environment_pins_the_runtime_and_nothing_else_configures_tabby(self) -> None:
        env = self.snapshot["env"]
        self.assertEqual(env["TABBY_ROOT"], f"{RUNTIME}/src/tabbyAPI")
        self.assertEqual(env["PYTHONPATH"], f"{REPO}/scripts/lib")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(env["PATH"].startswith(f"{RUNTIME}/runtime/bin:"))
        self.assertEqual([key for key in env if key.upper().startswith("TABBY_")],
                         ["TABBY_ROOT"])

    def test_containment_and_safety(self) -> None:
        systemd = self.snapshot["systemd"]
        self.assertEqual(systemd["unit"], "mimo26flash-engine")
        properties = systemd["properties"]
        # exllamav3's CUDA memory is not charged to the unit; it holds about
        # 3-4 GB anon plus the pack's page cache. A low MemoryHigh bounds that
        # cache while the weights stream in: at 16G it reached 12-14 GiB and
        # took MemFree to 1.2 GiB, and the kernel refused a GPU allocation
        # (NVRM, smoke-2026-10-13-r2).
        self.assertEqual((properties["MemoryHigh"], properties["MemoryMax"]), ("8G", "16G"))
        self.assertEqual((properties["MemorySwapMax"], properties["OOMPolicy"],
                          properties["KillMode"], properties["User"]),
                         ("0", "kill", "control-group", "bmarti44"))
        self.assertEqual(systemd["server_log"],
                         "/home/dsv4/ds4-project/engine-switch/mimo26flash.server.log")
        safety = self.profile["safety"]
        self.assertEqual(safety["kill_floor_gib"], 8)
        self.assertLessEqual(safety["minimum_start_gib"], 115)
        self.assertGreaterEqual(safety["startup_timeout_seconds"], 1800)

    def test_engine_pin_matches_the_build_manifest(self) -> None:
        build = json.loads(BUILD.read_text())
        engine = self.model["engines"]["tabby-exl3-mimo-r001"]
        self.assertEqual(engine["binary_path"],
                         "{cache_root}/mimo-v26-flash/native-runtime-001/runtime/bin/python3.12")
        self.assertEqual(engine["binary_sha256"], build["runtime"]["python_sha256"])
        self.assertEqual(engine["build_manifest"],
                         "configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json")

    def test_digests_match_the_weights_manifest(self) -> None:
        weights = json.loads(WEIGHTS.read_text())
        self.assertIs(weights["all_verified"], True)
        checks = self.snapshot["digest_checks"]
        self.assertEqual(checks[0], {
            "path": f"{RUNTIME}/runtime/bin/python3.12",
            "sha256": self.model["engines"]["tabby-exl3-mimo-r001"]["binary_sha256"]})
        for role, directory in (("model", MODEL_DIR), ("draft_model", DRAFT_DIR)):
            name = self.profile["artifact_roles"][role]
            listed = {entry["name"]: entry["sha256"]
                      for entry in weights["artifacts"][name]["files"]}
            pinned = {os.path.basename(c["path"]): c["sha256"] for c in checks
                      if os.path.dirname(c["path"]) == directory}
            self.assertTrue(pinned, role)
            for file_name, digest in pinned.items():
                self.assertEqual(listed[file_name], digest, f"{role}/{file_name}")
        model_pins = {os.path.basename(c["path"]) for c in checks
                      if os.path.dirname(c["path"]) == MODEL_DIR}
        self.assertTrue({"config.json", "model.safetensors.index.json",
                         "quantization_config.json", "tokenizer.json",
                         "tokenizer_config.json", "chat_template.jinja",
                         "preprocessor_config.json"} <= model_pins)

    def test_weights_manifest_covers_each_served_artifact(self) -> None:
        weights = json.loads(WEIGHTS.read_text())
        for role, directory in (("model", MODEL_DIR), ("draft_model", DRAFT_DIR)):
            artifact = weights["artifacts"][self.profile["artifact_roles"][role]]
            self.assertEqual(
                artifact["local_root"].replace("{model_root}", "/home/bmarti44/models"),
                directory)
            files = artifact["files"]
            self.assertEqual(len({f["name"] for f in files}), len(files))
            self.assertEqual(sum(f["bytes"] for f in files), artifact["total_bytes"])
            for entry in files:
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                if entry["name"].endswith(".safetensors"):
                    self.assertIs(entry["lfs_verified"], True, entry["name"])
        # The served pack carries the official release's vision preprocessor
        # config (every EXL3 pack ships the stale pre-Oct-9 file).
        served = {f["name"]: f for f in weights["artifacts"][
            self.profile["artifact_roles"]["model"]]["files"]}
        official = {f["name"]: f for f in weights["artifacts"]["mopd-release"]["files"]}
        self.assertEqual(served["preprocessor_config.json"]["sha256"],
                         official["preprocessor_config.json"]["sha256"])
        self.assertEqual(served["preprocessor_config.json"]["source_repo"],
                         "XiaomiMiMo/MiMo-V2.6-Flash-MOPD")

    def test_gate_results_are_lint_exempt(self) -> None:
        patterns = json.loads(LINT_ALLOWLIST.read_text())["exempt_paths"]
        for path in ("results/mimo-v26-flash-gates/smoke-2026-10-13/summary.json",
                     "results/mimo-v26-flash-gates/smoke-2026-10-13/raw.jsonl",
                     "results/mimo-v26-flash-gates/qualify-x/kit/a.json"):
            self.assertTrue(any(fnmatch.fnmatch(path, p) for p in patterns), path)

    def test_host_lists_the_dev_port(self) -> None:
        host = json.loads(SPARK_HOST.read_text())
        self.assertEqual(host["ports"]["dev"][SLUG], 8019)


@unittest.skipUnless(HOST_BOUND, "host-bound: reads the local runtime and weights")
class OnThisHost(unittest.TestCase):
    def test_runtime_sources_match_the_build_manifest(self) -> None:
        build = json.loads(BUILD.read_text())
        for name in ("tabbyAPI", "exllamav3"):
            source = Path(RUNTIME) / "src" / name
            if not source.is_dir():
                self.skipTest(f"{source} is not present on this host")
            head = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD", "HEAD^{tree}"],
                check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(head, [build["sources"][name]["revision"],
                                    build["sources"][name]["tree"]], name)
            dirty = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain",
                 "--untracked-files=no"],
                check=True, capture_output=True, text=True).stdout
            self.assertEqual(dirty, "", name)

    def test_served_directories_hold_exactly_the_manifest_files(self) -> None:
        if not os.path.isdir(MODEL_DIR):
            self.skipTest("MiMo weights are not present on this host")
        weights = json.loads(WEIGHTS.read_text())
        profile = profile_resolver.load_profile(SLUG, PROFILE_FILE)
        for role, directory in (("model", MODEL_DIR), ("draft_model", DRAFT_DIR)):
            tabby_launch.check_model_parent(Path(directory))
            artifact = weights["artifacts"][profile["artifact_roles"][role]]
            self.assertEqual(sorted(os.listdir(directory)),
                             sorted(f["name"] for f in artifact["files"]), role)
            for entry in artifact["files"]:
                path = Path(directory) / entry["name"]
                self.assertEqual(path.stat().st_size, entry["bytes"], path)

    def test_pinned_files_verify_and_the_shim_environment_is_clean(self) -> None:
        if not os.path.isdir(MODEL_DIR):
            self.skipTest("MiMo weights are not present on this host")
        snapshot = rendered()
        for check in snapshot["digest_checks"]:
            self.assertEqual(sha256(Path(check["path"])), check["sha256"], check["path"])
        tabby_launch.check_environment(
            Path(snapshot["env"]["TABBY_ROOT"]), [Path(MODEL_DIR), Path(DRAFT_DIR)],
            snapshot["env"])


if __name__ == "__main__":
    unittest.main()
