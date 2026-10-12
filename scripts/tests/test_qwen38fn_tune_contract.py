"""Qwen3.8-Flash-Next lossless speed-lever screen: runtime and arm contract.

Owner decision 2026-10-11: tune Flash-Next with lossless levers only (no FP8
side-layer cast). Runtime native-runtime-002 is runtime-001 (vLLM 72d59adc2 +
PR #58439) plus Python-only patches committed under patches/qwen38fn/. Each
speed lever sits behind a startup switch that stays off unless an arm profile
sets it:

  L1  QWEN38FN_PLE_WILLNEED=1: madvise(MADV_WILLNEED) the decode step's PLE
      rows before faulting them, lookup started at its own layer
      (jschmied/vllm#4 on #58439).
  L2  QWEN4EXP_DRAFT_VOCAB=65536 (the drafter scores token ids [0, 65536))
      plus an NVFP4 draft lm_head (vLLM #59973), draft side only.
  L3  k=4 at one running request, k=3 at two to four.
  L4  BF16 KV for the MTP layer (--kv-cache-dtype-skip-layers 48, in the
      nightly as #60023), with a larger arena to keep 1M tokens.

The arm profiles extend _tune.json and each differs from the qualified main
profile by exactly its declared levers. None is a switch alias: the switch keeps
serving the main profile on runtime-001 throughout the screen.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver

SLUG = "qwen3.8-flash-next"
MAIN = "cuda-spark-128g-1m.json"
HOST_BOUND = os.environ.get("FRONTIER_HOST_BOUND_TESTS", "1") != "0"
MANIFEST = ROOT / "configs/build-manifests/qwen38-flash-next-runtime-002.json"
PATCH_DIR = ROOT / "patches/qwen38fn"
RUNTIME_002 = "/home/bmarti44/.cache/qwen38-flash-next/native-runtime-002"
RUNTIME_001_MANIFEST = ROOT / "configs/build-manifests/qwen38-flash-next-runtime.json"
SERIES = [
    "0001-vllm-pr58439-fa537316.runtime.diff",
    "0002-vllm-pr59528.diff",
    "0003-vllm-pr58449.diff",
    "0004-vllm-pr58737.diff",
    "0005-jschmied-vllm-pr4.diff",
    "0006-local-gate-ple-willneed.diff",
    "0007-salient-0002-draft-vocab.diff",
    "0008-salient-0003-pr59973.diff",
    "0009-local-draft-vocab-init.diff",
]
ARENA_L4 = 20_600_000_000
LEVERS = {
    "L1": {"env": {"QWEN38FN_PLE_WILLNEED": "1"}},
    "L2": {"env": {"QWEN4EXP_DRAFT_VOCAB": "65536"},
           "spec": {"draft_lm_head_quantization": "nvfp4"}},
    "L3": {"spec": {"num_speculative_tokens": 4,
                    "num_speculative_tokens_per_batch_size": [[1, 1, 4], [2, 4, 3]]}},
    "L4": {"args": ["--kv-cache-dtype-skip-layers", "48"], "arena": ARENA_L4},
}
ARMS = {
    "a1": [],
    "a2": ["L1"],
    "a3": ["L1", "L2"],
    "a4": ["L1", "L2", "L3"],
    "a5": ["L1", "L2", "L3", "L4"],
    # Amended arm: the screen dropped L3 (a4 held 1,033,876 KV tokens at a
    # 3,216-token block, short of 1M), so L4 is screened on a3, the last kept arm.
    "a5b": ["L1", "L2", "L4"],
}


def arm_file(arm: str) -> str:
    return f"cuda-spark-128g-1m-tune-{arm}.json"


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def without(argv: list[str], flags: set[str]) -> list[str]:
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in flags:
            skip = True
            continue
        out.append(item)
    return out


def diff_paths(text: str) -> list[str]:
    return re.findall(r"^diff --git a/(\S+) b/", text, flags=re.M)


def tree_sha256(site_packages: Path) -> tuple[str, int]:
    root = site_packages / "vllm"
    rows = []
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in rel.split("/") or rel.endswith(".pyc"):
                continue
            rows.append(f"{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    rows.sort()
    return hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows)


class TuneArmContract(unittest.TestCase):
    def render(self, profile_file: str) -> dict:
        profile = profile_resolver.load_profile(SLUG, profile_file)
        return profile_resolver.resolve(
            profile, profile_resolver.load_model(SLUG),
            profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json"))

    def test_every_arm_is_the_main_launch_plus_exactly_its_levers(self):
        main = self.render(MAIN)
        main_spec = json.loads(flag_value(main["argv"], "--speculative-config"))
        for arm, levers in ARMS.items():
            with self.subTest(arm=arm):
                got = self.render(arm_file(arm))
                self.assertEqual(
                    got["binary"], RUNTIME_002 + "/runtime/bin/python3.12")

                env = {k: v.replace("native-runtime-001", "native-runtime-002")
                       for k, v in main["env"].items()}
                spec = dict(main_spec)
                arena = int(flag_value(main["argv"], "--kv-cache-memory-bytes"))
                extra_args: list[str] = []
                for lever in levers:
                    env.update(LEVERS[lever].get("env", {}))
                    spec.update(LEVERS[lever].get("spec", {}))
                    extra_args += LEVERS[lever].get("args", [])
                    arena = LEVERS[lever].get("arena", arena)
                self.assertEqual(got["env"], env)
                self.assertEqual(
                    json.loads(flag_value(got["argv"], "--speculative-config")), spec)
                self.assertEqual(
                    int(flag_value(got["argv"], "--kv-cache-memory-bytes")), arena)
                varying = {"--speculative-config", "--kv-cache-memory-bytes",
                           "--kv-cache-dtype-skip-layers"}
                self.assertEqual(without(got["argv"], varying),
                                 without(main["argv"], varying))
                if extra_args:
                    self.assertEqual(
                        flag_value(got["argv"], extra_args[0]), extra_args[1])
                else:
                    self.assertNotIn("--kv-cache-dtype-skip-layers", got["argv"])
                self.assertEqual(got["diagnostics_unset"], main["diagnostics_unset"])

    def test_arms_are_estimated_dev_profiles_the_switch_cannot_serve(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                p = profile_resolver.load_profile(SLUG, arm_file(arm))
                self.assertEqual(p["status"]["state"], "estimated")
                self.assertNotIn("switch_alias", p)
                self.assertNotIn("switch", p)
                self.assertEqual(p["engine"], "vllm-qwen38fn-r002")
                self.assertEqual(p["profile_id"], f"{SLUG}/{arm_file(arm)[:-5]}")

    def test_bench_block_size_follows_the_largest_k(self):
        # 30_bench_speed_blocks.py subtracts K = k + 1 tokens per event.
        for arm, levers in ARMS.items():
            with self.subTest(arm=arm):
                p = profile_resolver.load_profile(SLUG, arm_file(arm))
                want = 5 if "L3" in levers else 4
                self.assertEqual(
                    p["qualification_options"]["speed_max_tokens_per_event"], want)

    def test_memory_model_counts_the_larger_l4_arena(self):
        main = profile_resolver.load_profile(SLUG, MAIN)["memory_model"]
        for arm, levers in ARMS.items():
            with self.subTest(arm=arm):
                mm = profile_resolver.load_profile(SLUG, arm_file(arm))["memory_model"]
                for key in ("resident_weights_gib", "overhead_gib", "floor_gib",
                            "kv_bytes_per_token"):
                    self.assertEqual(mm[key], main[key])
                if "L4" in levers:
                    self.assertGreaterEqual(mm["extra_gib"], ARENA_L4 / 2**30)
                    self.assertLess(mm["extra_gib"], ARENA_L4 / 2**30 + 0.1)
                else:
                    self.assertEqual(mm["extra_gib"], main["extra_gib"])

    def test_main_profile_still_serves_runtime_001(self):
        main = self.render(MAIN)
        self.assertIn("native-runtime-001", main["binary"])
        self.assertEqual(main["status"], "qualified")

    def test_model_json_pins_the_r002_engine(self):
        model = profile_resolver.load_model(SLUG)
        r1 = model["engines"]["vllm-qwen38fn-r001"]
        r2 = model["engines"]["vllm-qwen38fn-r002"]
        self.assertEqual(r2["binary_path"],
                         "{cache_root}/qwen38-flash-next/native-runtime-002/runtime/bin/python3.12")
        # Same interpreter bytes: the patches live in site-packages, which the
        # build manifest's vllm_tree digest covers.
        self.assertEqual(r2["binary_sha256"], r1["binary_sha256"])
        self.assertEqual(r2["build_manifest"],
                         "configs/build-manifests/qwen38-flash-next-runtime-002.json")


class Runtime002Manifest(unittest.TestCase):
    def manifest(self) -> dict:
        return json.loads(MANIFEST.read_text())

    def test_series_is_committed_and_pinned_in_order(self):
        m = self.manifest()
        self.assertEqual([p["file"] for p in m["patches"]],
                         [f"patches/qwen38fn/{name}" for name in SERIES])
        for entry in m["patches"]:
            with self.subTest(patch=entry["file"]):
                data = (ROOT / entry["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        # 0001 is runtime-001's applied #58439 diff, byte for byte.
        r1 = json.loads(RUNTIME_001_MANIFEST.read_text())
        self.assertEqual(m["patches"][0]["sha256"],
                         r1["patches"][0]["applied_diff"]["sha256"])

    def test_patches_touch_only_python_under_vllm(self):
        # No csrc: runtime-002 needs no CUDA build.
        for name in SERIES:
            with self.subTest(patch=name):
                paths = diff_paths((PATCH_DIR / name).read_text())
                self.assertTrue(paths)
                for path in paths:
                    self.assertTrue(path.startswith("vllm/") and path.endswith(".py"), path)

    def test_manifest_records_base_runtime_and_tree_method(self):
        m = self.manifest()
        self.assertEqual(m["runtime_root"], "{cache_root}/qwen38-flash-next/native-runtime-002/runtime")
        self.assertEqual(m["base_runtime"]["build_manifest"],
                         "configs/build-manifests/qwen38-flash-next-runtime.json")
        self.assertEqual(m["vllm"]["commit"], "72d59adc2c76de8f77aad640c7a04d663a970735")
        self.assertRegex(m["vllm_tree"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("relpath", m["vllm_tree"]["method"])
        self.assertEqual(m["cuda_compile"], "none")

    @unittest.skipUnless(HOST_BOUND, "needs the local runtime-002 tree")
    def test_runtime_002_tree_matches_the_manifest(self):
        m = self.manifest()
        site = Path(RUNTIME_002) / "runtime/lib/python3.12/site-packages"
        digest, files = tree_sha256(site)
        self.assertEqual(digest, m["vllm_tree"]["sha256"])
        self.assertEqual(files, m["vllm_tree"]["files"])


class SwitchesResolveAtStartup(unittest.TestCase):
    """AGENTS.md: switches resolve once at initialization, never per token."""

    def source(self, rel: str) -> str:
        site = Path(RUNTIME_002) / "runtime/lib/python3.12/site-packages"
        return (site / rel).read_text()

    @staticmethod
    def env_reads(tree: ast.AST) -> list[ast.AST]:
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and n.attr == "environ"]

    @unittest.skipUnless(HOST_BOUND, "needs the local runtime-002 tree")
    def test_ple_switch_is_read_once_at_module_scope(self):
        tree = ast.parse(self.source("vllm/models/qwen4_exp/nvidia/ple_pageable.py"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            self.assertFalse(self.env_reads(fn), f"environ read inside {fn.name}")
        names = [t.id for n in tree.body if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name)]
        self.assertIn("PLE_WILLNEED", names)

    @unittest.skipUnless(HOST_BOUND, "needs the local runtime-002 tree")
    def test_draft_vocab_is_read_only_while_building_the_drafter(self):
        tree = ast.parse(self.source("vllm/models/qwen4_exp/nvidia/mtp.py"))
        callers = []
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "_draft_vocab_size"):
                    callers.append(fn.name)
            if fn.name in {"compute_logits", "_reduced_compute_logits",
                           "_reduced_vocab_logits", "_draft_rows"}:
                self.assertFalse(self.env_reads(fn), f"environ read inside {fn.name}")
        self.assertEqual(callers, ["__init__"])


if __name__ == "__main__":
    unittest.main()
