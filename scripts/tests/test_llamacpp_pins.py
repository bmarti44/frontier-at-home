#!/usr/bin/env python3
"""llama.cpp pin registry (configs/pins/llamacpp.json) and its builder.

Every llama.cpp engine a profile can launch from {cache_root} is built from
one registered pin by scripts/14_build_llamacpp_pin.sh. Off-Spark profiles
on upstream llama.cpp share the registry's portable pin; Spark-qualified
engines keep the exact pin their qualification evidence was measured on.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver as resolver  # noqa: E402

REGISTRY = ROOT / "configs/pins/llamacpp.json"
BUILDER = ROOT / "scripts/14_build_llamacpp_pin.sh"
UPSTREAM = "https://github.com/ggml-org/llama.cpp"
PIN_DIR = re.compile(r"^\{cache_root\}/(llamacpp-[^/]+)/src/build/bin/llama-server$")


def registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def engine_pin(engine: dict) -> str | None:
    match = PIN_DIR.match(engine.get("binary_path", ""))
    return match.group(1) if match else None


class RegistryShape(unittest.TestCase):
    def test_pins_are_immutable_references(self) -> None:
        pins = registry()["pins"]
        self.assertTrue(pins)
        for pin_id, pin in pins.items():
            with self.subTest(pin=pin_id):
                self.assertRegex(pin_id, r"^llamacpp-[a-z0-9.-]+$")
                self.assertTrue(pin["repository"].startswith("https://github.com/"))
                self.assertRegex(pin["commit"], r"^[0-9a-f]{40}$")
                self.assertEqual(len([k for k in ("tag", "branch") if pin.get(k)]), 1,
                                 "exactly one of tag/branch names the ref")
                self.assertTrue(pin["rationale"])

    def test_portable_pin_is_registered_upstream(self) -> None:
        data = registry()
        portable = data["portable"]
        self.assertIn(portable, data["pins"])
        self.assertEqual(data["pins"][portable]["repository"], UPSTREAM)


class ProfilesUseRegisteredPins(unittest.TestCase):
    def test_every_cache_root_llamacpp_engine_is_registered(self) -> None:
        pins = registry()["pins"]
        for model_slug in sorted({slug for slug, _ in resolver.list_profiles()}):
            for name, engine in resolver.load_model(model_slug)["engines"].items():
                pin = engine_pin(engine)
                if pin is None:
                    continue
                with self.subTest(engine=f"{model_slug}:{name}"):
                    self.assertIn(pin, pins)

    def test_off_spark_upstream_profiles_use_the_portable_pin(self) -> None:
        data = registry()
        for model_slug, profile_file in resolver.list_profiles():
            profile = resolver.load_profile(model_slug, profile_file)
            if profile.get("hardware_class") == "spark" or "engine" not in profile:
                continue
            engine = resolver.load_model(model_slug)["engines"][profile["engine"]]
            pin = engine_pin(engine)
            if pin is None or data["pins"][pin]["repository"] != UPSTREAM:
                continue  # forks (Laguna) carry their own pin
            with self.subTest(profile=f"{model_slug}/{profile_file}"):
                self.assertEqual(pin, data["portable"])

    def test_spark_build_manifests_match_their_pin(self) -> None:
        pins = registry()["pins"]
        for model_slug in sorted({slug for slug, _ in resolver.list_profiles()}):
            for name, engine in resolver.load_model(model_slug)["engines"].items():
                pin, manifest = engine_pin(engine), engine.get("build_manifest")
                if pin is None or not manifest:
                    continue
                with self.subTest(engine=f"{model_slug}:{name}"):
                    built = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
                    self.assertEqual(built["commit"], pins[pin]["commit"])


class BuilderPlan(unittest.TestCase):
    def plan(self, *args: str) -> dict:
        result = subprocess.run(["bash", str(BUILDER), "--dry-run", *args],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_dry_run_resolves_pin_paths_and_flags(self) -> None:
        portable = registry()["portable"]
        plan = self.plan("--pin", portable, "--host-class", "vulkan")
        self.assertEqual(plan["commit"], registry()["pins"][portable]["commit"])
        self.assertTrue(plan["build_directory"].endswith(f"/.cache/{portable}/src/build"))
        self.assertTrue(plan["manifest"].endswith(
            f"configs/build-manifests/{portable}-vulkan.json"))
        self.assertIn("-DGGML_VULKAN=ON", plan["cmake_flags"])

    def test_spark_class_keeps_the_unsuffixed_manifest(self) -> None:
        plan = self.plan("--pin", "llamacpp-qwen38-9d77fa17", "--host-class", "cuda-spark")
        self.assertTrue(plan["manifest"].endswith(
            "configs/build-manifests/llamacpp-qwen38-9d77fa17.json"))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121", plan["cmake_flags"])

    def test_unknown_pin_fails_closed(self) -> None:
        result = subprocess.run(["bash", str(BUILDER), "--dry-run", "--pin",
                                 "llamacpp-nope", "--host-class", "cpu"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown pin", result.stderr)


if __name__ == "__main__":
    unittest.main()
