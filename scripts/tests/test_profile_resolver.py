#!/usr/bin/env python3
"""Contracts for the declarative profile resolver (docs/PROFILE-SCHEMA.md)."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver as resolver  # noqa: E402

SPARK_HOST_PATH = ROOT / "configs/hosts/spark-aba1.json"
CATALOG = ROOT / "models/catalog.json"


def spark_host() -> dict:
    return resolver.load_host(SPARK_HOST_PATH)


class CommittedProfilesValidate(unittest.TestCase):
    def test_every_committed_profile_loads_and_renders(self) -> None:
        host = spark_host()
        for model_slug, profile_file in resolver.list_profiles():
            with self.subTest(profile=f"{model_slug}/{profile_file}"):
                model = resolver.load_model(model_slug)
                profile = resolver.load_profile(model_slug, profile_file)
                if profile["backend"] not in host["backends"]:
                    continue  # off-host profile; rendering needs its host class
                snapshot = resolver.resolve(profile, model, host)
                self.assertEqual(snapshot["profile_id"], profile["profile_id"])
                for value in snapshot.get("argv", []):
                    self.assertNotRegex(
                        value, r"\{(model|mmproj|draft_model|binary|port|repo|"
                        r"model_root|cache_root|state_root|verb)\}",
                        "placeholder survived rendering",
                    )
                for value in snapshot["env"].values():
                    self.assertNotRegex(
                        value, r"\{(port|repo|model_root|cache_root|state_root)\}"
                    )

    def test_render_is_deterministic(self) -> None:
        host = spark_host()
        model = resolver.load_model("qwen3.8-27b")
        profile = resolver.load_profile("qwen3.8-27b", "cuda-spark-128g.json")
        first = resolver.resolve(profile, model, host)
        second = resolver.resolve(profile, model, host)
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )

    def test_profile_directories_match_catalog_slugs(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        slugs = {model["slug"] for model in catalog["models"]}
        for directory in sorted(resolver.PROFILES_ROOT.iterdir()):
            if directory.is_dir():
                self.assertIn(
                    directory.name, slugs,
                    f"configs/profiles/{directory.name} is not a catalog slug",
                )


class SchemaFailsClosed(unittest.TestCase):
    def load_valid(self) -> dict:
        return resolver.load_profile("qwen3.8-27b", "cuda-spark-128g.json")

    def test_unknown_top_level_key_rejected(self) -> None:
        profile = self.load_valid()
        profile["surprise"] = 1
        with self.assertRaises(resolver.ProfileError):
            resolver._require_keys(
                profile, resolver.PROFILE_KEYS, set(), "synthetic"
            )

    def test_partial_base_is_not_servable(self) -> None:
        with self.assertRaises(resolver.ProfileError):
            resolver.load_profile("qwen3.8-27b", "_base-cuda-spark.json")

    def test_unknown_placeholder_is_a_hard_error(self) -> None:
        with self.assertRaises(resolver.ProfileError):
            resolver._substitute("{mystery_token}", {}, "synthetic")

    def test_missing_placeholder_value_is_a_hard_error(self) -> None:
        with self.assertRaises(resolver.ProfileError):
            resolver._substitute("{draft_model}", {}, "synthetic")

    def test_json_chat_template_kwargs_are_not_placeholders(self) -> None:
        rendered = resolver._substitute(
            '{"reasoning_effort":"low"}', {}, "synthetic"
        )
        self.assertEqual(rendered, '{"reasoning_effort":"low"}')

    def test_unsupported_backend_fails_with_reason(self) -> None:
        host = spark_host()
        model = resolver.load_model("glm-5.2")
        profile = resolver.load_profile("glm-5.2", "cuda-spark-128g.json")
        broken = copy.deepcopy(profile)
        broken["backend"] = "apple-silicon"
        with self.assertRaises(resolver.ProfileError) as context:
            resolver.resolve(broken, model, host)
        self.assertIn("CUDA kernel patches", str(context.exception))


class FeasibilityMath(unittest.TestCase):
    def test_qwen_1m_rejected_on_16_gib(self) -> None:
        model = resolver.load_model("qwen3.8-27b")
        profile = resolver.load_profile("qwen3.8-27b", "cuda-spark-128g-1m.json")
        mac = copy.deepcopy(profile)
        mac["hardware_class"] = "mac"
        fit = resolver.feasibility(mac, model, 16)
        self.assertEqual(fit["verdict"], "infeasible")

    def test_qwen_1m_admitted_on_128_gib_spark(self) -> None:
        model = resolver.load_model("qwen3.8-27b")
        profile = resolver.load_profile("qwen3.8-27b", "cuda-spark-128g-1m.json")
        fit = resolver.feasibility(profile, model, 128)
        self.assertIn(fit["verdict"], {"feasible", "estimated-tight"})

    def test_streaming_engine_charges_resident_footprint_only(self) -> None:
        model = resolver.load_model("glm-5.2")
        profile = resolver.load_profile("glm-5.2", "cuda-spark-128g.json")
        fit = resolver.feasibility(profile, model, 128)
        self.assertEqual(fit["weights_gib"], 0.0)
        self.assertNotEqual(fit["verdict"], "infeasible")

    def test_mac_usable_memory_uses_wired_limit(self) -> None:
        self.assertEqual(resolver.usable_gib("mac", 32, 8), 24)
        self.assertEqual(resolver.usable_gib("mac", 128, 8), 96)
        self.assertEqual(resolver.usable_gib("spark", 128, 8), 114)
        self.assertEqual(resolver.usable_gib("any", 64, 8), 56)
        self.assertIsNone(resolver.usable_gib("dgpu", 64, 8))


class HostSelection(unittest.TestCase):
    def test_explicit_path_wins(self) -> None:
        host = resolver.load_host(SPARK_HOST_PATH)
        self.assertEqual(host["host_id"], "spark-aba1")

    def test_missing_host_fails_closed(self) -> None:
        with self.assertRaises(resolver.ProfileError):
            resolver.load_host("/nonexistent/host.json")


if __name__ == "__main__":
    unittest.main()
