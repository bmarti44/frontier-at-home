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


def mac_host() -> dict:
    """The example Mac host with concrete paths (the resolver never expands {HOME})."""
    host = resolver.load_host(ROOT / "configs/hosts/example-macbook-32g.json")
    host["ram_gib"] = 192
    host["paths"] = {
        key: value.replace("{HOME}", "/Users/someone")
        for key, value in host["paths"].items()
    }
    return host


class AppleSiliconProfilesRender(unittest.TestCase):
    """The Spark-host render test skips these; render them on a Mac host."""

    def apple_profiles(self):
        for model_slug, profile_file in resolver.list_profiles():
            profile = resolver.load_profile(model_slug, profile_file)
            if profile["backend"] == "apple-silicon":
                yield model_slug, profile_file, profile

    def test_every_apple_silicon_profile_renders_on_a_mac_host(self) -> None:
        host = mac_host()
        rendered = 0
        for model_slug, profile_file, profile in self.apple_profiles():
            with self.subTest(profile=f"{model_slug}/{profile_file}"):
                snapshot = resolver.resolve(
                    profile, resolver.load_model(model_slug), host
                )
                for value in snapshot["argv"]:
                    self.assertNotIn("{", value.replace('{"', ""),
                                     "placeholder survived rendering")
                rendered += 1
        self.assertGreater(rendered, 0)

    def test_qwen38_27b_mac_weights_are_digest_checked(self) -> None:
        # check/93 skip artifacts without a sha256 silently, so a Mac would
        # serve an unverified Q8/Q6_K file.
        host = mac_host()
        model = resolver.load_model("qwen3.8-27b")
        for model_slug, profile_file, profile in self.apple_profiles():
            if model_slug != "qwen3.8-27b":
                continue
            with self.subTest(profile=profile_file):
                snapshot = resolver.resolve(profile, model, host)
                checked = {check["path"] for check in snapshot["digest_checks"]}
                for role in profile["artifact_roles"]:
                    artifact = model["artifacts"][profile["artifact_roles"][role]]
                    path = artifact["path"].replace(
                        "{model_root}", host["paths"]["model_root"]
                    )
                    self.assertIn(path, checked, f"{role} is not digest-checked")


class HostSelection(unittest.TestCase):
    def test_explicit_path_wins(self) -> None:
        host = resolver.load_host(SPARK_HOST_PATH)
        self.assertEqual(host["host_id"], "spark-aba1")

    def test_missing_host_fails_closed(self) -> None:
        with self.assertRaises(resolver.ProfileError):
            resolver.load_host("/nonexistent/host.json")


if __name__ == "__main__":
    unittest.main()


class Ds4ServingTopology(unittest.TestCase):
    """ds4 profiles declare slots x per-session ctx without vLLM flags."""

    def profile(self) -> dict:
        return resolver.load_profile("deepseek-v4.1-flash", "cuda-spark-128g-1m.json")

    def test_committed_ds4_topology_is_valid(self) -> None:
        resolver.validate_serving_topology(self.profile())

    def mutated(self, **changes) -> dict:
        profile = copy.deepcopy(self.profile())
        args = profile["launch"]["args"]
        for flag, value in changes.items():
            flag = {"ctx": "-c", "sessions": "--batched-session"}[flag]
            if value is None:
                index = args.index(flag)
                del args[index:index + 2]
            else:
                args[args.index(flag) + 1] = value
        return profile

    def test_ctx_disagreeing_with_request_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.mutated(ctx="1048576"))

    def test_missing_batched_session_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.mutated(sessions=None))

    def test_wrong_session_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.mutated(sessions="2"))

    def test_aggregate_mismatch_is_rejected(self) -> None:
        profile = self.profile()
        profile["serving"]["parallel_slots"] = 2
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(profile)

    def test_vllm_profiles_keep_the_glm53_contract(self) -> None:
        profile = resolver.load_profile("glm-5.3-flash", "cuda-spark-128g-1m.json")
        resolver.validate_serving_topology(profile)
        broken = copy.deepcopy(profile)
        broken["serving"].pop("max_images")
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(broken)


class TabbyServingTopology(unittest.TestCase):
    """TabbyAPI (exllamav3) profiles: --max-seq-len is the per-request cap,
    --max-batch-size the slot count, and --cache-size the explicit KV pool,
    which must hold every slot's full context in 256-token pages."""

    ARGS = ["-B", "-m", "tabby_launch", "--max-seq-len", "262144",
            "--max-batch-size", "1", "--cache-size", "262144"]

    def profile(self, args=None, slots=1, cap=262144, context_cap=262144) -> dict:
        return {"launch": {"args": list(self.ARGS if args is None else args)},
                "serving": {"parallel_slots": slots, "request_context_cap": cap},
                "context_cap": context_cap}

    def mutated(self, flag, value) -> list:
        args = list(self.ARGS)
        index = args.index(flag)
        if value is None:
            del args[index:index + 2]
        else:
            args[index + 1] = value
        return args

    def test_matching_topology_is_valid(self) -> None:
        resolver.validate_serving_topology(self.profile())
        resolver.validate_serving_topology(self.profile(
            args=["--max-seq-len", "65536", "--max-batch-size", "4",
                  "--cache-size", "262144"],
            slots=4, cap=65536))

    def test_max_seq_len_disagreeing_with_request_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(
                self.profile(args=self.mutated("--max-seq-len", "131072")))

    def test_batch_size_must_be_explicit_and_match_slots(self) -> None:
        for value in (None, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolver.validate_serving_topology(
                        self.profile(args=self.mutated("--max-batch-size", value)))

    def test_cache_size_must_be_explicit_cover_all_slots_and_page_aligned(self) -> None:
        for value in (None, "131072", "262300", "262145", "2e5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolver.validate_serving_topology(
                        self.profile(args=self.mutated("--cache-size", value)))

    def test_aggregate_mismatch_and_equals_forms_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.profile(context_cap=1048576))
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.profile(
                args=["--max-seq-len=262144", "--max-batch-size", "1",
                      "--cache-size", "262144"]))
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(self.profile(
                args=self.ARGS + ["--max-seq-len", "262144"]))

    def test_tabby_profile_without_a_serving_block_is_rejected(self) -> None:
        profile = self.profile()
        del profile["serving"]
        with self.assertRaises(ValueError):
            resolver.validate_serving_topology(profile)
