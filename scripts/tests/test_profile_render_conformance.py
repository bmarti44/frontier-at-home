#!/usr/bin/env python3
"""Migrated profiles must render byte-identically to the captured launch truth.

scripts/tests/fixtures/profile-conformance/<alias>.json captures the exact
production launch commands of scripts/52_engine_switch.sh (proven equal to
the live script by test_fixture_extraction.py while that temporary test
exists). This test renders each migrated profile in configs/profiles/
against configs/hosts/spark-aba1.json and asserts the snapshot equals the
fixture: same binary, argv, env, systemd properties, and safety parameters.
Together the two tests guarantee profile JSON == production behavior.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver  # noqa: E402

FIXTURES = ROOT / "scripts/tests/fixtures/profile-conformance"
SPARK_HOST = ROOT / "configs/hosts/spark-aba1.json"

ALIAS_TO_PROFILE = {
    "dsv4": ("deepseek-v4-flash", "cuda-spark-128g-1m-fast.json"),
    "glm52": ("glm-5.2", "cuda-spark-128g.json"),
    "qwen38": ("qwen3.8-27b", "cuda-spark-128g.json"),
    "qwen38-1m": ("qwen3.8-27b", "cuda-spark-128g-1m.json"),
    "laguna": ("laguna-s-2.1", "cuda-spark-128g.json"),
}

# Fixture keys that are launch truth; anything else in the fixture is
# metadata about the fixture itself.
COMPARED_KEYS = (
    "mechanism", "binary", "argv", "env", "systemd",
    "memory_guard", "memwatch", "runuser", "delegate",
)


def render(alias: str) -> dict:
    model_slug, profile_file = ALIAS_TO_PROFILE[alias]
    host = profile_resolver.load_host(SPARK_HOST)
    model = profile_resolver.load_model(model_slug)
    profile = profile_resolver.load_profile(model_slug, profile_file)
    return profile_resolver.resolve(profile, model, host)


class RenderedProfilesMatchFixtures(unittest.TestCase):
    def test_every_alias_renders_to_its_fixture(self) -> None:
        for alias in sorted(ALIAS_TO_PROFILE):
            with self.subTest(alias=alias):
                with open(FIXTURES / f"{alias}.json", encoding="utf-8") as stream:
                    fixture = json.load(stream)
                snapshot = render(alias)
                self.assertEqual(snapshot["switch_alias"], alias)
                for key in COMPARED_KEYS:
                    if key in fixture:
                        self.assertEqual(
                            snapshot.get(key), fixture[key],
                            f"{alias}: {key} diverges from production truth",
                        )

    def test_aliases_map_one_to_one(self) -> None:
        seen: dict[str, str] = {}
        for model_slug, profile_file in profile_resolver.list_profiles():
            profile = profile_resolver.load_profile(model_slug, profile_file)
            alias = profile.get("switch_alias")
            if alias is None:
                continue
            self.assertNotIn(
                alias, seen,
                f"switch_alias {alias} claimed by both {seen.get(alias)} "
                f"and {model_slug}/{profile_file}",
            )
            seen[alias] = f"{model_slug}/{profile_file}"
        self.assertEqual(
            sorted(seen), ["dsv4", "glm52", "laguna", "qwen38", "qwen38-1m"]
        )


if __name__ == "__main__":
    unittest.main()
