#!/usr/bin/env python3
"""Render conformance for the profile-rendering engine switch.

scripts/52_engine_switch.sh renders launch argv, env, and containment from
configs/profiles/ (docs/PROFILE-SCHEMA.md). Its test-only `render` verb must
assemble, for every switch alias, exactly the launch snapshot captured in
scripts/tests/fixtures/profile-conformance/ — the production truth recorded
before the refactor (with the fixture harness's test-root paths mapped back
to production paths). Together with test_profile_render_conformance.py
(profiles render to the same fixtures) this closes the loop:
profile JSON == switch assembly == captured production behavior.

Historical note: before the cutover this module targeted the staged
52_engine_switch_next.sh and additionally re-ran the whole lifecycle suite
against it; the lifecycle suite now runs against the production script
directly in test_engine_switch.py, which also owns the shared render
helpers used here.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest

import test_engine_switch as base

FIXTURES = base.FIXTURES
ALIASES = base.SWITCH_ALIASES


class RenderMatchesFixtures(unittest.TestCase):
    COMPARED_KEYS = (
        "mechanism", "binary", "argv", "env", "systemd", "runuser", "delegate",
    )

    def test_every_alias_assembles_to_its_fixture(self) -> None:
        for alias in ALIASES:
            with self.subTest(alias=alias):
                with open(FIXTURES / f"{alias}.json", encoding="utf-8") as stream:
                    fixture = json.load(stream)
                snapshot = base.render_switch_snapshot(alias)
                self.assertEqual(snapshot["alias"], alias)
                for key in self.COMPARED_KEYS:
                    if key in fixture:
                        self.assertEqual(
                            snapshot.get(key), fixture[key],
                            f"{alias}: assembled {key} diverges from the "
                            f"captured production truth",
                        )

    def test_render_requires_test_mode(self) -> None:
        result = subprocess.run(
            ["bash", str(base.SCRIPT), "render", "qwen38"],
            cwd=base.ROOT,
            env={"PATH": os.environ["PATH"]},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test-only verb", result.stderr)


if __name__ == "__main__":
    unittest.main()
