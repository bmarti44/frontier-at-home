#!/usr/bin/env python3
"""Guard the tuned DSV4 agent-latency serving profile against silent reverts.

The 2026-07-27/28 commits (1878837, 81501fc) silently replaced the measured
fast-prefill profile (docs/speed-tuning-2026-07-23.md) with small buffers,
regressing a novel 19K-token agent request from ~59 s to ~90-100 s TTFT.
Nothing failed, because no test pinned the performance-critical values or kept
the systemd unit and the engine-switch launcher in sync.

If this test fails after a memory-motivated edit: the correct fix is the
measured admission path (DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=5 in
scripts/21_serve_llamacpp.sh), not shrinking the buffers. See
docs/speed-tuning-2026-07-23.md and the plan notes in the unit file comments.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/profiles/dsv4-1m-fast.env"
SERVICE = ROOT / "configs/systemd/deepseek-v4-flash-llamacpp.service"
SWITCH = ROOT / "scripts/52_engine_switch.sh"

# The qualified 1M+fast profile: million-token context with the measured
# fast-prefill buffers (ggerganov's canonical DGX Spark config, ~434 tok/s
# prefill vs ~208 at ub=256).
EXPECTED = {
    "DSV4_UBATCH": "2048",
    "DSV4_BATCH": "2048",
    "DSV4_UBATCH_LARGE": "1",
    "CTX": "1048576",
    "DSV4_PARALLEL": "1",
    "DSV4_NO_MMAP": "1",
    "DSV4_SPEC_TYPE": "none",
    "DSV4_MEASURED_HEADLESS_OVERHEAD_GIB": "5",
    "DSV4_MEM_FLOOR_GIB": "14",
    "DSV4_WATCHDOG_FLOOR_GIB": "14",
    "DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB": "14",
}

ASSIGN_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(\S+)$")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = ASSIGN_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_service_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("Environment="):
            continue
        match = ASSIGN_RE.match(line[len("Environment="):])
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_switch_launcher(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    match = re.search(r"dsv4_launcher\(\)\s*\{(.*?)\n\}", source, re.DOTALL)
    if not match:
        return {}
    values: dict[str, str] = {}
    for token in re.findall(r"([A-Z][A-Z0-9_]*)=(\S+)", match.group(1)):
        values.setdefault(token[0], token[1].rstrip("\\").strip())
    return values


REVERT_HINT = (
    "The tuned agent-latency profile appears to have been changed. If this is "
    "an OOM-motivated revert, fix the admission math instead (measured "
    "overhead path in scripts/21_serve_llamacpp.sh); see "
    "docs/speed-tuning-2026-07-23.md."
)


class ProfileConformanceTests(unittest.TestCase):
    def test_profile_file_exists(self):
        self.assertTrue(PROFILE.is_file(), f"{PROFILE} missing. {REVERT_HINT}")

    def test_profile_file_matches_tuned_values(self):
        values = parse_env_file(PROFILE)
        for key, expected in EXPECTED.items():
            self.assertEqual(
                values.get(key), expected,
                f"{PROFILE.name}: {key} must be {expected}. {REVERT_HINT}",
            )

    def test_systemd_unit_matches_profile(self):
        unit = parse_service_environment(SERVICE)
        for key, expected in EXPECTED.items():
            self.assertEqual(
                unit.get(key), expected,
                f"{SERVICE.name}: Environment={key} must be {expected}. "
                f"{REVERT_HINT}",
            )

    def test_engine_switch_matches_profile(self):
        launcher = parse_switch_launcher(SWITCH)
        self.assertTrue(launcher, f"dsv4_launcher() not found in {SWITCH}")
        for key, expected in EXPECTED.items():
            self.assertEqual(
                launcher.get(key), expected,
                f"{SWITCH.name} dsv4_launcher(): {key} must be {expected}. "
                f"{REVERT_HINT}",
            )

    def test_launcher_admission_supports_fast_buffers(self):
        """The serve script must accept overhead=5 with ub<=2048, or the
        profile above cannot start (this is the admission-gate fix that makes
        the fast buffers safe under the 14 GiB qualification floor)."""
        source = (ROOT / "scripts/21_serve_llamacpp.sh").read_text(
            encoding="utf-8")
        self.assertIn(
            "DSV4_MEASURED_HEADLESS_OVERHEAD_GIB must be 0 or 3 or 5",
            source,
            "serve script must allow DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=5",
        )
        self.assertIn(
            "measured headless overhead 5 requires DSV4_UBATCH <= 2048",
            source,
        )


if __name__ == "__main__":
    unittest.main()
