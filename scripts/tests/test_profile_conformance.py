#!/usr/bin/env python3
"""Guard the tuned DSV4 agent-latency serving profile against silent reverts.

The 2026-07-27/28 commits (1878837, 81501fc) silently replaced the measured
fast-prefill profile (docs/speed-tuning-2026-07-23.md) with small buffers,
regressing a novel 19K-token agent request from ~59 s to ~90-100 s TTFT.
Nothing failed, because no test pinned the performance-critical values or kept
the serving surfaces in sync.

The authoritative copy now lives in the declarative profile
configs/profiles/deepseek-v4-flash/cuda-spark-128g-1m-fast.json
(docs/PROFILE-SCHEMA.md). This test pins the tuned values independently
(anti-revert) and asserts every surface that still carries a copy agrees:
the rendered profile, the legacy env file (until deleted), the systemd
unit, and the engine-switch launcher (until the switch launches from the
profile itself).

If this test fails after a memory-motivated edit: the correct fix is the
measured admission path (DSV4_MEASURED_HEADLESS_OVERHEAD_GIB in
scripts/21_serve_llamacpp.sh), not shrinking the buffers. See
docs/speed-tuning-2026-07-23.md.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import profile_resolver  # noqa: E402

PROFILE_JSON = ("deepseek-v4-flash", "cuda-spark-128g-1m-fast.json")
LEGACY_ENV = ROOT / "configs/profiles/dsv4-1m-fast.env"
SERVICE = ROOT / "configs/systemd/deepseek-v4-flash-llamacpp.service"
SWITCH = ROOT / "scripts/52_engine_switch.sh"
SPARK_HOST = ROOT / "configs/hosts/spark-aba1.json"

# The 1M+fast profile: million-token context with the measured fast-prefill
# buffers (ggerganov's canonical DGX Spark config, ~434 tok/s prefill vs ~208
# at ub=256). The context-scaled compute buffers at CTX=1M/ub=2048 leave a
# measured steady state of ~9.8 GiB free (memwatch BREACH 2026-08-01), so the
# owner accepted lowering the watchdog floor to 8 GiB — a deliberately
# thinner crash barrier, chosen over giving up either 1M context or prefill
# speed. The 12 GiB overhead charge is the measured non-weight, non-KV
# footprint (~11.3 GiB) at this exact configuration. Two slots (512k each)
# protect the agent conversation cache from utility-call eviction; a single
# request is therefore capped at 512k tokens.
EXPECTED = {
    "DSV4_UBATCH": "2048",
    "DSV4_BATCH": "2048",
    "DSV4_UBATCH_LARGE": "1",
    "CTX": "1048576",
    "DSV4_PARALLEL": "2",
    "DSV4_NO_MMAP": "1",
    "DSV4_SPEC_TYPE": "none",
    "DSV4_MEASURED_HEADLESS_OVERHEAD_GIB": "12",
    "DSV4_MEM_FLOOR_GIB": "8",
    "DSV4_WATCHDOG_FLOOR_GIB": "8",
    "DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB": "8",
}

ASSIGN_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(\S+)$")

REVERT_HINT = (
    "The tuned agent-latency profile appears to have been changed. If this is "
    "an OOM-motivated revert, fix the admission math instead (measured "
    "overhead path in scripts/21_serve_llamacpp.sh); see "
    "docs/speed-tuning-2026-07-23.md."
)


def rendered_profile_env() -> dict[str, str]:
    host = profile_resolver.load_host(SPARK_HOST)
    model_slug, profile_file = PROFILE_JSON
    model = profile_resolver.load_model(model_slug)
    profile = profile_resolver.load_profile(model_slug, profile_file)
    return profile_resolver.resolve(profile, model, host)["env"]


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


class ProfileConformanceTests(unittest.TestCase):
    def assert_expected(self, values: dict[str, str], label: str) -> None:
        for key, expected in EXPECTED.items():
            self.assertEqual(
                values.get(key), expected,
                f"{label}: {key} must be {expected}. {REVERT_HINT}",
            )

    def test_declarative_profile_matches_tuned_values(self):
        self.assert_expected(rendered_profile_env(), "rendered profile env")

    def test_legacy_env_file_matches_tuned_values(self):
        # The .env file is documentation-only and scheduled for deletion once
        # the refactored engine switch launches from the profile JSON; while
        # it exists it must not drift.
        self.assertTrue(LEGACY_ENV.is_file(), f"{LEGACY_ENV} missing. {REVERT_HINT}")
        self.assert_expected(parse_env_file(LEGACY_ENV), LEGACY_ENV.name)

    def test_systemd_unit_matches_profile(self):
        self.assert_expected(parse_service_environment(SERVICE), SERVICE.name)

    def test_engine_switch_matches_profile(self):
        launcher = parse_switch_launcher(SWITCH)
        if launcher:
            # Pre-cutover switch: literal env pairs in the launcher body.
            self.assert_expected(launcher, f"{SWITCH.name} dsv4_launcher()")
            return
        # Post-cutover switch: the launcher env renders from the dsv4
        # profile, whose rendered values are pinned above; assert the
        # switch actually launches from that profile.
        source = SWITCH.read_text(encoding="utf-8")
        self.assertIn(
            "configs/profiles/deepseek-v4-flash/cuda-spark-128g-1m-fast.json",
            source, REVERT_HINT,
        )
        self.assertIn("read_profile_array launcher_env dsv4 env", source,
                      REVERT_HINT)

    def test_launcher_admission_supports_fast_buffers(self):
        """The serve script must accept overhead=12 with ub<=2048, or the
        profile above cannot start (this is the admission-gate fix that makes
        the fast buffers safe under the 14 GiB qualification floor)."""
        source = (ROOT / "scripts/21_serve_llamacpp.sh").read_text(
            encoding="utf-8")
        self.assertIn(
            "DSV4_MEASURED_HEADLESS_OVERHEAD_GIB must be 0 or 3 or 5 or 12",
            source,
            "serve script must allow DSV4_MEASURED_HEADLESS_OVERHEAD_GIB=12",
        )
        self.assertIn(
            "measured headless overhead 12 requires CTX=1048576",
            source,
        )
        # The 5 GiB charge was measured at 64K context; the FA mask and other
        # context-scaled buffers make it unsafe at 1M (memwatch BREACH to
        # 9.79 GiB on 2026-08-01), so the launcher must scope it.
        self.assertIn(
            "measured headless overhead 5 requires CTX <= 131072",
            source,
        )
        self.assertIn(
            "DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB must be 0 or 8 or 14",
            source,
        )


if __name__ == "__main__":
    unittest.main()
