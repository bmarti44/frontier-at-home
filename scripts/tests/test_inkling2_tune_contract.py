"""Inkling 2 Mini lever screen arms (DIAGNOSTIC): each arm is the 1M carrier plus one lever.

results/inkling2-gates/tune-screen-2026-11-23/PREREG.md screens each lossless
lever against a0 (the unchanged cuda-spark-128g-1m profile) in its own fresh
start. These checks pin every arm profile to exactly its lever, so a startup
difference between two arms can only come from that lever, and pin arms.json
to the profiles it names.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import profile_resolver  # noqa: E402

SLUG = "inkling-2-mini"
BASE = "cuda-spark-128g-1m"
SCREEN = ROOT / "results/inkling2-gates/tune-screen-2026-11-23"
DSPARK = '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
ARMS = {
    # arm: (profile leaf, lever)
    "a0": (BASE, None),
    "a1": (f"{BASE}-tune-b4096", "batch4096"),
    "a2": (f"{BASE}-tune-b8192", "batch8192"),
    "a3": (f"{BASE}-tune-pow2", "max_ctas_pow2"),
    "a4": (f"{BASE}-tune-dspark", "dspark7"),
    "a5": (f"{BASE}-tune-humming", "moe_humming"),
}


def set_arg(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = value


def insert_before_middleware(argv: list[str], *items: str) -> None:
    at = argv.index("--middleware")
    argv[at:at] = list(items)


def expected(base: dict, lever: str) -> dict:
    """The base profile with exactly one lever applied (the arm-specific
    identity fields are compared separately)."""
    p = copy.deepcopy(base)
    argv = p["launch"]["args"]
    if lever == "batch4096":
        set_arg(argv, "--max-num-batched-tokens", "4096")
    elif lever == "batch8192":
        set_arg(argv, "--max-num-batched-tokens", "8192")
    elif lever == "max_ctas_pow2":
        p["launch"]["env"]["INKLING2_MAX_CTAS_POW2"] = "1"
    elif lever == "dspark7":
        insert_before_middleware(argv, "--speculative-config", DSPARK)
        p["qualification_options"]["speed_max_tokens_per_event"] = 8
    elif lever == "moe_humming":
        insert_before_middleware(argv, "--moe-backend", "humming")
    else:
        raise AssertionError(lever)
    return p


IDENTITY = ("profile_id", "purpose", "status", "bench", "launch.log_name", "containment.unit")


def strip_identity(profile: dict) -> dict:
    p = copy.deepcopy(profile)
    for key in IDENTITY:
        if "." in key:
            outer, inner = key.split(".")
            p[outer].pop(inner)
        else:
            p.pop(key)
    return p


class Inkling2TuneArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = profile_resolver.load_profile(SLUG, f"{BASE}.json")
        cls.arms_doc = json.loads((SCREEN / "arms.json").read_text())

    def test_arms_json_names_each_arm_profile_and_lever(self):
        self.assertEqual(set(self.arms_doc), set(ARMS))
        for arm, (leaf, lever) in ARMS.items():
            with self.subTest(arm=arm):
                entry = self.arms_doc[arm]
                self.assertEqual(entry["profile"], f"{SLUG}/{leaf}")
                self.assertEqual(entry["lever"], lever)
                self.assertEqual(entry["base"], None if arm == "a0" else "a0")

    def test_each_arm_is_the_carrier_plus_exactly_its_lever(self):
        for arm, (leaf, lever) in ARMS.items():
            if lever is None:
                continue
            with self.subTest(arm=arm):
                profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
                self.assertEqual(strip_identity(profile), strip_identity(expected(self.base, lever)))

    def test_arm_identity_is_a_diagnostic_dev_profile(self):
        stacks = set()
        for arm, (leaf, lever) in ARMS.items():
            if lever is None:
                continue
            with self.subTest(arm=arm):
                profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
                self.assertEqual(profile["profile_id"], f"{SLUG}/{leaf}")
                self.assertEqual(profile["status"]["state"], "estimated")
                self.assertIn("DIAGNOSTIC", profile["purpose"])
                self.assertNotIn("switch_alias", profile)
                self.assertNotIn("switch", profile)
                self.assertEqual(profile["bench"]["stack_label"], f"{SLUG}-{leaf}")
                self.assertEqual(profile["containment"]["unit"], f"inkling2-{leaf.split('-tune-')[1]}")
                stacks.add(profile["bench"]["stack_label"])
        self.assertEqual(len(stacks), len(ARMS) - 1)

    def test_only_the_dspark_arm_streams_blocks(self):
        for arm, (leaf, lever) in ARMS.items():
            profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
            with self.subTest(arm=arm):
                k = profile["qualification_options"].get("speed_max_tokens_per_event")
                self.assertEqual(k, 8 if lever == "dspark7" else None)

    def test_every_arm_renders_on_this_host(self):
        host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
        model = profile_resolver.load_model(SLUG)
        for arm, (leaf, _lever) in ARMS.items():
            with self.subTest(arm=arm):
                profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
                snapshot = profile_resolver.resolve(profile, model, host)
                self.assertEqual(snapshot["status"], "estimated")


if __name__ == "__main__":
    unittest.main()
