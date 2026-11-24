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
    "a0r": (BASE, None),   # a0 again, last in the window: the drift control
}
COMBINED = {
    # arm: (profile leaf, levers in the order they are applied); PREREG's a6 rules
    "a6": (f"{BASE}-tune-dspark-humming-pow2", ["dspark7", "moe_humming", "max_ctas_pow2"]),
}
EXTRA = {
    # arm: the arms.json keys beyond profile/lever(s)/base (PREREG.md)
    "a0r": {"repeat_of": "a0"},
    "a4": {"min_concurrency": 3.0},   # DSpark's draft KV shares the arena
    "a6": {"min_concurrency": 3.0},
}
LEVER_ARMS = [arm for arm, (_leaf, lever) in ARMS.items() if lever is not None]


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
# M4 qualified the serving profile after the screen: its status, switch fields and the measured
# admission and overhead changed. The screen's arm files stay as they ran, so these fields are
# not compared between an arm and the serving profile.
QUALIFICATION = ("switch_alias", "switch", "safety.minimum_start_gib", "safety.note",
                 "memory_model.overhead_gib", "memory_model.basis")


def screen_a0(serving: dict) -> dict:
    """The screen's a0: the serving profile without the levers it took from the carrier
    (DSpark, humming, #69288) and at effort low, the profile as the screen ran it."""
    p = copy.deepcopy(serving)
    argv = p["launch"]["args"]
    for flag in ("--speculative-config", "--moe-backend"):
        at = argv.index(flag)
        del argv[at:at + 2]
    argv[argv.index("--default-chat-template-kwargs") + 1] = '{"reasoning_effort":"low"}'
    del p["launch"]["env"]["INKLING2_MAX_CTAS_POW2"]
    del p["qualification_options"]["speed_max_tokens_per_event"]
    return p


def strip_identity(profile: dict) -> dict:
    p = copy.deepcopy(profile)
    for key in IDENTITY:
        if "." in key:
            outer, inner = key.split(".")
            p[outer].pop(inner)
        else:
            p.pop(key)
    for key in QUALIFICATION:
        if "." in key:
            outer, inner = key.split(".")
            p[outer].pop(inner, None)
        else:
            p.pop(key, None)
    return p


class Inkling2TuneArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = screen_a0(profile_resolver.load_profile(SLUG, f"{BASE}.json"))
        cls.arms_doc = json.loads((SCREEN / "arms.json").read_text())

    def test_arms_json_names_each_arm_profile_and_lever(self):
        self.assertEqual(set(self.arms_doc), set(ARMS) | set(COMBINED))
        for arm, (leaf, levers) in COMBINED.items():
            with self.subTest(arm=arm):
                self.assertEqual(self.arms_doc[arm], {"profile": f"{SLUG}/{leaf}", "levers": levers, "base": "a0",
                                                      **EXTRA.get(arm, {})})
        for arm, (leaf, lever) in ARMS.items():
            with self.subTest(arm=arm):
                self.assertEqual(self.arms_doc[arm], {"profile": f"{SLUG}/{leaf}", "lever": lever,
                                                      "base": None if lever is None else "a0",
                                                      **EXTRA.get(arm, {})})

    def test_a_retry_keeps_its_original_lever_and_floor(self):
        for arm, entry in self.arms_doc.items():
            if "retry_of" in entry:
                with self.subTest(arm=arm):
                    original = self.arms_doc[entry["retry_of"]]
                    self.assertEqual(entry["lever"], original["lever"])
                    self.assertEqual(entry.get("min_concurrency"), original.get("min_concurrency"))

    def test_each_arm_is_the_carrier_plus_exactly_its_lever(self):
        for arm, (leaf, lever) in ARMS.items():
            if lever is None:
                continue
            with self.subTest(arm=arm):
                profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
                self.assertEqual(strip_identity(profile), strip_identity(expected(self.base, lever)))

    def test_a_combined_arm_is_the_carrier_plus_exactly_its_levers(self):
        for arm, (leaf, levers) in COMBINED.items():
            with self.subTest(arm=arm):
                want = self.base
                for lever in levers:
                    want = expected(want, lever)
                profile = profile_resolver.load_profile(SLUG, f"{leaf}.json")
                self.assertEqual(strip_identity(profile), strip_identity(want))
                self.assertEqual(profile["profile_id"], f"{SLUG}/{leaf}")
                self.assertEqual(profile["status"]["state"], "estimated")
                self.assertIn(f"DIAGNOSTIC screen arm {arm}", profile["purpose"])
                self.assertNotIn("switch_alias", profile)
                self.assertEqual(profile["bench"]["stack_label"], f"{SLUG}-{leaf}")
                self.assertEqual(profile["containment"]["unit"], f"inkling2-{leaf.split('-tune-')[1]}")
                host = profile_resolver.load_host(ROOT / "configs/hosts/spark-aba1.json")
                snapshot = profile_resolver.resolve(profile, profile_resolver.load_model(SLUG), host)
                self.assertEqual(snapshot["status"], "estimated")

    def test_every_dspark_arm_has_the_draft_kv_floor(self):
        for arm, entry in self.arms_doc.items():
            levers = entry.get("levers") or [entry.get("lever")]
            with self.subTest(arm=arm):
                self.assertEqual(entry.get("min_concurrency"), 3.0 if "dspark7" in levers else None)

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
        self.assertEqual(len(stacks), len(LEVER_ARMS))

    def test_only_the_dspark_arm_streams_blocks(self):
        for arm, (leaf, lever) in ARMS.items():
            profile = self.base if leaf == BASE else profile_resolver.load_profile(SLUG, f"{leaf}.json")
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
                # The serving profile (arm a0's leaf) is qualified since M4; the screen arms are not.
                self.assertEqual(snapshot["status"], "qualified" if leaf == BASE else "estimated")


if __name__ == "__main__":
    unittest.main()
