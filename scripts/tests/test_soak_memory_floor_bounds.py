#!/usr/bin/env python3
"""The soak memory floor is a safety gate, not a caller-selected assertion.

The owner set one soak floor for every profile on 2026-10-11: 5 GiB. Each
profile's own memwatch kill floor (8 GiB or more) is the memory guard; the
soak's floor only rejects a run that got near exhaustion, and the soak's other
gates (errors, degradation, health, duration) judge stability. Before that the
default was 12 GiB, which a profile with an 8 GiB owner floor failed by
construction (results/qwen38fn-gates/qualify-2026-10-11-c8).

`DSV4_SOAK_MEM_FLOOR_GIB` may still raise the floor for a run. An unrestricted
`float()` would also accept values that delete the gate entirely:

  * `0` or a negative number: every observation clears the floor.
  * `-inf`: same, unconditionally.
  * `nan`: every comparison against it is False, so `min_mem >= floor` is False --
    which fails closed here, but any sibling comparison written the other way
    around would silently pass. A gate whose semantics depend on which side of the
    operator NaN lands is not a gate.

Two properties are asserted:

1. The floor must be finite and at least the repository floor of 5 GiB.
2. The artifact must state whether the run is admissible to the frozen
   2026-08 DSV4 decision procedure. `scripts/34_decision.py` independently
   recomputes the 12 GiB floor its evidence was produced under, so a soak at
   a lower floor is not evidence for that decision. The distinction has to
   be in the artifact, not in a memory of how the run was launched.
"""

import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "35_soak.py"


def load_module():
    spec = importlib.util.spec_from_file_location("soak", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SoakFloorBoundTests(unittest.TestCase):
    def load_with_floor(self, value):
        env = dict(os.environ)
        if value is None:
            env.pop("DSV4_SOAK_MEM_FLOOR_GIB", None)
        else:
            env["DSV4_SOAK_MEM_FLOOR_GIB"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return load_module()

    def test_default_floor_is_the_repository_floor_of_five(self):
        module = self.load_with_floor(None)
        self.assertEqual(module.MEM_FLOOR_GIB, 5.0)
        self.assertEqual(module.MINIMUM_FLOOR_GIB, 5.0)

    def test_the_repository_floor_is_accepted_as_an_override(self):
        module = self.load_with_floor("5.0")
        self.assertEqual(module.MEM_FLOOR_GIB, 5.0)

    def test_owner_set_operational_floor_is_accepted(self):
        module = self.load_with_floor("8.0")
        self.assertEqual(module.MEM_FLOOR_GIB, 8.0)

    def test_zero_floor_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load_with_floor("0")

    def test_negative_floor_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load_with_floor("-1")

    def test_negative_infinity_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load_with_floor("-inf")

    def test_nan_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load_with_floor("nan")

    def test_floor_below_the_repository_floor_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load_with_floor("4.999")

    def test_absurdly_high_floor_is_rejected(self):
        # 119.7 GiB is the whole machine; a floor above it can never be met and
        # signals a units error (MiB for GiB) rather than an intended gate.
        with self.assertRaises(SystemExit):
            self.load_with_floor("8192")

    def test_decision_procedure_eligibility_stays_at_its_frozen_twelve(self):
        # scripts/34_decision.py is the frozen 2026-08 DSV4 decision; it keeps
        # the 12 GiB floor its evidence was recorded under.
        self.assertFalse(self.load_with_floor(None).qualification_eligible_floor())
        self.assertFalse(self.load_with_floor("8.0").qualification_eligible_floor())
        qualifying = self.load_with_floor("12.0")
        self.assertTrue(qualifying.qualification_eligible_floor())
        self.assertEqual(qualifying.QUALIFICATION_FLOOR_GIB, 12.0)


if __name__ == "__main__":
    unittest.main()
