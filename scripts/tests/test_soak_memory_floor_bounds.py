#!/usr/bin/env python3
"""The soak memory floor is a safety gate, not a caller-selected assertion.

`DSV4_SOAK_MEM_FLOOR_GIB` exists because the 1M-fast profile's measured steady
state (~9.8 GiB free) cannot satisfy the 12 GiB default, so the gate would fail by
construction rather than by fault. That is a legitimate reason to let a run declare
the floor it is held to -- but an unrestricted `float()` also accepts values that
delete the gate entirely:

  * `0` or a negative number: every observation clears the floor.
  * `-inf`: same, unconditionally.
  * `nan`: every comparison against it is False, so `min_mem >= floor` is False --
    which fails closed here, but any sibling comparison written the other way
    around would silently pass. A gate whose semantics depend on which side of the
    operator NaN lands is not a gate.

Two properties are asserted:

1. The floor must be finite and at least the deployed watchdog floor. A soak may
   never claim to hold the box to less than the watchdog itself enforces, because
   below that value the watchdog -- not the soak -- decides the outcome.
2. The artifact must state whether the run is qualification-eligible.
   `scripts/34_decision.py` independently recomputes a 12 GiB floor, so a soak run
   at 8 GiB is operational evidence and cannot be fed to the decision procedure as
   qualifying evidence. The distinction has to be in the artifact, not in a memory
   of how the run was launched.
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

    def test_default_floor_is_twelve(self):
        module = self.load_with_floor(None)
        self.assertEqual(module.MEM_FLOOR_GIB, 12.0)

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

    def test_floor_below_the_watchdog_floor_is_rejected(self):
        # The deployed profile enforces DSV4_WATCHDOG_FLOOR_GIB=8. Below that the
        # watchdog decides the outcome, so a lower soak floor is not measurable.
        with self.assertRaises(SystemExit):
            self.load_with_floor("7.999")

    def test_absurdly_high_floor_is_rejected(self):
        # 119.7 GiB is the whole machine; a floor above it can never be met and
        # signals a units error (MiB for GiB) rather than an intended gate.
        with self.assertRaises(SystemExit):
            self.load_with_floor("8192")

    def test_qualification_eligibility_is_derived_from_the_floor(self):
        operational = self.load_with_floor("8.0")
        self.assertFalse(operational.qualification_eligible_floor())
        qualifying = self.load_with_floor("12.0")
        self.assertTrue(qualifying.qualification_eligible_floor())


if __name__ == "__main__":
    unittest.main()
