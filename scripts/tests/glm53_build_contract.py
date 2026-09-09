#!/usr/bin/env python3
"""Fail before CUDA compilation if vLLM metadata rejects the pinned runtime."""
import argparse
from importlib.metadata import version
from pathlib import Path
import unittest

from packaging.requirements import Requirement


def suite(source):
    class MetadataContract(unittest.TestCase):
        def test_flashinfer_requirement_accepts_installed_pinned_distribution(self):
            rows = (source / "requirements/cuda.txt").read_text().splitlines()
            requirements = [Requirement(row) for row in rows
                            if row.startswith("flashinfer-python")]
            self.assertEqual(len(requirements), 1)
            self.assertIn(version("flashinfer-python"), requirements[0].specifier)

    return unittest.defaultTestLoader.loadTestsFromTestCase(MetadataContract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite(parser.parse_args().source))
    raise SystemExit(0 if result.wasSuccessful() else 1)
