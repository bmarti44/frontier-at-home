#!/usr/bin/env python3
"""Fail before CUDA compilation if vLLM metadata rejects the pinned runtime."""
import argparse
import importlib.util
from importlib.metadata import version
from pathlib import Path
import re
import sys
import unittest

from packaging.requirements import Requirement

sys.dont_write_bytecode = True


def suite(source, pristine=None):
    class MetadataContract(unittest.TestCase):
        def test_rust_builds_cannot_re_resolve_cargo_lock(self):
            spec = importlib.util.spec_from_file_location("candidate_build_rust", source / "tools/build_rust.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            extensions = module.rust_extensions(optional=False)
            self.assertEqual(len(extensions), 2)
            for extension in extensions:
                with self.subTest(target=extension.target):
                    self.assertIn("--locked", extension.args or [])

        def test_flashinfer_requirement_accepts_installed_pinned_distribution(self):
            rows = (source / "requirements/cuda.txt").read_text().splitlines()
            requirements = [Requirement(row) for row in rows
                            if row.startswith("flashinfer-python")]
            self.assertEqual(len(requirements), 1)
            self.assertIn(version("flashinfer-python"), requirements[0].specifier)

        def test_cuda_transitive_sources_use_commits(self):
            cutlass = (source / "CMakeLists.txt").read_text()
            self.assertRegex(cutlass, r'set\(CUTLASS_REVISION "[0-9a-f]{40}"\)')
            triton = (source / "cmake/external_projects/triton_kernels.cmake").read_text()
            self.assertNotIn('"v3.5.1"', triton)

        @unittest.skipIf(pristine is None, "supply --pristine for source-scope gate")
        def test_no_spec_baseline_preserves_original_cache_and_mtp_sources(self):
            for relative in ("vllm/v1/core/kv_cache_utils.py", "vllm/models/glm5next/nvidia/mtp.py"):
                with self.subTest(path=relative):
                    self.assertEqual((source / relative).read_bytes(), (pristine / relative).read_bytes())

    return unittest.defaultTestLoader.loadTestsFromTestCase(MetadataContract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pristine", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(suite(args.source, args.pristine))
    raise SystemExit(0 if result.wasSuccessful() else 1)
