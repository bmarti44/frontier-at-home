#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class W4ServingContainmentTest(unittest.TestCase):
    def test_containment_forwards_exact_topk_flag(self) -> None:
        source = (ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh").read_text()
        self.assertIn("  DS4_CUDA_TOPK2048_CUB \\\n", source)


if __name__ == "__main__":
    unittest.main()
