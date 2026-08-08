#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/89_score_w7_cache_generation.py"
SPEC = importlib.util.spec_from_file_location("w7_cache_generation_scorer", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


GOOD = """\
ds4: CUDA backend initialized
0807 15:10:02 ds4-server: listening on http://127.0.0.1:8097
ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB
ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1
0807 15:10:06 ds4-server: completion prompt done 3.500s
0807 15:10:07 ds4-server: shutdown requested, draining requests
"""


class W7CacheGenerationGateTest(unittest.TestCase):
    def test_accepts_completed_resume_without_false_reload(self) -> None:
        result = MODULE.score_text(GOOD)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["observed"]["false_generation_flush_count"], 0)

    def test_rejects_one_false_generation_flush(self) -> None:
        mutated = GOOD.replace(
            "ds4: GLM sync branch=indexed_resume",
            "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
            "ds4: GLM sync branch=indexed_resume",
        )
        self.assertEqual(MODULE.score_text(mutated)["verdict"], "FAIL")

    def test_rejects_missing_cache_coverage(self) -> None:
        self.assertEqual(
            MODULE.score_text(GOOD.replace("ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB\n", ""))["verdict"],
            "FAIL",
        )

    def test_rejects_unfinished_resume(self) -> None:
        self.assertEqual(
            MODULE.score_text(GOOD.replace("0807 15:10:06 ds4-server: completion prompt done 3.500s\n", ""))["verdict"],
            "FAIL",
        )

    def test_ignores_startup_and_post_shutdown_noise(self) -> None:
        noise = "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
        self.assertEqual(MODULE.score_text(noise + GOOD + noise)["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
