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
0807 15:10:03 ds4-server: completion ctx=5044..5066:22 prompt start
ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB
ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1
0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s
0807 15:10:07 ds4-server: shutdown requested, draining requests
"""
HTTP = "200\n"
RESPONSE = '{"choices":[{"finish_reason":"length","text":""}],"usage":{"prompt_tokens":5066}}\n'
RC = "0\n"
CONTAINMENT = "SAFE_RUN_DONE rc=0 killed=no dir=/home/bmarti44/.local/state/glm52-crashlog/w7-test\n"


def score(text: str = GOOD, *, http: str = HTTP, response: str = RESPONSE,
          rc: str = RC, containment: str = CONTAINMENT) -> dict[str, object]:
    return MODULE.score_text(
        text,
        http_status=http,
        response_text=response,
        containment_rc=rc,
        containment_stdout=containment,
    )


class W7CacheGenerationGateTest(unittest.TestCase):
    def test_accepts_completed_resume_without_false_reload(self) -> None:
        result = score()
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["observed"]["false_generation_flush_count"], 0)

    def test_rejects_one_false_generation_flush(self) -> None:
        mutated = GOOD.replace(
            "ds4: GLM sync branch=indexed_resume",
            "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
            "ds4: GLM sync branch=indexed_resume",
        )
        self.assertEqual(score(mutated)["verdict"], "FAIL")

    def test_rejects_missing_cache_coverage(self) -> None:
        self.assertEqual(
            score(GOOD.replace("ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB\n", ""))["verdict"],
            "FAIL",
        )

    def test_rejects_unfinished_resume(self) -> None:
        self.assertEqual(
            score(GOOD.replace("0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s\n", ""))["verdict"],
            "FAIL",
        )

    def test_ignores_startup_and_post_shutdown_noise(self) -> None:
        noise = "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
        self.assertEqual(score(noise + GOOD + noise)["verdict"], "PASS")

    def test_rejects_unrelated_later_completion(self) -> None:
        bad = GOOD.replace(
            "ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1\n"
            "0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s\n",
            "ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1\n"
            "0807 15:10:04 ds4-server: completion ctx=5044..5066:22 request failed\n"
            "0807 15:10:05 ds4-server: completion ctx=0..7:7 prompt start\n"
            "0807 15:10:06 ds4-server: completion ctx=0..7:7 prompt done 1.000s\n",
        )
        self.assertEqual(score(bad)["verdict"], "FAIL")

    def test_rejects_fatal_after_completion(self) -> None:
        bad = GOOD.replace(
            "0807 15:10:07 ds4-server: shutdown requested",
            "ds4: CUDA GLM prefill failed\n0807 15:10:07 ds4-server: shutdown requested",
        )
        self.assertEqual(score(bad)["verdict"], "FAIL")

    def test_rejects_bad_http_response_or_containment(self) -> None:
        self.assertEqual(score(http="500\n")["verdict"], "FAIL")
        self.assertEqual(score(response="{}\n")["verdict"], "FAIL")
        self.assertEqual(score(rc="1\n")["verdict"], "FAIL")
        self.assertEqual(score(containment="SAFE_RUN_DONE rc=0 killed=yes dir=/tmp/x\n")["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
