#!/usr/bin/env python3
"""The soak decode window must span the tokens `completion_tokens` counts.

`35_soak.py` timed decode between the first and last delta carrying `content`, then
divided `completion_tokens - 1` by that window. `completion_tokens` counts EVERY
generated token, reasoning included. For a non-reasoning model the two agree and
the measurement is correct. For DeepSeek-V4-Flash-0731 they do not:

  * a 256-token budget is spent almost entirely inside `reasoning_content`, so
    usually there are no content deltas at all and the run raises "insufficient
    content chunks to measure decode" -- 134 of 135 requests failed this way on
    2026-08-10;
  * when a few content deltas do arrive at the very end, all 256 tokens are
    divided by the width of that short tail. The surviving rep reported
    89.97 tok/s against a ~19.7 tok/s published baseline: a 4.6x fabricated
    speedup, not a measurement.

The second failure mode is the dangerous one. Errors are loud; an inflated number
that passes every gate is not. Had enough reps survived, the soak would have
reported a large speed win that does not exist.

The fix is the one already applied to 32_golden_tests.py and 30_bench_speed.py:
treat reasoning and content as generated output for timing purposes, so the window
covers the same tokens the denominator counts.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "35_soak.py"


def load_module():
    spec = importlib.util.spec_from_file_location("soak_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sse(events):
    for event in events:
        yield f"data: {json.dumps(event)}\n".encode("utf-8")
    yield b"data: [DONE]\n"


class _Response:
    status = 200

    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self._lines)


def delta(**fields):
    return {"choices": [{"delta": fields}]}


USAGE = {"usage": {"completion_tokens": 256}, "choices": []}


class SoakReasoningDecodeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.client = self.module.Client.__new__(self.module.Client)
        self.client.base_url = "http://127.0.0.1:8013"
        self.client.api_key = None
        self.client.timeout = 30

    def run_stream(self, events):
        with mock.patch.object(
            self.module.urllib.request, "urlopen",
            return_value=_Response(sse(events)),
        ):
            return self.client.stream_decode({"model": "m", "messages": []})

    def test_reasoning_only_stream_is_measurable(self):
        """The common case for 0731 at a small budget: no content at all."""
        events = [delta(reasoning_content="a"), delta(reasoning_content="b"),
                  delta(reasoning_content="c"), USAGE]
        result = self.run_stream(events)
        self.assertIsNotNone(
            result["first_generated_at"],
            "a stream that generated 256 tokens of reasoning is not an unmeasurable "
            "stream; it is a stream this harness refused to look at",
        )
        self.assertIsNotNone(result["last_generated_at"])
        self.assertGreater(result["last_generated_at"], result["first_generated_at"])

    def test_window_starts_at_the_first_generated_token_not_the_first_content(self):
        """This is the inflation bug: the denominator counts what the window misses."""
        events = [delta(reasoning_content="r")] * 8 + [
            delta(content="x"), delta(content="y"), USAGE
        ]
        result = self.run_stream(events)
        self.assertLessEqual(
            result["first_generated_at"],
            result["first_content_at"],
            "timing must begin at the first generated token; starting at the first "
            "content delta divides every reasoning token by the content tail and "
            "inflates the rate",
        )

    def test_both_fields_are_recorded_so_the_split_is_auditable(self):
        events = [delta(reasoning_content="r"), delta(content="c"), USAGE]
        result = self.run_stream(events)
        for key in ("first_generated_at", "last_generated_at", "first_content_at"):
            self.assertIn(key, result)

    def test_an_empty_stream_is_still_rejected(self):
        """Reasoning-awareness must not turn a dead stream into a pass."""
        result = self.run_stream([USAGE])
        self.assertIsNone(result["first_generated_at"])


if __name__ == "__main__":
    unittest.main()
