#!/usr/bin/env python3
"""scripts/30_bench_speed_blocks.py: block-aware decode for speculative servers.

The wrapper runs the byte-frozen scripts/30_bench_speed.py unchanged except for
two substitutions: early stop counts client-tokenized completion tokens rather
than SSE events, and decode is (tokens - K) / (t_last - t_first) where K bounds
the tokens one event can carry (1 + MTP draft length). Nothing here touches a
server: streams are synthetic dicts shaped like Client.stream_chat output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "30_bench_speed_blocks.py"
FROZEN = ROOT / "scripts" / "30_bench_speed.py"
SMOKE = ROOT / "results" / "qwen38fn-gates" / "smoke-2026-10-11"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WordTokenizer:
    """One token per whitespace-separated word."""

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Encoding(list(range(len(text.split()))))


def stream(events: int, words_per_event: int, interval_ns: int = 100_000_000) -> dict:
    start = 1_000_000_000
    stamps = [start + i * interval_ns for i in range(events)]
    content = " ".join(["w"] * (events * words_per_event))
    return {
        "response_id": "chatcmpl-blocks",
        "request_sha256": "0" * 64,
        "request_bytes": 10,
        "request_started_ns": start - 500_000_000,
        "first_content_at_ns": stamps[0],
        "last_content_at_ns": stamps[-1],
        "generated_text": content,
        "generated_reasoning": "",
        "generated_content": content,
        "field_transitions": 0,
        "usage": {"prompt_tokens": 64, "completion_tokens": events * words_per_event},
        "done": True,
        "finish_reason": "length",
        "data_chunks": events + 2,
        "token_timestamps_ns": stamps,
    }


class BlockAwareRep(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = load(SCRIPT, "bench_speed_blocks_under_test")
        self.blocks.install(4)

    def run_rep(self, fake_stream: dict) -> dict:
        bench = self.blocks.bench
        client = mock.Mock()
        client.stream_chat.return_value = fake_stream
        with mock.patch.object(bench, "make_preamble", return_value="Preamble words."):
            return bench.run_rep(client, WordTokenizer(), WordTokenizer(), "default",
                                 "fixture text", 0, 0, True, 384, 200)

    def test_block_streamed_rep_is_valid_and_decode_counts_tokens(self) -> None:
        rep = self.run_rep(stream(events=100, words_per_event=3))
        self.assertTrue(rep["valid"], rep.get("error"))
        elapsed = 99 * 0.1
        self.assertAlmostEqual(rep["decode_tok_s"], (300 - 4) / elapsed)
        # The frozen event-based value is kept beside it, never used for medians.
        self.assertAlmostEqual(rep["event_decode_tok_s"], 99 / elapsed)
        self.assertEqual(rep["max_tokens_per_event"], 4)

    def test_early_stop_counts_tokens_not_events(self) -> None:
        rep = self.run_rep(stream(events=60, words_per_event=3))
        self.assertFalse(rep["valid"])
        self.assertIn("early stop: 180 client tokens, minimum is 200", rep["error"])

    def test_more_tokens_than_the_block_width_allows_is_rejected(self) -> None:
        # 6 tokens per event cannot come from 1 + 3 draft tokens: something
        # between the engine and the client coalesced events.
        rep = self.run_rep(stream(events=50, words_per_event=6))
        self.assertFalse(rep["valid"])
        self.assertIn("exceeds 4 tokens per event", rep["error"])

    def test_more_events_than_tokens_is_still_rejected(self) -> None:
        errors = self.blocks.observable_output_errors(200, 201, 200, 0)
        self.assertTrue(any("timestamps cannot outnumber tokens" in e for e in errors))

    def test_frozen_module_seen_by_other_importers_is_untouched(self) -> None:
        frozen = load(FROZEN, "bench_speed_frozen_check")
        self.assertTrue(any("early stop" in e for e in frozen.observable_output_errors(300, 100, 200)))
        self.assertIsNot(frozen, self.blocks.bench)

    def test_output_metadata_names_the_method_and_frozen_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speed.json"
            self.blocks.bench.write_result(out, {"metadata": {}, "cells": []})
            meta = json.loads(out.read_text())["metadata"]
        self.assertEqual(meta["max_tokens_per_event"], 4)
        self.assertIn("block-aware", meta["decode_method"])
        self.assertEqual(meta["frozen_bench_sha256"], hashlib.sha256(FROZEN.read_bytes()).hexdigest())

    def test_cli_rejects_raw_timing_logs_and_single_token_events(self) -> None:
        for extra in (["--max-tokens-per-event", "1"],
                      ["--max-tokens-per-event", "4", "--token-timing-log", "/tmp/x.log"]):
            argv = [str(SCRIPT), "--base-url", "http://127.0.0.1:1", "--out", "/tmp/o.json",
                    "--stack-label", "s", *extra]
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", argv), \
                    self.assertRaises(SystemExit) as raised:
                load(SCRIPT, "bench_speed_blocks_cli").main()
            self.assertEqual(raised.exception.code, 2)

    def test_reproduces_the_smoke_block_aware_decode(self) -> None:
        # The smoke scorer (PREREG amendment 1) used the same formula with K=4.
        speed = json.loads((SMOKE / "kit" / "speed" / "speed.json").read_text())
        summary = json.loads((SMOKE / "summary.json").read_text())
        for cell in speed["cells"]:
            expected = [d["decode"] for d in summary["checks"]["speed"][str(cell["ctx_tokens"])]["detail"]]
            got = [self.blocks.block_decode_tok_s(rep["token_timestamps_ns"],
                                                  rep["client_completion_tokens"], 4)
                   for rep in cell["reps"]]
            with self.subTest(ctx=cell["ctx_tokens"]):
                self.assertEqual(len(got), len(expected))
                for g, e in zip(got, expected):
                    self.assertAlmostEqual(g, e, places=9)


if __name__ == "__main__":
    unittest.main()
