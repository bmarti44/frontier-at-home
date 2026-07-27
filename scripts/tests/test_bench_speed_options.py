#!/usr/bin/env python3

import importlib.util
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "30_bench_speed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_speed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchOptionTests(unittest.TestCase):
    def test_decisive_subset_options(self):
        bench = load_module()
        argv = [
            str(SCRIPT),
            "--base-url",
            "http://127.0.0.1:8011",
            "--out",
            "/tmp/out.json",
            "--stack-label",
            "test",
            "--context-levels",
            "0",
            "--max-tokens",
            "160",
            "--min-completion-tokens",
            "128",
            "--seed",
            "123",
            "--model-id",
            "glm-5.2",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bench.parse_args()
        self.assertEqual(args.context_levels, (0,))
        self.assertEqual(args.max_tokens, 160)
        self.assertEqual(args.min_completion_tokens, 128)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.model_id, "glm-5.2")
        self.assertEqual(args.tokenizer_path, bench.TOKENIZER_PATH)
        self.assertEqual(
            args.tokenizer_sha256,
            bench.DEFAULT_TOKENIZER_SHA256,
        )
        self.assertIsNone(args.output_tokenizer_path)
        self.assertIsNone(args.output_tokenizer_sha256)

    def test_any_invalid_or_missing_rep_fails_the_cell(self):
        bench = load_module()
        self.assertTrue(bench.reps_are_complete([{"valid": True}], 1))
        self.assertFalse(bench.reps_are_complete([{"valid": False}], 1))
        self.assertFalse(bench.reps_are_complete([], 1))
        self.assertFalse(bench.reps_are_complete([{"valid": True}], 2))

    def test_tokenizer_artifact_must_match_expected_hash(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_bytes(b"substituted tokenizer")
            with self.assertRaisesRegex(RuntimeError, "tokenizer SHA-256 mismatch"):
                bench.verify_tokenizer_hash(path, "0" * 64)

    def test_tokenizer_is_loaded_from_the_same_bytes_that_were_hashed(self):
        bench = load_module()
        original = b'{"frozen":"original"}'
        expected = hashlib.sha256(original).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_bytes(original)
            raw, actual = bench.read_verified_tokenizer_bytes(path, expected)
            path.write_bytes(b'{"frozen":"replacement"}')
        self.assertEqual(raw, original)
        self.assertEqual(actual, expected)

    def test_observable_output_not_server_usage_controls_decode_validity(self):
        bench = load_module()
        self.assertEqual(bench.observable_output_errors(157, 157, 128), [])
        self.assertTrue(bench.observable_output_errors(157, 160, 128))
        self.assertTrue(bench.observable_output_errors(127, 127, 128))

    def test_parses_strict_raw_token_timestamps(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "noise\n"
                "DS4_TOKEN_TIMING request=chatcmpl-a index=1 monotonic_ns=100 token=7\n"
                "DS4_TOKEN_TIMING request=chatcmpl-a index=2 monotonic_ns=250 token=8\n",
                encoding="utf-8",
            )
            timing = bench.read_token_timing(
                path,
                0,
                expected_request="chatcmpl-a",
                expected_count=2,
            )
        self.assertEqual(timing["request"], "chatcmpl-a")
        self.assertEqual(timing["indices"], [1, 2])
        self.assertEqual(timing["monotonic_ns"], [100, 250])
        self.assertEqual(timing["token_ids"], [7, 8])

    def test_rejects_missing_or_nonmonotonic_raw_timestamps(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "DS4_TOKEN_TIMING request=x index=1 monotonic_ns=200 token=7\n"
                "DS4_TOKEN_TIMING request=x index=3 monotonic_ns=100 token=8\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                bench.read_token_timing(
                    path,
                    0,
                    expected_request="x",
                    expected_count=2,
                )

    def test_raw_token_timing_rejects_malformed_identity_and_count(self):
        bench = load_module()
        mutations = (
            (
                "DS4_TOKEN_TIMING malformed\n",
                "malformed",
            ),
            (
                "DS4_TOKEN_TIMING request=other index=1 monotonic_ns=100 token=7\n"
                "DS4_TOKEN_TIMING request=other index=2 monotonic_ns=200 token=8\n",
                "request",
            ),
            (
                "DS4_TOKEN_TIMING request=chatcmpl-a index=1 monotonic_ns=100 token=7\n",
                "count",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            for contents, error in mutations:
                with self.subTest(error=error):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, error):
                        bench.read_token_timing(
                            path,
                            0,
                            expected_request="chatcmpl-a",
                            expected_count=2,
                        )

    def test_raw_timing_is_bound_to_exact_client_visible_bytes(self):
        bench = load_module()

        class Encoding:
            def __init__(self, ids):
                self.ids = ids

        class Tokenizer:
            pieces = {
                7: "alpha",
                8: " beta",
                9: "</think>",
                10: "answer",
            }
            sequences = {
                "": [],
                "alpha": [7],
                "alpha beta": [7, 8],
                "alphaanswer": [7, 10],
                "answer": [10],
                "<think>": [],
                "</think>": [9],
            }

            def get_vocab_size(self, with_added_tokens=True):
                self.assert_added = with_added_tokens
                return 11

            def decode(self, ids, skip_special_tokens=False):
                return "".join(self.pieces[token] for token in ids)

            def encode(self, text, add_special_tokens=False):
                return Encoding(self.sequences[text])

        tokenizer = Tokenizer()
        self.assertEqual(
            bench.raw_visible_output_errors(
                tokenizer, [7, 8], "alpha beta", ""
            ),
            [],
        )
        # GLM's reasoning separator is generated but intentionally omitted from
        # the concatenated reasoning_content/content bytes.
        self.assertEqual(
            bench.raw_visible_output_errors(
                tokenizer, [7, 9, 10], "alpha", "answer"
            ),
            [],
        )
        # Canonical reasoning and answer tokens stay separate across the hidden
        # boundary even if concatenating their bytes would select another BPE.
        tokenizer.pieces.update({1: "a", 2: "aa"})
        tokenizer.sequences.update({"a": [1], "aa": [2]})
        self.assertEqual(
            bench.raw_visible_output_errors(
                tokenizer, [1, 9, 1], "a", "a"
            ),
            [],
        )
        self.assertTrue(
            bench.raw_visible_output_errors(
                tokenizer, [7, 8], "unrelated", ""
            )
        )
        self.assertTrue(
            bench.raw_visible_output_errors(
                tokenizer, [999_999_999], "alpha", ""
            )
        )

        # Identical decoded bytes are insufficient: a longer noncanonical
        # decomposition would inflate N and therefore the reported throughput.
        tokenizer.pieces.update({1: "a", 2: "aaaaaaaa"})
        tokenizer.sequences["a" * 128] = [2] * 16
        self.assertTrue(
            bench.raw_visible_output_errors(
                tokenizer, [1] * 128, "a" * 128, ""
            )
        )

    def test_raw_clock_must_match_independent_client_wall_interval(self):
        bench = load_module()
        self.assertEqual(
            bench.raw_timing_envelope_errors(10.0, 20.0, 30.0),
            [],
        )
        self.assertTrue(
            bench.raw_timing_envelope_errors(0.000001, 20.0, 30.0)
        )
        self.assertTrue(
            bench.raw_timing_envelope_errors(10.0, 20.0, 20.0)
        )

    def test_rep_preserves_client_timestamps_and_exact_output_identities(self):
        bench = load_module()

        class Encoding:
            def __init__(self, ids):
                self.ids = ids

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return Encoding(list(range(len(text))))

        timestamps_ns = [2_000_000_000 + index * 10_000_000 for index in range(128)]
        reasoning = "x" * 128

        class Client:
            def stream_chat(self, payload):
                return {
                    "response_id": "chatcmpl-evidence",
                    "request_sha256": "a" * 64,
                    "request_started_ns": 1_000_000_000,
                    "first_content_at_ns": timestamps_ns[0],
                    "last_content_at_ns": timestamps_ns[-1],
                    "generated_text": reasoning,
                    "generated_reasoning": reasoning,
                    "generated_content": "",
                    "usage": {"completion_tokens": 128, "prompt_tokens": 32},
                    "done": True,
                    "data_chunks": 130,
                    "token_timestamps_ns": timestamps_ns,
                }

        with mock.patch.object(bench, "make_preamble", return_value="p"):
            rep = bench.run_rep(
                Client(),
                Tokenizer(),
                Tokenizer(),
                "deepseek-v4-flash",
                "",
                0,
                1,
                False,
                max_tokens=128,
                min_completion_tokens=128,
            )

        self.assertTrue(rep["valid"])
        self.assertEqual(rep["response_id"], "chatcmpl-evidence")
        self.assertEqual(rep["request_sha256"], "a" * 64)
        self.assertEqual(rep["client_request_started_ns"], 1_000_000_000)
        self.assertEqual(rep["client_first_content_ns"], timestamps_ns[0])
        self.assertEqual(rep["client_last_content_ns"], timestamps_ns[-1])
        self.assertEqual(rep["sse_token_timestamps_ns"], timestamps_ns)
        self.assertEqual(rep["token_timestamps_ns"], timestamps_ns)
        self.assertEqual(
            rep["generated_reasoning_sha256"],
            hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(rep["generated_reasoning_bytes"], 128)
        self.assertEqual(
            rep["generated_content_sha256"],
            hashlib.sha256(b"").hexdigest(),
        )
        self.assertEqual(rep["generated_content_bytes"], 0)
        self.assertIsNone(rep["raw_client_timing_ratio"])

    def test_fabricated_raw_timing_cannot_make_short_output_valid(self):
        bench = load_module()

        class Encoding:
            ids = [1]

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return Encoding()

            def decode(self, ids, skip_special_tokens=False):
                return "x" * len(ids)

            def get_vocab_size(self, with_added_tokens=True):
                return 100

        class Client:
            def __init__(self, timing):
                self.timing = timing

            def stream_chat(self, payload):
                self.timing.write_text(
                    "".join(
                        "DS4_TOKEN_TIMING request=chatcmpl-fake "
                        f"index={index} monotonic_ns={index} token=99\n"
                        for index in range(1, 129)
                    ),
                    encoding="utf-8",
                )
                return {
                    "response_id": "chatcmpl-fake",
                    "request_sha256": "b" * 64,
                    "request_started_ns": 1_000_000_000,
                    "first_content_at_ns": 2_000_000_000,
                    "last_content_at_ns": 2_000_000_000,
                    "generated_text": "x",
                    "generated_reasoning": "x",
                    "generated_content": "",
                    "usage": {"completion_tokens": 128, "prompt_tokens": 32},
                    "done": True,
                    "data_chunks": 3,
                    "token_timestamps_ns": [2_000_000_000],
                }

        with tempfile.TemporaryDirectory() as tmp:
            timing = Path(tmp) / "server.log"
            timing.write_text("", encoding="utf-8")
            with mock.patch.object(bench, "make_preamble", return_value="p"):
                rep = bench.run_rep(
                    Client(timing),
                    Tokenizer(),
                    Tokenizer(),
                    "glm-5.2",
                    "",
                    0,
                    1,
                    False,
                    max_tokens=128,
                    min_completion_tokens=128,
                    token_timing_log=timing,
                )
        self.assertFalse(rep["valid"])
        self.assertIn("mismatch", rep["error"])


if __name__ == "__main__":
    unittest.main()
