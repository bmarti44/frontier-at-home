#!/usr/bin/env python3

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "38_bench_vision.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_vision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_evalset(root: Path) -> Path:
    images = root / "images"
    images.mkdir()
    image = images / "one.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    cases = root / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "Which?",
                "options": ["x", "y"],
                "answer": "A",
                "image": "images/one.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    files = {}
    for path in (cases, image):
        files[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    (root / "pins.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "MMMU/MMMU",
                "revision": "a" * 40,
                "split": "validation",
                "rows": 1,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return cases


class LetterExtractionTests(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def test_extracts_single_and_anchored_answers(self):
        examples = {
            "C": "C",
            "**b**": "B",
            "Reasoning.\nAnswer: (D)": "D",
            "I first considered A. Final answer is C.": "C",
            r"work \boxed{B}": "B",
            "Some explanation\nA.": "A",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(self.bench.extract_answer_letter(text, 4), expected)

    def test_rejects_prose_out_of_range_and_ambiguous_mentions(self):
        for text in (
            "I considered A but did not finish",
            "Option Z",
            "",
            "AB",
            "Answer: A or B",
            "Final answer is A/B",
            r"\boxed{A} and B",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.bench.extract_answer_letter(text, 4))
        self.assertIsNone(self.bench.extract_answer_letter("Answer: E", 4))


class PinVerificationTests(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def make_evalset(self, root: Path) -> Path:
        return make_evalset(root)

    def test_accepts_complete_matching_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = self.make_evalset(Path(temporary))
            pins = self.bench.verify_pins(cases)
            rows = self.bench.load_cases(cases, pins)
            self.assertEqual(rows[0]["answer"], "A")

    def test_refuses_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.make_evalset(root)
            # Same byte count isolates the cryptographic check from the size check.
            (root / "images" / "one.png").write_bytes(b"\x89PNG\r\n\x1a\nmutated")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                self.bench.verify_pins(cases)

    def test_refuses_unpinned_extra_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.make_evalset(root)
            (root / "images" / "extra.jpg").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
                self.bench.verify_pins(cases)

    def test_refuses_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.make_evalset(root)
            pins_path = root / "pins.json"
            pins = json.loads(pins_path.read_text(encoding="utf-8"))
            pins["files"]["../escape.png"] = {
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
            pins_path.write_text(json.dumps(pins), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsafe pinned path"):
                self.bench.verify_pins(cases)


class BenchmarkBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def run_benchmark(self, temporary: str, client) -> tuple[int, Path, list[dict], dict, dict]:
        root = Path(temporary) / "eval"
        root.mkdir()
        cases = make_evalset(root)
        out = Path(temporary) / "out"
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--base-url",
            "http://127.0.0.1:9",
            "--cases",
            str(cases),
            "--out",
            str(out),
        ]
        with (
            mock.patch.object(self.bench, "Client", return_value=client),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(stdout),
        ):
            return_code = self.bench.main()
        transcripts = [
            json.loads(line)
            for line in (out / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        result = json.loads(stdout.getvalue().splitlines()[-1])
        return return_code, cases, transcripts, summary, result

    def test_request_error_is_counted_and_fails_run(self):
        class FailingClient:
            def get_model(self):
                return "test-model", {"data": [{"id": "test-model"}]}

            def chat(self, payload):
                raise RuntimeError("synthetic transport failure")

        with tempfile.TemporaryDirectory() as temporary:
            return_code, _, transcripts, summary, result = self.run_benchmark(
                temporary, FailingClient()
            )
        self.assertEqual(return_code, 1)
        self.assertFalse(result["ok"])
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertIn("synthetic transport failure", transcripts[0]["error"])

    def test_non_stop_content_and_reasoning_are_not_scored(self):
        class TruncatedClient:
            def get_model(self):
                return "test-model", {"data": [{"id": "test-model"}]}

            def chat(self, payload):
                response = {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "Answer: A", "reasoning_content": "Answer: A"},
                        }
                    ]
                }
                return "Answer: A", "Answer: A", "length", response, 0.1

        with tempfile.TemporaryDirectory() as temporary:
            return_code, _, transcripts, summary, result = self.run_benchmark(
                temporary, TruncatedClient()
            )
        self.assertEqual(return_code, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertIsNone(transcripts[0]["parsed"])
        self.assertEqual(transcripts[0]["finish_reason"], "length")

    def test_reasoning_content_is_not_an_answer_fallback(self):
        class ReasoningOnlyClient:
            def get_model(self):
                return "test-model", {"data": [{"id": "test-model"}]}

            def chat(self, payload):
                response = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "No final choice in content.",
                                "reasoning_content": "Answer: A",
                            },
                        }
                    ]
                }
                return "No final choice in content.", "Answer: A", "stop", response, 0.1

        with tempfile.TemporaryDirectory() as temporary:
            return_code, _, transcripts, summary, result = self.run_benchmark(
                temporary, ReasoningOnlyClient()
            )
        self.assertEqual(return_code, 0)
        self.assertTrue(result["ok"])
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["invalid_count"], 1)
        self.assertIsNone(transcripts[0]["parsed"])
        self.assertEqual(transcripts[0]["reasoning_content"], "Answer: A")

    def test_transcript_hashes_the_exact_image_buffer_sent(self):
        replacement = b"\x89PNG\r\n\x1a\nreplacement-after-pin-check"

        class MutatingClient:
            def __init__(self):
                self.cases: Path | None = None
                self.sent = b""

            def get_model(self):
                assert self.cases is not None
                (self.cases.parent / "images" / "one.png").write_bytes(replacement)
                return "test-model", {"data": [{"id": "test-model"}]}

            def chat(self, payload):
                url = payload["messages"][1]["content"][0]["image_url"]["url"]
                self.sent = base64.b64decode(url.split(",", 1)[1])
                response = {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "A"}}
                    ]
                }
                return "A", "", "stop", response, 0.1

        client = MutatingClient()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "eval"
            # run_benchmark creates this exact path before get_model is called.
            client.cases = root / "cases.jsonl"
            return_code, _, transcripts, summary, result = self.run_benchmark(temporary, client)
        sent_digest = hashlib.sha256(replacement).hexdigest()
        self.assertEqual(return_code, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(client.sent, replacement)
        self.assertEqual(transcripts[0]["sent_image"]["sha256"], sent_digest)
        redacted_url = transcripts[0]["request"]["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(redacted_url.startswith("data:image/png;base64,"))
        self.assertIn("REDACTED: exact sent image bytes", redacted_url)
        self.assertIn(sent_digest, redacted_url)


if __name__ == "__main__":
    unittest.main()
