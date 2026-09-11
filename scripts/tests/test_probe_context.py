#!/usr/bin/env python3
"""Unit tests for scripts/50_probe_context.py against a fake OpenAI SSE server."""

import importlib.util
import json
import re
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "50_probe_context.py"
FIXTURE = ROOT / "fixtures" / "ctx-32k.txt"


def load_module():
    spec = importlib.util.spec_from_file_location("probe_context", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


class FakeHandler(BaseHTTPRequestHandler):
    mode = "correct"  # or "wrong"
    prompt_overhead = 7  # pretend chat-template tokens
    seen: list[dict] = []

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeHandler.seen.append(payload)
        text = payload["messages"][0]["content"]
        asked = re.search(r"secret codes for (\w+), (\w+) and (\w+), in that order", text).groups()
        codes = [re.search(rf"The secret code for {animal} is (\d{{8}})\.", text).group(1) for animal in asked]
        if FakeHandler.mode == "wrong":
            codes[0] = "00000000"
        answer = ", ".join(codes) + ", NO_EXTRA_RECORD"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        rid = "chatcmpl-fake"
        pieces = answer.split(", ")
        for i, piece in enumerate(pieces):
            chunk = {"id": rid, "choices": [{"index": 0, "delta": {"content": piece + (", " if i < len(pieces) - 1 else "")},
                                             "finish_reason": "stop" if i == len(pieces) - 1 else None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        usage = {"prompt_tokens": len(text.split()) + self.prompt_overhead, "completion_tokens": 9}
        self.wfile.write(f"data: {json.dumps({'id': rid, 'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class ProbeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.paragraphs = [p.strip() for p in FIXTURE.read_text().split("\n\n") if p.strip()]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.probe = load_module()
        self.probe.load_tokenizer = lambda path, sha: WhitespaceTokenizer()
        FakeHandler.mode = "correct"
        FakeHandler.seen = []

    def run_main(self, tmp: str, phase: str = "both", tokens: int = 2048, slots: int = 2) -> tuple[int, Path]:
        out = Path(tmp) / "out"
        code = self.probe.main([
            "--base-url", self.base_url, "--model", "fake-model", "--out", str(out),
            "--stack-label", "unit-test", "--slots", str(slots), "--tokens-per-slot", str(tokens),
            "--tokenizer-path", str(FIXTURE), "--tokenizer-sha256", "0" * 64,
            "--seed", "7", "--max-tokens", "16", "--mem-floor-gib", "0", "--phase", phase,
            "--profile-id", "unit", "--extra-body", '{"chat_template_kwargs": {"x": 1}}',
        ])
        return code, out

    def test_prompt_is_deterministic_and_exact(self):
        tok = WhitespaceTokenizer()
        a = self.probe.build_slot_prompt(tok, self.paragraphs, 20000, 42, 0)
        b = self.probe.build_slot_prompt(tok, self.paragraphs, 20000, 42, 0)
        other = self.probe.build_slot_prompt(tok, self.paragraphs, 20000, 42, 1)
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(a["text"], other["text"])
        self.assertEqual(len(tok.encode(a["text"])), 20000)
        self.assertEqual(a["client_prompt_tokens"], 20000)
        self.assertEqual(len(re.findall(r"^slot salt [0-9a-f]{16}$", a["text"], re.M)), 4)
        self.assertNotEqual(a["needles"][0]["code"], other["needles"][0]["code"])

    def test_needles_sit_at_expected_positions(self):
        tok = WhitespaceTokenizer()
        p = self.probe.build_slot_prompt(tok, self.paragraphs, 20000, 42, 3)
        self.assertEqual(len(p["needles"]), 6)
        for needle, fraction in zip(p["needles"], self.probe.NEEDLE_FRACTIONS):
            self.assertAlmostEqual(needle["relative_position"], fraction, delta=0.01, msg=needle)
            self.assertEqual(tok.encode(p["text"][: needle["char_offset"]]).__len__(), needle["token_offset"])
            self.assertIn(f"The secret code for {needle['animal']} is {needle['code']}.", p["text"])
        for control in p["controls"]:
            self.assertNotIn(f"secret code for {control}", p["text"])
        self.assertEqual(len(p["asked_codes"]), 3)
        self.assertEqual(p["asked_codes"][0], p["target_code"])
        for animal, code in zip(p["asked_animals"], p["asked_codes"]):
            self.assertIn(f"The secret code for {animal} is {code}.", p["text"])
        self.assertIn(f"codes for {p['asked_animals'][0]}, {p['asked_animals'][1]} and {p['asked_animals'][2]}", p["text"])
        self.assertEqual(p["target_needle_index"], self.probe.TARGET_ORDER[3])
        targets = {self.probe.build_slot_prompt(tok, self.paragraphs, 4096, 1, s)["target_needle_index"]
                   for s in range(4)}
        self.assertTrue({0, 5} <= targets)

    def test_correct_answers_pass_and_write_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_main(tmp)
            self.assertEqual(code, 0)
            summary = json.loads((out / "summary.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["needles_correct"], 2)
        self.assertEqual(summary["total_tokens"], 2 * (2048 + FakeHandler.prompt_overhead))
        self.assertTrue(summary["memory_floor_pass"])
        self.assertIsNotNone(summary["max_ttft_s"])
        self.assertEqual(summary["per_slot"][0]["prompt_token_delta"], FakeHandler.prompt_overhead)
        self.assertEqual(manifest["stack_label"], "unit-test")
        self.assertEqual(manifest["profile_id"], "unit")
        self.assertEqual(manifest["models_response"]["status"], 200)
        self.assertEqual(len(manifest["phases"]["full"][0]["needles"]), 6)
        self.assertIn("uname", manifest["host"])
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds.count("request"), 4)  # 2 short + 2 full
        self.assertEqual(kinds.count("memory_samples"), 1)
        mem = next(r for r in rows if r["kind"] == "memory_samples")
        self.assertGreaterEqual(len(mem["samples"]), 1)
        self.assertGreaterEqual(len(mem["health"]), 1)
        self.assertEqual(FakeHandler.seen[0]["chat_template_kwargs"], {"x": 1})
        self.assertEqual(FakeHandler.seen[0]["temperature"], 0)
        self.assertTrue(FakeHandler.seen[0]["stream"])

    def test_score_answer_requires_order_marker_and_no_extra_codes(self):
        asked = ["11111111", "22222222", "33333333"]
        ok = self.probe.score_answer("11111111, 22222222, 33333333, NO_EXTRA_RECORD", asked)
        self.assertTrue(ok["needle_correct"] and ok["controls_ok"])
        self.assertFalse(self.probe.score_answer("22222222, 11111111, 33333333, NO_EXTRA_RECORD", asked)["needle_correct"])
        self.assertFalse(self.probe.score_answer("11111111, 22222222, 33333333", asked)["controls_ok"])
        extra = self.probe.score_answer("11111111, 22222222, 33333333, 44444444, NO_EXTRA_RECORD", asked)
        self.assertTrue(extra["needle_correct"])
        self.assertFalse(extra["controls_ok"])

    def test_wrong_answer_fails(self):
        FakeHandler.mode = "wrong"
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_main(tmp, phase="full")
            summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(summary["needles_correct"], 0)
        self.assertFalse(summary["all_slots_ok"])
        self.assertFalse(summary["checks"]["needles_all_correct"])

    def test_short_only_run_is_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_main(tmp, phase="short")
            summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertIn("NOT a full-context qualification", summary["scope"])
        self.assertEqual(summary["total_tokens"], 0)

    def test_memory_sampler_records_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemTotal:       131072000 kB\nMemAvailable:   12582912 kB\n")
            sampler = self.probe.MemorySampler(interval_s=0.02, meminfo=meminfo)
            sampler.start()
            time.sleep(0.15)
            sampler.stop.set()
            sampler.join()
        self.assertGreaterEqual(len(sampler.samples), 3)
        summary = sampler.summary(10)
        self.assertAlmostEqual(summary["min_mem_available_gib"], 12.0)
        self.assertTrue(summary["memory_floor_pass"])
        self.assertFalse(sampler.summary(12.5)["memory_floor_pass"])

    def test_tokenizer_sha_mismatch_is_usage_error(self):
        probe = load_module()
        with self.assertRaises(probe.UsageError):
            probe.load_tokenizer(FIXTURE, "0" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            code = probe.main(["--base-url", self.base_url, "--model", "m", "--out", tmp, "--stack-label", "x",
                               "--tokenizer-path", str(FIXTURE), "--tokenizer-sha256", "0" * 64])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
