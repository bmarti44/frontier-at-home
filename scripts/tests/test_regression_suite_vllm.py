"""agent-gate against vLLM: usage-derived reuse counts, block-granular floors.

vLLM returns no llama.cpp ``timings``; prefix reuse is read from
``usage.prompt_tokens_details.cached_tokens`` (``--enable-prompt-tokens-details``)
and hits land on whole KV blocks. A local fake server stands in for both
engines; the llama.cpp path must stay identical to the pinned script.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/regression-suite.py"
# Last change to the script on main; the llama.cpp path must match it exactly.
LLAMA_PINNED_FROM = "d897c7e8"
BLOCK = 3200
AGENT_GATE = ("prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput")


def load(path: Path, tmp: Path):
    spec = importlib.util.spec_from_file_location(f"regression_suite_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fixture = tmp / "fixture.txt"
    if not fixture.exists():
        fixture.write_text("".join(f"Record {i:05d}: stable reference text. " for i in range(1200)))
    mod.FIXTURE = str(fixture)
    # Distinct but reproducible session ids, so both scripts send identical bytes.
    counter = itertools.count()
    mod.uuid = types.SimpleNamespace(uuid4=lambda: f"{next(counter):08d}-fixed")
    return mod


def lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class FakeEngine:
    """One character = one token. Reuse is the longest common prefix with any
    earlier prompt, capped at total - 1 (the last prompt token is always run).
    vllm modes round reuse down to whole BLOCK-token blocks like vLLM does."""

    def __init__(self, mode: str):
        self.mode = mode
        self.seen: list[str] = []
        self.requests: list[dict] = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append(body)
                total, cached = engine.account(body["messages"])
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for event in engine.stream_events(total, cached):
                        self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                raw = json.dumps(engine.document(total, cached)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def account(self, messages):
        prompt = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        reuse = min(max((lcp(prompt, s) for s in self.seen), default=0), len(prompt) - 1)
        self.seen.append(prompt)
        if self.mode == "llama":
            return len(prompt), reuse
        if self.mode == "vllm-nocache":
            return len(prompt), 0
        cached = reuse // BLOCK * BLOCK
        if self.mode == "vllm-lossy":
            cached = max(0, cached - BLOCK)
        # vLLM with MTP/EAGLE drops one more trailing block on every hit
        # (single_type_kv_cache_manager drop_eagle_block).
        if self.mode == "vllm-mtp":
            cached = max(0, cached - BLOCK)
        if self.mode == "vllm-mtp-lossy":
            cached = max(0, cached - 2 * BLOCK)
        return len(prompt), cached

    def usage(self, total, cached):
        usage = {"prompt_tokens": total, "completion_tokens": 3, "total_tokens": total + 3,
                 "prompt_tokens_details": {"cached_tokens": cached, "created_cache_tokens": None,
                                           "multimodal_tokens": None}}
        if self.mode == "vllm-nodetails":
            usage["prompt_tokens_details"] = None
        return usage

    def document(self, total, cached):
        message = {"role": "assistant", "content": "Fake reply.", "reasoning": None}
        doc = {"id": "chatcmpl-fake", "object": "chat.completion",
               "choices": [{"index": 0, "message": message, "finish_reason": "length"}]}
        if self.mode == "llama":
            # 1 ms per evaluated token: 1,000 tok/s.
            doc["timings"] = {"prompt_n": total - cached, "cache_n": cached,
                              "prompt_ms": float(total - cached), "prompt_per_second": 1000.0}
        else:
            doc["usage"] = self.usage(total, cached)
        return doc

    def stream_events(self, total, cached):
        def chunk(delta, finish=None):
            return json.dumps({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        yield chunk({"role": "assistant", "content": ""})
        yield chunk({"content": "Fake"})
        yield chunk({"content": " reply."})
        yield chunk({}, "length")
        yield json.dumps({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "choices": [],
                          "usage": self.usage(total, cached)})


class VllmAgentGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.engines: list[FakeEngine] = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()

    def engine(self, mode):
        engine = FakeEngine(mode)
        self.engines.append(engine)
        return engine

    def run_gate(self, mode, *extra):
        engine = self.engine(mode)
        mod = load(SCRIPT, self.tmp)
        out = self.tmp / f"{mode}.json"
        rc = mod.main(["agent-gate", "--base", engine.base, "--json", str(out), *extra])
        return rc, json.loads(out.read_text()), engine

    def vllm_args(self):
        return ("--engine", "vllm", "--model", "default", "--prefix-block-tokens", str(BLOCK))

    def test_llamacpp_path_matches_the_pinned_script(self):
        pinned = self.tmp / "regression-suite-pinned.py"
        pinned.write_text(subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{LLAMA_PINNED_FROM}:scripts/dev/regression-suite.py"],
            check=True, capture_output=True, text=True).stdout)
        runs = []
        for path in (pinned, SCRIPT):
            engine = self.engine("llama")
            mod = load(path, self.tmp)
            for name in AGENT_GATE:
                mod.TESTS[name](engine.base)
            runs.append((mod.RESULTS, engine.requests))
        self.assertEqual(runs[0], runs[1])
        self.assertTrue(all(r["pass"] for r in runs[1][0]), runs[1][0])
        self.assertTrue(all(body["model"] == "d" for body in runs[1][1]))

    def test_llamacpp_json_report_keeps_its_shape(self):
        rc, report, _ = self.run_gate("llama")
        self.assertEqual(rc, 0)
        self.assertEqual(set(report), {"when", "base", "results"})

    def test_block_granular_reuse_passes_every_agent_gate_test(self):
        rc, report, engine = self.run_gate("vllm", *self.vllm_args())
        self.assertEqual(rc, 0, report)
        self.assertEqual([r["test"] for r in report["results"]], list(AGENT_GATE))
        self.assertTrue(all(body["model"] == "default" for body in engine.requests))
        prefill = report["results"][-1]["detail"]
        self.assertIn("client-measured TTFT", prefill)
        self.assertTrue(engine.requests[-1]["stream"])
        self.assertEqual(engine.requests[-1]["stream_options"], {"include_usage": True})
        self.assertEqual(report["engine"], "vllm")
        self.assertEqual(report["prefix_block_tokens"], BLOCK)

    def test_strict_llamacpp_floors_reject_block_granular_reuse(self):
        rc, report, _ = self.run_gate("vllm", "--engine", "vllm", "--model", "default",
                                      "--prefix-block-tokens", "1")
        self.assertEqual(rc, 1)
        failed = {r["test"] for r in report["results"] if not r["pass"]}
        self.assertIn("prefix-cache", failed)

    def test_missing_reuse_fails_the_reuse_tests(self):
        rc, report, _ = self.run_gate("vllm-nocache", *self.vllm_args())
        self.assertEqual(rc, 1)
        verdicts = {r["test"]: r["pass"] for r in report["results"]}
        self.assertEqual(verdicts, {"prefix-cache": False, "turn-continuation": False,
                                    "slot-thrash": False, "prefill-throughput": True})

    def test_one_block_short_of_full_reuse_fails_the_repeat(self):
        rc, report, _ = self.run_gate("vllm-lossy", *self.vllm_args())
        self.assertEqual(rc, 1)
        prefix = next(r for r in report["results"] if r["test"] == "prefix-cache")
        self.assertFalse(prefix["pass"], prefix)

    def test_mtp_block_drop_passes_when_declared(self):
        rc, report, _ = self.run_gate("vllm-mtp", *self.vllm_args(), "--spec-block-drop")
        self.assertEqual(rc, 0, report)
        self.assertTrue(all(r["pass"] for r in report["results"]), report)
        self.assertIs(report["spec_block_drop"], True)

    def test_mtp_block_drop_fails_the_repeat_when_not_declared(self):
        rc, report, _ = self.run_gate("vllm-mtp", *self.vllm_args())
        self.assertEqual(rc, 1)
        prefix = next(r for r in report["results"] if r["test"] == "prefix-cache")
        self.assertFalse(prefix["pass"], prefix)

    def test_a_block_short_of_mtp_reuse_still_fails(self):
        rc, report, _ = self.run_gate("vllm-mtp-lossy", *self.vllm_args(), "--spec-block-drop")
        self.assertEqual(rc, 1)
        prefix = next(r for r in report["results"] if r["test"] == "prefix-cache")
        self.assertFalse(prefix["pass"], prefix)

    def test_missing_prompt_tokens_details_is_a_named_error(self):
        rc, report, _ = self.run_gate("vllm-nodetails", *self.vllm_args())
        self.assertEqual(rc, 1)
        for result in report["results"]:
            with self.subTest(test=result["test"]):
                self.assertFalse(result["pass"])
                self.assertIn("--enable-prompt-tokens-details", result["detail"])

    def test_vllm_engine_requires_the_block_size(self):
        mod = load(SCRIPT, self.tmp)
        with self.assertRaises(SystemExit) as raised:
            mod.main(["agent-gate", "--base", "http://127.0.0.1:9", "--engine", "vllm"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
