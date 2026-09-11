#!/usr/bin/env python3
"""Unit tests for scripts/49_score_teacher_windows.py against a synthetic dataset and fake server."""

import hashlib
import importlib.util
import json
import math
import socket
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "49_score_teacher_windows.py"
VOCAB, WINDOW_LEN, MODEL = 16, 6, "fake-model"


def load_script():
    spec = importlib.util.spec_from_file_location("score_teacher_windows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: Path, logits: np.ndarray) -> None:
    payload = logits.astype("<f4").tobytes()
    header = json.dumps({"logits": {"dtype": "F32", "shape": list(logits.shape), "data_offsets": [0, len(payload)]}}).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + payload)


def build_dataset(root: Path, rng: np.random.Generator):
    (root / "logits").mkdir(parents=True)
    arrays = root / "calibration" / "panel-v1" / "arrays"
    arrays.mkdir(parents=True)
    entries, data = [], {}
    for i in range(2):
        window_id = f"final-{i:04d}"
        ids = rng.integers(0, VOCAB, size=WINDOW_LEN, dtype=np.int32)
        logits = rng.normal(size=(WINDOW_LEN - 1, VOCAB)).astype(np.float32) * 3
        np.save(arrays / f"{window_id}.tokens.npy", ids)
        write_safetensors(root / "logits" / f"window-{i:04d}.safetensors", logits)
        entries.append({"window_id": window_id, "path": f"logits/window-{i:04d}.safetensors", "domain": ["code", "prose"][i],
                        "sha256": hashlib.sha256((root / "logits" / f"window-{i:04d}.safetensors").read_bytes()).hexdigest(),
                        "bytes": (root / "logits" / f"window-{i:04d}.safetensors").stat().st_size,
                        "token_ids_sha256": hashlib.sha256((arrays / f"{window_id}.tokens.npy").read_bytes()).hexdigest(),
                        "prediction_positions": WINDOW_LEN - 1})
        data[window_id] = (ids.tolist(), logits)
    (root / "dataset-manifest.json").write_text(json.dumps(
        {"vocab_size": VOCAB, "dataset_sha256": "d" * 64, "model_revision": "rev", "logit_files": entries}))
    return data


def candidate_response(ids, logprob_rows, model=MODEL, tamper=False):
    """Build a well-formed vLLM completions response from per-position candidate log-distributions."""
    rows = [None]
    for target, logp in zip(ids[1:], logprob_rows):
        order = np.argsort(-logp, kind="stable")
        rank = {int(t): r + 1 for r, t in enumerate(order)}
        top = int(order[0])
        row = {str(target): {"logprob": float(logp[target]), "rank": rank[target], "decoded_token": "x"}}
        row[str(top)] = {"logprob": float(logp[top]), "rank": 1, "decoded_token": "y"}
        rows.append(row)
    if tamper:
        rows[1] = {}
    return {"id": "cmpl-1", "object": "text_completion", "model": model,
            "choices": [{"index": 0, "text": "a", "finish_reason": "length", "token_ids": [1],
                         "prompt_token_ids": ids, "prompt_logprobs": rows}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}


class FakeServer:
    def __init__(self, dists: dict, tamper_window=None):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"object": "list", "data": [{"id": MODEL}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                ids = payload["prompt"]
                key = tuple(ids)
                self._send(candidate_response(ids, dists[key], tamper=(key == tamper_window)))

        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ScoreTeacherWindowsTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "dataset"
        rng = np.random.default_rng(7)
        self.data = build_dataset(self.root, rng)
        # candidate distributions: log-softmax of teacher logits plus noise
        self.dists = {}
        for window_id, (ids, logits) in self.data.items():
            noisy = logits.astype(np.float64) + rng.normal(size=logits.shape) * 0.5
            self.dists[tuple(ids)] = noisy - np.log(np.exp(noisy).sum(axis=1, keepdims=True))

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, base_url, out, extra=()):
        return self.mod.main(["--base-url", base_url, "--model", MODEL, "--dataset-dir", str(self.root),
                              "--out", str(out), "--stack-label", "test", *extra])

    def expected(self, window_id):
        ids, logits = self.data[window_id]
        l64 = logits.astype(np.float64)
        lse = np.log(np.exp(l64).sum(axis=1))
        t_nll = lse - l64[np.arange(WINDOW_LEN - 1), ids[1:]]
        t_top = l64.argmax(axis=1)
        cand = self.dists[tuple(ids)]
        c_nll = -cand[np.arange(WINDOW_LEN - 1), ids[1:]]
        c_top = cand.argmax(axis=1)
        return {"tokens": WINDOW_LEN - 1, "teacher_nll_sum": float(t_nll.sum()), "candidate_nll_sum": float(c_nll.sum()),
                "teacher_top1_correct": int((t_top == ids[1:]).sum()), "candidate_top1_correct": int((c_top == ids[1:]).sum()),
                "candidate_top1_agrees_teacher": int((c_top == t_top).sum())}

    def test_teacher_reduction_matches_numpy(self):
        entry = json.loads((self.root / "dataset-manifest.json").read_text())["logit_files"][0]
        ids, _ = self.data["final-0000"]
        result = self.mod.reduce_teacher(self.root / entry["path"], entry["sha256"], ids, VOCAB)
        exp = self.expected("final-0000")
        self.assertAlmostEqual(result["nll_sum"], exp["teacher_nll_sum"], places=9)
        self.assertEqual(result["top1_correct"], exp["teacher_top1_correct"])
        self.assertEqual(result["tokens"], WINDOW_LEN - 1)

    def test_candidate_reduction(self):
        ids, _ = self.data["final-0001"]
        reduced = self.mod.reduce_prompt_response(candidate_response(ids, self.dists[tuple(ids)]), ids, MODEL, VOCAB)
        exp = self.expected("final-0001")
        self.assertAlmostEqual(reduced["nll_sum"], exp["candidate_nll_sum"], places=9)
        self.assertEqual(reduced["top1_correct"], exp["candidate_top1_correct"])
        self.assertEqual(reduced["top1_ids"], [int(t) for t in self.dists[tuple(ids)].argmax(axis=1)])
        with self.assertRaises(ValueError):
            self.mod.reduce_prompt_response(candidate_response(ids, self.dists[tuple(ids)], model="other"), ids, MODEL, VOCAB)

    def test_end_to_end_summary_and_cache(self):
        server = FakeServer(self.dists)
        self.addCleanup(server.close)
        out = Path(self.tmp.name) / "out1"
        # Force a FAIL-free gate so verdict depends only on numbers; use generous gates.
        code = self.run_script(server.url, out, ["--nll-gate", "100", "--top1-gate-pp", "100"])
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        self.assertEqual([r["window_id"] for r in rows], ["final-0000", "final-0001"])
        expected = {}
        for row in rows:
            exp = expected[row["window_id"]] = self.expected(row["window_id"])
            for key in ("teacher_nll_sum", "candidate_nll_sum"):
                self.assertAlmostEqual(row[key], exp[key], places=9)
            for key in ("tokens", "teacher_top1_correct", "candidate_top1_correct", "candidate_top1_agrees_teacher"):
                self.assertEqual(row[key], exp[key], key)
            self.assertIsNone(row["error"])
            self.assertEqual(row["prompt_tokens"], WINDOW_LEN)
            self.assertTrue((out / "responses" / f"{row['window_id']}.json").is_file())
        # request template check
        self.assertEqual({k: v for k, v in server.requests[0].items() if k != "prompt"},
                         {"model": MODEL, "add_special_tokens": False, "prompt_logprobs": 1, "return_token_ids": True,
                          "n": 1, "max_tokens": 1, "stream": False, "temperature": 0})
        summary = json.loads((out / "summary.json").read_text())
        weights = [e["tokens"] for e in expected.values()]
        deltas = [(e["candidate_nll_sum"] - e["teacher_nll_sum"]) / e["tokens"] for e in expected.values()]
        losses = [100.0 * (e["teacher_top1_correct"] - e["candidate_top1_correct"]) / e["tokens"] for e in expected.values()]
        agree = [e["candidate_top1_agrees_teacher"] / e["tokens"] for e in expected.values()]
        total = sum(weights)
        m = summary["metrics"]
        self.assertAlmostEqual(m["delta_nll_mean"], sum(d * w for d, w in zip(deltas, weights)) / total, places=9)
        self.assertAlmostEqual(m["top1_loss_pp_mean"], sum(l * w for l, w in zip(losses, weights)) / total, places=9)
        self.assertAlmostEqual(m["top1_agreement_mean"], sum(a * w for a, w in zip(agree, weights)) / total, places=9)
        # upper bound: equal weights, n=2 -> mean + t95(1) * stdev/sqrt(2)
        import statistics
        import sys
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from glm52_goal import _t95
        self.assertAlmostEqual(m["delta_nll_upper_95"],
                               statistics.fmean(deltas) + _t95(1) * statistics.stdev(deltas) / math.sqrt(2), places=9)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual((summary["windows_scored"], summary["windows_failed"]), (2, 0))
        self.assertEqual(set(summary["per_domain"]), {"code", "prose"})
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["logprobs_mode_assumed"], "raw_logprobs")
        self.assertEqual(manifest["vocab_size"], VOCAB)
        self.assertEqual([w["teacher_cache_hit"] for w in manifest["windows"]], [False, False])
        self.assertEqual(manifest["models_response"]["data"][0]["id"], MODEL)

        # Teacher cache hit: delete the logits files and rerun; must still score identically.
        for entry in json.loads((self.root / "dataset-manifest.json").read_text())["logit_files"]:
            (self.root / entry["path"]).unlink()
        out2 = Path(self.tmp.name) / "out2"
        self.assertEqual(self.run_script(server.url, out2, ["--nll-gate", "100", "--top1-gate-pp", "100"]), 0)
        manifest2 = json.loads((out2 / "manifest.json").read_text())
        self.assertEqual([w["teacher_cache_hit"] for w in manifest2["windows"]], [True, True])
        self.assertEqual((out2 / "raw.jsonl").read_text().count("teacher_nll_sum"), 2)
        rows2 = [json.loads(line) for line in (out2 / "raw.jsonl").read_text().splitlines()]
        self.assertEqual([r["teacher_nll_sum"] for r in rows2], [r["teacher_nll_sum"] for r in rows])

    def test_token_sha_mismatch_exits_2(self):
        arrays = self.root / "calibration" / "panel-v1" / "arrays"
        np.save(arrays / "final-0001.tokens.npy", np.zeros(WINDOW_LEN, dtype=np.int32))
        out = Path(self.tmp.name) / "out"
        self.assertEqual(self.run_script("http://127.0.0.1:9", out), 2)
        self.assertFalse((out / "summary.json").exists())

    def test_missing_logits_file_exits_2(self):
        (self.root / "logits" / "window-0000.safetensors").unlink()
        self.assertEqual(self.run_script("http://127.0.0.1:9", Path(self.tmp.name) / "out"), 2)

    def test_reducer_rejection_fails_window(self):
        bad = tuple(self.data["final-0001"][0])
        server = FakeServer(self.dists, tamper_window=bad)
        self.addCleanup(server.close)
        out = Path(self.tmp.name) / "out"
        self.assertEqual(self.run_script(server.url, out, ["--nll-gate", "100", "--top1-gate-pp", "100"]), 1)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual([f["window_id"] for f in summary["failures"]], ["final-0001"])
        self.assertIn("final-0001", summary["fail_reasons"][0])
        self.assertEqual((summary["windows_scored"], summary["windows_failed"]), (1, 1))
        rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        self.assertIsNone(rows[0]["error"])
        self.assertIn("native truth/top1 logprobs are missing", rows[1]["error"])
        self.assertIsNone(rows[1]["candidate_nll_sum"])
        self.assertIsNotNone(m := summary["metrics"].get("delta_nll_mean"))
        self.assertIsNone(summary["metrics"]["delta_nll_upper_95"])  # single window: no variance
        self.assertTrue(math.isfinite(m))

    def test_gate_failure(self):
        server = FakeServer(self.dists)
        self.addCleanup(server.close)
        out = Path(self.tmp.name) / "out"
        # candidate is noisy, so delta NLL is positive; a zero gate must fail
        self.assertEqual(self.run_script(server.url, out, ["--nll-gate", "-1", "--windows", "0-1"]), 1)
        summary = json.loads((out / "summary.json").read_text())
        self.assertFalse(summary["checks"]["delta_nll_mean_le_gate"])


if __name__ == "__main__":
    unittest.main()
