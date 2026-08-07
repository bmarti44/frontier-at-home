#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/85_score_w7_resume_equivalence.py"
SPEC = importlib.util.spec_from_file_location("w7_scorer", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PRIMARY_SHA = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
LIVE_SHA = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"
N_VOCAB = 154880


class W7ScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.strict = self._arm("strict")
        self.candidate = self._arm("candidate")
        self.cold = self._arm("cold")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _arm(self, name: str) -> Path:
        arm = self.root / name
        arm.mkdir()
        meta = {
            "schema_version": 1,
            "arm": name,
            "containment_rc": 0,
            "request_sha256": {"primary": PRIMARY_SHA},
        }
        if name != "cold":
            meta["request_sha256"]["live"] = LIVE_SHA
        (arm / "arm.json").write_text(json.dumps(meta))
        (arm / "primary-http-status").write_text("200\n")
        (arm / "primary-response.json").write_text(json.dumps({"usage": {"prompt_tokens": 5066}}))
        (arm / "trace-result.json").write_text(json.dumps({"verdict": "PASS", "checks": {"fixtures": True}}))
        if name != "cold":
            (arm / "live-http-status").write_text("200\n")
            (arm / "live-response.json").write_text(json.dumps({"usage": {"prompt_tokens": 5055}}))
        selected = "a.kv"
        (arm / "kv-before.sha256").write_text(f"{'1' * 64}  {selected}\n")
        (arm / "kv-after.sha256").write_text(f"{'1' * 64}  {selected}\n")
        if name == "candidate":
            log = (
                "kv cache hit text tokens=5044 file=" + str(arm / "kv" / selected) + "\n"
                "ds4: GLM restored-frontier diagnostic: authoritative checkpoint=5044 compact_rows=5044 prior_frontier=5044\n"
                "ds4: GLM sync start=5044 prompt=5066 suffix=22\n"
            )
            names = [
                "logits.sync1.start0.prompt5055.suffix5055",
                "logits.sync2.start5044.prompt5066.suffix22",
            ]
        elif name == "strict":
            log = (
                "kv cache hit text tokens=5044 file=" + str(arm / "kv" / selected) + "\n"
                "ds4: GLM resume guard: prompt (5066) extends/diverges past evaluated frontier 5055 (checkpoint 5044)\n"
                "ds4: GLM sync start=0 prompt=5066 suffix=5066\n"
            )
            names = [
                "logits.sync1.start0.prompt5055.suffix5055",
                "logits.sync2.start0.prompt5066.suffix5066",
            ]
        else:
            log = "ds4: GLM sync start=0 prompt=5066 suffix=5066\n"
            names = ["logits.sync1.start0.prompt5066.suffix5066"]
        (arm / "server.log").write_text(log)
        values = [0.0] * N_VOCAB
        values[1] = 0.25
        values[2] = 1.0
        values[3] = -0.5
        for filename in names:
            (arm / filename).write_bytes(struct.pack(f"<{N_VOCAB}f", *values))
        return arm

    def test_passing_equivalence(self) -> None:
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["observed"]["candidate_argmax"], 2)
        self.assertEqual(result["observed"]["max_abs_logit_delta"], 0.0)

    def test_rejects_logit_drift(self) -> None:
        target = self.candidate / "logits.sync2.start5044.prompt5066.suffix22"
        values = [0.0] * N_VOCAB
        values[1], values[2], values[3] = 0.25, 0.98, -0.5
        target.write_bytes(struct.pack(f"<{N_VOCAB}f", *values))
        self.assertEqual(MODULE.score(self.strict, self.candidate, self.cold)["verdict"], "FAIL")

    def test_rejects_nonfinite_and_stale_selected_kv(self) -> None:
        target = self.candidate / "logits.sync2.start5044.prompt5066.suffix22"
        values = [0.0] * N_VOCAB
        values[1], values[2], values[3] = math.nan, 1.0, -0.5
        target.write_bytes(struct.pack(f"<{N_VOCAB}f", *values))
        (self.candidate / "kv-after.sha256").write_text(f"{'2' * 64}  a.kv\n")
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["candidate_logits_finite"])
        self.assertFalse(result["checks"]["selected_kv_unchanged"])

    def test_rejects_missing_trace_and_wrong_request(self) -> None:
        (self.strict / "trace-result.json").unlink()
        meta = json.loads((self.candidate / "arm.json").read_text())
        meta["request_sha256"]["primary"] = "0" * 64
        (self.candidate / "arm.json").write_text(json.dumps(meta))
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["traces_pass"])
        self.assertFalse(result["checks"]["request_hashes_exact"])


if __name__ == "__main__":
    unittest.main()
