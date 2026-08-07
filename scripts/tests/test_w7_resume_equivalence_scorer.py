#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
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
        self.original_primary = MODULE.PRIMARY_SHA256
        self.original_live = MODULE.LIVE_SHA256
        self.primary_bytes = b'{"fixture":"primary"}'
        self.live_bytes = b'{"fixture":"live"}'
        MODULE.PRIMARY_SHA256 = hashlib.sha256(self.primary_bytes).hexdigest()
        MODULE.LIVE_SHA256 = hashlib.sha256(self.live_bytes).hexdigest()
        self.strict = self._arm("strict")
        self.candidate = self._arm("candidate")
        self.cold = self._arm("cold")
        MODULE.write_evidence_contract(
            self.root, {"test_sha256": "9" * 64},
            ["strict", "candidate", "cold"],
        )

    def tearDown(self) -> None:
        MODULE.PRIMARY_SHA256 = self.original_primary
        MODULE.LIVE_SHA256 = self.original_live
        self.tmp.cleanup()

    def _arm(self, name: str) -> Path:
        arm = self.root / name
        arm.mkdir()
        meta = {
            "schema_version": 1,
            "arm": name,
            "containment_rc": 0,
            "request_sha256": {"primary": MODULE.PRIMARY_SHA256},
        }
        if name != "cold":
            meta["request_sha256"]["live"] = MODULE.LIVE_SHA256
        (arm / "arm.json").write_text(json.dumps(meta))
        (arm / "primary-request.json").write_bytes(self.primary_bytes)
        if name != "cold":
            (arm / "live-request.json").write_bytes(self.live_bytes)
        (arm / "primary-http-status").write_text("200\n")
        (arm / "primary-response.json").write_text(json.dumps({"usage": {"prompt_tokens": 5066}}))
        trace_checks = {
            "trace_exactly_two_requests": True,
            "trace_request_ids_exact": True,
            "trace_request_bytes_exact": True,
            "trace_rendered_bytes_exact": True,
            "trace_token_vectors_exact": True,
        }
        observations = [
            {"request_sha256": MODULE.LIVE_SHA256, "rendered_sha256": "3" * 64,
             "token_count": 5055, "token_ids_sha256": "4" * 64},
            {"request_sha256": MODULE.PRIMARY_SHA256, "rendered_sha256": "5" * 64,
             "token_count": 5066, "token_ids_sha256": "6" * 64},
        ]
        (arm / "trace-result.json").write_text(json.dumps({
            "schema_version": 1, "checks": trace_checks, "observed": observations,
            "error": None, "verdict": "PASS",
        }))
        (arm / "trace-scorer.rc").write_text("0\n")
        if name != "cold":
            (arm / "live-http-status").write_text("200\n")
            (arm / "live-response.json").write_text(json.dumps({"usage": {"prompt_tokens": 5055}}))
        selected = f"{'a' * 40}.kv"
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
        (arm / "containment.rc").write_text("0\n")
        (arm / "containment.stdout").write_text(
            f"SAFE_RUN_DONE rc=0 killed=no dir={arm}/safety-source\n"
        )
        safety = arm / "safety"
        safety.mkdir()
        (safety / "main.log").write_text(
            "executed_candidate_verified executed_binary_sha256=" + "7" * 64 + "\n"
            "cgroup_final current_bytes=1 peak_bytes=2 swap_current_bytes=0 "
            "events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n"
            "wrapper and descendant checks clean\n"
            "SAFE_RUN end rc=0 killed=no\n"
        )
        (safety / "samples.log").write_text(
            "2026-01-01T00:00:00+00:00 mem_avail_kb=50000000 eng_rss_kb=1 "
            "read_bytes=1 cgroup_current_bytes=1 cgroup_peak_bytes=2 "
            "cgroup_swap_current_bytes=0\n"
        )
        (safety / "kernel.log").write_text("-- No entries --\n")
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
        (self.candidate / "kv-after.sha256").write_text(
            f"{'2' * 64}  {'a' * 40}.kv\n"
        )
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

    def test_rejects_forged_trace_and_mutated_actual_request(self) -> None:
        (self.strict / "trace-result.json").write_text(
            json.dumps({"verdict": "PASS", "checks": {"invented": True}})
        )
        (self.cold / "primary-request.json").write_bytes(self.primary_bytes + b" ")
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertFalse(result["checks"]["traces_pass"])
        self.assertFalse(result["checks"]["request_hashes_exact"])

    def test_rejects_cross_arm_checkpoint_substitution(self) -> None:
        selected = f"{'b' * 40}.kv"
        digest = "8" * 64
        (self.candidate / "kv-before.sha256").write_text(f"{digest}  {selected}\n")
        (self.candidate / "kv-after.sha256").write_text(f"{digest}  {selected}\n")
        log = (self.candidate / "server.log").read_text().replace(
            f"{'a' * 40}.kv", selected
        )
        (self.candidate / "server.log").write_text(log)
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertFalse(result["checks"]["selected_kv_cross_arm_exact"])

    def test_rejects_missing_safety_evidence(self) -> None:
        (self.candidate / "safety" / "samples.log").unlink()
        result = MODULE.score(self.strict, self.candidate, self.cold)
        self.assertFalse(result["checks"]["safety_evidence_pass"])


if __name__ == "__main__":
    unittest.main()
