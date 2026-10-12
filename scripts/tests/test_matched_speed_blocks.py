"""Shared matched-speed tool: five fresh-server ABBA/BAAB blocks (AGENTS.md).

scripts/97_matched_speed_blocks.py compares two profiles' block-aware speed on
a dev port. `plan` derives each block's order from public randomness, `run`
starts every arm fresh in that order (contained, sampled, kernel-logged), and
`score` reduces the starts to paired per-block log ratios with one-sided 95%
bounds. Scoring fails closed on any missing, unsafe, invalid or out-of-order
start.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/97_matched_speed_blocks.py"
RANDOMNESS = "ab" * 32
T95_DF4 = 2.131846786326649


def load():
    spec = importlib.util.spec_from_file_location("matched_speed_blocks", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def speed_doc(decode0: float, decode28: float, prefill: float, ttft0: float, valid=True) -> dict:
    return {"suite_valid": valid, "cells": [
        {"ctx_tokens": 0, "median_decode": decode0, "median_ttft": ttft0, "valid": valid,
         "reps": [{"prefill_tok_s": 350.0}] * 3},
        {"ctx_tokens": 28672, "median_decode": decode28, "median_ttft": 13.0, "valid": valid,
         "reps": [{"prefill_tok_s": prefill}] * 3}]}


def write_campaign(root: Path, tool, factor=lambda block: 1.10, mutate=None) -> None:
    orders = tool.block_orders(RANDOMNESS, 5, "test-domain")
    plan = {"schema_version": 1, "domain": "test-domain", "randomness": RANDOMNESS,
            "blocks": orders, "a_profile": "m/a", "b_profile": "m/b"}
    (root / "plan.json").write_text(json.dumps(plan))
    for block, position, arm in tool.schedule(orders):
        d = root / f"block-{block}" / f"{position}-{arm}"
        d.mkdir(parents=True)
        f = factor(block) if arm == "B" else 1.0
        jitter = 1.0 + 0.002 * position
        start = {"block": block, "position": position, "arm": arm,
                 "profile": plan["a_profile" if arm == "A" else "b_profile"],
                 "failed": None, "kill_lines": [], "swap_growth_gib": 0.0,
                 "low_mem_available_gib": 10.0}
        doc = speed_doc(25.8 * f * jitter, 34.9 * f * jitter, 2680.0 / f, 0.27)
        if mutate:
            mutate(block, position, arm, start, doc, d)
        (d / "start.json").write_text(json.dumps(start))
        (d / "speed.json").write_text(json.dumps(doc))


class MatchedSpeedBlocks(unittest.TestCase):
    def setUp(self):
        self.tool = load()

    def score(self, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root, self.tool, **kwargs)
            return self.tool.score(root)

    def test_block_orders_are_deterministic_abba_or_baab(self):
        a = self.tool.block_orders(RANDOMNESS, 5, "d")
        self.assertEqual(a, self.tool.block_orders(RANDOMNESS, 5, "d"))
        self.assertEqual(len(a), 5)
        self.assertTrue(set(a) <= {"ABBA", "BAAB"})
        self.assertNotEqual(
            [self.tool.block_orders(f"{i:064x}", 5, "d") for i in range(8)].count(a), 8)

    def test_schedule_runs_every_block_in_its_order(self):
        steps = self.tool.schedule(["ABBA", "BAAB"])
        self.assertEqual([(b, p, x) for b, p, x in steps],
                         [(0, 0, "A"), (0, 1, "B"), (0, 2, "B"), (0, 3, "A"),
                          (1, 0, "B"), (1, 1, "A"), (1, 2, "A"), (1, 3, "B")])

    def test_ratio_bounds_are_paired_block_log_ratios(self):
        factors = {0: 1.08, 1: 1.12, 2: 1.10, 3: 1.09, 4: 1.11}
        result = self.score(factor=lambda block: factors[block])
        self.assertEqual(result["verdict"], "COMPLETE")
        d0 = result["metrics"]["decode_0"]
        logs = []
        for block in range(5):
            a = statistics.fmean([25.8 * (1 + 0.002 * p) for p, arm in
                                  enumerate(self.tool.block_orders(RANDOMNESS, 5, "test-domain")[block])
                                  if arm == "A"])
            b = statistics.fmean([25.8 * factors[block] * (1 + 0.002 * p) for p, arm in
                                  enumerate(self.tool.block_orders(RANDOMNESS, 5, "test-domain")[block])
                                  if arm == "B"])
            logs.append(math.log(b / a))
        mean, sem = statistics.fmean(logs), statistics.stdev(logs) / math.sqrt(5)
        self.assertAlmostEqual(d0["ratio"], math.exp(mean), places=12)
        self.assertAlmostEqual(d0["lower_95"], math.exp(mean - T95_DF4 * sem), places=12)
        self.assertAlmostEqual(d0["upper_95"], math.exp(mean + T95_DF4 * sem), places=12)
        self.assertLess(result["metrics"]["prefill_28672"]["ratio"], 1.0)
        self.assertIn("ttft_0", result["metrics"])

    def assertNoResult(self, result, fragment):
        self.assertEqual(result["verdict"], "NO_RESULT")
        self.assertTrue(any(fragment in r for r in result["reasons"]), result["reasons"])

    def test_missing_start_fails_closed(self):
        def drop(block, position, arm, start, doc, d):
            if block == 2 and position == 1:
                start["drop"] = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root, self.tool, mutate=drop)
            for d in root.glob("block-2/1-*"):
                (d / "speed.json").unlink()
            self.assertNoResult(self.tool.score(root), "block-2/1")

    def test_unsafe_or_failed_start_fails_closed(self):
        for field, value, fragment in [("kill_lines", ["NVRM: Xid 79"], "kill"),
                                       ("swap_growth_gib", 0.5, "swap"),
                                       ("failed", "KV short", "failed")]:
            with self.subTest(field=field):
                def mutate(block, position, arm, start, doc, d, field=field, value=value):
                    if block == 3 and position == 2:
                        start[field] = value
                self.assertNoResult(self.score(mutate=mutate), fragment)

    def test_invalid_or_non_finite_speed_fails_closed(self):
        def invalid(block, position, arm, start, doc, d):
            if block == 1 and position == 0:
                doc["cells"][1]["valid"] = False
        self.assertNoResult(self.score(mutate=invalid), "invalid")

        def nan(block, position, arm, start, doc, d):
            if block == 4 and position == 3:
                doc["cells"][0]["median_decode"] = float("nan")
        self.assertNoResult(self.score(mutate=nan), "finite")

    def test_start_of_the_wrong_profile_or_without_a_finite_low_point_fails_closed(self):
        def swapped(block, position, arm, start, doc, d):
            if block == 2 and position == 0:
                start["profile"] = "m/a" if arm == "B" else "m/b"
        self.assertNoResult(self.score(mutate=swapped), "profile")
        for value in (None, float("nan"), "7.2"):
            with self.subTest(low=value):
                def low(block, position, arm, start, doc, d, value=value):
                    if block == 1 and position == 3:
                        start["low_mem_available_gib"] = value
                self.assertNoResult(self.score(mutate=low), "low point")

    def test_campaign_stops_at_the_first_failed_or_unsafe_start(self):
        # Review round 1 (reviewer 2 M4): no further 100 GiB loads after an
        # NVRM/Xid/OOM line, swap growth, a low point under the floor or a failed start.
        clean = {"failed": None, "kill_lines": [], "swap_growth_gib": 0.0, "low_mem_available_gib": 10.0}
        self.assertEqual(self.tool.unsafe_start(clean, 8.0), [])
        for change, fragment in [({"kill_lines": ["kernel.log: NVRM: Xid 79"]}, "kill"),
                                 ({"swap_growth_gib": 0.5}, "swap"),
                                 ({"low_mem_available_gib": 7.9}, "low point"),
                                 ({"low_mem_available_gib": None}, "low point"),
                                 ({"failed": "launch exit 1"}, "failed")]:
            with self.subTest(change=change):
                reasons = self.tool.unsafe_start({**clean, **change}, 8.0)
                self.assertTrue(any(fragment in r for r in reasons), reasons)
        ran = []

        def run_one(step):
            ran.append(step)
            return {**clean, "kill_lines": ["kernel.log: Xid 13"]} if step == 1 else clean
        with self.assertRaises(SystemExit) as stop:
            self.tool.run_steps([0, 1, 2, 3], run_one, 8.0, lambda message: None)
        self.assertEqual(ran, [0, 1])
        self.assertIn("kill", str(stop.exception))
        ran.clear()
        self.tool.run_steps([0, 1], lambda step: ran.append(step) or clean, 8.0, lambda message: None)
        self.assertEqual(ran, [0, 1])

    def test_sample_lines_are_the_format_scorers_read(self):
        # Review round 2 (reviewer 2 H1): a scorer must parse what the sampler writes.
        sample = self.tool._meminfo()
        self.assertEqual(set(sample), set(self.tool.MEMINFO_KEYS))
        line = self.tool.meminfo_line("2026-10-12T00:00:00Z", sample)
        self.assertEqual(line.split(" ", 1)[0], "2026-10-12T00:00:00Z")
        self.assertEqual(dict(re.findall(r"(\w+):=(\d+)", line)), {k: str(v) for k, v in sample.items()})
        self.assertTrue({"MemAvailable", "SwapFree"} <= set(self.tool.MEMINFO_KEYS))

    def test_resume_stops_at_an_unsafe_start_already_on_disk(self):
        # Review round 2 (reviewer 2 M1): rerunning after an unsafe stop must not launch the rest.
        clean = {"failed": None, "kill_lines": [], "swap_growth_gib": 0.0, "low_mem_available_gib": 10.0}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "block-0" / "1-B"
            self.assertIsNone(self.tool.existing_start(d, resume=True))
            d.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                self.tool.existing_start(d, resume=False)
            with self.assertRaises(SystemExit):     # a directory without start.json: a start that died
                self.tool.existing_start(d, resume=True)
            (d / "start.json").write_text(json.dumps({**clean, "kill_lines": ["kernel.log: Xid 13"]}))
            existing = self.tool.existing_start(d, resume=True)
            self.assertEqual(existing["kill_lines"], ["kernel.log: Xid 13"])
            (d / "start.json").write_text("{not json")
            self.assertEqual(self.tool.existing_start(d, resume=True), {})
        launched = []

        def run_one(step):
            if step == 0:
                return {**clean, "kill_lines": ["kernel.log: Xid 13"]}   # read back from disk on resume
            launched.append(step)
            return clean
        with self.assertRaises(SystemExit):
            self.tool.run_steps([0, 1, 2], run_one, 8.0, lambda message: None)
        self.assertEqual(launched, [])
        self.assertTrue(self.tool.unsafe_start({}, 8.0))   # an unreadable record is not safe to pass

    def test_start_out_of_plan_order_fails_closed(self):
        def wrong(block, position, arm, start, doc, d):
            if block == 0 and position == 0:
                start["arm"] = "B" if arm == "A" else "A"
        self.assertNoResult(self.score(mutate=wrong), "order")

    def test_extra_start_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root, self.tool)
            (root / "block-0" / "4-A").mkdir()
            self.assertNoResult(self.tool.score(root), "unexpected")

    def test_dry_run_lists_contained_fresh_starts_in_plan_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            subprocess.run([sys.executable, str(TOOL), "plan", "--randomness", RANDOMNESS,
                            "--domain", "d", "--blocks", "2",
                            "--a-profile", "qwen3.8-flash-next/cuda-spark-128g-1m",
                            "--b-profile", "qwen3.8-flash-next/cuda-spark-128g-1m-tune-a4",
                            "--out", str(plan)], check=True)
            out = subprocess.run([sys.executable, str(TOOL), "run", "--plan", str(plan),
                                  "--out", str(Path(tmp) / "run"), "--dry-run",
                                  "--release-tree", "/r/one", "--release-tree", "/r/two"],
                                 check=True, capture_output=True, text=True).stdout
        lines = [json.loads(line) for line in out.splitlines()]
        orders = self.tool.block_orders(RANDOMNESS, 2, "d")
        self.assertEqual([(x["block"], x["position"], x["arm"]) for x in lines],
                         list(self.tool.schedule(orders)))
        for line in lines:
            self.assertIn("systemd-run", line["launch"][0])
            self.assertIn("MemorySwapMax=0", line["launch"])
            self.assertIn("93_profile_serve.sh", " ".join(line["launch"]))
            want_k = "5" if line["arm"] == "B" else "4"
            bench = line["bench"]
            self.assertEqual(bench[bench.index("--max-tokens-per-event") + 1], want_k)
            self.assertEqual(bench[bench.index("--context-levels") + 1], "0,28672")
            # Every start first releases the named trees' page cache (a previous
            # start on another runtime leaves its libraries cached).
            self.assertEqual(line["release_trees"], ["/r/one", "/r/two"])

    def test_drand_round_is_the_first_after_the_freeze_and_relays_must_agree(self):
        t = self.tool
        frozen = 1_760_000_000
        want = (frozen - 1595431050) // 30 + 2
        self.assertEqual(t.first_drand_round_after(frozen), want)
        self.assertGreater(t.drand_publication_time(want), frozen)
        doc = {"round": want, "randomness": "cd" * 32, "signature": "ef" * 96, "previous_signature": "01" * 96}
        seen = []

        def fetch(relay, number):
            seen.append((relay, number))
            return dict(doc)

        clock, sleeps = [t.drand_publication_time(want) - 40.0], []

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
        receipt = t.drand_receipt(frozen, "f" * 40, fetch=fetch, verify=lambda d: seen.append("verified"),
                                  now=lambda: clock[0], sleep=sleep)
        self.assertTrue(sleeps, "waits until the round is published")
        self.assertEqual([r for r, _ in seen[:3]], list(t.DRAND_RELAYS))
        self.assertEqual(seen[3], "verified")
        self.assertEqual((receipt["round"], receipt["randomness"]), (want, "cd" * 32))
        self.assertEqual(receipt["frozen_commit"], "f" * 40)
        later = lambda: t.drand_publication_time(want) + 60.0
        docs = iter([dict(doc), dict(doc, randomness="ce" * 32), dict(doc)])
        with self.assertRaisesRegex(ValueError, "disagree"):
            t.drand_receipt(frozen, "f" * 40, fetch=lambda r, n: next(docs), verify=lambda d: None,
                            now=later, sleep=lambda s: None)

        def reject(d):
            raise RuntimeError("BLS verification failed")
        with self.assertRaisesRegex(RuntimeError, "BLS"):
            t.drand_receipt(frozen, "f" * 40, fetch=lambda r, n: dict(doc), verify=reject,
                            now=later, sleep=lambda s: None)

    def test_the_bundled_verifier_accepts_a_real_round_and_rejects_a_tampered_one(self):
        fixture = json.loads((ROOT / "results/glm52-gates/W7-resume-production-drand-verifier-fixture.json")
                             .read_text())
        doc = {**fixture, "round": int(fixture["round"])}
        self.tool.verify_drand_round(doc)
        tampered = dict(doc, randomness=("0" if doc["randomness"][0] != "0" else "1") + doc["randomness"][1:])
        with self.assertRaises(RuntimeError):
            self.tool.verify_drand_round(tampered)

    def test_arena_guard_refusal_is_not_a_kill_line(self):
        refusal = ("RuntimeError: host page cache released under /m; MemFree 19.85 GiB: refusing to "
                   "allocate the KV arena; page cache outside /m would leave the driver short (NVRM)")
        self.assertEqual(self.tool.kill_lines("server.log", refusal + "\n"), [])
        self.assertEqual(len(self.tool.kill_lines("server.log", "NVRM: Xid 79\n")), 1)
        self.assertEqual(len(self.tool.kill_lines("kernel.log", refusal + "\n")), 1)

    def test_kernel_kill_pattern_ignores_case_like_the_gate_scorers(self):
        # The kernel's global OOM line is capitalized; the smoke and gate scorers match with re.I.
        self.assertEqual(len(self.tool.kill_lines("kernel.log", "Out of memory: Killed process 7 (python3)\n")), 1)
        self.assertEqual(len(self.tool.kill_lines("kernel.log", "nvrm: xid 79\n")), 1)

    def test_sigterm_runs_the_start_cleanup(self):
        # A SIGTERM to `run` must unwind through _run_start's finally (stop the
        # engine, wait for the scope) instead of orphaning it until RuntimeMaxSec.
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "cleanup"
            script = (
                "import importlib.util, os, signal, sys, time\n"
                f"spec = importlib.util.spec_from_file_location('m', {str(TOOL)!r})\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "m.exit_on_sigterm()\n"
                "try:\n"
                "    os.kill(os.getpid(), signal.SIGTERM)\n"
                "    time.sleep(10)\n"
                "finally:\n"
                f"    open({str(marker)!r}, 'w').write('ran')\n")
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 143, proc.stderr)
            self.assertEqual(marker.read_text(), "ran")


if __name__ == "__main__":
    unittest.main()
