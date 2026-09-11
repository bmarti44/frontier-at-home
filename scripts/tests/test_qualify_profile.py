#!/usr/bin/env python3
"""scripts/94_qualify_profile.py: dry-run planning, parsers, summary rendering.

Nothing here touches a server: the dry-run resolves the GLM 1M profile through
scripts/92_resolve_profile.py render with a temporary host file and only writes
manifest.json; the parsers are exercised on small synthetic documents shaped
like the real outputs under results/qwen38-gates/ and results/laguna-gates/.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "94_qualify_profile.py"
SPARK_HOST = ROOT / "configs" / "hosts" / "spark-aba1.json"
GLM_PROFILE = "glm-5.3-flash/cuda-spark-128g-1m"


def load_module():
    spec = importlib.util.spec_from_file_location("qualify_profile", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kit = load_module()


class DryRunGlmProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="qualify-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.host = self.tmp / "host.json"
        shutil.copy(SPARK_HOST, self.host)

    def test_dry_run_writes_manifest_with_planned_cells(self) -> None:
        out = self.tmp / "bundle"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--profile", GLM_PROFILE, "--host", str(self.host),
             "--dry-run", "--out", str(out), "--port", "8099"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn("dry-run:", proc.stdout)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertTrue(manifest["dry_run"])
        self.assertFalse((out / "summary.json").exists())
        profile = manifest["profile"]
        self.assertEqual(profile["profile_id"], GLM_PROFILE)
        self.assertEqual(profile["served_model"], "glm-5.3-flash")
        self.assertEqual(profile["slots"], 4)
        self.assertEqual(profile["tokens_per_slot"], 262144)
        self.assertEqual(profile["context_cap"], 1048576)
        self.assertEqual(profile["kill_floor_gib"], 10)
        self.assertTrue(profile["tokenizer_path"].endswith("/tokenizer.json"))
        self.assertTrue(profile["digest_checks"])
        self.assertIn("30_bench_speed.py", manifest["scripts"])
        self.assertIn("93_profile_serve.sh", manifest["scripts"])
        self.assertIn("head", manifest["git"])
        self.assertIn("uname", manifest["host"])
        self.assertEqual(manifest["cells_selected"], ["speed", "toolcall", "vision", "media", "teacher", "accuracy", "context"])
        cells = manifest["cells"]
        # No GLM encoder is registered; model.json declares reference_logits so teacher is planned.
        self.assertIsNotNone(cells["accuracy"]["skip_reason"])
        self.assertIn("encoder", cells["accuracy"]["skip_reason"])
        self.assertIsNone(cells["teacher"]["skip_reason"])
        self.assertIn("49_score_teacher_windows.py", " ".join(cells["teacher"]["argv"]))
        self.assertIn("teacher-logits/glm-5.3-flash", " ".join(cells["teacher"]["argv"]))
        toolcall = cells["toolcall"]["argv"]
        self.assertIn("http://127.0.0.1:8099/v1", toolcall)
        self.assertEqual(toolcall[toolcall.index("--model") + 1], "glm-5.3-flash")
        context = cells["context"]["argv"]
        self.assertEqual(context[context.index("--slots") + 1], "4")
        self.assertEqual(context[context.index("--tokens-per-slot") + 1], "250128")  # 262144 - 12016 headroom
        self.assertEqual(context[context.index("--mem-floor-gib") + 1], "10")
        # The kit validates output token ids against the on-disk tokenizer digest,
        # or the profile's expected digest when the weights are not on this host.
        expected_sha = profile["tokenizer_sha256"] or profile.get("tokenizer_sha256_expected")
        if expected_sha:
            speed = cells["speed"]["argv"]
            self.assertEqual(speed[speed.index("--output-tokenizer-sha256") + 1], expected_sha)
            self.assertIn("--extra-body", speed)
        else:
            self.assertIsNotNone(cells["speed"]["skip_reason"])

    def test_dry_run_honours_cells_skip_and_soak(self) -> None:
        out = self.tmp / "bundle2"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--profile", GLM_PROFILE, "--host", str(self.host), "--dry-run",
             "--out", str(out), "--cells", "speed,context", "--skip", "context", "--with-soak"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["cells_selected"], ["speed", "soak"])
        self.assertIsNone(manifest["cells"]["soak"]["skip_reason"])


class Parsers(unittest.TestCase):
    def test_parse_speed_matches_30_bench_speed_shape(self) -> None:
        document = {
            "metadata": {"stack_label": "x"},
            "cells": [
                {"ctx_tokens": 0, "median_decode": 20.1, "median_ttft": 0.4, "valid": True,
                 "reps": [{"prefill_tok_s": 30.0, "decode_tok_s": 20.0}, {"prefill_tok_s": 40.0, "decode_tok_s": 20.2}]},
                {"ctx_tokens": 28672, "median_decode": 19.9, "median_ttft": 59.2, "valid": True,
                 "reps": [{"prefill_tok_s": 510.6}, {"prefill_tok_s": 510.9}]},
            ],
            "suite_valid": True,
        }
        parsed = kit.parse_speed(document)
        self.assertEqual(parsed["levels"]["0"]["median_decode_tok_s"], 20.1)
        self.assertEqual(parsed["levels"]["0"]["median_prefill_tok_s"], 35.0)
        self.assertEqual(parsed["levels"]["28672"]["median_ttft_s"], 59.2)
        self.assertAlmostEqual(parsed["levels"]["28672"]["median_prefill_tok_s"], 510.75)
        self.assertTrue(parsed["suite_valid"])

    def test_parse_toolcall_matches_39_shape(self) -> None:
        document = {"suite": "toolcall-v1", "per_case": [{"id": "a", "passed": True}, {"id": "b", "passed": False}],
                    "passed": 19, "total": 20, "score": 0.95}
        self.assertEqual(kit.parse_toolcall(document), {"passed": 19, "total": 20, "score": 0.95, "suite": "toolcall-v1"})
        # Fallback: count per_case when the totals are absent.
        self.assertEqual(kit.parse_toolcall({"per_case": document["per_case"]})["passed"], 1)

    def test_parse_vision_matches_38_summary_shape(self) -> None:
        document = {"ok": True, "suite": "mmmu-val-100", "n": 100, "correct": 64, "accuracy": 0.64,
                    "invalid_count": 3, "error_count": 0}
        parsed = kit.parse_vision(document)
        self.assertEqual(parsed["accuracy"], 0.64)
        self.assertEqual(parsed["n"], 100)

    def test_parse_teacher_matches_49_summary_shape(self) -> None:
        document = {"verdict": "PASS", "metrics": {"delta_nll_mean": 0.004, "delta_nll_upper_95": 0.008,
                                                    "top1_loss_pp_mean": 0.2, "top1_loss_pp_upper_95": 0.4,
                                                    "top1_agreement_mean": 0.97},
                    "windows_scored": 25, "windows_failed": 0, "fail_reasons": []}
        parsed = kit.parse_teacher(document)
        self.assertEqual(parsed["verdict"], "PASS")
        self.assertEqual(parsed["delta_nll_upper_95"], 0.008)
        self.assertEqual(parsed["top1_loss_pp_mean"], 0.2)

    def test_parse_accuracy_matches_31_shape(self) -> None:
        parsed = kit.parse_accuracy({
            "gsm8k": {"suite": "gsm8k", "split": "holdout", "n": 100, "correct": 98, "accuracy": 0.98,
                      "wilson95": [0.93], "invalid_count": 0},
            "humaneval": {"suite": "humaneval", "split": "all", "n": 164, "correct": 150, "accuracy": 0.9146},
        })
        self.assertEqual(parsed["suites"]["gsm8k"]["accuracy"], 0.98)
        self.assertEqual(parsed["suites"]["humaneval"]["n"], 164)

    def test_parse_context_matches_50_summary_shape(self) -> None:
        document = {"verdict": "PASS", "total_tokens": 1048576, "min_mem_available_gib": 18.4,
                    "memory_floor_gib": 10, "needles_correct": 4, "needles_total": 4, "max_ttft_s": 900.0,
                    "checks": {"needles_all_correct": True}}
        parsed = kit.parse_context(document)
        self.assertEqual(parsed["verdict"], "PASS")
        self.assertEqual(parsed["total_tokens"], 1048576)
        self.assertEqual(parsed["min_mem_available_gib"], 18.4)

    def test_parse_soak_matches_35_shape(self) -> None:
        document = {"kind": "soak", "pass": True, "failed_gates": [], "n_requests": 96, "n_errors": 0,
                    "decode_overall_median_tok_s": 14.03, "mem_available_min_gib": 20.9,
                    "duration_seconds_actual": 1804.5}
        parsed = kit.parse_soak(document)
        self.assertTrue(parsed["pass"])
        self.assertEqual(parsed["n_requests"], 96)

    def test_parse_cell_evidence_reads_files_from_cell_dir(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="qualify-parse-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "speed").mkdir()
        (tmp / "speed" / "speed.json").write_text(json.dumps({"cells": [], "suite_valid": False}))
        self.assertEqual(kit.parse_cell_evidence("speed", tmp / "speed", {})["levels"], {})
        self.assertIsNone(kit.parse_cell_evidence("vision", tmp / "vision", {}))
        (tmp / "accuracy").mkdir()
        (tmp / "accuracy" / "acc-gsm8k.json").write_text(json.dumps({"accuracy": 0.5, "n": 2}))
        parsed = kit.parse_cell_evidence("accuracy", tmp / "accuracy", {"suites": {"gsm8k": [], "mmlu-pro": []}})
        self.assertEqual(list(parsed["suites"]), ["gsm8k"])


def _resolved(targets=None):
    return {"profile_id": "m/p", "stack_label": "m-p", "served_model": "m", "qualification_targets": targets}


def _records():
    return {
        "speed": {"status": "OK", "exit_code": 0, "wall_s": 10.0, "evidence": "speed/speed.json",
                  "result": {"levels": {"0": {"median_decode_tok_s": 20.0, "median_ttft_s": 0.3,
                                              "median_prefill_tok_s": 35.0, "valid": True},
                                        "28672": {"median_decode_tok_s": 19.0, "median_ttft_s": 60.0,
                                                  "median_prefill_tok_s": 500.0, "valid": True}},
                             "suite_valid": True}},
        "toolcall": {"status": "OK", "exit_code": 0, "wall_s": 5.0, "evidence": "toolcall/toolcall.json",
                     "result": {"passed": 18, "total": 20, "score": 0.9}},
        "vision": {"status": "OK", "exit_code": 0, "wall_s": 5.0, "evidence": "vision/summary.json",
                   "result": {"accuracy": 0.66, "n": 100}},
        "teacher": {"status": "SKIPPED", "skip_reason": "no reference_logits", "evidence": "teacher/summary.json"},
        "accuracy": {"status": "SKIPPED", "skip_reason": "no encoder", "evidence": "accuracy/"},
        "context": {"status": "OK", "exit_code": 0, "wall_s": 3600.0, "evidence": "context/summary.json",
                    "result": {"verdict": "PASS", "total_tokens": 1048576, "min_mem_available_gib": 18.0}},
        "media": {"status": "OK", "exit_code": 0, "wall_s": 60.0, "evidence": "media/summary.json",
                  "result": {"verdict": "PASS", "checks": {"images_at_max_answered": True, "video_direction_right": True}}},
    }


CELLS = ["speed", "toolcall", "vision", "media", "teacher", "accuracy", "context"]


class SummaryFormatting(unittest.TestCase):
    def finalize(self, targets=None, records=None, baseline=None):
        return kit.finalize_summary(_resolved(targets), records or _records(), CELLS, Path("/bundle"), baseline,
                                    {"start_unix": 1.0, "end_unix": 2.0, "start_iso": "a", "end_iso": "b"})

    def test_no_targets_is_measured(self) -> None:
        summary = self.finalize()
        self.assertEqual(summary["verdict"], "MEASURED")
        self.assertEqual(summary["kind"], "qualify-profile")
        self.assertTrue(all(r["status"] == "measured" for r in summary["rows"]))
        md = kit.render_summary_md(summary)
        self.assertIn("| speed | decode_tok_s@0 | 20.00 | - | measured | speed/speed.json |", md)
        self.assertIn("| teacher | - | - | - | SKIPPED | no reference_logits |", md)
        self.assertNotIn("baseline", md.splitlines()[0])

    def test_targets_pass(self) -> None:
        targets = {"decode_tok_s_min": {"0": 17.46, "28672": 18.5}, "prefill_tok_s_min": {"28672": 400},
                   "ttft_s_max": {"0": 0.5}, "context_pass": True, "toolcall_min": 18, "vision_min": 0.64,
                   "delta_nll_max": 0.01}
        summary = self.finalize(targets)
        self.assertEqual(summary["verdict"], "PASS")
        statuses = {(r["cell"], r["metric"]): r["status"] for r in summary["rows"]}
        self.assertEqual(statuses[("speed", "decode_tok_s@28672")], "PASS")
        self.assertEqual(statuses[("speed", "ttft_s@0")], "PASS")
        self.assertEqual(statuses[("speed", "ttft_s@28672")], "measured")
        self.assertEqual(statuses[("toolcall", "passed")], "PASS")
        self.assertEqual(statuses[("context", "verdict")], "PASS")
        md = kit.render_summary_md(summary)
        self.assertIn("| speed | decode_tok_s@28672 | 19.00 | >= 18.50 | PASS |", md)
        self.assertIn("| speed | ttft_s@0 | 0.3 | <= 0.5 | PASS |", md)

    def test_target_miss_fails_overall(self) -> None:
        summary = self.finalize({"decode_tok_s_min": {"28672": 26.71}})
        self.assertEqual(summary["verdict"], "FAIL")
        row = next(r for r in summary["rows"] if r["metric"] == "decode_tok_s@28672")
        self.assertEqual(row["status"], "FAIL")

    def test_errored_cell_fails_even_without_targets(self) -> None:
        records = _records()
        records["vision"] = {"status": "FAIL", "exit_code": 1, "status_reason": "script exit 1",
                             "log_tail": "Traceback ...\nRuntimeError: boom", "evidence": "vision/summary.json",
                             "result": None}
        summary = self.finalize(None, records)
        self.assertEqual(summary["verdict"], "FAIL")
        md = kit.render_summary_md(summary)
        self.assertIn("## Failed cells", md)
        self.assertIn("RuntimeError: boom", md)
        self.assertIn("| vision | - | - | - | FAIL | vision/summary.json (exit 1) |", md)

    def test_gate_script_fail_verdict_is_visible(self) -> None:
        records = _records()
        records["context"] = {"status": "FAIL", "exit_code": 1, "status_reason": "script exit 1", "log_tail": "",
                              "evidence": "context/summary.json",
                              "result": {"verdict": "FAIL", "total_tokens": 900000, "min_mem_available_gib": 9.0}}
        summary = self.finalize({"context_pass": True}, records)
        self.assertEqual(summary["verdict"], "FAIL")
        row = next(r for r in summary["rows"] if r["cell"] == "context" and r["metric"] == "verdict")
        self.assertEqual(row["status"], "FAIL")

    def test_baseline_column(self) -> None:
        baseline = {"kind": "qualify-profile", "_source": "results/qwen38-gates/qualify-2026-08-18/summary.json",
                    "rows": [{"cell": "speed", "metric": "decode_tok_s@28672", "measured": 18.36},
                             {"cell": "toolcall", "metric": "passed", "measured": 19}]}
        summary = self.finalize(None, None, baseline)
        row = next(r for r in summary["rows"] if r["metric"] == "decode_tok_s@28672")
        self.assertEqual(row["baseline"], 18.36)
        md = kit.render_summary_md(summary)
        self.assertIn("| cell | metric | measured | target | baseline | status | evidence |", md)
        self.assertIn("| speed | decode_tok_s@28672 | 19.00 | - | 18.36 | measured | speed/speed.json |", md)
        self.assertIn("| toolcall | passed | 18 | - | 19 | measured |", md)

    def test_load_baseline_results_dir(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="qualify-baseline-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "summary.json").write_text(json.dumps({"kind": "qualify-profile", "rows": []}))
        args = kit.parse_args(["--profile", GLM_PROFILE, "--baseline-results", str(tmp)])
        baseline = kit.load_baseline(args)
        self.assertEqual(baseline["_source"], str(tmp / "summary.json"))

    def test_load_baseline_profile_without_kit_bundle_reports_error(self) -> None:
        args = kit.parse_args(["--profile", GLM_PROFILE, "--baseline", "qwen3.8-27b/cuda-spark-128g-1m"])
        baseline = kit.load_baseline(args)
        self.assertIn("_error", baseline)
        self.assertEqual(baseline["rows"], [])


class Helpers(unittest.TestCase):
    def test_argv_value_first_occurrence_and_equals_form(self) -> None:
        argv = ["--served-model-name", "glm", "default", "--port=8015", "--served-model-name", "other"]
        self.assertEqual(kit.argv_value(argv, "--served-model-name"), "glm")
        self.assertEqual(kit.argv_value(argv, "--port"), "8015")
        self.assertIsNone(kit.argv_value(argv, "--alias"))

    def test_select_cells_rejects_unknown(self) -> None:
        with self.assertRaises(kit.KitError):
            kit.select_cells(kit.parse_args(["--profile", GLM_PROFILE, "--cells", "speed,bogus"]))
        self.assertEqual(kit.select_cells(kit.parse_args(["--profile", GLM_PROFILE, "--cells", "context,speed"])),
                         ["speed", "context"])


if __name__ == "__main__":
    unittest.main()


class PartialRerunKeepsEarlierCells(unittest.TestCase):
    def test_previous_bundle_cells_are_kept_and_reparsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "toolcall").mkdir()
            (out / "toolcall" / "toolcall.json").write_text(json.dumps({"passed": 20, "total": 20}))
            (out / "manifest.json").write_text(json.dumps({"cells": {
                "toolcall": {"argv": ["x"], "skip_reason": None, "evidence": "toolcall/toolcall.json",
                             "status": "OK", "exit_code": 0, "wall_s": 3.0},
                "teacher": {"argv": None, "skip_reason": "no reference_logits", "evidence": "teacher/summary.json",
                            "status": "SKIPPED"},
                "vision": {"argv": ["y"], "skip_reason": None, "evidence": "vision/summary.json",
                           "status": "OK", "exit_code": 0, "wall_s": 9.0},
                "speed": {"argv": ["z"], "skip_reason": None, "evidence": "speed/speed.json"},  # never ran
            }}))
            kept = kit.load_previous_bundle(out, ["vision"])
        self.assertEqual(sorted(kept), ["teacher", "toolcall"])  # vision re-runs; speed had no status
        self.assertEqual(kept["toolcall"]["record"]["status"], "OK")
        self.assertEqual(kept["toolcall"]["record"]["result"]["passed"], 20)
        self.assertEqual(kept["teacher"]["record"]["status"], "SKIPPED")
        self.assertIn("reference_logits", kept["teacher"]["record"]["status_reason"])

    def test_evidence_without_manifest_entry_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "toolcall").mkdir()
            (out / "toolcall" / "toolcall.json").write_text(json.dumps({"passed": 19, "total": 20}))
            (out / "manifest.json").write_text(json.dumps({"cells": {}}))
            kept = kit.load_previous_bundle(out, ["vision"])
        self.assertEqual(kept["toolcall"]["record"]["status"], "OK")
        self.assertIn("recovered", kept["toolcall"]["record"]["status_reason"])
        self.assertEqual(kept["toolcall"]["record"]["result"]["passed"], 19)

    def test_missing_evidence_for_kept_cell_is_a_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "manifest.json").write_text(json.dumps({"cells": {
                "toolcall": {"argv": ["x"], "skip_reason": None, "evidence": "toolcall/toolcall.json",
                             "status": "OK", "exit_code": 0, "wall_s": 3.0}}}))
            kept = kit.load_previous_bundle(out, ["speed"])
        self.assertEqual(kept["toolcall"]["record"]["status"], "FAIL")
        self.assertEqual(kit.load_previous_bundle(Path(tmp) / "absent", ["speed"]), {})


class ResummarizeMode(unittest.TestCase):
    def test_resummarize_selects_no_cell_and_implies_no_launch(self) -> None:
        args = kit.parse_args(["--profile", GLM_PROFILE, "--resummarize", "--out", "/tmp/x"])
        self.assertTrue(args.no_launch)
        self.assertEqual(kit.select_cells(args), [])
        with self.assertRaises(SystemExit):
            kit.parse_args(["--profile", GLM_PROFILE, "--resummarize"])
