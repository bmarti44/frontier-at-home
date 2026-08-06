"""Behavior and mutation tests for the fixed W3 campaign scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/85_score_w3_performance_campaign.py"
SPEC = importlib.util.spec_from_file_location("w3_campaign_scorer", SCORER)
assert SPEC and SPEC.loader
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class W3PerformanceCampaignTests(unittest.TestCase):
    def make_pair(self, root: Path, index: int, order: str, ratio: float = 0.8) -> Path:
        pair = root / f"pair-{index:02d}"
        pair.mkdir()
        arms = {}
        for arm, seconds in (("off", 10.0), ("on", 10.0 * ratio)):
            crash = pair / f"crash-{arm}"
            crash.mkdir()
            lines = []
            for phase in range(2):
                request = f"{index}-{arm}-{phase}"
                for token_index in range(1, 130):
                    step = seconds * 1_000_000_000 / 128
                    stamp = int((phase * 200 + token_index - 1) * step)
                    lines.append(
                        f"DS4_TOKEN_TIMING request={request} index={token_index} "
                        f"monotonic_ns={stamp} token={token_index}\n"
                    )
            cmd = crash / "cmd.log"
            cmd.write_text("".join(lines), encoding="utf-8")
            arms[arm] = {
                "arm": arm,
                "safe_returncode": 0,
                "independent_completion_tokens": 129,
                "independent_warm_completion_tokens": 129,
                "generated_sha256": f"generated-{index}",
                "crash_evidence": str(crash),
                "crash_artifact_sha256": {"cmd.log": digest(cmd)},
            }
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "arm_order": order,
            "required_completion_tokens": 129,
            "binary_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "repository_head": "e" * 40,
            "freeze_sha256": "f" * 64,
            "request_sha256": f"{index:064x}",
            "public_randomness": {
                "round": 7000000 + index,
                "randomness": f"{index + 100:064x}",
                "signature": f"{index + 200:0192x}",
                "freeze_floor_round": 6999999,
            },
            "checks": {"all": True},
            "arms": arms,
        }
        (pair / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifacts = {
            "summary.json": digest(pair / "summary.json"),
        }
        manifest = {
            "schema_version": 1,
            "binary_sha256": summary["binary_sha256"],
            "model_sha256": summary["model_sha256"],
            "tokenizer_sha256": summary["tokenizer_sha256"],
            "repository_head": summary["repository_head"],
            "freeze_sha256": summary["freeze_sha256"],
            "request_sha256": summary["request_sha256"],
            "arm_order": order,
            "required_completion_tokens": 129,
            "artifact_sha256": artifacts,
        }
        (pair / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return pair

    def run_scorer(self, pairs: list[Path], output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/env", "-i", "HOME=/nonexistent", "PATH=/usr/bin:/bin",
             "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "/usr/bin/python3", "-I", "-B",
             str(SCORER), "--output", str(output), "--repo", str(output.parent),
             "--freeze", str(output.parent / "freeze.json"),
             *map(str, pairs)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_internally_consistent_synthetic_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orders = ["off-on", "on-off", "on-off", "off-on"] * 2 + [
                "off-on", "on-off"
            ]
            pairs = [self.make_pair(root, i, order) for i, order in enumerate(orders)]
            result = self.run_scorer(pairs, root / "result")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "result/summary.json").exists())

    def test_fixed_completed_time_bound(self):
        baseline = [10.0] * 5
        candidate = [8.0] * 5
        self.assertAlmostEqual(
            SCORER_MODULE.upper_ratio(candidate, baseline), 0.8, places=14
        )

    def test_ambient_python_is_rejected(self):
        result = subprocess.run(
            ["python3", str(SCORER), "--output", "/tmp/not-created"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("isolated Python", result.stderr)

    def test_malformed_timing_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cmd.log"
            rows = []
            for phase in range(2):
                for index in range(1, 130):
                    rows.append(
                        f"DS4_TOKEN_TIMING request=r{phase} index={index} "
                        f"monotonic_ns={phase * 1000 + index} token={index}\n"
                    )
            rows.insert(5, "DS4_TOKEN_TIMING malformed\n")
            path.write_text("".join(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed token timing"):
                SCORER_MODULE.measured_timing(path, digest(path))

    def test_reused_crash_evidence_is_rejected(self):
        rows = []
        for index in range(10):
            arms = {}
            for arm in ("off", "on"):
                arms[arm] = {
                    "completed_seconds": 8.0 if arm == "on" else 10.0,
                    "crash_identity": "shared",
                    "cmd_log_sha256": ("a" if arm == "off" else "b") * 64,
                    "executed_identity": ("1", "2", "c" * 64, "3:4"),
                    "request_id": f"{index}-{arm}",
                }
            rows.append({
                "path": f"/pair/{index}",
                "request_sha256": f"{index:064x}",
                "randomness_round": 100 + index,
                "randomness": f"{1000 + index:064x}",
                "bindings": {"same": True},
                "arms": arms,
            })
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "validated CLI authority"):
                SCORER_MODULE.write_result(Path(temporary) / "result", rows)

    def test_scorer_recomputes_runtime_evidence(self):
        source = SCORER.read_text(encoding="utf-8")
        for contract in (
            'cmd_log.count("direct expert-slot arena mapping enabled")',
            'cmd_log.count("direct expert-slot dispatch layer=")',
            'fault_re.findall(cmd_log + "\\n" + main_log + "\\n" + kernel_log)',
            'memory-preflight.json',
            'stored checks differ from recomputed runtime evidence',
            'response-bound timing receipt is invalid',
            '__name__ != "__main__"',
            'warm[1] == measured[0]',
            'measured[1] == final',
            'oom_group_kill (\\d+)',
            'right - left <= 0.6',
            'memory samples do not cover execution',
            'canonical environment is invalid',
        ):
            self.assertIn(contract, source)

    def test_dispatch_chain_mutations_fail(self):
        SCORER_MODULE.validate_dispatch_chain("off", (0, 0), (0, 0), 0)
        SCORER_MODULE.validate_dispatch_chain("on", (0, 2), (2, 4), 4)
        for arm, warm, measured, final in (
            ("on", (0, 1), (1, 2), 1),
            ("on", (0, 0), (0, 1), 1),
            ("off", (0, 1), (1, 1), 1),
            ("on", (2, 1), (1, 2), 2),
        ):
            with self.subTest(arm=arm, warm=warm, measured=measured, final=final):
                with self.assertRaises(ValueError):
                    SCORER_MODULE.validate_dispatch_chain(arm, warm, measured, final)

    def test_cgroup_event_mutations_fail(self):
        prefix = "2026-08-06T00:00:00.000000000+00:00 cgroup_verified memory_swap_max=0\n"
        clean = (
            prefix + "2026-08-06T00:01:00.000000000+00:00 cgroup_final "
            "current_bytes=1 peak_bytes=2 swap_current_bytes=0 "
            "events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n"
        )
        SCORER_MODULE.validate_cgroup_safety(clean)
        for field in ("max", "oom", "oom_kill", "oom_group_kill"):
            mutated = clean.replace(f"{field} 0", f"{field} 1")
            with self.subTest(field=field), self.assertRaises(ValueError):
                SCORER_MODULE.validate_cgroup_safety(mutated)

    def test_memory_sample_coverage_mutations_fail(self):
        main = (
            "2026-08-06T00:00:00.000000000+00:00 executed_candidate_verified x\n"
            "2026-08-06T00:00:01.000000000+00:00 executed candidate exited; x\n"
        )
        def row(offset: str, available: int = 20_971_520) -> str:
            return (f"2026-08-06T00:00:{offset}+00:00 mem_avail_kb={available} "
                    "eng_rss_kb=1 read_bytes=1 cgroup_current_bytes=1 "
                    "cgroup_peak_bytes=1 cgroup_swap_current_bytes=0")
        clean = "\n".join((row("00.100000000"), row("00.400000000"),
                            row("00.700000000"), row("01.000000000"))) + "\n"
        self.assertEqual(SCORER_MODULE.validate_memory_samples(clean, main), 20.0)
        for mutated in (
            row("00.100000000") + "\n",
            clean.replace(row("00.400000000") + "\n", ""),
            clean.replace("cgroup_swap_current_bytes=0", "cgroup_swap_current_bytes=1", 1),
            clean + "malformed\n",
        ):
            with self.assertRaises(ValueError):
                SCORER_MODULE.validate_memory_samples(mutated, main)

    def test_historical_identity_survives_byte_identical_copy(self):
        recorded = "41:195:570:1000000:1000001"
        self.assertEqual(
            SCORER_MODULE.validate_historical_file_identity(recorded, 570), recorded
        )
        for mutated, size in (("41:195:571:1000000:1000001", 570),
                              ("not-an-identity", 570)):
            with self.assertRaises(ValueError):
                SCORER_MODULE.validate_historical_file_identity(mutated, size)
        source = SCORER.read_text(encoding="utf-8")
        self.assertNotIn('stat_identity(Path("/run/user/1000/ds4-engine.lock"))', source)
        self.assertNotIn("file_identity(response_path)", source)

    def test_mutations_fail_closed(self):
        mutations = ("short", "schedule", "fixture", "digest", "safety", "nan")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                orders = ["off-on", "on-off", "on-off", "off-on"] * 2 + [
                    "off-on", "on-off"
                ]
                pairs = [self.make_pair(root, i, order) for i, order in enumerate(orders)]
                target = pairs[0]
                summary_path = target / "summary.json"
                summary = json.loads(summary_path.read_text())
                if mutation == "short":
                    cmd = Path(summary["arms"]["on"]["crash_evidence"]) / "cmd.log"
                    cmd.write_text("\n".join(cmd.read_text().splitlines()[:-1]) + "\n")
                    summary["arms"]["on"]["crash_artifact_sha256"]["cmd.log"] = digest(cmd)
                elif mutation == "schedule":
                    summary["arm_order"] = "on-off"
                elif mutation == "fixture":
                    summary["arms"]["on"]["generated_sha256"] = "different"
                elif mutation == "digest":
                    summary["arms"]["on"]["crash_artifact_sha256"]["cmd.log"] = "0" * 64
                elif mutation == "safety":
                    summary["checks"]["all"] = False
                elif mutation == "nan":
                    summary["arms"]["on"]["safe_returncode"] = float("nan")
                summary_path.write_text(json.dumps(summary) + "\n")
                manifest = json.loads((target / "manifest.json").read_text())
                manifest["artifact_sha256"]["summary.json"] = digest(summary_path)
                (target / "manifest.json").write_text(json.dumps(manifest) + "\n")
                result = self.run_scorer(pairs, root / "result")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((root / "result/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
