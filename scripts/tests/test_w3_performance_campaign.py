"""Behavior and mutation tests for the fixed W3 campaign scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/85_score_w3_performance_campaign.py"


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
            "model_sha256": "m" * 64,
            "tokenizer_sha256": "t" * 64,
            "repository_head": "r" * 40,
            "freeze_sha256": "f" * 64,
            "request_sha256": f"{index:064x}",
            "public_randomness": {
                "round": 7000000 + index,
                "randomness": f"{index + 100:064x}",
                "signature": f"{index + 200:192x}",
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
            ["python3", "-I", str(SCORER), "--output", str(output),
             *map(str, pairs)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_valid_five_block_campaign_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orders = ["off-on", "on-off", "on-off", "off-on"] * 2 + [
                "off-on", "on-off"
            ]
            pairs = [self.make_pair(root, i, order) for i, order in enumerate(orders)]
            result = self.run_scorer(pairs, root / "result")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "result/summary.json").read_text())
            self.assertEqual(summary["status"], "PASS")
            self.assertLessEqual(summary["completed_time_ratio_upper_95"], 0.95)
            self.assertEqual(len(summary["baseline_seconds"]), 5)
            self.assertEqual(len(summary["candidate_seconds"]), 5)

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
