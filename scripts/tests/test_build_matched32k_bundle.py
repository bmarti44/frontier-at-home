#!/usr/bin/env python3
"""Tests for the matched-32K committed-shape bundle builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "57_build_matched32k_bundle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_matched32k_bundle", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildMatched32KBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo = root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "results" / "glm52-gates").mkdir(parents=True)
        (self.repo / "scripts" / "glm52_goal.py").write_bytes(b"frozen scorer\n")
        self.out = root / "campaign-r1234"
        self.out.mkdir()
        self.fixture_sha256 = "9" * 64
        self.raw = (
            (
                '{"fixture_sha256":"%s","record_type":"matched_arm",'
                '"value":1}\r\n' % self.fixture_sha256
            ).encode()
            + (
                '{"fixture_sha256":"%s","record_type":"matched_arm",'
                '"value":2}\n' % self.fixture_sha256
            ).encode()
        )
        (self.out / "raw.jsonl").write_bytes(self.raw)
        receipt = b'{"round":1234}\n'
        (self.out / "retained" / "scripts").mkdir(parents=True)
        (self.out / "retained" / "randomness-receipt.json").write_bytes(receipt)
        (self.out / "retained" / "scripts" / "glm52_goal.py").write_bytes(
            b"retained frozen scorer\n"
        )
        self.receipt_sha256 = hashlib.sha256(receipt).hexdigest()
        self._write_json(
            "raw.jsonl.identity.json",
            {
                "schema_version": 2,
                "record_type": "matched_campaign_identity",
                "candidate_hash": "a" * 40,
                "freeze_commit": "b" * 40,
                "randomness_receipt_sha256": self.receipt_sha256,
                "glm_binary_sha256": "d" * 64,
                "glm_model_sha256": "e" * 64,
                "glm_profile_sha256": "3" * 64,
                "dsv4_binary_sha256": "f" * 64,
                "dsv4_configuration_sha256": "4" * 64,
                "dsv4_serving_weights_manifest_sha256": "1" * 64,
            },
        )
        self._write_json("terminal-memory.json", {"available_gib": 111.25})
        self._write_json(
            "retained-manifest.json",
            {
                "schema": "test",
                "freeze_commit": "b" * 40,
                "randomness_receipt_sha256": self.receipt_sha256,
            },
        )
        self._write_json(
            "campaign-preflight.json",
            {
                "dsv4_shards": [
                    {"sha256": "2" * 64},
                    {"sha256": "5" * 64},
                ]
            },
        )

    def _write_json(self, name: str, value) -> None:
        (self.out / name).write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def scorer(gate, scorer_id, rows):
        if gate != "parity" or scorer_id != "parity.performance.v1":
            raise AssertionError("unexpected scorer registration")
        return {
            "checks": {"synthetic": True},
            "rows": len(list(rows)),
            "verdict": "PASS",
        }

    def test_writes_bound_three_file_bundle_and_preserves_raw_bytes(self):
        bundle, verdict = self.builder.build_bundle(
            self.out, self.repo, scorer=self.scorer
        )
        self.assertEqual(verdict, "PASS")
        self.assertEqual(
            {path.name for path in bundle.iterdir()},
            {"raw.jsonl", "summary.json", "manifest.json"},
        )
        self.assertEqual((bundle / "raw.jsonl").read_bytes(), self.raw)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(
            manifest["raw_sha256"], hashlib.sha256(self.raw).hexdigest()
        )
        self.assertEqual(
            manifest["summary_sha256"],
            hashlib.sha256((bundle / "summary.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["scorer_sha256"],
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["frozen_scorer_sha256"],
            hashlib.sha256(
                (self.out / "retained/scripts/glm52_goal.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(manifest["scorer_module"], __name__)
        self.assertNotEqual(
            manifest["scorer_sha256"],
            hashlib.sha256(
                (self.repo / "scripts/glm52_goal.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(manifest["fixture_sha256"], self.fixture_sha256)
        self.assertEqual(manifest["drand_round"], 1234)
        self.assertEqual(manifest["glm_profile_sha256"], "3" * 64)
        self.assertEqual(manifest["dsv4_configuration_sha256"], "4" * 64)
        self.assertEqual(
            manifest["dsv4_serving_weights_manifest_sha256"], "1" * 64
        )
        self.assertNotIn("dsv4_model_sha256", manifest)
        for manifest_key, sidecar_name in (
            ("identity_sha256", "raw.jsonl.identity.json"),
            ("retained_manifest_sha256", "retained-manifest.json"),
            ("preflight_sha256", "campaign-preflight.json"),
        ):
            self.assertEqual(
                manifest[manifest_key],
                hashlib.sha256((self.out / sidecar_name).read_bytes()).hexdigest(),
            )

    def test_default_scorer_rejects_post_freeze_drift(self):
        with self.assertRaisesRegex(ValueError, "post-freeze scorer drift"):
            self.builder.build_bundle(self.out, self.repo)

    def test_default_scorer_accepts_matching_frozen_copy(self):
        scorer_path = Path(
            sys.modules[self.builder.score_registered_gate.__module__].__file__
        )
        (self.out / "retained/scripts/glm52_goal.py").write_bytes(
            scorer_path.read_bytes()
        )
        scorer_module = sys.modules[self.builder.score_registered_gate.__module__]
        with mock.patch.object(
            scorer_module,
            "_score_parity",
            return_value={"checks": {"synthetic": True}, "verdict": "PASS"},
        ):
            bundle, verdict = self.builder.build_bundle(self.out, self.repo)
        self.assertEqual(verdict, "PASS")
        manifest = json.loads((bundle / "manifest.json").read_text())
        expected = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
        self.assertEqual(manifest["scorer_sha256"], expected)
        self.assertEqual(manifest["frozen_scorer_sha256"], expected)

    def test_lineage_mismatch_raises(self):
        retained_path = self.out / "retained-manifest.json"
        original = json.loads(retained_path.read_text(encoding="utf-8"))
        for field, replacement, message in (
            ("freeze_commit", "0" * 40, "freeze_commit mismatch"),
            (
                "randomness_receipt_sha256",
                "0" * 64,
                "randomness_receipt_sha256 mismatch",
            ),
        ):
            with self.subTest(field=field):
                retained = dict(original)
                retained[field] = replacement
                self._write_json("retained-manifest.json", retained)
                with self.assertRaisesRegex(ValueError, message):
                    self.builder.build_bundle(self.out, self.repo, scorer=self.scorer)

    def test_receipt_digest_mismatch_raises(self):
        (self.out / "retained" / "randomness-receipt.json").write_text(
            '{"round":9999}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "receipt digest mismatch"):
            self.builder.build_bundle(self.out, self.repo, scorer=self.scorer)

    def test_disagreeing_fixture_digests_raise(self):
        (self.out / "raw.jsonl").write_text(
            '{"fixture_sha256":"one"}\n{"fixture_sha256":"two"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "disagree on fixture_sha256"):
            self.builder.build_bundle(self.out, self.repo, scorer=self.scorer)

    def test_bundle_inside_out_dir_raises(self):
        nested_repo = self.out / "repo"
        (nested_repo / "scripts").mkdir(parents=True)
        (nested_repo / "scripts" / "glm52_goal.py").write_text(
            "frozen scorer\n", encoding="utf-8"
        )
        (nested_repo / "results" / "glm52-gates").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "must not be inside OUT_DIR"):
            self.builder.build_bundle(self.out, nested_repo, scorer=self.scorer)

    def test_refuses_existing_bundle_directory(self):
        expected = (
            self.repo
            / "results"
            / "glm52-gates"
            / "lossless-plateau-candidate15-matched32k-pass"
        )
        expected.mkdir()
        sentinel = expected / "keep"
        sentinel.write_text("untouched", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            self.builder.build_bundle(self.out, self.repo, scorer=self.scorer)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
