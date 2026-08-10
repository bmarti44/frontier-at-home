#!/usr/bin/env python3
"""The DeepSeek comparison profile must identify the GGUF generation it serves.

`configs/dsv4-profile.json` is the approved DeepSeek identity that
`56_collect_matched_evidence.py` records as the baseline arm of every GLM-vs-DeepSeek
matched comparison. Before this change it was internally inconsistent:

    binary_sha256          -> llama-server            (the llama.cpp serving stack)
    configuration_sha256   -> deepseek-v4-flash-llamacpp.service   (llama.cpp)
    weights_manifest_sha256-> configs/build-manifests/ds4-weights.json   (ds4!)
    model_files            -> ds4 base/mtp/dspark_drafter shards     (ds4!)

So the profile named the llama.cpp *configuration* but the *ds4* weights, and the
UD-Q2_K_XL GGUF generation the endpoint actually loads was bound nowhere. Worse,
`56_collect_matched_evidence.py` validated only `binary_sha256` and
`configuration_sha256` and never read any weights field at all.

The consequence is silent cross-campaign contamination: swapping the served weights
from the pre-0731 release to 0731 changes no value that a matched-evidence run
records, so a GLM candidate measured against pre-0731 and one measured against 0731
produce evidence that claims the same DeepSeek baseline.

The ds4 fields are deliberately left alone -- `test_engine_switch.py` binds them to
the ds4 weights schema on purpose. This adds the serving identity alongside them.
"""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs" / "dsv4-profile.json"
SERVING_MANIFEST = ROOT / "weights" / "unsloth-ud-q2_k_xl" / "manifest.json"
SERVING_PIN = ROOT / "configs" / "pins" / "unsloth-ud-q2_k_xl.json"
COLLECTOR = ROOT / "scripts" / "56_collect_matched_evidence.py"


class ServingWeightIdentityTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(PROFILE.read_text(encoding="utf-8"))

    def test_profile_records_the_serving_weights_manifest_digest(self):
        self.assertIn("serving_weights_manifest_sha256", self.profile)
        self.assertEqual(
            self.profile["serving_weights_manifest_sha256"],
            hashlib.sha256(SERVING_MANIFEST.read_bytes()).hexdigest(),
            "the profile must track the llama.cpp weights manifest actually loaded, "
            "not a stale or unrelated digest",
        )

    def test_profile_records_the_serving_release_identity(self):
        release = self.profile.get("serving_weights_release")
        self.assertIsInstance(release, dict)
        pin = json.loads(SERVING_PIN.read_text(encoding="utf-8"))
        self.assertEqual(release.get("repo"), pin["repo"])
        self.assertEqual(release.get("revision"), pin["revision"])

    def test_serving_identity_is_distinct_from_the_ds4_weights_identity(self):
        # Guards against a future edit that "simplifies" these into one field.
        ds4_weights = hashlib.sha256(
            (ROOT / "configs/build-manifests/ds4-weights.json").read_bytes()
        ).hexdigest()
        self.assertEqual(self.profile["weights_manifest_sha256"], ds4_weights)
        self.assertNotEqual(
            self.profile["serving_weights_manifest_sha256"], ds4_weights
        )

    def test_schema_version_was_bumped_so_consumers_must_notice(self):
        self.assertEqual(self.profile["schema_version"], 3)

    def test_collector_validates_the_serving_weights_identity(self):
        source = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("serving_weights_manifest_sha256", source)
        self.assertIn(
            'schema_version") != 3',
            source,
            "the collector must reject the old schema rather than silently accept "
            "a profile that carries no serving identity",
        )

    def test_collector_compares_against_the_live_manifest(self):
        source = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn(
            "unsloth-ud-q2_k_xl",
            source,
            "recording the digest is not enough; the collector must hash the "
            "manifest on disk and reject a mismatch, or a profile edit alone "
            "would relabel the baseline",
        )


if __name__ == "__main__":
    unittest.main()
