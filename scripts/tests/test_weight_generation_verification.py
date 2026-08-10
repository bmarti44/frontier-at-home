#!/usr/bin/env python3
"""The serving path must verify weight generation by hash, not by size.

DeepSeek-V4-Flash-0731 and the pre-0731 release it replaced ship shards 2 and 3
at BYTE-IDENTICAL sizes:

    shard 1   pre-0731 5,256,864   0731 5,257,664   <- differs
    shard 2   pre-0731 49,437,013,568   0731 49,437,013,568   <- identical
    shard 3   pre-0731 47,390,237,120   0731 47,390,237,120   <- identical

Only shard 1 distinguishes the two releases by size. `21_serve_llamacpp.sh`
defaults `DSV4_VERIFY_WEIGHTS` to `size`, and `52_engine_switch.sh` starts the
engine through an `env -i` whitelist that did not set it. A weight set that mixed
generations -- 0731 shard 1 with pre-0731 shards 2 and 3, which is exactly the
state a crash partway through a multi-shard swap leaves behind -- therefore
passed verification and loaded.

The engine would then serve two thirds of a superseded model's weights under the
new release's identity, with no error anywhere. `52_engine_switch.sh` must force
full hash verification.
"""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SWITCH = ROOT / "scripts" / "52_engine_switch.sh"
LAUNCHER = ROOT / "scripts" / "21_serve_llamacpp.sh"
MANIFEST = ROOT / "weights" / "unsloth-ud-q2_k_xl" / "manifest.json"


class WeightGenerationVerificationTests(unittest.TestCase):
    def test_switcher_forces_full_weight_verification(self):
        source = SWITCH.read_text(encoding="utf-8")
        self.assertIn(
            "DSV4_VERIFY_WEIGHTS=full",
            source,
            "the production switcher must not inherit the size-only default: "
            "shards 2 and 3 are byte-identical in size across the 0731 boundary, "
            "so a mixed-generation weight set passes a size check",
        )

    def test_full_verification_is_inside_the_launcher_env_whitelist(self):
        source = SWITCH.read_text(encoding="utf-8")
        match = re.search(
            r"dsv4_launcher\(\)\s*\{(.*?)\n\}", source, re.DOTALL
        )
        self.assertIsNotNone(match, "dsv4_launcher() not found")
        body = match.group(1)
        self.assertIn("env -i", body)
        self.assertIn(
            "DSV4_VERIFY_WEIGHTS=full",
            body,
            "env -i clears the environment, so setting the variable anywhere "
            "outside this whitelist does not reach the launcher",
        )

    def test_launcher_still_accepts_only_size_or_full(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_VERIFY_WEIGHTS must be size or full", source)

    def test_shard_sizes_alone_cannot_identify_the_release(self):
        # Guards the premise. If a future release changes every shard size, this
        # test should be revisited deliberately rather than silently relaxed.
        import json

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        sizes = {entry["bytes"] for entry in manifest["files"]}
        pre_0731_sizes = {5256864, 49437013568, 47390237120}
        self.assertTrue(
            sizes & pre_0731_sizes,
            "installed shard sizes no longer overlap the pre-0731 release; "
            "the size-collision premise of this test has changed",
        )


if __name__ == "__main__":
    unittest.main()
