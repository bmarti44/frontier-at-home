#!/usr/bin/env python3
"""The protected context-qualification model must be verified by digest.

`64_context_submit.sh` copies the serving weights into
/var/lib/dsv4-context/models/deepseek-v4-flash so a context run measures an
immutable, root-owned model the unprivileged account cannot alter mid-run. That is
sound. Its validity check was not:

    [[ ! -f $protected_path || -L $protected_path ]] ||
    [[ $(stat -c %u ...) != 0 ]] ||
    [[ $(stat -c %h ...) != 1 ]]

Existence, ownership, and link count. Nothing compares the copy's CONTENT to the
weights the endpoint actually serves. A copy taken before a weight swap therefore
stays "valid" forever, and the swap on 2026-08-09 produced exactly that state: the
protected copy held pre-0731 shards (shard 1 at 5,256,864 bytes) while the endpoint
served 0731 (5,257,664). A root-mode context run would have measured the superseded
release and labelled the result with the current candidate hash.

Byte size cannot substitute for the digest here: shards 2 and 3 are byte-identical
in size across the 0731 boundary, so only shard 1 differs by size at all.

The chain that makes reading the manifest safe from a root script is already
enforced above this point: the candidate hash is verified with `git rev-parse` and
`git status --porcelain` must be empty, so the manifest content is bound to the
verified commit rather than to whatever is currently on disk.
"""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "scripts" / "64_context_submit.sh"


class ContextSubmitDigestTests(unittest.TestCase):
    def setUp(self):
        self.source = SUBMIT.read_text(encoding="utf-8")

    def test_validity_check_compares_the_protected_copy_by_digest(self):
        self.assertIn(
            "expected_digest",
            self.source,
            "the protected copy must be compared against the serving manifest, not "
            "merely stat-ed",
        )
        self.assertIn("sha256sum", self.source)

    def test_expected_digests_come_from_the_serving_weights_manifest(self):
        self.assertIn("$MODEL_SOURCE/manifest.json", self.source)

    def test_a_digest_mismatch_forces_a_refresh_rather_than_a_silent_run(self):
        # model_valid=false is the refresh trigger; a mismatch must reach it.
        window = self.source[
            self.source.index("model_valid=true") : self.source.index(
                'if ! "$model_valid"'
            )
        ]
        self.assertIn("expected_digest", window)
        self.assertIn("model_valid=false", window)

    def test_the_freshly_copied_shards_are_verified_before_being_published(self):
        """A copy that silently truncates must not become the protected model."""
        window = self.source[
            self.source.index('if ! "$model_valid"') : self.source.index(
                'OUT=$ATTEMPT_ROOT'
            )
        ]
        # shard_digest() is the sha256sum wrapper defined above this block.
        self.assertIn("shard_digest", window)
        self.assertIn('shard_digest "$model_temporary/$name"', window)
        self.assertIn('shard_digest "$source_path"', window)
        self.assertRegex(
            window,
            r"die [\"'][^\"']*(digest|sha256)",
            "a post-copy digest mismatch must die, not warn",
        )

    def test_the_manifest_is_required_not_optional(self):
        self.assertRegex(
            self.source,
            r"die [\"'][^\"']*manifest",
            "a missing or unreadable manifest must fail closed",
        )

    def test_stat_based_checks_are_retained(self):
        # The digest check is additional, not a replacement: link count 1 is what
        # makes the copy an independent inode rather than a hardlink to the
        # user-writable original.
        self.assertIn("stat -c %h", self.source)
        self.assertIn("stat -c %u", self.source)


if __name__ == "__main__":
    unittest.main()
