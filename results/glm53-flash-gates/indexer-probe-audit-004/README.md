Indexer candidate 4 audit
=========================

Campaign 49, candidate c18d3896. All 249 scoped CPU tests pass. The only implementation change views existing pinned raw-key staging as BF16 when copying to BF16 keys, preserving bits rather than converting uint16 values numerically. The exact failing expression and all 256 GPU tail mismatches were reproduced and committed before correction. Nine focused tests pass; the secret-lint self-test passes. Frozen numerical reference, controller/freezer and runtime are unchanged. A new freeze and public seed are required for the next attempt. Prior failures remain unchanged.
