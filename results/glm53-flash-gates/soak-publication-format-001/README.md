# Durability publication checksum format: RED

The unchanged staged secret lint returned 1 for the public raw checksum at
`$.raw.uncompressed_sha256`; gitleaks itself reported no leaks. A preceding
unsupported `--pre-commit` invocation returned 2 and is preserved separately.
The original manifest and publisher are retained in `failure.tar.gz` alongside
the actual lint output. The measured durability verdict remains FAIL.

Prospective correction: express the same raw digest and size through the existing
nested `raw.uncompressed.{path,sha256,size_bytes}` format used by startup001/002.
Acceptance requires identical raw/archive/summary bytes, an exact reversible
metadata mapping to the original manifest, a matching decompressed raw digest
and size, and the unchanged staged secret lint passing. Do not alter the lint,
scorer, fixtures, frozen source or measured verdict. This is publication metadata;
it does not create a new model candidate or reinterpret the failed attempt.
