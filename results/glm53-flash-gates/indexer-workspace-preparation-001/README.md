# Indexer workspace preparation001: FAIL

Four fresh512MiB baseline arms passed their individual correctness, identity and
host checks. The single-request pair was byte-identical; the four-request pair
differed in ordered indices. Cache, tail, valid logits and consumed pool sets
matched, but downstream attention-output equivalence was not measured. The fixed
protocol stopped before either64MiB arm. No smaller-buffer saving, speed,
full-model fidelity or context result is established.

Original terminal-host.json is missing. Separate post-run checks verified all98
frozen file bindings but found149 changed inventoried bytecode files and830
extra bytecode files. The original runtime inventory failed; full accounting
and changed/extra bytes are retained. All four processes/cgroups were gone and host
memory recovered. The original failure is preserved, including the secondary
inventory failure. No serving profile or default changed.

The split archive reconstructs in numeric part order. Its member hashes and the
whole compressed hash are recorded in manifest.json.gz.
