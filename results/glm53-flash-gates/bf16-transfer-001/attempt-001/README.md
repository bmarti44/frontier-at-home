# Partial-file HTTP diagnostic: PASS

Four transfers of the same first 256 MiB of pinned BF16 shard 115 completed in
the frozen, publicly seeded ABBA order. One connection and four disjoint range
connections returned identical bytes. Exact response framing, range coverage,
host observations, process identity and terminal cleanup passed. All payload
buffers were discarded; no downloaded sample was used for inference.

This is a small transport diagnostic. Two observations per arm support only the
elapsed-time comparison in the unchanged frozen summary, with no confidence
bound or guarantee for complete shards. It does not verify the complete shard's
LFS digest, produce native reference probabilities, or qualify GPU memory,
fidelity, context or serving speed. The native one-layer attempt002 stays FAIL.

The host retained at least 120,975,948 KiB available across 127 external samples.
Global swap counters and used swap did not change; sampled cgroup swap was zero.
The identity guard collected 146 samples and passed. The wrapper exited zero,
with no surviving process group or cgroup. No model or service was started and
no swap setting changed.

The archive preserves all 48 original files, including frozen source, metadata,
public seed, invocation, payload/host/identity observations and terminal receipts.
All 19 frozen file bindings and every archived byte were checked. The 278 raw
source lines are also retained losslessly in raw.jsonl.gz. The publication copies
the original summary.json unchanged. Source corrections, genuine RED records,
rejection tests and focused reviewer receipts remain in the parent directory.
