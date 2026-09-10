# Bounded transport diagnostic: not model qualification

The first-shard stage in native attempt002 took about215seconds against a total
600-second probe limit, before any GPU stage. The next bounded transport option
is four disjoint HTTP ranges into the same buffer. This test investigates only
that option while the independent host-swap prerequisite awaits owner action.
No native retry, inference, serving change or swap/service setting change occurs.

Freeze this source, its scorer/tests, standard-library Python and the existing
Drand verifier/node, with the pinned model revision/shard metadata, before a
later public seed. Use ABBA or BAAB selected by seed parity. A is one range;
B is four equal, disjoint ranges of the same first256MiB of shard115. Each arm
allocates one256MiB buffer. Exact HTTP206, Content-Range, Content-Length and
identity encoding are required before reading into disjoint slices. Reject short
or oversized bodies. All four complete payload SHA256 values must match.
This equality does not verify the complete shard's LFS digest and no bytes from
this test may be used for inference. Discard payload buffers after their digest.

Run alone under the shared inference lock, with GLM stopped and at least110GiB
available. Reuse the unchanged guarded GLM capture/identity/host scorer with fresh user
containment: MemoryHigh32GiB, MemoryMax34GiB (the hardened wrapper minimum),
MemorySwapMax0, OOMPolicykill, KillModecontrol-group, and180-second timeout.
This ceiling is not a reservation. Keep a64GiB whole-host kill floor and require
34+64GiB to fit measured start memory. The earlier512/768MiB proposal is replaced
by this explicit envelope; no equivalence is claimed. Retain
actual invocation, cgroup observations, interpreter/process identity, host
observations, errors and terminal cleanup. No GPU libraries are imported. A
failed prerequisite or transfer remains FAIL/NO_RESULT; do not silently retry.

The fixed transport scorer requires all four arms in the seeded order, complete
range coverage, byte equality, valid positive durations and unchanged observed
host swap counters. Preserve raw elapsed times and report only a diagnostic
ratio of the two mean durations. Two observations each give no confidence bound
and cannot establish full-file throughput, native GPU memory, reference fidelity,
context or serving speed. This test cannot admit a serving configuration.

Apply framing and malformed-evidence controls. Keep prior native attempt002 FAIL,
closed probe/controller/scorers and every serving profile byte-identical.
