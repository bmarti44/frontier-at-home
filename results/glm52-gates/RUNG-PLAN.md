# GLM-5.2 performance qualification: active rung plan

Owner course correction accepted 2026-07-31. This document supersedes the
W1-W11 execution order, but preserves those identifiers so old evidence stays
traceable. The existing G0-G5 gate evidence, `glm_safe_run.sh` witness, and the
two persistent sol reviewers remain the evidence and review mechanism. Do not
build another attestation framework.

## Goal and stopping rule

On one DGX Spark, move GLM-5.2 toward the matched DSV4 reference of 18.4 decode
tokens/s, warm agent-turn TTFT below 2 seconds, and 467 prefill tokens/s on a
32K-class fixture. Priority is decode, then TTFT, then best-effort prefill.

Fidelity is a budget. Exhaust byte-identical changes first. After every
adopted change, run the fixed 100-case `glm52-openrouter-100` NLL suite against
the 0.4515 NLL / 0.834 top-1 reference. Report every nonzero delta and the
performance it purchased to the owner. Only the owner may approve a lossy
change. Stop climbing permanently when the performance bars are met.

The measured streaming ceiling is not an open question: a token touches about
5.8 GB of expert weights, and the current 68 GB cache leaves faithful decode
at roughly 6-8 tokens/s even after I/O improvements. Reaching 18.4 requires a
resident or approximation path; prefill parity is not a reason to spend
fidelity.

## Branch and source reconciliation

- Active repository branch: `glm52-rung0-io-submission`.
- The unrelated RED test from commit `10c9cc0` is reverted on this branch. It
  belonged to the retired attestation work, not an engine behavior.
- The separate `glm52-w8-ckvstore` clone is preserved as historical work. Its
  uncommitted test edit and generated binaries are not merged.
- Engine changes will be carried into this repository as a small, reviewable
  patch plus production-path tests. The actually compiled `ds4.c` and
  `ds4_cuda.cu` must be copied to a reviewer-readable path before review;
  reviewers must ignore `vendor/ds4/`.

Identity audit at reconciliation:

- Qualification stack v4.9 recorded server prefix: `625cdef11d86`.
- Vendored `ds4-iq2xxs-down-cuda.patch` SHA-256 prefix: `bb2146ec760c`.
- Retired W8 clone HEAD: `90aeaaf72eff6ea2ffc4afd39876577df6bc9d78`;
  its local server SHA-256 prefix is `3980804edef7`.
- The last sealed affine candidate was commit `01cf7ca6c3c2a71cc0ee89496ecfba1f04f7f1d8`
  with server SHA-256 prefix `70c2a09488d2`.

Those three binaries are different candidates. No W8 or affine result may be
attributed to v4.9, and none is the current production DSV4 process. The
installed GLM tree remains unverified until its source and binary are exported
through the existing delegated control surface during a safe GLM window.

## Ranked execution plan and pre-registered gates

### Rung 0.1 - coalesced expert I/O and slab layout (W3/W8 transport)

Implement a default-off expert-I/O mode that submits aligned O_DIRECT reads at
QD8-QD32 and reads one contiguous gate/up/down slab per expert. Repacking is a
separate offline artifact with a checksummed map; it never rewrites the source
GGUF in place.

Acceptance before adoption:

- unchanged arm records a genuine behavioral RED: tensor-separated/QD1-style
  submission and no contiguous expert-slab map;
- repack round-trip is byte-identical for every expert tensor and rejects a
  corrupted map or slab;
- both arms log distinct effective modes and use the same expert IDs, prompt,
  output length, source tensors, and deterministic dispatch;
- temp-0 output is byte-identical and the 100-case NLL delta is exactly zero;
- five fresh-server ABBA/BAAB blocks show a positive decode lower confidence
  bound, with no warm-TTFT regression above 5%; report prefill without using it
  as an adoption gate;
- measured completed I/O throughput and queue depth come from harness timing,
  not engine self-report; no OOM, Xid, swap growth, timeout, or survivor;
- the safe wrapper is used with RLIMIT_AS 400 GiB and cgroup limits sized from
  arena plus RSS, while preserving at least 10 GiB available memory.

### Rung 0.2 - cross-layer theta-star prefetch (W2/W3)

Predict layer L+1 routing from layer L hidden state, fetch asynchronously, and
handoff completed entries through a staging queue so the single-threaded arena
map remains single-owner. Require sol review of event lifetime and ownership
before a serving run. Gate on byte identity, equal access streams, no stale
slot reuse, and a positive decode lower bound versus Rung 0.1.

### Rung 0.3 - zero-copy expert-slot GEMV (W3)

Consume pinned arena pointers directly, hold slot ownership through a CUDA
completion event, and remove the compact-copy path. The hit-only microgate must
show at least 5% lower completed time with identical output before serving A/B;
the serving gate is the same byte-identity and positive decode-bound gate.

### Rung 0.4 - flat-access-aware admission (W2)

Use the committed G4a trace first. A deterministic replay must gain at least
3 percentage points of byte-weighted hit rate over SLRU before engine work.
Then require byte-identical serving and a positive decode lower bound. Do not
use popularity priors or importance tiers contradicted by the flat router.

### Rung 0.5 - multi-turn TTFT (W7)

Keep the strict resume guard. First resolve the L40 same-lineage store/load
round-trip with branch-matched suffix probes and logit comparison. Any guard
relaxation requires `max|delta logit| < 1e-2`, matching argmax, correct
checkpoint selection, sol pre-registration, and owner sign-off. Then test, in
order, removal of the redundant 920 MiB re-store, live rewind, and small-suffix
batch prefill. The target is measured warm agent-turn TTFT below 2 seconds;
projections are never reported as measurements.

### Fidelity-free prefill work after decode/TTFT (W4/W5/W6)

Exact top-k, bit-safe F16 indexer storage, and wider K-tile reuse remain valid
only after the higher-priority decode and TTFT work. Require identical selected
IDs, order, tie behavior, logits, and output. Prefill gain is best-effort and
cannot authorize a fidelity trade.

### Rung 1 - MTP one-time retest

Run `mtp_ab.sh` once, only after Rung 0.1-0.3 land. Require deterministic
self-replay, the 100-case NLL result, a measured decode gain, and owner approval
for any nonzero fidelity delta. Do not retest the current integration again.

### Rung 2 - bounded lossy streaming (W1/W8/W9)

The affine-INT8 compact cache is explicitly lossy and is not qualification
plumbing. Its existing NLL campaign measured delta NLL
`-0.003998243080469773` with one-sided upper bound
`0.009672485037041306`, but it has no accepted performance purchase and is not
adopted. Any revisit must quantify that purchase and present it to the owner.

The next streaming approximation, if authorized, is a no-stall resident coarse
sketch plus full streamed refinement. Test intermediate bit rates separately;
each requires the full NLL suite, deterministic self-replay, retrieval checks,
and owner approval. Real 512-wide captures and query-weighted error are the
offline falsifier for sub-FP8 work before kernels.

### Rung 3 - resident model (W9/W10)

This is the only plausible path to 18.4 decode. It requires an owner cost
decision before rented compute or large downloads. Verify Hugging Face shard
sums and local disk first; deletion requires owner approval and a log. Test a
QTIP-class roughly 1.75 bpw representation and/or REAP-style pruning fully
resident with `-ngl 999`; never repeat the measured-dead `--cpu-moe` setup.

### Context and switching (W8/W11)

Preserve exact NVMe cKV as the fidelity-free 1M route; keep affine compact cKV
separate as a lossy experiment. The final GLM profile must directly process at
least 1,000,000 tokens under a 1,048,576 cap, pass multi-position retrieval and
negative controls with `max_tokens >= 64`, complete generation, and retain at
least 10 GiB available memory. Switching/rollback is re-run only after a
candidate earns adoption; DSV4 and GLM engine forks remain separate.

## Final decision table

Every rung appends one same-fixture row containing context, prefill tokens/s,
decode tokens/s, warm agent-turn TTFT, NLL delta and confidence bound, top-1
delta and confidence bound, memory low point, and verdict. The least-loss row
that meets the bars wins. If no row reaches parity, preserve the best qualified
GLM profile and issue a reviewed numerical NO_GO rather than climbing without
owner authorization.
