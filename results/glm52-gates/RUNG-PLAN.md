# GLM-5.2 performance qualification: active rung plan

Owner course correction accepted 2026-08-01. This document supersedes the
W1-W11 execution order, but preserves those identifiers so old evidence stays
traceable. The existing G0-G5 gate evidence, `glm_safe_run.sh` witness, and the
two persistent sol reviewers remain the evidence and review mechanism. Do not
build another attestation framework.

The native agent goal predates this correction and cannot be renamed while it
is active. This file and the active agent execution plan are therefore the
authoritative interpretation of that goal: maximize GLM performance with the
least fidelity spend, rather than completing old W identifiers in numerical
order. Old controller statuses are historical bookkeeping, not the work queue.

## Goal and stopping rule

On one DGX Spark, move GLM-5.2 as close as possible to the matched DSV4
reference of 18.4 decode tokens/s, warm agent-turn TTFT below 2 seconds, and
467 prefill tokens/s on a 32K-class fixture. Priority is decode, then TTFT,
then best-effort prefill. Prefill parity is likely unreachable and is never a
reason to spend fidelity.

Fidelity is a budget. Exhaust byte-identical changes first, then consider the
smallest measured fidelity spend in order of cost-effectiveness. After every
candidate change, run the fixed 100-case `glm52-openrouter-100` NLL suite
against the 0.4515 NLL / 0.834 top-1 reference. Report every nonzero delta and
the performance it purchased to the owner. A statistical threshold is not
authorization: only the owner may approve a lossy change. Stop climbing
permanently when the performance bars are met.

The measured streaming ceiling is not an open question: a token touches about
5.8 GB of expert weights, and the current 68 GB cache leaves faithful decode
at roughly 6-8 tokens/s even after I/O improvements. Reaching 18.4 requires a
resident or approximation path; prefill parity is not a reason to spend
fidelity.

Do not re-run the measured dead ends: expert keep-N, lossless or entropy
compression of routed weights, shared-basis/MoBE, REAP Q2_K with `--cpu-moe`,
prefill chunking, or purchased NVMe-oF. MTP gets exactly one retest after the
first three fidelity-free I/O levers land.

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

Current Rung 0.1 candidate reconciliation:

- engine source worktree: `/tmp/glm52-rung0-engine`, clean commit
  `cb95fc37ef005d1b95315793f72d98f79f90533f`. This is the reviewed slab
  commit plus the previously tested `score_official` all-token top-1 column;
- repository patch: `results/glm52-gates/harness/ds4-expert-slab-io.patch`,
  SHA-256 prefix `4a60e135c479`;
- the patch reverse-applies cleanly to the candidate commit, proving the
  repository patch describes that source delta;
- verified immutable sidecar:
  `/home/bmarti44/.cache/glm52-rung0-artifacts/glm52-experts-v2.slab`,
  190,028,697,600 bytes, SHA-256 prefix `62961905a685` (the full digest is in
  `G6-rung0-io-sidecar-build.json`);
- no current Rung 0.1 binary exists yet. A binary hash will be recorded only
  after DSV4 is safely stopped, the measured memory envelope is known, and a
  clean `-j2` build is frozen under `/home/bmarti44/.cache/glm52-*`.

The retired W8 branch is not an implementation dependency. Its affine-INT8
cache result remains a separate lossy datum and cannot be merged into Rung 0.

The shared-lock ACL is deployed only by
`scripts/71_install_glm_benchmark_lock_acl.sh`. It grants `bmarti44` read/write
access to the one inference lock and nothing else; it does not grant service
control, sudoers access, or arbitrary root execution. The campaign continues
to fail closed until that ACL is installed and independently verified.

After every Rung 0 campaign attempt, restore the installed DSV4 1M-fast
profile without rerunning an installer or rotating its API key:

1. require `pgrep -x ds4-server` and `pgrep -x llama-server` to be empty;
2. require swap use below 1 GiB;
3. run `systemctl reset-failed deepseek-v4-flash-llamacpp.service`, restart
   `dsv4-engine-restore.service`, then start `dsv4-guard.timer` as root;
4. verify `scripts/52_engine_switch.sh status --json`, authenticated model
   identity/completion, unauthenticated rejection, and the 1M process command.

Do not use `scripts/61_restore_dsv4_user.sh`: that historical helper launches
an 8K profile and violates the repository's current largest-context policy.

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

Before the full 68 GB-cache campaign, run one contained cache-off startup probe
to measure non-arena peak RSS. Set `MemoryHigh` from that measurement plus the
68 GB arena and explicit margin; never reuse the rejected 68/71 GiB cgroup.
The production DSV4 engine, guard timer, and guard service must all be stopped,
the shared inference lock must be held, and both engine names must be absent
before every arm. The campaign is blocked until both persistent reviewers
report no high or critical safety or measurement issue.

Implementation scope is capped. Reuse `glm_safe_run.sh`, the existing fixed
ratio/ABBA scorer, committed raw logs, and the existing sol-review pattern.
Only add code needed to observe client token timing, effective slab mode,
external I/O, process identity, and OOM/Xid/swap/survivor failures. Do not add
commitment, invocation-receipt, or cryptographic self-attestation layers.

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
