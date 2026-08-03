# GLM-5.2 performance qualification: active rung plan

Owner course correction accepted 2026-08-01 and owner-approved literature
revision accepted 2026-08-02. This document supersedes the W1-W11 execution
order, but preserves those identifiers so old evidence stays traceable. The
existing G0-G5 gate evidence, `glm_safe_run.sh` witness, and the two persistent
Sol xhigh reviewers remain the evidence and review mechanism. Do not build
another attestation framework.

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

The measured physics are not open questions:

- GLM-5.2 has 256 experts per routed layer, eight active plus one shared, 75
  routed layers, and about 9.28 MiB per expert. Decode touches about 5.8 GB of
  expert weights per token.
- The router is flat: top-2 carries about 42% of gate mass and top-6 about 85%.
  Popularity priors and importance-tier cache policy are therefore weak.
- A 2048-token prefill touches about 99% of experts, or roughly a 170 GiB
  sweep. The production indexer/top-k boundary is compiled at 2048, so prefill
  chunking does not avoid the sweep.
- All-miss NVMe bottoms out around 1.83 tok/s. The 68 GB SLRU cache measured
  about 77% hits and roughly 107 ms/token of kernel-side reads, leaving a
  faithful streamed ceiling near 6-8 tok/s on the current stack.
- All lossless work together is bounded around 7-10 decode tok/s and 75-100
  prefill tok/s. Reaching roughly 18 tok/s decode or 450 tok/s prefill requires
  residency. These are engineering bounds, not benchmark results.

After the lossless plateau is measured, stop and present it to the owner. Do
not automatically spend fidelity to pursue residency.

Do not re-run the measured dead ends: expert keep-N/skipping, expert merging,
lossless or stream-separated entropy compression, shared-basis/MoBE,
REAP Q2_K with `--cpu-moe`, prefill chunking, or purchased NVMe-oF. Also do not
retry acceptance speculation as a primary decode lever, tree speculation,
layer-skip drafting, batch-2 expert sharing, CPU expert placement,
BuddyMoE-style substitution without a cheap NLL falsifier, MoE-SVD/D2-MoE
per-expert low-rank compression, AQLM/PV-Tuning, or standalone CALDERA.

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

- engine source worktree: `/tmp/glm52-score-official`, clean commit
  `e637b6f1eaaf9fbc5f08874d5f2a28e5ac618004`. It includes persistent pinned
  slab staging only for the ON path, allocation/CUDA-memory telemetry, exact
  EVP SHA-256 for the full O_DIRECT identity reads, and the corrected build
  target for the real official scorer. The original pageable OFF path remains
  deliberately unchanged;
- frozen serving binary:
  `/home/bmarti44/.cache/glm52-rung0-e637-server/ds4-server`, SHA-256
  `5a7caa3e7fded039797e6a0158dd4687b932d3b3c5f225c05ac7a656021fbd1a`;
- frozen quality binary:
  `/home/bmarti44/.cache/glm52-rung0-e637-quality/ds4-server`, SHA-256
  `3f4f6d197a37369ec20413e7ee77b87508803511106d0250ecc45671ac01e349`.
  It identifies itself as `score_official`, consumes the 100-case manifest,
  and is not the earlier accidental `ds4-eval` binary;
- both binaries were independently clean-built twice at no more than `-j2`
  and were byte-identical within their respective pairs. Candidate commits
  `5858be8`, `5294c32`, `95b3844`, and `e637b6f` preserve the scorer-build and
  full-hash RED/fix lineage on top of `840767e`;
- the existing repository patch series begins with
  `results/glm52-gates/harness/ds4-expert-slab-io.patch`,
  `ds4-expert-slab-pinned-staging.patch`, and
  `ds4-expert-slab-pinned-on-only.patch`, plus the allocation and CUDA-memory
  telemetry patches. The final EVP/scorer diffs must be exported from the
  exact `e637b6f` tree before promotion; until then that clean commit is the
  candidate source of truth;
- verified immutable sidecar:
  `/home/bmarti44/.cache/glm52-rung0-artifacts/glm52-experts-v2.slab`,
  190,028,697,600 bytes, SHA-256 prefix `62961905a685` (the full digest is in
  `G6-rung0-io-sidecar-build.json`);
- the fresh memory envelope passed at
  `/home/bmarti44/.local/state/glm52-rung0-e637-mem-20260802a/` with a 68 GB
  arena, 29,053,083,648 non-arena peak bytes, 88.299553 GiB minimum available,
  zero swap/cgroup/Xid failures, and no survivor;
- the bounded slab-on canary passed and all preceding attempts are preserved
  in `R0-slab-canary-attempts-2026-08-02.json`. The passing arm generated 160
  tokens at a control-config 1.504518 tok/s, observed 29,679 slab reads and
  peak queue depth eight, allocated eight 9,744,384-byte pinned buffers, kept
  24.878883 GiB available at the low point, and recorded zero cgroup events,
  swap, Xid, OOM, or survivor. Its 277.226-second TTFT includes the evidence-
  only 401 GB identity scan and is not a serving-profile TTFT;
- the completed five-block A/B on the same frozen `e637b6f` binary measured
  **2.2977581991 tok/s OFF** and **1.5485937026 tok/s ON** on the independent
  client-wall clock. The decode lower 95% ratio is **0.6732533994** and the
  control-config TTFT upper 95% ratio is **1.4938309019**, so slab ON is
  decisively diagnose a regression. The corrected client-wall formula was
  frozen after observation, so the formal gate is **NO_RESULT**, not a
  preregistered terminal FAIL; slab ON is still rejected from engineering
  adoption. These TTFTs isolate the slab in a stripped control
  configuration and are not serving-profile TTFT;
- the four-arm B/A/A/B 100-case campaign was deterministic and byte-identical:
  token-weighted delta NLL **0.0**, top-1 loss **0.0 pp**, baseline mean NLL
  **0.45145226406**, and hosted-reference top-1 agreement **0.83384090914**.
  Every arm had zero cgroup/OOM/swap/Xid/survivor failure and at least
  24.69 GiB available. `R0-e637-slab-final-2026-08-03.json` binds the raw,
  manifest, receipt, post-hoc summary, and quality hashes and records the
  formal **NO_RESULT**;
- post-reconciliation public randomness round `6342798` is authenticated and
  bound to the two frozen hashes. Its campaign attempt is terminal
  `NO_RESULT`: two complete arms and the third arm's requests were safe and
  valid, but a normal exit between `/proc` identity reads triggered a
  fail-closed rc=11 and aborted the schedule.
  `R0-e637-campaign-attempt-2026-08-02.json` preserves the raw bindings.
  Commits `62fbd4d` and `a4109b1` reproduce and repair that race. The later
  completed campaign above supersedes this attempt without deleting it.

The retired W8 branch is not an implementation dependency. Its affine-INT8
cache result remains a separate lossy datum and cannot be merged into Rung 0.

The shared-lock ACL is deployed only by
`scripts/71_install_glm_benchmark_lock_acl.sh`. It grants `bmarti44` read/write
access to the one inference lock and nothing else; it does not grant service
control, sudoers access, or arbitrary root execution. The campaign continues
to fail closed until that ACL is installed and independently verified.

DSV4 is idle test infrastructure on this machine, not a production service.
Agents must not pause to ask the owner to stop, restart, or restore it. They may
stop an identity-verified DSV4 process automatically when an exclusive evidence
run requires the machine, and restore it only when the active task calls for a
DSV4 validation. When restoration is required, do it without rerunning an
installer or rotating its API key:

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
The DSV4 engine, guard timer, and guard service must all be stopped, the shared
inference lock must be held, and both engine names must be absent before every
arm. This is an automated exclusivity check, not an owner-coordinated production
maintenance window. The campaign is blocked until both persistent reviewers
report no high or critical safety or measurement issue.

Implementation scope is capped. Reuse `glm_safe_run.sh`, the existing fixed
ratio/ABBA scorer, committed raw logs, and the existing sol-review pattern.
Only add code needed to observe client token timing, effective slab mode,
external I/O, process identity, and OOM/Xid/swap/survivor failures. Do not add
commitment, invocation-receipt, or cryptographic self-attestation layers.

### NVMe characterization and submission gate

This gate runs immediately after the current `e637b6f` slab A/B has a terminal
result and before Rung 0.5. No engine, evidence arm, or GPU measurement may run
concurrently. Use fio in read-only O_DIRECT mode against the expert-slab
sidecar file only; never open the raw NVMe device and never use a writable fio
mode. Sweep 1, 4, 9.28, and 16 MiB blocks at iodepth 1, 4, 8, 16, and 32 with
both one and four io_uring jobs for 60 seconds per cell. These 40 headline
cells rotate the QD order once per block-size group and balance the numjobs
order. Add two separately labeled layer-78 tail
cells (QD1/QD16 at exactly 12,386,304 bytes) and three separately labeled
16-MiB sequential cells (QD1/QD16/QD32) for the identity-scan comparator; none
of those five diagnostics may be mixed into the 40-cell serving curve.

Capture `nvme smart-log` temperature and throttle state before and after every
cell when the unprivileged controller device permits it. On this installed
host `/dev/nvme0` is root-only and the standing no-sudo rule applies, so the
mandatory fallback binds the target filesystem device through sysfs to its
exact NVMe controller and records controller hwmon alarm/max/critical values
plus partition diskstats every second. A cell waits below 70 C, stays at least
2 C below the sysfs maximum, and rejects a thermal alarm, a SMART transition
when SMART is available, or a second-half bandwidth decay greater than 15%.
The committed evidence must contain the full GB/s curve, thermal trace,
fio command/config hashes, slab device/inode/size/hash identity, and explicit
safety/exclusivity checks. The matched 9.28 MiB cell uses the engine's exact
`9,732,096`-byte O_DIRECT request, not a rounded `9.28m` fio value: the frozen
sidecar index contains 19,200 such records across routed layers 3--77, plus 256
`12,386,304`-byte records for layer 78. Report the latter separately rather
than silently mixing request sizes. If the exact-size QD1 cell does not
reproduce the roughly 4.8 GB/s engine observation, diagnose the method before
using any cell to recalibrate the plan. Sustained thermally stable bandwidth,
not burst peak, is the planning constant.

The target path, byte count, and expected full digest come only from the
committed passing `G6-rung0-io-sidecar-build.json`; the CLI cannot substitute a
different self-consistent file/digest pair. The fixed scorer accepts QD1 method
reproduction only within 25% of 4.8 GB/s;
otherwise the characterization is `NO_RESULT` and cannot recalibrate physics.
The matched serving reference is the median across start-temperature-matched
exact-size QD16/QD32 cells of
`min(fio average, second-half device bandwidth)`. The future slab engine target
is exactly 80% of that value. The identity-scan reference and 80% target use the
same formula over the distinct sequential QD16/QD32 cells.

If high-QD fio materially exceeds 4.8 GB/s, audit and correct the engine's
submission batching, io_uring depth, pinned staging-buffer count, and
completion-to-compute overlap. All eight routed experts across the available
prefetch horizon should be eligible to remain in flight; one staging buffer is
still QD1 regardless of ring depth. The eventual slab engine must sustain at
least 80% of the matched 9.28 MiB fio bandwidth. This work is coupled to the
Rung 0.5 expert-address oracle because cross-token lookahead may be required to
keep QD16+ occupied.

The same fio curve also gates an evidence-only identity-scan acceleration. The
live `b1fd7e6` campaign preflight read the 211 GB model at roughly 0.66 GB/s,
making serialized hashing a repeated multi-minute setup cost. Implement bounded
high-QD O_DIRECT reads feeding the unchanged digest algorithm, with the source
artifact opened read-only. Acceptance requires exact digest parity with the
stock scanner on model and slab, a one-byte corruption mutation that fails
closed, bounded memory, and at least 80% of the matched fio sequential rate.
This changes evidence setup time only; it is not a serving-performance claim.

After the sweep, re-derive rather than scale by analogy: all-miss decode,
cache-hit/miss decode with compute overlap, the faithful streamed decode
ceiling, and the ub2048 streamed-prefill ceiling. The currently documented
6-8 decode, 75-100 prefill, and 7-10 lossless-plateau ranges are provisional
until that artifact lands. The owner decision to stop or authorize a lossy
rung is blocked on the recalibrated measured plateau.

### DSV4 bounded cold-load acceleration

The owner has approved a DSV4 experiment after both the `e637b6f` campaign and
the fio gate. It is strictly a cold-start quality-of-life change: the serving
path, `configs/profiles/dsv4-1m-fast.env`, and admission budget
`90.2 + 4.0 + 12 + 8` remain byte-for-byte unchanged. DSV4 is not serving
production traffic here, so the agent should take the exclusive window itself
after confirming no campaign, fio, or engine process is active; do not ask the
owner to stop or restart it.

Start with an external bounded readahead window of 2-4 GiB over the stock
loader, paced ahead of its file offset and issuing `POSIX_FADV_DONTNEED` behind
the consumed range. This has the smallest serving-code surface. If it cannot
reach the gate, option A is an O_DIRECT/io_uring loader at the fio-supported
QD16-32, reading aligned superblocks directly into allocated weight storage and
copying only unaligned GGUF edges. At no point may the loader retain a second
90.2 GiB weight copy; transient memory is bounded to the declared window and
must fail closed before the host safety floor.

Build to a side install and pre-verify the existing switch/restore path before
changing the selected backend. Acceptance requires at least three stock and
three candidate cold loads, candidate load time in the 15-30 second target and
at least 50% of the matched sustained fio sequential rate, deterministic
in-memory tensor samples or exact-replay first-token logits matching stock,
unchanged profile conformance, and the complete `regression-suite.py agent-gate`
(prefix cache, turn continuation, slot thrash, and novel-19K prefill at or above
350 tok/s). Any steady-state regression is a rejection. Evidence labels this
as cold-start only. If the direct loader is adopted, record it as a candidate
for future GLM resident/rung-3 loading; do not schedule that reuse yet.

### R0-UPGRADE a - corrected cross-layer prefetch (W2/W3)

The first post-slab addition is the two-step shared-expert router correction:
compute the shared expert on layer L's state, add it to the residual, then run
layer L+1's router. This costs three small matrix multiplies and no model I/O.
The measured GLM-5.2 result reported in colibri issue 200 is a prefetch-recall
increase from 73.6% to 76.7%; each point avoids about 58 MB/token of wasted
NVMe traffic on this model.

Fetch asynchronously and hand completed entries through a staging queue so the
single-threaded arena map remains single-owner. Require Sol review of event
lifetime and ownership before a serving run. Gate on byte-identical target
output, equal target access streams, no stale slot reuse, measured recall/I/O,
and a positive completed-time lower bound versus Rung 0.1.

### R0-UPGRADE b - trained expert predictor (W2)

Train a small frozen low-rank predictor from calibration routing traces, in the
style of [SpecPrefetch](https://arxiv.org/abs/2607.24787). The literature range
is roughly 1.6-13M parameters and 3-4 additional recall points over gate replay,
with the largest value on slow storage. Freeze training data, split, seed and
weights before serving confirmation. The target model's computation and output
must remain byte-identical. Accept only a measured useful-prefetch and
completed-time gain with no cache-thrash or memory-safety regression.

### R0-UPGRADE c - least-stale admission/eviction (W2)

Replace recency/popularity policy with the online form of the existing Belady
analysis: evict the expert whose predicted next use is farthest away. Compare
against SLRU and the committed G4a trace; Apple's SpecMD work reports up to 85x
fewer collision misses than LRU, but only this machine's replay and serving
measurements count. Require byte identity and a positive completed-time/cache-
miss result. The previously preregistered 3 percentage-point hit-rate threshold
remains the minimum for engine adoption.

### Remaining lossless transport - zero-copy expert-slot GEMV (W3)

Consume pinned arena pointers directly, hold slot ownership through a CUDA
completion event, and remove the compact-copy path. The hit-only microgate must
show at least 5% lower completed time with identical output before serving A/B;
the serving gate is byte identity and a positive decode lower bound. Any new
device-copy path must use pinned memory or carry an explicit measured
justification in review.

### Rung 0.5 - MTP as an expert-address oracle

Use the MTP head K tokens ahead only to predict expert IDs and keep the NVMe
queue full across token boundaries. Wrong draft tokens may still route to
overlapping expert sets; a prediction miss causes a stall, never a change to
the target output. Measure expert-set accuracy, useful-prefetch rate, pollution,
queue depth, overlap and decode. [MoE-SpeQ](https://arxiv.org/abs/2511.14102)
reports 90.9% expert-set accuracy and 2.5-4.8x over offload baselines;
[SP-MoE](https://arxiv.org/abs/2510.10302) reports roughly 88% plus a cutoff-
depth rule; related expert-prefetch evidence is reported in
[arXiv:2508.21706](https://arxiv.org/abs/2508.21706). These are hypotheses for
this box, not claimed results.

Layer-level lookahead alone cannot fill roughly 5.8 GB/token at the measured
NVMe rate: it is reliable only a few layers ahead while about a second of I/O
lookahead is needed. Oracle prefetch therefore uses token-level lookahead while
leaving target computation exact. This is the intended path toward the faithful
6-8 tok/s streaming ceiling.

### Rung 0.5 prototype - calibrated union-probe prediction

This byte-identical prototype predicts the set union of experts needed over
future tokens; it never chooses the target model's executed experts. A miss may
stall and a false positive may consume bandwidth or cache, but neither may alter
target outputs. No NLL gate applies. Decode non-regression and byte identity do.

The lineage is explicit. Griffioen and Appleton's 1994 predictive file
prefetching work introduced probability thresholds; [TIP at SOSP
1995](https://doi.org/10.1145/224057.224064) and Vellanki and Chervenak's
[SC 1999 cost-benefit scheme](https://dblp.org/rec/conf/sc/VellankiC99.html)
framed prefetch economics; [TransFetch](https://arxiv.org/abs/2205.02269)
predicts unordered address sets over a future window;
[MoE-SpeQ](https://arxiv.org/abs/2511.14102) uses a draft model to predict
future-token expert needs; and
[DraftExpert](https://arxiv.org/abs/2607.24434) formalizes the cost of marginal
expert-set expansion. The proposed cheap direct multi-token-union probe,
per-expert conformal calibration, and NVMe queue-slot allocation by marginal
probability are the research additions. They must be described as a composition
of that lineage, not as invention of predictive prefetching.

[WiSP](https://arxiv.org/abs/2606.21868) found little single-stream decode value
from predicted expert prefetch when bandwidth, rather than prediction accuracy,
was limiting; its stronger result came from working-set allocation. The owner's
review reports a 46-55% prefetch regression in its relevant configurations, and
the current incomplete GLM slab diagnostics also show slab-on slower than
slab-off. Therefore every prediction feeds two independently gated consumers:

1. calibrated admission/eviction, ranking residents and candidates by the
   probability of use within the next K tokens; and
2. idle-slot prefetch, which may submit only below fio's measured sustainable
   in-flight depth, in descending marginal-probability order, without displacing
   demand reads.

The admission result and the prefetch result are attributed separately. An
admission-only win with rejected prefetch is a successful endpoint.

#### P0 - frozen trace corpus

Prefer extending existing `DS4_TOKEN_TIMING`/`LOADPROF` records rather than a
new telemetry framework. For each sampled token/layer record the eight routed
expert IDs, top-32 gate IDs/logits stored as FP16, and a compact gate-input
feature (initial candidate: 4-bit quantized, with layer subsampling allowed).
Use the existing 100-case quality prompts plus frozen long agent transcripts,
with train/calibration/test splits fixed by content hashes before training.
Target at least one million token-layer routing events. Telemetry is rate-limited
and default-off; if it cannot piggyback on an already-approved quality run, P0
waits rather than consuming an unplanned GPU evidence window. Raw trace schema,
quantization error, dropped-event count, and fixture lineage are committed.

#### P1 - direct-union probe and decisive baselines

Train per-layer rank 8, 16, and 32 heads in the style of
[SpecPrefetch](https://arxiv.org/abs/2607.24787), using gate input plus recent
expert history and multi-label BCE targets for the layer-local expert union at
K = 2, 4, and 8. Freeze optimizer, seed, splits, parameter count, training time,
and inference cost. On the untouched test split, report recall-versus-prefetch-
budget and precision/wasted-byte curves against:

- frequency-prior top N (mandatory null baseline for the flat router);
- gate replay;
- two-step shared-expert correction; and
- K-step MTP rollout, evaluated on exactly the same events and budgets.

If frequency prior is not beaten, stop the probe. If MTP rollout dominates the
probe and the cheaper baselines at equal end-to-end cost, merge the probe result
into the Rung 0.5 oracle as a refinement rather than creating another runtime
mechanism. Preserve the full comparison table either way.

Bootstrap status: the legacy G4a ID-only trace provides 96,000 token-layer
events, not the fixture-grouped P0 corpus. It measures mean/P95 true union sizes
of 14.096/16 at K=2, 24.498/30 at K=4, and 41.630/53 at K=8. A chronological
70/30 diagnostic frequency prior with a 64-expert budget reaches only 54.2561%
K=2 recall and 48.6062% K=8 recall, with effectively zero full-set coverage.
This is a frozen null-baseline diagnostic in `logs/g4a/trace-analysis.txt`, not
P1 acceptance; it confirms that flat-router popularity alone is insufficient.

#### P2 - split-conformal calibration

Use the frozen calibration split to turn raw per-expert scores into prediction
sets. The preregistered headline target is at least 90% empirical containment of
the true K-token union on the untouched test split; report finite-sample coverage
assumptions and confidence intervals rather than implying a guarantee under
distribution shift. Deliver per-layer reliability diagrams and coverage-versus-
set-size/wasted-byte curves. If the flat 256-expert router requires sets too
large for the fio-derived spare-bandwidth/cache budget, record the negative
result and stop before online integration.

#### P3 - online consumers after campaign and fio

P3 is blocked until the active `e637b6f` campaign is terminal and the fio curve
has established sustainable bandwidth and queue depth. Implement admission and
prefetch behind separate default-off flags and A/B each against the then-current
best OFF configuration. Both require matched requests, byte-identical target
outputs, no demand-read displacement, no safety regression, and decode
throughput whose lower confidence bound is non-negative versus that best OFF
configuration. Prefetch may use only measured spare queue slots and its coverage
budget. If combined mode wins, also run admission-only and prefetch-only arms so
the gain is attributable; if prefetch regresses, reject it and retain any passing
admission-only policy.

### Rung 0.6 - multi-turn TTFT (W7)

Keep the strict resume guard. First resolve the L40 same-lineage store/load
round-trip with branch-matched suffix probes and logit comparison. Any guard
relaxation requires `max|delta logit| < 1e-2`, matching argmax, correct
checkpoint selection, Sol pre-registration, and owner sign-off. Then test, in
order, removal of the redundant 920 MiB re-store, live rewind, and small-suffix
batch prefill. The target is measured warm agent-turn TTFT below 2 seconds;
projections are never reported as measurements.

### Fidelity-free prefill work after decode/TTFT (W4/W5/W6)

Exact top-k, bit-safe F16 indexer storage, and wider K-tile reuse remain valid
only after the higher-priority decode and TTFT work. Require identical selected
IDs, order, tie behavior, logits, and output. Prefill gain is best-effort and
cannot authorize a fidelity trade.

### Demoted acceptance speculation

MTP acceptance speculation is not a primary lever. With flat 8-of-256 routing,
the expected K-token expert union is
`256 * (1 - (31/32)^K)`, approximately `8K` for small K, so verification reads
nearly K tokens of weights and speedup approaches the acceptance rate. The
measured current integration gained only about 10%.
[Cascade](https://arxiv.org/abs/2506.20675) reports 2-3x data-movement
inflation for MoE speculation and [EcoSpec](https://arxiv.org/abs/2607.12696)
reclaimed about 1% of bandwidth on DeepSeek-V3.1. After all I/O work it may be
tested once as an optional final roughly 1.2x multiplier, informed by
[EVICT](https://arxiv.org/abs/2605.00342) and
[DraftExpert](https://arxiv.org/abs/2607.24434), with deterministic replay and
NLL evidence. Tree speculation and layer-skip drafting stay on the do-not-retry
list.

### Lossless plateau decision

After all Rung 0 work, run the same-fixture performance gauntlet and report the
measured plateau to the owner. The expected hard range is 7-10 decode tok/s and
75-100 prefill tok/s, but measurements alone populate the decision table. Stop
there until the owner chooses one of: accept the lossless profile, authorize a
bounded Rung 2/2.5 fidelity spend, or authorize the Rung 3 residency program.

### Rung 2 - bounded lossy streaming (W1/W8/W9)

The affine-INT8 compact cache is explicitly lossy and is not qualification
plumbing. Its existing NLL campaign measured delta NLL
`-0.003998243080469773` with one-sided upper bound
`0.009672485037041306`, but it has no accepted performance purchase and is not
adopted. Any revisit must quantify that purchase and present it to the owner.

If authorized, implement a no-stall resident coarse sketch plus full streamed
refinement in the style of [HOBBIT](https://arxiv.org/abs/2411.01433) and
[FloE](https://arxiv.org/abs/2505.05950). On a miss, serve the low-precision
resident tier immediately and importance-weight refinement by gate value. The
router's 42% top-2 gate mass may make low-weight approximations cheaper than
frequency alone suggests, but the 100-case NLL suite decides. Test intermediate
bit rates separately; every rate requires deterministic self-replay, retrieval,
quality and owner approval. Real 512-wide captures and query-weighted error are
the offline falsifier before kernels.

### Rung 2.5 - optional router-locality fine-tune

Only with owner authorization, fine-tune the router gates with a temporal-
locality loss and trust-KL anchor while keeping the 2-bit backbone frozen, in
the style of [ReMoE](https://arxiv.org/abs/2605.27081). Published results report
27% to 35% consecutive-token overlap and 1.77-1.99x decode on SSD-backed Jetson.
Routing changes are lossy: freeze data/splits/seeds and gate every candidate on
the paired NLL/top-1 suite before presenting performance.

### Rung 3 - resident model pipeline (W9/W10)

Quantization alone cannot make this model resident: roughly 1.75 bpw experts
still occupy about 163 GB, while true quant-only residency needs about 1.1 bpw.
The owner-approved pipeline is therefore:

1. Measure per-expert saliency `E[gate * ||output||]` over frozen streamed
   calibration data and prune 40-50% using the
   [REAP](https://arxiv.org/abs/2510.13999) method. Flat routing frequency does
   not imply flat output saliency. Published existence checks include 50%
   DeepSeek-V3.2 pruning and about 0.7 percentage-point loss on 50%-pruned
   Kimi-K2; the public `0xSero/DeepSeek-V4-Flash-0731-REAP` artifact is a
   workflow precedent on DGX Spark, not GLM quality evidence. This observer pass
   is prefill-bound and may take days; freeze and test the offline tensor
   surgery before serving.
2. Allocate bits among survivors by measured loss sensitivity rather than
   routing frequency, following
   [MC#/PMQ](https://arxiv.org/abs/2510.10962) and
   [BitsMoE](https://arxiv.org/abs/2606.00079). Target 40% prune plus a roughly
   1.7-bit tail, or about 90-95 GiB total.
3. Heal against IQ2_XXS teacher activations with block-wise
   [EfficientQAT Block-AP](https://arxiv.org/abs/2407.11062) and recalibrate the
   small router matrices. Optionally requantize survivors with existing
   ik_llama.cpp IQ2_KT/IQ1_KT trellis kernels.

This is the only plausible path to 18.4 decode. It requires owner cost and
fidelity decisions before rented compute or large downloads. Verify Hugging
Face shard sums through the API and local free disk before downloading; deletion
requires owner approval and a log. Never repeat `--cpu-moe`. BiMoE-style 1-bit
QAT is considered only if this pipeline fails its preregistered storage/NLL gate.

[TEAL](https://arxiv.org/abs/2408.14690)-style 25-35% intra-expert sparsity is a
separate optional lever with an expected 1.3-1.5x range.
[SIRIUS](https://arxiv.org/abs/2409.03856) reports reasoning-specific
degradation from contextual sparsity, so this agent workload requires a strict
reasoning/NLL gate; do not test 50% merely for margin.

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
