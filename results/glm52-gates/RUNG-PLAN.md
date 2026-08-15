# GLM-5.2 performance qualification: active rung plan

Owner course correction accepted 2026-08-01, owner-approved literature
revision accepted 2026-08-02, owner-directed external-research amendments
accepted 2026-08-15, and the owner-directed expert-validation course correction
accepted 2026-08-15. This document supersedes the W1-W11 execution
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
- The current production token budget measured by
  [`loadprof-2026-07-25.json`](loadprof-2026-07-25.json) is about 383 ms:
  fetch 227.5 ms, kernels 107.1 ms (70.6 gate/up plus 35.5 down), other
  44.1 ms, hit copy 2.6 ms, and fill 2.1 ms. Fetch realizes about 5.8 GB/s
  against the frozen 12.56 GB/s NVMe curve, the signature of synchronous QD1
  submission rather than a drive ceiling.
- Per-fetch SHA-256 is **inactive in the production path**. It exists only in
  the unadopted slab path, so checksum removal cannot explain or accelerate the
  current 2.3 tok/s serving profile. This corrects the superseded R-V premise
  below.
- With R-K in scope, faithful streamed decode is projected to move from about
  2.3 toward 4-6 tok/s. If the remaining lossless B-E items also pass, the
  planning plateau is 7-10 decode and 75-140 prefill tok/s. Reaching roughly
  18 tok/s decode or 450 tok/s prefill still requires residency. These are
  engineering estimates, not benchmark results.

External single-Spark results now independently anchor the demand-streaming
part of that bound. Upstream llama.cpp's GLM-5.2-UD GB10 experiment measured
about 2.20 decode tok/s at a 79% expert-cache hit rate, while its smaller cache
measured about 1.83 tok/s at 73%; its implementation and measurements are in
[llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294).
The owner-provided Pulsar audit reports 2.07-2.12 tok/s, consistent with this
repository's 1.82-2.33 tok/s range. These observations confirm the known
single-Spark demand-streaming plateau; they do not replace the matched local
plateau campaign. The remaining lossless levers below must be folded into that
campaign before any fidelity decision.

After the lossless plateau is measured, stop and present it to the owner. Do
not automatically spend fidelity to pursue residency.

Do not re-run the measured dead ends: expert keep-N/skipping, expert merging,
lossless or stream-separated entropy compression, shared-basis/MoBE,
REAP Q2_K with `--cpu-moe`, prefill chunking, or purchased NVMe-oF. Also do not
retry acceptance speculation as a primary decode lever, tree speculation,
layer-skip drafting, batch-2 expert sharing, CPU expert placement,
BuddyMoE-style substitution without a cheap NLL falsifier, MoE-SVD/D2-MoE
per-expert low-rank compression, AQLM/PV-Tuning, or standalone CALDERA.

The 2026-08-15 external review closes additional variants:

- Lossless compression of the already-quantized IQ2_XXS experts is closed by
  this repository's 99.2% retained-size result and by the quantized-model
  limits described in [On the Compressibility of Quantized Large Language
  Models](https://arxiv.org/abs/2403.01384) and
  [ZipNN](https://arxiv.org/abs/2411.05239); their useful lossless gains are on
  substantially higher-entropy BF16/FP16 storage, not this packed 2-bit form.
- GDS/cuFile is closed on this host. NVIDIA's
  [GDS release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
  say DGX Spark is compatibility-mode-only and must not load `nvidia-fs`; the
  [Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
  likewise documents the UMA platform's lack of peer-direct mechanisms.
- A GB10 hardware decompression-engine route is closed. NVIDIA's
  [nvCOMP DE support matrix](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
  lists B200/B300/GB200/GB300, not GB10.
- Fiddler/KTransformers-style CPU/GPU expert splitting remains closed because
  Spark exposes one 273 GB/s unified pool, so there is no discrete PCIe weight
  hop to avoid; the same UMA constraint is documented in the
  [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/index.html).
- Cross-layer spatial prefetch stays closed by the terminal local ABBA failure;
  external PCIe-offload claims do not transfer to this NVMe-bound UMA path.
- An internal-drive upgrade is closed: the installed Samsung PM9E1 already
  reaches its measured sustained ceiling. Purchased NVMe-oF remains separately
  owner-descoped and must not be implemented, prototyped, or designed against.
  A second Spark is equally owner-descoped. Preserve only historical context
  for a future owner decision: native NVMe multipath and
  md-RAID1 read balancing do not stripe one logical record; any future design
  would require an in-engine dual-source sub-record miss scheduler, while the
  owner-provided dual-Spark datapoint is 37-48 tok/s class. Do no current work
  against either option.

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
- The engine source of truth is the Entrpi/ds4 checkout pinned by
  `versions.lock`; `vendor/ds4/` is a stale snapshot and is never review or
  build authority. Never modify a harness file while an attempt is running;
  every launch uses a versioned immutable attempt directory.

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

#### Completed NVMe characterization (`2026-08-03d`)

The clean 45-cell read-only sweep is preserved in
`NVME-characterization-final-2026-08-03.json`. Its formal verdict is
**`NO_RESULT`**: the exact 9,732,096-byte QD1/single-job cell measured
9.354496968 GB/s, outside the frozen 3.6-6.0 GB/s method-reproduction band.
This is not silently relabeled as a PASS. All cells completed, fio reported
zero writes and trims, the 190 GB target's pre/post SHA-256 is identical, and
the maximum observed controller temperature was 60.85 C.

The complete curve nevertheless falsifies the raw-drive bottleneck hypothesis
and supplies bounded engineering diagnostics. Exact expert records peak at
12.56426434920906 GB/s sustained with one QD4 ring; QD8 is close, while QD16,
QD32 and four independent jobs are slower. Sequential 16 MiB reads peak at
11.711432152234353 GB/s sustained at QD1. Therefore the next engine target is
a single QD4-QD8 ring with at most eight persistent pinned buffers, not QD16-32
or multiple independent job pools. The diagnostic 80% targets are
10.051411479367248 GB/s for exact expert records and 9.369145721787483 GB/s
for sequential evidence scans.

At the exact-record drive rate, 5.8 GB of all-miss expert traffic is a
drive-only 0.4616267087985234 seconds/token (2.166252474001562 tok/s). With a
77% byte hit rate, the 1.334 GB/token miss component alone is
0.10617414302366038 seconds/token (9.418489017398096 tok/s). These are component
bounds, **not serving ceilings**. The e637 engine already approached 10 GB/s
when an all-miss layer exposed eight reads, but cached layers normally exposed
only one or two: a representative one-miss layer spent 1.01 ms in `pread`
inside a 5.77 ms fetch window. Compute, checksum/copy cost, realized trace
bytes and overlap remain measured inputs; total decode/prefill and the
lossless plateau therefore remain provisional until the matched engine-stage
gate below closes.

#### R-K - production hot-path recovery (top rung after matched 32K)

R-K is fidelity-free and precedes R-V, MTP, cache-policy, OS, and context work.
Its premise is the measured production budget above, not the unadopted slab
path. Each numbered item is separately preregistered and byte-identical unless
the item explicitly requires reporting a delta to the owner.

1. **IQ2 down-reference A/B.** Once the pinned Entrpi/ds4 tree is readable,
   inspect every `DS4_CUDA_IQ2_DOWN_REFERENCE` definition and branch to classify
   whether it is a correctness workaround or only a determinism preference.
   The engine default is off, while the current GLM profile opted into `1`
   without a completed A/B; the historical smoke was one-arm `NO_RESULT`.
   Parameterize the existing `glm_decisive_arm.sh` and compare `1` with `0`.
   Byte-identical outputs at `0` permit production adoption; any output/logit
   delta is reported to the owner before adoption. The measured kernel bucket
   puts up to roughly 106 ms/token in scope.
2. **Downgrade synchronization.** First A/B the already-existing
   `DS4_CUDA_END_STREAM_SYNC=1` flag, which is currently absent from the
   profile. Then replace the four stubbed signal/read-after/wait/commit
   `cudaDeviceSynchronize` readbacks with event record plus synchronization on
   the readback stream; their existing `event_value` argument must no longer be
   discarded. NVIDIA's
   [CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
   is the implementation authority. Require byte identity and explicit
   signal/read-after ordering tests. The target is roughly 80-90 device-wide
   barriers/token and a material share of the measured 44.1 ms other bucket.
3. **Recover non-slab fetch bandwidth.** On the unmodified non-slab file
   layout, deepen and overlap miss submission with one bounded QD4 ring. Do not
   introduce a sidecar, slab, or layout change. The frozen
   [`NVME-characterization-final-2026-08-03.json`](NVME-characterization-final-2026-08-03.json)
   shows 12.56 GB/s at single-ring QD4 and lower results at QD16/QD32. The
   bounded target is fetch 227.5 ms toward 110-130 ms with identical bytes and
   access accounting.
4. **Close F13 without relaxing first.** Build the missing reproduction:
   disk-checkpoint restore plus long-suffix extension through a BPE-junction
   live-cache miss, following
   [`docs/ds4-glm-resume-frontier-bug-2026-07-26.md`](../../docs/ds4-glm-resume-frontier-bug-2026-07-26.md).
   If divergence reproduces, fix and requalify it. Only a bounded preregistered
   non-reproduction may be presented to the owner as evidence for relaxing the
   strict guard. This item targets minutes per affected warm agent turn.
5. **One zero-risk hygiene candidate.** Hoist the roughly 51 per-launch
   `getenv()` calls from routed-MoE launch into the existing `g_cuda_*`
   initialization (about 4,000-4,600 calls/token); remove
   `DS4_CUDA_MOE_NO_ATOMIC_DOWN` from production because the atomic path needs
   `n_tokens >= 128` and is a decode no-op; gate the always-on access-stream
   SHA-256 behind an evidence-only environment flag; and make
   `glm_safe_run.sh` call `sync -d` every Nth 4 Hz sample while preserving every
   sample and the kill floor. Refresh the frozen environment digest and run one
   matched A/B. Do not alter `DS4_CUDA_STABLE_MODEL_REMAP=1` or
   `DS4_TOKEN_TIMING_LOG=1`.

The source prerequisite is currently unmet: `bmarti44` cannot read
`/home/dsv4/ds4-project/src/ds4-upstream-master`. Restore read/traverse access
before R-K source audit and copy the exact compiled `ds4.c` and `ds4_cuda.cu`
to a reviewer-readable immutable path. `vendor/ds4/` is stale and must never be
used as fallback review authority.

#### R-V - conditional configurable slab expert-stream validation

This owner decision is authoritative and is not a reviewer-overridable
proposal, but it is conditional: the production path does not currently use
the slab or per-fetch SHA-256. Do **no immediate R-V engine work**. If a future
slab candidate is adopted for an I/O win, it must expose the modes below.
Evidence-mode governance remains unchanged.

The slab-only measured one-miss fetch window is about 5.77 ms: approximately 1.01 ms in
`pread`, 3.95 ms in whole-record SHA-256, and 0.81 ms in remaining copy and
compute. The SHA term is independently consistent with
[`G6-rung0-io-accelerated-sha-falsifier.json`](G6-rung0-io-accelerated-sha-falsifier.json),
which measured 2.293743 GiB/s and 3.95149856 ms per record. The zero-fit-parameter
decomposition reproduces the observed 2.31-2.33 tok/s decode and brackets the
observed 23-32 tok/s prefill. This diagnoses that experimental slab path only;
it is not a diagnosis of current production.

If the slab is later adopted, add one exact, logged setting to the Entrpi/ds4
streaming source pinned by `versions.lock`:

- `DS4_EXPERT_VALIDATION=full` preserves the current byte-for-byte validation
  path. It is the mandatory mode for all evidence, gate, and fidelity runs.
- `DS4_EXPERT_VALIDATION=chunked` verifies eight approximately 1.2 MiB SHA-256
  sub-digests per record in parallel. Generate a frozen Merkle-style sidecar
  offline and cross-verify its root/record relationship against every existing
  whole-record digest before first use. Round 28 rejected non-cryptographic
  checksums; it does not reject this cryptographic construction.
- `DS4_EXPERT_VALIDATION=off` performs no per-fetch digest. It may become the
  production default only after the qualification gate below passes.

The mandatory production-integrity floor for `off` is a one-time full-artifact
SHA-256 at model install or server start, logged with the digest in the run
header; unchanged read/copy accounting with copied bytes reconciled to read
bytes; and a standalone offline scrub command that re-verifies the artifact on
demand. The serving profile must fail closed if the startup/install digest is
missing or wrong.

Pre-register the production `off` qualification before engine work. Use the
same binary, seed, fixtures, and fresh-server schedule in at least five
counterbalanced blocks:

- correctness: `off` output bytes and the complete logit sequence equal `full`
  exactly. Any mismatch is terminal `FAIL` for `off`; diagnose and fix the
  underlying fetch/copy bug before a new candidate;
- performance: decode lower-95 ratio versus `full` exceeds 2.0; report prefill
  and cold TTFT; warm TTFT upper-95 ratio is at most 1.05. The planning values
  are about 3.1x decode, 7.3 tok/s, 140 tok/s prefill, and 15 seconds for a
  2048-token cold window, but only measured harness values have authority;
- safety: zero swap, Xid, OOM, stale-prefetch, and model-generation events,
  with the existing memory floor and survivor checks unchanged.

Separately accelerate evidence mode without weakening its guarantee: hash full
records across 12-16 bounded in-flight workers during prefill, or adopt chunked
mode only after whole-record-to-sidecar equivalence and both persistent reviews
pass. Evidence mode never uses `off`.

If `off` qualifies, re-measure that slab candidate with `off` as its production
configuration and `full` reserved for evidence. Do not import the earlier
7-8+ estimate into the present non-slab plateau. This spends no model fidelity;
it changes integrity-check placement and is controlled by the exact correctness
gate above.

#### Rung 0.2 - collision-resistant evidence validation and bounded submission

Round-28 review falsified the proposed non-cryptographic repeated-record
checksum before implementation: a compensating multi-byte mutation can collide
while passing a one-byte mutation test. Do not build or adopt that design.
In `full` evidence mode, every demand or speculative slab read continues to
match the record's frozen SHA-256 before its bytes become CUDA-copy-eligible.
The historical candidate below attempted to overlap that unchanged guarantee.
R-V now separately governs configurable production validation and must not be
retroactively attributed to this failed prefetch candidate.

The next candidate is default-off and uses one bounded QD4-QD8 staging ring.
The background worker publishes a slot `READY` only after the exact record key,
offset, length, model generation and frozen SHA-256 all match. The main thread
is the only cache/admission owner. Each pinned buffer remains exclusively owned
from read submission through digest validation and until its synchronous CUDA
copy returns or its completion event fires; it cannot be recycled or published
to the arena sooner. A prediction miss or late completion falls back to the
unchanged demand path, which performs the same full SHA-256. No predicted bytes
may bypass validation.

Before implementation, freeze the three contemporaneous arms and scorer:
`A=slab-off`, `B=slab-on demand-SHA`, and `C=slab-on prefetch-SHA`. All arms use
one source/binary and identical request/access fixtures in five fresh-server,
counterbalanced blocks. A bounded probe continues only if C reduces median
completed-fetch time by at least 10% versus B, produces at least 128 tokens,
and has no safety or byte-identity failure. Final adoption is an
intersection-union gate: on both the client-wall and raw-token clocks, C's
decode 95% lower ratio must exceed 1.0 independently versus A and B; its warm
and cold TTFT 95% upper ratio must be at most 1.05 versus both. Because all
comparisons must pass, no favorable historical arm may substitute and no
single-comparison win is sufficient.

Telemetry is generation- and mode-labeled and reconciles read attempts into
SHA successes/failures, prefetch READY/late/stale/fallback outcomes, CUDA-copy
counts/bytes, and arena publications. Copy bytes can never exceed successfully
SHA-validated bytes; any digest mismatch permits zero fallback, CUDA copies or
publication and invalidates the slab globally. Timers separately cover read,
SHA, wait, copy and completed fetch and must be finite and nonnegative. Mutation
tests cover first/middle/last byte, compensating multi-byte edits, wrong equal-
size record, offset/length, truncation, stale generation, concurrent completion
and buffer reuse between validation and copy. Require byte-identical outputs,
the fixed quality suite if retained, and no swap/Xid/OOM/survivor.

Implementation status (2026-08-04): the final bounded successor used repo
candidate `6885a458496822405995c60dbcb2cfad96b0818a`, engine source
`3187250cae7c16cf62c6a401df6c7b5cb210e06a`, and byte-reproducible binary
SHA-256 `753ac03cc6e8d4b643727d366b2b5233033ab8d85702c9a9aa4cb4da9ae978c7`.
It implements the default-off synchronous-copy form of this contract. A shared
state authority is used by the production CUDA source for issue, full SHA-256,
READY publication, exact-identity claim, copy, arena publication, recycling,
model reload, and integrity invalidation. Its executable tests mutate real
buffers before and after validation, exercise leases/concurrency/ring
exhaustion, and the fixed scorer rejects partial wins and malformed telemetry.
The patch applies to a fresh `e637b6f` tree and two deterministic max-`-j2`
builds produced that same binary.

The preregistered bounded B/C probe is now terminal **FAIL**. Demand-SHA's
median completed fetch was 6.435168 ms; prefetch-SHA measured 6.815296 ms, a
C/B ratio of **1.0590704081074496** against the required `<=0.90`. Median
decode fell from **1.4809392882696675** to **0.9869088852045461 tok/s** and
warm control-config TTFT rose from **45.986219305** to **73.422772407 s**.
Matched outputs were byte-identical, both arms produced two complete 160-token
repetitions, candidate QD stayed at eight, minimum available memory was
24.774864 GiB, and there were no cgroup, swap, Xid, OOM, or survivor failures.
The regression coincided with 22,809 stale prefetches, 19,705 demand
fallbacks, and 22.3% more external bytes in C; those counters are correlated
diagnostics, not an isolated causal decomposition. Preserve the result in
`R0.2-prefetch-probe-6885a45-final-2026-08-04.json`; do not run the full
campaign or adopt this prefetch path. The TTFT values are evidence-control
numbers, not serving-profile TTFT.

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
reach the gate, option A is an O_DIRECT/io_uring loader using the measured
single-stream 16 MiB path (QD1 first; QD4 only if the loader's own trace proves
overlap is needed), reading aligned superblocks directly into allocated weight
storage and copying only unaligned GGUF edges. At no point may the loader retain a second
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

Implementation status (2026-08-04): the production-source candidate is engine
commit `3332c25` on parent `3187250`, reproduced by
`harness/ds4-shared-router-correction.patch`. The correction and its matched
trace probe are default-off. Review round 50 found that the first scorer could
count copied rows and that pending state was function-static; that candidate
was not run. The first live candidate then exposed a normal-layer/MTP-layer
boundary bug after one complete token sweep; its failed attempt is preserved.
The RED-first fix uses `glm_graph_normal_layer_count()` for every next-layer
predictor and prevents stale L78 state. The corrected trace uses graph-scoped lineage plus contiguous
event, position, and layer keys and fails closed while balancing command
ownership on read/router failures. `scripts/73_run_glm_shared_router_probe.py`
runs one request on each of two fresh contained servers and binds the binary,
model, tokenizer, environment, response, raw trace, cgroup, memory, and kernel
artifacts. Before any serving A/B, require at least 1,000 unique token-layer
rows, all compared layers, at least 14 positions, zero malformed rows,
byte/token-identical probe-off/on output, and an absolute top-8 recall gain of
at least 0.02 over stale gate replay. The fixed scorer is
`scripts/72_glm_shared_router_score.py`. A failure ends this item without a
serving campaign; a pass permits review and a small runtime-performance probe,
but is not itself an adoption result.

The corrected falsifier passed on candidate `f9812ec`: 13,838 unique matched
rows across 187 complete positions raised top-8 recall from
0.7596744471744472 to 0.7895378667437492, an absolute gain of
0.02986341956930194. Both arms completed 128 tokens with identical token IDs
and output bytes; containment had no pressure, OOM, swap, Xid, or survivor and
kept at least 24.522499084472656 GiB available. The terminal evidence is
`R0a-shared-router-f9812ec-final-2026-08-04.json`. This is a recall result, not
a serving-speed result: the trace arm synchronously reads and logs every set.
The small serving screen passed its 95% non-regression falsifier at 2.2740
tok/s corrected versus 2.3021 tok/s off (ratio 0.9878), so the item advanced to
the frozen five-block campaign.

Terminal serving result (2026-08-04): **FAIL; default-off; do not adopt.** The
20-arm `ABBA/BAAB/ABBA/BAAB/ABBA` fresh-server campaign produced identical
128-token outputs and no containment, OOM, swap-growth, Xid, or survivor
failure. Across five paired block means, corrected decode was consistently
slower: its one-sided 95% lower ratio was 0.9886772305621901, while adoption
required a value strictly above 1.0. TTFT passed its non-regression gate with a
one-sided 95% upper ratio of 1.0460583955158789 against the 1.05 ceiling. The
TTFT values are control-configuration timings and are not the adopted warm
prefix profile's 1.76-second TTFT. Terminal evidence is
`R0a-shared-router-campaign-e745e4c-final-2026-08-04.json`. This closes
R0-UPGRADE a without spending fidelity: better recall did not overcome the
correction's serving overhead. Continue to R0-UPGRADE b/c; do not compose this
rejected correction into their baseline.

For any future predictor that passes its offline gate, fetch asynchronously and
hand completed entries through a staging queue so the
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

Bootstrap diagnostic (2026-08-04): the 1,280-token, 75-layer G4a trace confirms
large oracle headroom at the production-sized 7,398-slot cache: LRU hit rate
0.737404 versus exact Belady 0.880451 (+14.304688 percentage points). A causal
history-only least-stale null, using the most recent inter-arrival interval and
scan-resistant admission bypass, reached only 0.605353 (-13.205078 points
versus LRU). This is `POSTHOC_DIAGNOSTIC_ONLY` because the old trace lacks
fixture boundaries and was already inspected. It nevertheless falsifies engine
work on standalone interval/history least-stale. R0-UPGRADE c is now coupled to
R0-UPGRADE b: reconsider it only if a held-out trained probe produces calibrated
future-use probabilities that beat the frequency prior. See
`R0c-causal-least-stale-diagnostic-2026-08-04.json`.

#### Bounded decaying-hotness cache-policy probe

This is distinct from the falsified least-stale policy. Replay upstream's
decaying route-hotness admission/eviction score with an LRU tiebreak against
the current SLRU using the existing `harness/slru_ab.sh` methodology and
byte-identical access-stream digests. The policy and its 79% GLM-5.2 cache-hit
observation are documented in
[llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294).
The local exact-Belady value of 88.0% is only the upper-bound headroom claim.
Pre-register and close offline unless replay beats SLRU by at least 2.0
percentage points. Only a passing replay opens a serving A/B. Serving remains
byte-identical and must show a positive completed-time bound. Re-rank the value
of each hit-rate point against the measured post-R-K fetch budget rather than
reusing the current 227.5 ms/token economics.

### Remaining lossless transport - zero-copy expert-slot GEMV (W3)

Consume pinned arena pointers directly, hold slot ownership through a CUDA
completion event, and remove the compact-copy path. The hit-only microgate must
show at least 5% lower completed time with identical output before serving A/B;
the serving gate is byte identity and a positive decode lower bound. Any new
device-copy path must use pinned memory or carry an explicit measured
justification in review.

Terminal result (2026-08-06): **FAIL; default-off; do not adopt.** Ten fresh
paired arms in five `ABBA/BAAB` blocks completed with exact 129-token output,
byte-identical OFF/ON responses, valid event/slot ownership, and no OOM, Xid,
swap growth, survivor, truncation, or memory-floor failure. Mean completed time
fell from `59.0415113424` to `57.0098507536` seconds, a `3.4439267281%`
geometric point improvement (approximately `2.1680` to `2.2452` tok/s in this
control configuration). The one-sided 95% upper candidate/baseline ratio was
`0.9690380685669832`; adoption required `<= 0.95`, so the measured improvement
is real but too small for W3's preregistered bar. The fixed scorer and all ten
pair bindings replayed exactly for both persistent reviewers. Nash scored the
final record 100 and Singer 99 with no critical, high, or medium issues. Singer
left one nonblocking linter-syntax note: uppercase hexadecimal is accepted in
the exact path-scoped binary-digest slot, while the frozen scorer requires the
lowercase digest actually recorded. Evidence is
`W3-performance-campaign-v8-fail/{manifest.json,raw.jsonl,summary.json}`. Do not
rerun or weaken this gate; continue to the MTP address-oracle work.

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

Rung 0.5 remains the primary oracle lever **after R-K**. Upstream merged
GLM-5.2 NextN/MTP support in
[llama.cpp PR #25980](https://github.com/ggml-org/llama.cpp/pull/25980), built on
the generic MTP work in
[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). Its reported
resident-model result is about 1.37x decode, position acceptance
`0.83/0.65/0.49`, and mean accepted length about 2.98; MTP-off graphs are
bit-identical. Upstream also reports that MTP-on greedy output can diverge at
near-tie tokens because batched verification changes floating-point reduction
order. Therefore its speed result does not satisfy this repository's stricter
byte-identity gate: any local divergence closes the serving arm. Before any
engine work, inspect the frozen IQ2_XXS artifact and conversion lineage to
prove the `blk.78` NextN head was not removed by the converter's `--no-mtp`
path. The NextN loading/conversion distinction is also documented in upstream
[issue #26290](https://github.com/ggml-org/llama.cpp/issues/26290).

The pre-registered falsifier is offline and uses only the frozen P0 corpus.
For every layer, compute the exact union of routed expert IDs across `k=2,3,4`
consecutive real decode tokens. If the median layer union at `k=3` is at least
20 experts, retain the demotion arithmetic and close the conditional serving
arm. If it is at most 16, open one bounded greedy-MTP serving probe that fetches
each union once per verification pass. It must preserve byte-identical target
output and its decode lower-95 ratio must exceed 1.10 to adopt. Values strictly
between 16 and 20 are an inconclusive offline `NO_RESULT` and do not authorize
engine work. Before any serving probe, re-derive its expected value from the
measured post-R-K per-layer fetch/kernel/other budget. This gate tests the
mechanism reported by
[SpecMoEOff](https://arxiv.org/abs/2508.21706),
[SP-MoE](https://arxiv.org/abs/2510.10302), and
[MoE-SpeQ](https://arxiv.org/abs/2511.14102) without assuming their offload
speedups transfer to this flat-router UMA system.

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

Status (2026-08-05): the raw-event volume and capture-safety floor now pass.
Candidate `2ff949c` captured 1,209,150 token-layer events from two distinct
8,061-token requests across exact routed layers 3 through 77. The fixed scorer
validated 450 layer/chunk events and 2,250 files (32,231,563,200 bytes), with
byte/token-identical OFF/ON outputs, zero pressure/OOM/swap events, and at least
31.934 GiB host memory available. Engine commit `4d878c2` prevents the prior
30-GiB trace page-cache accumulation by applying `POSIX_FADV_DONTNEED` only
after each trace file is fully written and fsynced; both persistent reviewers
qualified the run and Nash independently replayed the complete scorer. The
reviewed receipt is `R0b-union-corpus-pass-2ff949c.json`.

The two atomic compacted shards contain 604,575 rows each and are bound by
`R0b-union-corpus-compaction-pass-2ff949c.json`. Router probabilities retain a
maximum absolute error of `1.7881393432617188e-07` and FP16 top logits at most
`0.0078125`; the int4 hidden feature's worst per-chunk NRMSE is
`0.13815192062923207`. Keep the raw corpus until P1 is terminal and preserve a
small FP16 hidden-state holdout before interpreting predictor failures.

P0 is not otherwise complete: these two requests share one long-fixture
lineage. Before training, add the frozen 100-case quality prompts and long-agent
transcripts, reject duplicate/cross-split content hashes, and commit immutable
fixture-grouped train/calibration/test assignments. This remaining diversity
work may add examples but must not weaken or replace the qualified >=1M source.

The immutable quality split is preregistered in
`R0b-union-p0-split-plan.json`. It reuses the committed drand-round-6329090
five-block ordering rather than reshuffling after observing routing data:
blocks 00-02 are train (60 cases), block 03 is calibration (20), and block 04
is untouched test (20). The two qualified long requests stay together in train
because they share one fixture lineage. The first five cases of train block 02
are excluded from probe fitting and retain raw float32 hidden tensors plus FP16
and int4 views for a bounded, case-level feature-precision diagnostic; neither
calibration nor test is inspected for that decision. The suite exposes no
broader semantic-family provenance, so the result is explicitly a held-out
case split within this suite, not a claim that unknown paraphrases are
independent. The split contract requires a content-complete fixture digest,
exact 100-case request ledger, per-token/layer/field bijection, case-confined
K windows, split-specific shards, deterministic failure mutations, and a
precomputed disk/safety envelope before capture. Expected prompt tokens are
independently frozen from the exact GLM rendered-prompt template and tokenizer,
so identically truncated OFF/ON arms cannot pass by agreeing with each other.

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
An expert-history Markov baseline trained only on the first 70% and evaluated
on the final 30% raises recall materially: at budget 32, K=2 improves from
34.9832% to 54.7135% (+19.7303 percentage points), K=4 from 32.3048% to
45.8401% (+13.5353 points), and K=8 from 29.2904% to 38.4113% (+9.1209
points). Full-set coverage remains poor (0.6946% even at K=2/budget 32), so
calibrated sets may still prove uneconomic. These are reproducible post-hoc
diagnostics in `harness/g4a_trace_analysis.py`, not P1 acceptance. They show
that recent routing carries useful signal beyond flat popularity and justify
collecting the fixture-grouped hidden/logit P0 corpus; they do not justify an
online predictor yet.

Train-only P1 status (2026-08-05): candidate `6ade34d` completed the frozen
three-fold cross-validation over all 75 routed layers, 57 training requests,
and 1,324,350 valid K=4 events without a CUDA/Xid/OOM fault. At the
preregistered K=4, budget=32 selection point, macro-request recall was 25.9584%
for the fold-local frequency prior and 51.4214%, 58.1163%, and 63.7837% for
ranks 8, 16, and 32 respectively, so rank 32 is frozen. Its event-weighted
recall was 17.9106% versus 14.2103% for frequency. These are train-only
cross-validation measurements, not held-out performance. The bound receipt is
`R0c-union-probe-p1-cv-pass-6ade34d.json`; next comes the already-frozen
five-case FP16-versus-int4 feature diagnostic, followed by the separately
captured gate-replay/shared-correction/MTP baselines before the test split is
opened once.

Feature-precision status (2026-08-05): candidate `a416101` passed the complete
75-layer five-case diagnostic after exact deterministic retraining of every
rank-32 head. At K=4/budget=32, int4-minus-FP16 macro-request recall was
`+0.0008668429684277612` percentage points and event-weighted recall was
`+0.016643735847832186` points; top-32 overlap was
`0.9890536723163842`. This qualifies the compact int4 feature for P1 but is not
a held-out probe or serving result. The receipt is
`R0c-union-probe-p1-precision-pass-a416101.json`. Baseline semantics and common
coverage are frozen separately in
`R0c-union-probe-p1-baseline-freeze.json` before the untouched test split is
opened.

Held-out status (2026-08-05): the one authorized candidate-18 run opened the
sealed 20-case test mappings, completed nine requests and part of the tenth,
then failed closed when unrelated firmware-notifier AppArmor audit records
violated an over-broad literal-empty kernel-log assertion. There was no OOM,
Xid, survivor, or engine failure, but complete coverage and a fixed-scorer
summary do not exist. The attempt is immutable **FAILED** evidence and its
scientific result is terminal **NO_RESULT**. It may not be rerun, completed,
partially scored, combined with replacement cases, or aliased under a new
namespace. No held-out scientific values were inspected. P2 and P3 are
therefore skipped. The failure, terminal disposition, and final Nash/Singer
acceptance (94/94, no high or critical findings) are bound by
`R0c-union-probe-p1-baseline-c18-heldout-runtime-attestation-failure-2026-08-05.json`,
`R0c-union-probe-p1-baseline-terminal-no-result-2026-08-05.json`, and
`R0c-union-probe-p1-baseline-review-r159.json`. Future work returns to the
lossless engine bottleneck; the trained probe is not qualified for online use.

#### P2 - split-conformal calibration

Status: **SKIPPED**. P2 requires a complete accepted P1 held-out comparison;
the terminal P1 `NO_RESULT` does not satisfy that prerequisite. Do not open or
reuse the calibration split to manufacture a substitute result.

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

Status: **PASS for production correctness; TTFT target remains open.** The
graduated source at `3ba062e5433e56df7c6da70b58cc9757e7777d54` preserves the
global guard, authenticates KVC v2 records, selects the correct 5,044-token
checkpoint, and resumes without a diagnostic opt-in. The frozen production
campaign at `W7-resume-production-pass-2026-08-07.json` measured 154,880
candidate/cold f32 logits byte-identical (`max|delta| = 0`, argmax 14,109),
while the strict control restarted from zero as required. Both persistent
reviewers gave terminal runtime PASS in round 232 with no high or critical
issues. The resumed request took 3.58 seconds, so it does **not** meet the
sub-two-second target. Continue only with byte-identical removal of remaining
restore/store and suffix-processing overhead; do not weaken the global guard.

#### W7.1 - stable-model CUDA cache generation

Status: **production-coverage diagnostic PASS; matched promotion campaign
terminal FAIL.** Candidate 21 separates per-layer stable model-map remaps from the
authoritative model-load generation behind the logged, default-off
`DS4_CUDA_STABLE_MODEL_REMAP=1` flag. The fresh contained run at
`W7-cache-generation-candidate21-pass.json` reduced the unchanged-path RED of
374 false generation-change flushes to zero, completed exactly one
response-bound indexed resume, and measured a 1.615-second warm 22-token append
TTFT. Its low point was 49,603,476 kB MemAvailable; cgroup max/OOM/kill and swap
were zero. The complete committed package independently replays the exact fixed
scorer PASS, and Nash/Singer both verified it at 100 with no findings in round
276.

The complete matched campaign is preserved at
`W7-cache-generation-campaign-candidate10-fail.json`. Across five fresh-server
ABBA/BAAB blocks (20 runs), ON was byte- and logit-identical to OFF, reduced
false generation flushes from 300 per OFF run to zero, and reduced mean
control-config warm TTFT from 2.0318397955 seconds to 1.180619677 seconds. The
TTFT geometric ratio was 0.5809940066093183 with upper 95% bound
0.5977543267655646, passing the `<=0.95` requirement. Mean decode was
1.8143087702970946 tok/s OFF and 1.817778154931558 tok/s ON, but the decode
ratio lower 95% bound was 0.999854233137691, below the preregistered `>=1.00`
non-regression requirement. The fixed scorer therefore returned `FAIL`.

The run stayed safe (minimum 49,813,584 kB MemAvailable; zero cgroup swap,
OOM, Xid, or surviving descendants), and both persistent reviewers independently
approved the terminal FAIL at 100 with no findings in round 287. That original
verdict and its preregistered `>=1.00` decode bound remain permanent.

The owner subsequently created the separately named **W7.1a owner-adoption
gate**. Its owner-set primary criterion is warm-TTFT ratio upper-95 `<=0.95`;
its secondary decode non-regression criterion is lower-95 `>=0.995`. The same
round-287 evidence passes both at `0.5977543267655646` and
`0.999854233137691`, respectively. The binding-only adoption record is
`W7-cache-generation-W7.1a-owner-adoption.json`; Nash and Singer approved its
evidence binding at 100 with no findings in round 295. The exact binary is
  `eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2`.
That binary and `DS4_CUDA_STABLE_MODEL_REMAP=1` are therefore the GLM default.
W7.2 remains unauthorized. No NLL rerun was required because the full logit
sequence was byte-identical with maximum absolute delta zero. The campaign
timings remain control-configuration numbers, not a new serving-profile
measurement.

### Fidelity-free prefill work after decode/TTFT (W4/W5/W6)

Exact top-k, bit-safe F16 indexer storage, and wider K-tile reuse remain valid
only after the higher-priority decode and TTFT work. Require identical selected
IDs, order, tie behavior, logits, and output. Prefill gain is best-effort and
cannot authorize a fidelity trade.

W4 exact top-k is terminal **NO_RESULT** and is not adopted. Candidate 10 ran
all five fresh-server ABBA/BAAB blocks (20 complete 19,783-token arms) safely
and byte-identically, but the frozen scorer failed closed before evaluating the
serving metric. The older microgate stored its lower bound with the truncated
`t=2.131846786`; the serving scorer replayed it with
`t=2.131846786326649` and a `math.isclose` effective tolerance of
`2.982571005715445e-12`. The resulting `1.2098322343945256e-11` difference
was correctly rejected. Preserve that authoritative scorer `FAIL`; it is not
an authoritative performance FAIL or PASS.

For branch disposition only, the complete arm data were recomputed post hoc
with the preregistered serving formula. OFF averaged `536.9154678945` seconds
(`36.8456510996` prompt tok/s) and ON averaged `534.5364878795` seconds
(`37.0096344189` prompt tok/s). The five block ratios were
`[1.0048340785, 1.0026998695, 1.0008500127, 1.0067321986, 1.0071495086]`,
with lower-95 `1.0019003405`, far below the required `1.05`. These are
explicitly diagnostic-only values, not acceptance evidence. Nash and Singer
independently verified the artifact closure, calculation and safe execution in
round 310 (scores 97 and 100, no high/critical findings) and approved
terminating W4 without another multi-hour run. See
`W4-serving-candidate10-runtime-fail.json` and
`W4-serving-candidate10-review-r310.json`. W5/W6 remain eligible as distinct
mechanisms and must not reuse or reinterpret the W4 result.

W5 is paused after a convergence break and is not adopted. Its exact
default-off flag is
`DS4_CUDA_GLM_INDEXER_CACHE_F16=1`. The candidate may change only the
128-wide indexer-key cache on CUDA from F32 to FP16; compact latent and RoPE
storage, quality mode, selected-ID ordering/ties, and all other caches remain
unchanged. The fixed acceptance formula is the existing `W5` formula in
`scripts/glm52_goal.py`: `scores_identical`, `ids_identical`, and
`logits_identical` must all be true, and isolated candidate indexer allocation
bytes divided by isolated baseline indexer allocation bytes must equal exactly
`0.5`. Additionally, the production-path test requires non-indexer compact
allocation to be unchanged and explicit flag value `0` to equal default OFF.
Byte-identical generation remains the serving confirmation backbone; no NLL
budget is spent.

Candidate 1 preserved the intended production-path RED and implemented the
isolated F16 indexer allocation, but reviewer findings did not strictly shrink
under the repository convergence rule. Preserve the committed W5 attempt and
do not adopt or silently carry the representation into another gate. It may be
reopened only as its own bounded candidate with the named blockers resolved.

W6 K-tile reuse is terminal **NO_RESULT** and is not adopted. The corrected,
randomized CUDA microgate used the frozen canonical binary below.
SHA-256 `703237062fafa70e240ce6d525cd2fc9cb93fb774c78842398371285ea722d1a`.
Both wider variants preserved exact scores, ordered selected IDs, complete
writes, canaries, causal cases and ragged tails, but both were slower: width 2
had geometric speedup `0.9616597011` with lower-95 `0.9552284552`, and width 4
had geometric speedup `0.8807974994` with lower-95 `0.8756106404`, against the
fixed `1.05` requirement. Nash and Singer found no critical/high issues in the
terminal review. See `W6-indexer-tile-reuse-terminal.json`.

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

The Rung 0.5 offline union falsifier above is the only exception: it tests MTP
as an expert-address oracle and conditionally permits one exact greedy serving
probe. It does not reopen acceptance speculation as an unconditional primary
lever.

### Rung 0.7 - bounded Grace OS tuning

After the matched 32K campaign freezes, test AutoNUMA disabled, THP `always`,
measured proactive compaction, and then explicit hugepages for the pinned arena
as separate reversible arms. NVIDIA's
[Grace OS tuning guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
recommends disabling AutoNUMA for GPU-heavy Grace workloads, documents THP and
hugetlbfs tradeoffs, and warns that overly aggressive proactive compaction can
increase translation latency. Therefore do not bundle the knobs or assume a
win. Use the standard matched decode fixture, require byte-identical output,
and adopt an arm only when its decode improvement lower-95 is at least 2%.
Restore the prior host settings after every arm and after any failure.

### Lossless plateau decision

After all Rung 0 work, run the same-fixture performance gauntlet and report the
measured plateau to the owner. With R-K in scope, current production is expected
to move from about 2.3 toward 4-6 decode tok/s; with B-E passing, the broader
lossless planning range is 7-10 decode tok/s and 75-140 prefill tok/s.
Measurements alone populate the decision table. Hardware additions are not
continuation options under the 2026-08-15 owner directive: NVMe-oF and a second
Spark remain recorded context only. Stop there for the owner's choice between
the measured lossless profile and separately authorized fidelity/residency work.
If the MTP-union, decaying-hotness, and OS arms all pass, the planning envelope
is approximately `1.3-2.0x * 1.05-1.15x * 1.02-1.05x` on top of 2.33 tok/s,
which is consistent with the existing 6-10 tok/s lossless estimate. This is a
composition hypothesis, never a reported result.

### Rung 2 - bounded lossy streaming (W1/W8/W9)

Fidelity ordering is unchanged. R-K spends no model fidelity, and conditional
R-V changes integrity checking rather than model bytes. The packed
FP4/E2M1 compact-cKV experiment is
the active lossy frontier because its real-capture falsifier has cleared; it
still requires the fixed 100-case NLL/top-1 gate and owner adoption. Rung 3
residency remains the only plausible path to 18.4 tok/s and remains separately
owner-gated. No Rung 0, 0.5, 0.7, or context-qualification item spends fidelity.

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
the offline falsifier before kernels. That falsifier has now produced a bounded
**post-hoc diagnostic PASS** from the hash-bound real capture: plain E2M1
measured query-weighted relative RMSE `0.0408614719`, Hadamard E2M1
`0.0407874594`, and Hadamard plus per-channel correction `0.0407269026`, all
below the preregistered `0.05` offline ceiling across all eight stratified
layers and 29,609,024 evaluated pairs. The mathematical scorer and capture
were unchanged; only the blocked root publication wrapper was bypassed.
Therefore this result justifies a default-off packed FP4 implementation
experiment, but it is not adoption evidence and cannot replace the fixed
100-case NLL/top-1 gate. See
`W9-fp4-falsifier-diagnostic-4f4a9af-2026-08-15/`.

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

Current status (2026-08-15): **W8 exact F32 NVMe cKV candidate 9 is closed
FAIL.** In the reviewed 5,066-token model-backed smoke, the exact arm completed
only the first 2,048-token checkpoint (254.004 seconds) and then exceeded the
fixed 1,800-second request timeout before returning a response. The resident
arm was correctly not run after that failure. The attempt was host-safe
(minimum `MemAvailable` 51,590,940 KiB, zero cgroup swap/OOM events, no Xid,
no survivor), but it produced neither byte-identity evidence nor terminal
exact-I/O telemetry. Round 245 closed the result with both persistent reviewers
at 100 and no critical/high issues. A review-time accidental overwrite and
restoration of `exact/http-status` is disclosed; the local attempt is explicitly
non-pristine, and the committed pre-mutation digest record plus the independent
timeout/missing-arm rules support FAIL only. Do not extend the timeout or use
this candidate for direct 1M. W9 has since completed the real 512-wide capture
and cleared its offline FP4 error falsifier diagnostically. The next bounded
context route is a default-off packed FP4 compact-cache implementation followed
by the fixed 100-case NLL/top-1 gate; a materially different exact-storage
design would require a new candidate rather than rehabilitating W8.

After the matched-32K plateau campaign freezes, qualify the largest presently
feasible lossless serving context directly, moving from 32,768 toward the
current-cache estimate of roughly 207K. This is a context-capability gate, not
a speed claim and not a substitute for the final direct 1M gate advertised by
the official [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2). Do
not prove a ladder below a context already passed; use the largest bounded
candidate first under the normal OOM containment and retrieval controls.

## Final decision table

Every rung appends one same-fixture row containing context, prefill tokens/s,
decode tokens/s, warm agent-turn TTFT, NLL delta and confidence bound, top-1
delta and confidence bound, memory low point, and verdict. The least-loss row
that meets the bars wins. If no row reaches parity, preserve the best qualified
GLM profile and issue a reviewed numerical NO_GO rather than climbing without
owner authorization.
