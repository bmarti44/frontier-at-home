# GLM-5.3-Flash CUDA status

**GLM is runnable through named experimental profiles.** An earlier configuration
passed the direct aggregate million-token context check; current settings still
need confirmation. Qwen remains the recorded/reboot default.
GLM is currently stopped. The latest reference probe completed one native layer,
but its host-monitoring aggregation failed. Full model qualification remains incomplete.

The [passive unloaded control](host-swap-accounting-001/unloaded-control-001/README.md)
completed all 121 samples with no swap activity. The subsequent
[unchanged-profile preflight](soak-native-010/README.md) nevertheless recorded one
host swap-in page during the unloaded freeze/check interval. A second bounded
[control-warmup replay](soak-native-011/README.md) passed artifact and fresh-input
preparation, then recorded another swap-in page before GLM started. Direct cgroup
counters account for the latter page in Docker's cgroup; this identifies accounting
ownership, not the trigger for the read. Neither replay admitted a model workload.
The two environmental alternatives are **NO_RESULT**; no further automatic warmup
variant is planned.

The owner stopped Docker and its socket. The subsequent
[Docker-isolated replay](soak-native-012/README.md) verified artifacts and prepared
fresh inputs, then failed on one host swap-in page before GLM loaded. Containerd,
a separate service left running after Docker stopped, lost 4 KiB of process swap
and gained one major fault in the same interval. This is a correlation, not proof
of the cause. The complete 900-census observation also retains one later swap-in
page after the failed preflight. No model workload was admitted.

The owner subsequently stopped containerd and installed the reviewed
[scoped passwordless runtime grant](runtime-access-001/README.md). Noninteractive
operation was verified. The [replay with all three services stopped](soak-native-013/README.md)
still failed on one swap-in page before model loading. The event falls before the
first process census, so attribution is unavailable. The read-only freeze completed;
no new public seed, client fixtures or model workload followed. Later quiet samples
do not revise the failure. GLM and the three container runtime units remain stopped.

Ollama's installed `kimi-k3:cloud` supplied [100 synthetic prompts](kimi-corpus-001/README.md).
Their [pinned chat serialization](kimi-serialization-001/README.md) totals 32,837
input tokens, with a 516-token maximum. These are inputs, not native reference
probabilities. A [BF16 metadata audit](bf16-local-feasibility-001/README.md) ruled
out whole-model loading on this host. A [tiny CPU test](bf16-tiny-parity-004/README.md)
then verified stock checkpoint conversion and byte-identical full-load versus
partial-module outputs, independently reproduced with seven rejection controls.
It used synthetic weights kept in memory and stock Torch fallbacks. Real-weight
streaming, GPU memory feasibility and the native 100-case fidelity result remain
unverified. Earlier preparation failures are preserved.

The first real-layer preparation selected an obsolete packaging inventory and
failed before freeze. The [correction](bf16-one-layer-preflight-001/README.md)
reused the existing serving-profile inventory and verified all 61,401 runtime
files; the temporary bytecode quarantine was restored byte-for-byte. The
[real-weight attempt](bf16-one-layer-002/README.md) then passed freeze and public
seed checks but stopped at the first shard's verification/copy stage after five
host pages (20 KiB) were written to swap. Available memory stayed above 100 GiB,
and sampled probe cgroup swap stayed zero. No completed shard receipt, GPU
forward or native reference output was recorded. Cleanup succeeded; the attempt
is FAIL and native GPU feasibility is NO_RESULT, not a memory-capacity rejection.

A separate retrospective comparison accounts the five writes to the persistent
`system.slice` cgroup over a wider interval; `user.slice` gained none. This does
not identify a particular service or the trigger. Further automatic warmup or
daemon-stop variations are not proposed. The owner has now performed the
temporary pause of the exact `/swap.img` unit. Read-only observations confirm
the unit inactive, no active swap entries and zero SwapTotal with GLM stopped
and over 110 GiB available. The [next replay preparation](bf16-one-layer-003-preparation/README.md)
keeps the native probe and all memory, OOM/Xid and swap checks unchanged under
a fresh baseline. This removes the pending host prerequisite; it does not revise
the earlier failed attempts or complete model qualification.

The [unchanged native replay003](bf16-one-layer-003/README.md) then verified two
complete shards and part of the third within its ten-minute limit. It stopped
before GPU computation. The wrapper terminated a remaining child and exited15;
the guard's terminal summary is missing. The attempt remains FAIL despite zero
swap, ample memory and separately verified eventual process/cgroup cleanup.
The [reviewed download alternative](bf16-parallel-transfer-001/PROTOCOL.md) uses
four disjoint ranges into the same shard buffer, retaining complete LFS/header
verification and all native computation, staging, scoring and safety limits.

The [parallel native replay004](bf16-one-layer-004/README.md) completed all three
whole-shard checks and a real native layer forward with finite output. Its
identity guard and cleanup passed, but the outer host scorer rejected sampling
coverage, leaving the overall verdict FAIL. Native full-model reference and
paired fidelity remain unverified. The original failure and all raw output are
preserved; a scoped review is checking the teardown timing boundary.

The [terminal coverage correction](bf16-terminal-coverage-001/PROTOCOL.md) now
keeps active identity and memory cadence separate from the guard's existing
five-second verified teardown. All 29 regression/mutation tests and the focused
adversarial review passed. Original native004 remains FAIL; corrected native
confirmation is pending.

The [current-profile preparation008](context-direct-008/preparation-summary.json)
prepared four exact 250,128-token inputs but failed the separate 4,224-token startup
fixture before model loading. Its tokenizer boundary oscillated between 4,223 and
4,225 tokens. The [focused correction](context-short-padding-001/PROTOCOL.md) reuses
existing durability padding only for that short check; exact tokens, fixture hash,
instruction, markers and negative control passed independent verification. The
long-context branch and scorer are unchanged. Fresh freeze and confirmation follow.

The [current-scheduler context adapter](context-scheduler-003/PROTOCOL.md) is also
ready: it reuses the existing512/128 launch validator and unchanged direct-input,
retrieval, stream and score code. A reviewed short-launch validation correction
is included. This preparation is not a fresh million-token result.

A separate [small HTTP transfer diagnostic](bf16-transfer-001/attempt-001/README.md)
passed byte equality, host safety and cleanup for four transfers of the same
256 MiB shard prefix. Parallel ranges reduced elapsed download time in this
sample, making them a justified option for a future native-reference probe.
This does not verify a full shard or qualify model speed, native reference,
fidelity or context. That diagnostic changed no model, service or swap setting.
The later owner swap pause is recorded above.

The current full-context profile is the [second bounded scheduler configuration](soak-scheduler-002/PROTOCOL.md):
512-token batches with a 128-token prompt-chunk cap per conversation, retaining
all four 262,144-token slots and existing memory safeguards. The
[20-request necessary window passed](soak-native-005/README.md): all four workers
admitted five requests within 300 seconds and every reply was correct. Overall
qualification remains **FAIL** because one host swap-in page occurred before the
window and seven generated kernel files changed the frozen inputs. No full-duration
run was admitted. Shutdown, recovery, identity, default/proxy/guard and post-run
artifact checks passed; both persistent reviewers verified the complete archive.

The [prepared-kernel replay](soak-native-006/README.md) reached authenticated READY
and kept all 2,484 prepared files unchanged. It failed the host startup gate:
three pages were written to swap, so no test workload was admitted. GLM cgroup
swap and limit events stayed zero. A separate six-page swap-out occurred during
artifact verification after GLM stopped. The complete 900-second census preserves
both events, partial snapd correlations and read errors; attribution remains
unknown. Both reviewers verified the full archive and negative result.

The [final-warmup cleanup attempt](soak-native-007/README.md) passed startup,
correctness, the 20-request necessary window and host checks through clean shutdown.
It reported 40 MiB of unused CUDA allocator reservations released; this single
attempt does not establish causal host or performance improvement. Two new
DeepGEMM files keep the overall attempt **FAIL** and kernel confirmation **NO_RESULT**;
no full-duration client was admitted. Both reviewers verified the complete archive.

The latest [two-file cache replay](soak-native-008/README.md) kept all profiles,
startup flags, model settings and scorers unchanged. All 2,486 prepared files
stayed unchanged and the 4,224-input-token correctness smoke passed. The host
wrote one page to swap, increasing used swap by 4 KiB; the overall verdict is
**FAIL**, and neither the necessary window nor the full-duration test was admitted.
The complete external census retains a simultaneous snapd VmSwap increase;
this correlation does not establish causation or exempt the attempt from its gate.
GLM cgroup swap and limit events remained zero. Shutdown, recovery, identity,
default/proxy/guard and post-run artifact checks passed. GLM is stopped.

The [second bounded startup experiment](soak-native-009/README.md) used process-local
CPU heap trimming, with the preceding final-warmup cleanup disabled. It passed
authenticated startup, the actual worker's frozen library check, short correctness
and all 2,486 prepared-cache comparisons. A small measured memory release did not
close the host gate: one page was read from swap during the smoke. Used swap and
swap-out counters stayed unchanged; GLM cgroup swap and limit events remained zero.
The attempt is **FAIL**, with no necessary window or full-duration admission.
Both bounded reclamation alternatives have now been tried; this branch is
**NO_RESULT**, and no third allocator variant is planned.

Shutdown, recovery, identity and post-run artifact checks passed. The complete
census also retained 17 swap-in pages after GLM stopped, during read-only service
queries; this diagnostic interval is separate from the loaded failure. The next
independent investigation concerns host activity and a quiet qualification window.
Default/proxy/guard state remained unchanged. The legacy engine guard's existing
open circuit breaker prevented restarts; these checks do not claim a running
production model. That attempt changed no service or swap-policy settings; the
later owner-requested Docker stop is recorded above.

Current-configuration direct context, sustained-load confirmation, the fixed
100-case native BF16 fidelity reference, maximum media and production switching
remain pending. Qualified production performance is not yet measured.

The [preceding 256/64 test](soak-native-004/README.md) returned 17 correct replies,
but admissions were `[5,4,4,4]` against the required five per worker. The bounded
window failed and no full-duration run was admitted. Host swap use stayed unchanged,
but one page was read from swap; attribution is unknown. Fourteen generated kernel-
cache files require freeze/replay for confirmation. Clean shutdown, host memory,
identity, default/proxy/guard and post-run model/runtime checks passed. Both
persistent reviewers verified the complete published archive and negative result.

The [30-minute native durability attempt](soak-native-003/README.md) completed
all 68 requests correctly and drained normally. Its fixed verdict is **FAIL**:
the first/final five-minute windows admitted fewer than five requests per worker,
and whole-host swap use increased by 108 KiB after preflight. GLM's cgroup swap
samples and peak stayed zero; attribution of the host increase is unknown.
The remaining client checks passed. The guard confirmed a clean stop with no
surviving model processes, memory recovered to 114.882 GiB, and default/proxy/
guard state stayed unchanged. All 2,463 prepared inputs and closed runtime/model
inventories verified after stop. Independent scoped reviews and all 22 existing
scorer mutation/regression tests are preserved with the failed attempt.

Two earlier durability launches failed during CUDA allocation
([startup001](soak-startup-001/README.md), [startup002](soak-startup-002/README.md)).
The current experimental full-context profile enables startup-only allocator
cleanup and verified-model file-cache advice. The latter reclaimed cached model
pages before CUDA initialization, and the completed launch recorded no kernel
OOM/Xid. Both flags were enabled together; their individual effects and production
performance are not qualified. These flags are absent from the agent and
production profiles. The context result below used the preceding startup
configuration; this durability workload did not repeat the direct context test.

The [latest completed context attempt](context-direct-007/README.md) passed all
four final-answer retrieval checks and their negative controls. It processed
1,000,512 actual input tokens across four simultaneous requests. The configured
capacity is 1,048,576 tokens in total: four slots of 262,144 tokens each. This
establishes the declared aggregate capacity; a single request is capped at
262,144 tokens.

The native profile's start, status, authentication, READY reply and orderly stop
also passed. The guard confirmed cleanup, and available memory recovered above
110 GiB. The measured low point was 18.44664764404297 GiB, with zero cgroup swap
and no recorded Xid/OOM. All 2,463 prepared cache files stayed unchanged. Comparing
launch and post-run bytes found growth only in unfrozen usage metadata. Default,
proxy and guard state stayed unchanged. Both persistent reviewers independently
reproduced the fixed context scorer's PASS and found no high/critical issue in
this result.

Use [the quickstart](../../docs/GLM53-QUICKSTART.md) for the repo's
`93_profile_serve.sh` start/status/stop commands. The two profiles are:

- `glm-5.3-flash/cuda-spark-128g-agent-fast`: four 65,536-token slots for agent work.
- `glm-5.3-flash/cuda-spark-128g-1m-experimental`: four 262,144-token slots for large contexts.

Both profiles remain experimental. Paired fidelity, a passing durability gate and
production switching are pending. Qualified production performance is **not yet
measured**. The context run's short outputs do not support a decode-speed claim.

The successful candidate clarified only the test's answer-format instruction,
with a fresh freeze, public seed and a 4,224-token startup correctness check.
Weights, runtime, model settings and retrieval scorer were unchanged. This does
not establish a causal fix for the earlier asynchronous CUDA failure.

Earlier results remain preserved:

- [context-direct-006](context-direct-006/README.md): FAIL; all input tokens processed, but three requests exhausted their output allowance in reasoning without a final answer.
- [context-direct-004](context-direct-004/README.md): FAIL; synchronized diagnostic completed input processing but all four final answers were empty.
- [context-direct-003](context-direct-003/README.md): FAIL; CUDA illegal memory access and Xid31, no completed request.
- [profile-launch-002](profile-launch-002/README.md): PASS for actual native profile lifecycle after the preserved [first stop failure](profile-launch-001/README.md).
- [agent-fast-001](agent-fast-001/README.md): basic authenticated chat, tool arguments and round trip, four overlapping requests, four images and a 16-frame video passed using 224x224 media fixtures. Its later unexplained SIGTERM and interrupted shutdown remain a separate [terminal FAIL](agent-fast-001/terminal/summary.json).

The [native probability diagnostic](native-logprobs-001/README.md) passed exact
input/position alignment on one non-final window. It measured delta-NLL
0.043934924660812516 and top-1 accuracy loss 1.5144113336590133 percentage points
against the native BF16 teacher. These point values exceed the eventual limits,
but one non-final case is not the required 100-case gate. No performance benefit
or adoption approval is claimed for that loss. The bounded
[reference refresh](reference-availability-002/README.md) still found insufficient
public native reference coverage for 100 qualifying cases. Broader published
captures remain [retained privately](reference-followup-001/README.md).

The [native top-k diagnostic](context-topk-probe-001/README.md) was a null crash
reproduction: finite and all-NaN inputs at the failed batch geometry completed
without invalid or duplicate selections. No serving patch followed from it.
Earlier interrupted attempts, client corrections and failures remain in their
original evidence directories. The unchanged context scorer reproduced the new
result exactly; its 17 existing tests passed.

[Weight preparation completed](model-weights-001/README.md): 84,696,019,172 tensor
bytes with no new local quantization. Every launch verifies the final local
model/tokenizer inventory. The [reference binding check](reference-binding-001/README.md)
verified exact public tokenizer, BF16 output-head and final-normalization bytes;
this does not establish native reference equivalence or paired model fidelity.

The component history below describes earlier model-free work.

Integration is **in preparation**, not serving-qualified. All three pinned
CUDA builds and the 207-package dependency check passed. The packaged runtime
has 61,353 independently inventoried files. Native007 passed fourteen synthetic
kernel checks and the complete host/identity/freeze gate. Cache003 allocated
9,565,306,880 physical bytes, held four full reservations, rejected a fifth and
passed host/identity/freeze checks. Those earlier results are model-free: zero input tokens were processed during
those attempts. Real-weight bring-up is reported above.

MLA001 completed all five analytic attention cases with exact output agreement
and passed host/identity checks. Its combined verdict remains **NO_RESULT**:
kernels generated after freeze require separately frozen replay. MLA replay001 then passed the separate frozen confirmation: all outputs exact,
complete before/after inventories, BLS seed and host/identity/cleanup checks.
The sealed loader and replay reviews closed through campaign round 35. Prior
failures remain unchanged.

KDA preparation001 failed on raw nested metadata access; the pinned configuration
normalizer correction closed in campaign round 37. Preparation002 then passed
all five analytic cases and host/identity/cleanup checks. It remains NO_RESULT
because its compiled kernels were generated after freeze. Frozen KDA replay001 then passed all five cases with sealed compiled inputs,
zero mismatches/nonfinite values, full freeze/seed/host/identity checks and clean
containment exit. Both reviewers closed round 38, including the focused
effective-config alias correction. Seven finite selected configurations passed;
four historical failed trials remain preserved cache inputs.

The optional profile requests 1,048,576 aggregate tokens in four 262,144-token
slots. Production activation remains rejected; the default and rollback state
are unchanged. Performance is **not yet measured**.

## Frozen component reviews

| Component | Final candidate | Candidates | Campaign review round | Result |
| --- | --- | ---: | ---: | --- |
| Admission and inventory helpers | `95a804c3` | 1 plus focused medium fixes | 1 | Both reviewers: no critical/high; all three medium regressions closed |
| Request media policy | `eefea2ad` | 2 | 3 | Both reviewers: H1 closed; zero critical/high |
| Actual frame sampler and loader | `3a4a4aa7` | 2 | 5 | Both reviewers: F1 closed; zero critical/high |
| Source preparation | `7babbb1d` | 1 plus focused medium fixes | 6 | Relative paths, extra DFlash/MTP scope and mutable transitive tags closed |
| NVIDIA wheel metadata repair | `65c4448b` | 1 | 7 | Independent RECORD/payload check; zero critical/high |
| Contained build driver | `d3b2df87` | 1 plus focused Rust fix | 8 | Two CUDA/C++/Cargo jobs, pinned Rust, locked Cargo resolution |
| Service credential/access gate | `62c16764` | 1 | 11 | Both reviewers: zero critical/high; six CPU tests pass; actual dsv4 execution pending |
| Standalone runtime packager | `4a69041d` | 1 plus focused low alias fix `29b9e4bf` | 10 | Both reviewers: zero critical/high; six tests pass; relocation/permissions pending |
| Triton-only sealed cache | `994c6666` | 1 plus focused low regression | 9 | Both reviewers: zero critical/high; uninstalled and default-off |

Review approvals cover these components only. They do not qualify a runtime,
model, memory budget, fidelity choice, context capacity, or switching path.
The latest completed scoped CPU audit has 249 passing tests; these include
admission, media, existing profiles, switching and the model-free harness. Three source-function tests cover 48 sampler combinations, four
within-cap comparisons, and three actual loader/metadata cases.

## Reproduced source findings

`source-audit-001` executes exact functions from verified pristine source,
without model or CUDA imports. On the pinned Spark sparse backend, both
`auto` and `fp8` canonicalize to packed `fp8_ds_mla`. Consequently the profile
now explicitly says FP8. A faithful-KV baseline is not provided by that
backend; EXL3 weights and FP8 cache still require the paired fidelity gate.

The original frame sampler can select 1,200 frames despite `num_frames=16`.
The first method-only patch missed the normal media loader; its failed
candidate and genuine RED are retained. The reviewed replacement caps the
shared helper and sets the loader's `max_frames=16`. The request middleware
enforces four images OR one video across the conversation and rejects UUID,
ambiguous content keys, and media overrides before engine dispatch.

## Runtime preparation

Exact upstream commits and audited file hashes are in
`configs/build-manifests/glm53-flash-sources.json`. All 204 preparation
dependencies are installed in an isolated managed Python 3.12.13 environment;
`glm53-runtime-dependencies.json` records the expanded distribution lock.
`uv pip check` passes after the explicitly recorded cuSPARSELt metadata repair.
The original 195-package resolution remains preserved. These locks cannot
authorize a launch. The estimated profile selects the separate `dsv4` service
account, isolated Python (`-I -B`), empty capabilities, and an explicit
no-Inductor baseline. Its secondary API name `default` preserves client
compatibility without changing the startup default. The current build venv
is not yet a service-readable, frozen runtime; permission enforcement and
worker startup remain unqualified.

The NVIDIA repair preserves all library bytes and changes only WHEEL/RECORD;
it asserts local Linux AArch64 packaging, not manylinux compatibility. An
earlier instanttensor source build failed because system Python lacked headers;
the managed-interpreter retry passed. Both attempts are retained. Rust 1.95.0
is installed into a user-owned prefix from a pinned standalone archive.
The latest prepared source adds locked Cargo resolution; all seven source
contract tests pass. Each preparation attempt retains its complete source diff.

Dependency attempts are retained individually:

- `dependencies-001`: FAIL; stale FlashInfer auxiliary-index wheel URL (404).
- `dependencies-002`: FAIL; wheel-only resolution cannot use ARM64 instanttensor.
- `dependencies-003`: dependency resolution PASS under the existing inference
  lock and hardened GLM cgroup wrapper; no model or CUDA compilation.
- `dependencies-004`: NO_RESULT for containment; resolver produced its hash
  file but exited before the wrapper's process-group observation. The failed
  wrapper result is not a successful contained attempt.

A copied standalone CPython prefix passed native standard-library imports
and an actual spawned-child isolation check (`interpreter-relocation-001`).
Both processes retained `-I -B` and paths within the copied prefix. This used
a synthetic probe package, not serving dependencies or the dsv4 service.
Static sysconfig LIBDIR still names the original installation; native loader
auditing remains required.

## Model metadata audit

`model-layout-001` independently hashes both complete indexes and validates
132 bounded shard-header reads against sizes, tensor shapes, offsets and full
shard coverage. The pinned overlay plan replaces 315 BF16 tensors with 1,260
EXL3 tensors across 191 module keys. Overlay payload is 3,539,957,996 bytes;
it replaces 14,361,296,896 bytes, an estimated resident reduction of
10,821,338,900 bytes before allocator/runtime costs. This is metadata only: no
weight payload was downloaded and no memory-fit or fidelity result is claimed.
Full metadata inputs remain in the local `model-layout-001` archive, bound by
the committed closed inventory; the complete overlay plan is committed.

The [reference metadata audit](reference-layout-001/README.md) records a
`NO_RESULT` for native reference binding: no candidate captures or aligned
reference result exist yet. It distinguishes the published panel roles, actual
public coverage and post-final-norm replay semantics.

## Load sizing

The [reproducible storage census](load-sizing-audit-001/README.md) counts
84,696,019,172 selected payload bytes after dense overlay and MTP exclusion.
At a fixed 115 GiB available-memory assumption, a cache-off full resident load
cannot preserve a 40 GiB floor: **NO_GO for that envelope**. With the measured
cache and an 18 GiB floor, 9,891,630,876 bytes remain for all runtime, loading,
allocator and workspace overhead; feasibility remains **NO_RESULT**. The next
step is bounded actual module loading with persistent pinned staging and
separate constructor/transfer/finalization memory observations.

## Component loading

All four separately frozen component probes pass: [288-expert MoE](load-moe-001/README.md),
[mixed KDA](load-kda-001/README.md), [MLA](load-mla-001/README.md) and
[full embedding](load-ordinary-001/README.md). They verify byte identity through
persistent pinned staging, actual loaders and final native handles, exact storage
relationships, phase memory observations and complete host/identity/hash/seed/
cleanup. Both reviewers closed the two evidence defects at candidate2 / campaign41;
225 scoped CPU tests pass. All attempts had zero new swap and no generated runtime
artifacts. Inputs are synthetic, not model weights.

The MoE component retains2,124,585,984 bytes, including288 MiB shared scratch,
12 KiB GTensorCache and20,736 bytes of pointer tables. The full embedding transfer
requires at least2,537,553,920 bytes of simultaneous destination and temporary
storage; its actual allocation observations pass that lower bound. These are
component costs, not a full-model fit or production-performance result.

The [two-layer growth probe](load-growth-001/README.md) also passes. Both modules
remain live and independent, while shared scratch/cache identities remain stable.
The observed retained-phase increase is 4,500 KiB RSS/PSS-anonymous and
1,822,587,392 CUDA allocation bytes. This is a probe observation, not established
production overhead per layer. Cgroup peak is below live CUDA allocation, so the
external whole-host floor remains mandatory.

Convolution preparation001 completed all eight cases and host checks, retained
as [NO_RESULT](conv-preflight-001/README.md) because kernels were generated after
freeze. The separate [sealed replay001](conv-replay-001/README.md) passed all eight
cases with unchanged compiled inputs, full host/identity/cleanup checks and no
new runtime artifacts. Both reviews closed at campaign44; 235 scoped CPU tests
pass. The independent execution review confirmed the narrow kernel result.

The independent [indexer CPU reference](indexer-fixture-review-001.md) passed both
persistent reviews at candidate 1 / campaign 45. The native capture review
closed memory-consistency and FP32 APE findings at candidate 2 / campaign 47.
The first contained [attempt](indexer-preflight-001/README.md) failed during
circular imports before computation; the import correction closed at candidate
3 / campaign 48. The [second attempt](indexer-preflight-002/README.md) reached
actual decode-2 and failed exact tail bytes. Its 256 mismatches were traced to
numeric conversion of uint16 staging during BF16 key reload. All captured valid
logits matched in a separate postmortem, which does not replace the failed verdict.
The one-line bit-preserving copy correction has a [249-test audit](indexer-probe-audit-004/README.md)
and passed both [focused reviews](indexer-probe-review-004.md) as candidate 4 /
campaign 49. Both failed attempts, their raw captures and generated kernels remain preserved.
The corrected [preparation003](indexer-preflight-003/README.md) passed all six
synthetic cases and host checks: 268,434,423 valid logits matched byte-for-byte,
as did all cache, tail and selected-index checks. It remains **NO_RESULT**
because this run generated kernels after freeze. The lossless archive was
restored and rescored with the frozen scorer; all 89 captured files and 65
generated state files match their originals. Separate sealed confirmation is
next; this is not a full-model or context-capability result.

These historical synthetic results do not establish cold checkpoint I/O behavior.
Current full-model bring-up, context007 PASS and prior failures are recorded above.

## Remaining gates

The experimental profile's actual prepared kernels remained unchanged through
context007, including terminal checks. For eventual production service admission,
the [runtime-compilation audit](jit-audit-001.md) still requires effective
read-only access and rejection of unprepared compiler paths in the actual
service identity. The earlier failed user-systemd namespace probes remain
preserved; requested mount properties alone do not prove enforcement.

The next independent gate is [30-minute native durability](soak-native-001/PROTOCOL.md)
with four clients and short retrieval requests. Its CPU compatibility witness
shows that the old soak parser misses native GLM reasoning; the new evidence-only
client reuses the existing GLM stream and retrieval validators. No server change
or repeated context ladder is planned for this gate.

Direct aggregate context passed on an earlier configuration, and the experimental
lifecycle passed; current settings still need direct confirmation. The 100-case
paired fidelity gate, sustained operation, qualified performance, and production
switching/rollback remain pending. Public native reference coverage is still
insufficient for fidelity qualification. Owner adoption of any measured nonzero
fidelity delta remains separate from passing statistical bounds. Qwen stays the
default; qualified GLM production performance is not yet measured.
