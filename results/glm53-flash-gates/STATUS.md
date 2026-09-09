# GLM-5.3-Flash CUDA status

Integration is **in preparation**, not serving-qualified. All three pinned
CUDA builds and the 207-package dependency check passed. The packaged runtime
has 61,353 independently inventoried files. Native007 passed fourteen synthetic
kernel checks and the complete host/identity/freeze gate. Cache003 allocated
9,565,306,880 physical bytes, held four full reservations, rejected a fifth and
passed host/identity/freeze checks. These are model-free results: zero input
tokens were processed and no model weight payload has been downloaded or loaded.

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
campaign 49. The next contained run requires a fresh freeze and public seed. Both failed attempts,
their raw captures and generated kernels remain preserved. No complete indexer
kernel result exists yet. Preparation cannot claim sealed-kernel qualification.

Next: finish indexer and vision workspace probes. Do not multiply fresh-process RSS by
layer count or add it blindly to CUDA allocations. The synthetic input was
freshly generated, so these runs do not establish cold checkpoint I/O behavior.

## Remaining gates

The [runtime-compilation audit](jit-audit-001.md) identifies an additional
large-load admission requirement: prewarm/freeze the actual selected kernels
and reject cache misses before compilation. Fourteen Triton cache/control-flow
tests pass, but worker startup wiring, cache immutability, FlashInfer AOT,
DeepGEMM, CuTe and Inductor closure remain. Two user-systemd read-only namespace
probes failed and are preserved; requested mount properties are not evidence
of enforcement on this host.

The three clean builds, runtime inventory, bounded native checks and cache
allocation are complete. Convolution preparation and sealed confirmation are
complete. Finish remaining indexer kernels and the measured full workspace envelope before downloading weights.
Resolve existing plugin environment reads and diagnostic overhead before any
production qualification. Complete the Python-specific monitored lifecycle,
measured memory envelope, artifact identity, 100-case paired fidelity,
four-slot aggregate occupancy, multimodal correctness, authenticated switching,
rollback, soak and persistent runtime review. Owner adoption of any measured
nonzero fidelity delta remains separate from passing statistical bounds.
