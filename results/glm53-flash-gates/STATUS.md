# GLM-5.3-Flash CUDA status

Integration is **in preparation**, not serving-qualified. All three pinned
CUDA builds and the 207-package dependency check passed. The packaged runtime
has 61,353 independently inventoried files. Native007 passed fourteen synthetic
kernel checks and the complete host/identity/freeze gate. Cache003 allocated
9,565,306,880 physical bytes, held four full reservations, rejected a fifth and
passed host/identity/freeze checks. These are model-free results: zero input
tokens were processed and no weight payload has been downloaded or loaded.

MLA001 completed all five analytic attention cases with exact output agreement
and passed host/identity checks. Its combined verdict remains **NO_RESULT**:
kernels generated after freeze require separately frozen replay. MLA replay001 then passed the separate frozen confirmation: all outputs exact,
complete before/after inventories, BLS seed and host/identity/cleanup checks.
The sealed loader and replay reviews closed through campaign round 35. Prior
failures remain unchanged.

KDA preparation001 failed on raw nested metadata access; the pinned configuration
normalizer correction closed in campaign round 37. Preparation002 then passed
all five analytic cases and host/identity/cleanup checks. It remains NO_RESULT
because its compiled kernels were generated after freeze. Frozen KDA replay is
under review in round 38; seven finite autotune winners and four failed trials
are preserved separately from authoritative output evidence.

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
The latest completed scoped CPU audit has 201 passing tests; these include
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

## Remaining gates

The [runtime-compilation audit](jit-audit-001.md) identifies an additional
large-load admission requirement: prewarm/freeze the actual selected kernels
and reject cache misses before compilation. Fourteen Triton cache/control-flow
tests pass, but worker startup wiring, cache immutability, FlashInfer AOT,
DeepGEMM, CuTe and Inductor closure remain. Two user-systemd read-only namespace
probes failed and are preserved; requested mount properties are not evidence
of enforcement on this host.

The three clean builds, runtime inventory, bounded native checks and cache
allocation are complete. Finish remaining KDA/indexer
kernels and the measured full workspace envelope before downloading weights.
Resolve existing plugin environment reads and diagnostic overhead before any
production qualification. Complete the Python-specific monitored lifecycle,
measured memory envelope, artifact identity, 100-case paired fidelity,
four-slot aggregate occupancy, multimodal correctness, authenticated switching,
rollback, soak and persistent runtime review. Owner adoption of any measured
nonzero fidelity delta remains separate from passing statistical bounds.
