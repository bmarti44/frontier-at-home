# GLM-5.3-Flash CUDA status

Integration is **in preparation**, not serving-qualified. No GLM-5.3 weights
have been downloaded or loaded. No CUDA build has run. The optional profile
requests 1,048,576 aggregate tokens in four 262,144-token slots; the default
and recorded previous profile are unchanged. Production activation rejects
before state, lock, or engine changes. Performance is **not yet measured**.

## Frozen component reviews

| Component | Final candidate | Candidates | Campaign review round | Result |
| --- | --- | ---: | ---: | --- |
| Admission and inventory helpers | `95a804c3` | 1 plus focused medium fixes | 1 | Both reviewers: no critical/high; all three medium regressions closed |
| Request media policy | `eefea2ad` | 2 | 3 | Both reviewers: H1 closed; zero critical/high |
| Actual frame sampler and loader | `3a4a4aa7` | 2 | 5 | Both reviewers: F1 closed; zero critical/high |

Review approvals cover these components only. They do not qualify a runtime,
model, memory budget, fidelity choice, context capacity, or switching path.
The complete admission/media/existing profile/switch suite has 87 passing
tests. Three source-function tests cover 48 sampler combinations, four
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
`configs/build-manifests/glm53-flash-sources.json`. Dependencies are resolved
and distribution hashes recorded in `glm53-flash-dependencies.json`; they
have not been installed. These source/dependency locks are not a built-runtime
manifest and cannot authorize a launch.

Dependency attempts are retained individually:

- `dependencies-001`: FAIL; stale FlashInfer auxiliary-index wheel URL (404).
- `dependencies-002`: FAIL; wheel-only resolution cannot use ARM64 instanttensor.
- `dependencies-003`: dependency resolution PASS under the existing inference
  lock and hardened GLM cgroup wrapper; no model or CUDA compilation.
- `dependencies-004`: NO_RESULT for containment; resolver produced its hash
  file but exited before the wrapper's process-group observation. The failed
  wrapper result is not a successful contained attempt.

## Remaining gates

Install the immutable dependency set and clean-build the pinned runtime with
at most two jobs after a stable 110 GiB start-memory check. Freeze source
patches, packages, native extensions, interpreter and loader dependencies;
verify imports and bounded native-kernel parity before downloading weights.
Resolve existing plugin environment reads and diagnostic overhead before any
production qualification. Complete the Python-specific monitored lifecycle,
measured memory envelope, artifact identity, 100-case paired fidelity,
four-slot aggregate occupancy, multimodal correctness, authenticated switching,
rollback, soak and persistent runtime review. Owner adoption of any measured
nonzero fidelity delta remains separate from passing statistical bounds.
