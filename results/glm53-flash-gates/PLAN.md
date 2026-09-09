# GLM-5.3-Flash / CUDA qualification campaign

Status: REAL MODEL SERVING — authenticated text, tool calls, and four concurrent
short requests passed in development attempt 008. Full context, paired fidelity,
media, and production switching remain pending; GLM remains optional and the
recorded Qwen default is unchanged.

The model-free build, cache, attention, convolution/indexer, and loader work is
preserved in the evidence archive. Real weights are downloaded, checked against
the local inventory at launch, and loaded successfully. See [STATUS.md](STATUS.md)
for the latest actual serving result; the acceptance formulas below are unchanged.

Next: the bounded media retry uses the working 128-token prefill configuration,
then direct aggregate-context and remaining qualification gates. Do not repeat
already frozen component reviews or add more startup guard infrastructure.

Target: one ARM64 DGX Spark, 1,048,576 aggregate context tokens in four
262,144-token slots; text, reasoning, tools, four images or one video sampled
to at most 16 frames per request. The existing default and authenticated
endpoint remain unchanged. Interactive speed is the optimization priority;
this does not waive fidelity, safety, evidence, or owner-adoption gates.

The agent harness goal, rather than a repository goal implementation, owns
campaign progress. Claim branch: `claim-model/glm-5.3-flash/cuda`.

## Starting facts

- Repository baseline: `21db1e51` (clean tree).
- Recorded profile: `qwen38-1m`; its systemd engine was inactive at inspection.
- Restore service and guard timer active; no listener on production port 8013.
- MemAvailable approximately 116 GiB; existing swap occupancy approximately
  155 MiB. Qualification rejects new swap activity; never erase baseline data.
- Model filesystem has approximately 135 GiB free. Downloads require a
  complete disk budget including runtime/build/evidence headroom.

## Ordered work

1. Commit production-path acceptance tests and capture genuine RED on unchanged
   code. Preserve prior profile render fixtures.
2. Implement an estimated, default-off profile, verified runtime/artifact
   manifests, model-specific rendering, and opt-in lifecycle integration.
3. Audit pinned EXL3 K2 + dense-overlay runtime compatibility before large
   downloads. Use isolated worktrees and source-built immutable runtimes.
4. Preflight memory; clean-build with at most two jobs and at least 110 GiB
   MemAvailable. Freeze source, packages, extensions, weights, tokenizer,
   processor, fixtures, scorer, and configuration before public randomness.
5. Qualify correctness, 100-case paired fidelity, four-slot occupancy,
   multimodal behavior, five matched speed blocks, soak, switching, and review.
6. Restore the previous profile after each qualification window. Promote only
   with complete passing evidence and the owner's measured fidelity decision.

## Fixed acceptance

The serving configuration must specify 1,048,576 total tokens, exactly four
slots, and a 262,144-token request cap. Four distinct simultaneous requests
must each evaluate at least 250,000 input tokens; at least 1,000,000 tokens
must be collectively resident before eviction/reuse. Retrieval at multiple
positions and negative controls must pass in every slot; generated token
timestamps and completed output are mandatory. This is an aggregate-context
claim, not a single-request 1M claim.

Fidelity uses exactly 100 paired cases: token-weighted delta-NLL <= 0.01 and
one-sided 95% upper bound <= 0.01; top-1 loss and its upper bound <= 0.5
percentage points. Invalid/missing references produce NO_RESULT, not PASS.
Changing these formulas requires a new candidate, never editing a frozen run.

Use the global inference lock, hardened GLM wrapper, fresh systemd cgroup,
MemorySwapMax=0, OOMPolicy=kill, KillMode=control-group, stable start-memory
gate, external MemAvailable sampler, process/runtime identity verification,
and timeout. Keep the GLM 18 GiB floor initially (40 GiB for a cache-off
streaming measurement probe); any observation below 10 GiB fails. Include
weights, pinned staging, KV/index/recurrent state, vision, workspaces, CUDA
graphs, and allocator reserve in the measured budget. Never run two large
models or compile CUDA with a loaded model.

Each authoritative attempt requires manifest.json, raw.jsonl, summary.json,
frozen hashes, and post-freeze verifiable public randomness. Missing arms,
unequal fixtures, duplicate IDs, nonfinite/malformed rows, stale runtime,
missing coverage, scorer failure, OOM/Xid/swap/timeout/truncation, and surviving
descendants fail. Retain negative attempts. Performance comes only from a
qualified production path with diagnostics disabled; until then it is not
measured. Follow AGENTS.md for diagnostic overhead, review, and convergence.

## Research cautions

- GLM-5.2's streaming cache sizes, architecture, accepted quantization deltas,
  and inherited binary provenance are not GLM-5.3 acceptance evidence.
- The EXL3 recipe's longest run used an unpublished workspace patch.
- The public fidelity-suite correction of 2026-09-08 identifies post-final-
  normalization captures; old pre-normalization scorer claims cannot qualify
  a candidate. Shared-head replay cannot assess changed output-head error.
- Pin exact commits and hashes; neither a floating wheel install nor engine
  capacity self-reports establish correctness or full-context occupancy.

Sources: https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
and https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1
