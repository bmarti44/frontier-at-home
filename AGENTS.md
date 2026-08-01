# Agent contribution guide

This repository is both a production service and an evidence archive for serving
very large models on one DGX Spark. Treat host stability, reproducibility, and
honest negative results as product requirements—not cleanup work.

## What success means

When adding or optimizing a model:

1. Keep the existing authenticated endpoint and rollback path working.
2. Target the model's largest useful context directly. A smaller prompt may be
   used for a fast correctness, fidelity, or kernel smoke test, but it is not a
   context-capability result.
3. Measure behavior with repository scorers and raw host observations. Engine
   self-reports and agent narration are not acceptance evidence.
4. Preserve failed and null attempts. If the measured roofline or quality gate
   fails, say so and move to the next bounded alternative.
5. Never trade whole-system stability for benchmark progress.

Before starting claimed work, set a persistent goal for the chosen model and
backend with the goal tool built into your agent harness. That is the entire
agent-goal requirement: do not create a repository-specific substitute. Read
the existing evidence before assuming that an inherited claim is true.

## Start every session with facts

Before changing code:

```bash
git status --short
git log -5 --oneline
systemctl is-active dsv4-engine-restore.service dsv4-guard.timer
awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
```

Then read your current harness goal and its status before acting.

Also inspect the exact running process, command line, context cap, and listener
before stopping or replacing anything. Preserve a dirty tree and unrelated user
changes. Use a dedicated worktree for upstream engine experiments.

Inherited patches, binaries, conclusions, and benchmark summaries are hypotheses
until reproduced from a clean source state.

## The required change workflow

Use this sequence for production-path changes:

1. Write the production-path test and fixed acceptance formula.
2. Commit the test and capture a genuine RED result on unchanged code.
3. Implement the smallest useful change behind an exact, logged, default-off
   diagnostic flag.
4. Commit the implementation and make the worktree clean.
5. Stop the active large model safely and wait for enough memory to build.
6. Clean-build with at most two parallel jobs.
7. Freeze the source, binary, scorer, model, tokenizer/fixture, and configuration
   hashes.
8. Obtain verifiable public randomness after the freeze. Use it to select or
   permute the confirmation fixtures and arm order.
9. Run equal-fixture arms in fresh containment. A post-freeze code, test, scorer,
   or fixture change creates a new candidate and requires a new seed.
10. Preserve raw evidence, calculate the verdict with the fixed scorer, and run
    mutation tests that demonstrate malformed or broken evidence is rejected.
11. Review the frozen candidate. Continue fixing until there are no verified
    high or critical issues. Reviewers choose their own scores; do not prescribe
    a scoring formula.

Do not add a permanent semantic variant merely because a diagnostic flag passes.
First use the flag to prove the representation or algorithm. Then implement and
requalify the single intended production path.

## UMA and OOM safety

The Spark has unified CPU/GPU memory. A GPU allocation can freeze the entire host,
not merely fail one process.

- Never run two large model processes concurrently.
- Never build CUDA code while a loaded model leaves only production-sized
  headroom. Stop the identity-verified engine and wait for at least 110 GiB
  `MemAvailable` before a clean GLM build or load.
- Use the inference lock and the hardened wrapper:
  `results/glm52-gates/harness/glm_safe_run.sh`.
- For current GLM qualification runs, use a fresh systemd cgroup with
  `MemorySwapMax=0`, `OOMPolicy=kill`, and `KillMode=control-group`. Size
  `MemoryHigh` and `MemoryMax` from a measured preflight that includes both
  the pinned expert arena and engine RSS. Do not reuse the old 68/71 GiB
  limits: a 68 GiB arena plus roughly 30 GiB engine RSS will fight those
  limits and can destabilize the host.
- Use a 40 GiB whole-system kill floor for the cache-off measurement probe. A
  full 68 GB-arena campaign may use the preregistered 18 GiB kill floor only
  when its cgroup ceiling is derived from that probe, the arithmetic preserves
  the floor against physical memory, and external sampling still rejects any
  arm below 10 GiB available. The production profile has its own validated
  floor.
- Require stable start memory, a wall-clock timeout, continuous process/binary
  identity checks, and timestamped memory sampling.
- Treat OOM, cgroup kill, swap use, Xid, short output, timeout, missing process
  identity, or a surviving descendant as a failed attempt.

Do not reboot as a routine recovery step. Do not ask the user to repeat `sudo`
commands when the installed, delegated systemd/Docker control path can perform
the exact operation safely. Never broaden those controls or run Codex itself as
root. A reboot or new privilege is a last resort and requires explicit user
authorization.

## Long-context qualification

For a 1M profile, the capability gate is direct:

- configured cap exactly `1,048,576`;
- at least `1,000,000` actual input tokens processed;
- deterministic retrieval at multiple positions plus negative controls;
- completed generation with timestamped output tokens;
- no truncation, OOM, Xid, or unexpected swap;
- at least 10 GiB available memory at the measured low point.

Do not spend time proving a ladder below a context that has already passed. Use
shorter inputs only when they cheaply falsify correctness or fidelity before the
expensive direct run, and label them as such.

Lossy cache or weight changes require the fixed 100-case paired suite:

- token-weighted `ΔNLL <= 0.01`;
- one-sided 95% upper bound `<= 0.01`;
- top-1 loss and its one-sided 95% upper bound `<= 0.5` percentage points.

An improved average does not override a failed confidence bound.
Passing these statistical limits also does not authorize a fidelity spend.
Report every nonzero delta together with the measured performance it bought;
only the repository owner decides whether to adopt it. Exhaust byte-identical
levers first, and never spend fidelity merely to improve prefill.

## Evidence contract

Every authoritative attempt must contain:

- `manifest.json`: source/diff, binary, scorer, model, tokenizer/fixture, public
  randomness, arm input, and configuration hashes;
- `raw.jsonl`: paired case data or token timestamps, CUDA timing, cache/I/O
  counters, memory samples, process identity, and failures;
- `summary.json`: exact formulas, unrounded metrics, confidence intervals,
  checks, and verdict.

Missing arms, unequal fixtures, duplicate IDs, stale binaries, malformed rows,
NaN/Inf, missing coverage, scorer failure, or self-authored replacement data are
failures. Never discard an inconvenient attempt.

Keep benchmark prompts and case names out of production decision branches. Scan
diffs for prompt/hash detection, canned results, disabled assertions,
benchmark-only paths, and timing manipulation.

## Performance and switching

Measure decode from token timestamps as `(N-1)/(tN-t1)` over at least 128 output
tokens. Measure prefill as actual evaluated tokens divided by synchronized
prefill time. Use five fresh-server ABBA/BAAB blocks for matched comparisons and
derive confidence intervals in the harness.

The operator interface must remain:

```bash
scripts/52_engine_switch.sh status --json
sudo scripts/52_engine_switch.sh glm52
sudo scripts/52_engine_switch.sh dsv4
```

The switch must serialize on one lock, verify hashes, stop only the
identity-verified process group, validate authentication/health/semantics and
memory monitoring, commit state only after validation, and restore the previous
profile on failure. DeepSeek remains the safe default until a new model passes
quality, safety, direct-1M, switching, and review gates.

## Working style

- Lead status updates with the outcome in plain language.
- Explain whether a small run is a fidelity check or a capability result.
- Report exact progress and safety state during long runs, but avoid narrating
  every shell command.
- Prefer bounded falsifiers before expensive kernel work.
- Do not over-engineer a failed branch. Preserve its evidence and continue with
  the next justified alternative.
- Keep commits small and descriptive. Do not rewrite or erase prior attempts.
- Use the two persistent reviewers already assigned to the goal. Do not create a
  new reviewer each round.

If blocked, provide a reviewer the exact failing assertion and evidence. Verify
the diagnosis locally, try the best two bounded alternatives, then mark the
branch `NO_RESULT` if neither works.

## Claiming a model integration

The discovery queue is [`models/catalog.json`](models/catalog.json). Claim work
before implementing it:

1. Choose a catalog `slug` and one listed `claim_backends` value.
2. Create the exact branch
   `claim-model/<catalog-slug>/<backend>` in your fork.
3. Open a draft PR immediately with
   `.github/PULL_REQUEST_TEMPLATE/model-integration-claim.md`.
4. Before engine changes, set a persistent model/backend goal with the goal tool
   bundled with your agent harness.
5. Keep that PR open and update its description with the baseline, hardware,
   memory safeguards, and evidence location as facts become available.

The `pull_request_target` claim workflow deliberately does not check out or
execute fork content. It reads the current trusted default-branch catalog,
validates the head branch, and applies `claim:<model>`, `backend:<backend>`, and
`status:self-declared` labels. Those labels drive the live claim badges in the
README. The workflow also reconciles stale managed labels after PR changes and
default-branch catalog changes. Never add fork checkout or fork-authored
scripts to that privileged workflow, and pin every external action to a full
reviewed commit SHA.

A claim is self-declared coordination metadata, not proof of ongoing activity,
an endorsement, or a reservation. Multiple claims may coexist, especially for
different backends. Close a claim PR when work stops so the status badge
becomes accurate again. A model marked `reference_only` cannot be claimed
until public local weights and a compatible license are documented in the
catalog.

### Architecture claim mapping

Use the backend that is actually being implemented and qualified:

| Target architecture | Branch backend |
| --- | --- |
| NVIDIA CUDA, including DGX Spark and Jetson | `cuda` |
| Apple Silicon with MLX or Metal | `apple-silicon` |
| AMD Strix Halo or discrete AMD with HIP | `rocm` |
| A cross-vendor Vulkan implementation | `vulkan` |
| Intel Arc/Xe with oneAPI, Level Zero, or SYCL | `intel-xe` |
| Qualcomm Snapdragon native acceleration | `qualcomm` |
| Tenstorrent Tensix | `tenstorrent` |
| CPU-first or CPU/offload | `cpu` |

The model and architecture are claimed together in the branch name. Examples:
`claim-model/glm-5.2/apple-silicon`,
`claim-model/deepseek-v4-flash/cuda`, and
`claim-model/kimi-k3/rocm`. If one PR genuinely qualifies multiple primary
backends, use separate claim PRs so status, evidence, and review remain
unambiguous.

## Adding another model

For a new model family:

1. Pin and independently hash model/tokenizer artifacts.
2. Add a profile with an exact maximum context and conservative memory budget.
3. Reproduce a clean baseline and roofline before optimizing.
4. Capture real production-path tensors before choosing a lossy format.
5. Run an offline error curve; stop before kernel work when it falsifies the
   storage, runtime, or fidelity budget.
6. Add the default-off production-path arm, frozen paired quality gate, then the
   packed implementation.
7. Attempt the largest context directly under containment.
8. Add switching, rollback, stale-PID, wrong-model, startup-death, low-memory,
   auth-rejection, and reboot-restore tests.
9. Preserve a reviewed `PASS`, `FAIL`, `NO_RESULT`, or `NO_GO`; never substitute
   an optimistic narrative for a terminal measurement.
