# Backend contract

What a serving backend must provide so that every gate, suite, and
qualification rule in this repo works unchanged on new hardware. The
evaluation stack is architecture-neutral on purpose: if your backend
satisfies this page, you inherit the speed cells, fidelity suites,
tool-call probe, template-fidelity harness, and the claim/qualification
flow without modification.

The repo's reference backend is `cuda` (NVIDIA GB10 DGX Spark, llama.cpp
family). Other claimable backends (`apple-silicon`, `rocm`, `vulkan`,
`intel-xe`, `qualcomm`, `tenstorrent`, `cpu`) implement THIS contract on
their own platform; the registry and dispatcher (`configs/backends.json`,
`scripts/91_serve.sh`) give them a uniform entry point.

## 1. Serving surface (hard requirements)

- OpenAI-compatible HTTP endpoint on `127.0.0.1` (never a public bind):
  - `GET /health` → 200 when ready to serve.
  - `GET /v1/models` → exactly one model whose `id` names the served
    model (identity check; gates verify it).
  - `POST /v1/completions` — plain completions; the accuracy harness
    posts pre-rendered prompts here and applies no chat template
    server-side. Must honor `temperature: 0`, `seed`, `max_tokens`, and
    an ignore-EOS mechanism (flag or extra-body) for strict speed cells.
  - `POST /v1/chat/completions` — used by the tool-call suite and
    canaries; must honor `tools` if the model claims tool-calling.
- **Per-token SSE streaming** for strict speed measurement: one content
  event ≈ one token. If a speculative decoder emits multi-token blocks,
  strict 30_bench cells are INVALID on that config — measure raw, and
  characterize the drafted config with wall-clock probes (see the
  qwen38-sglang DSpark precedent).
- Tokenization introspection for BOS/EOS auditing: `POST /tokenize`
  (llama.cpp style) or an equivalent the gate scripts can call to prove
  single-leading-BOS behavior (see the Laguna `add_default_bos_token`
  lesson in NEW-MODEL-PLAYBOOK.md).

## 2. Lifecycle (hard requirements)

- A serve entry point with `start | stop | status` verbs that:
  - verifies the engine binary against a build manifest digest and the
    weights against `weights/<slug>/manifest.json` BEFORE launch
    (fail closed; a missing weights manifest may warn during early
    integration only);
  - confirms port ownership and `/v1/models` identity after launch;
  - stops by verified process identity (never by name/pkill), and
    reports honest status.
- Registered in `configs/backends.json` so
  `scripts/91_serve.sh --model <slug> --backend <name> start` routes to
  it. Unimplemented (model, backend) pairs must fail closed with a
  pointer to this document — never silently fall back to another
  backend.

## 3. Memory safety (platform-appropriate, non-negotiable in spirit)

The GB10 implementation (memwatch MemAvailable floor + capped systemd
unit + 100 GiB release gate between loads) is Linux/UMA-specific. A new
backend must provide the same three guarantees with its platform's
mechanisms:

1. **A whole-system guard that survives accounting blind spots.** On
   GB10, cgroups cannot see CUDA unified-memory allocations, so the
   guard watches global MemAvailable and SIGKILLs the verified engine
   group below a floor. Find your platform's equivalent blind spot
   before trusting any per-process limit (Metal unified memory has the
   same character on Apple Silicon).
2. **One large model at a time**, enforced by a residency lock file the
   serve scripts take exclusively.
3. **A release gate between loads**: block until memory is actually
   free and stable before loading the next model (03_memory_guard.py is
   portable Linux; port the idea, not necessarily the file).

Two hard-won rules that transfer to any UMA platform: derive memory
budgets from measured KV-bytes/token and measured steady-state under
load, not from datasheet arithmetic (the Laguna 524K shape passed
arithmetic and breached in practice); and treat lazy allocation as
real — steady state under sustained load is the number that matters.

## 4. What you inherit for free

- `scripts/30_bench_speed.py`, `scripts/31_bench_accuracy.py` (holdout
  ledger, config-evidence binding), `scripts/39_bench_toolcall.py`,
  `scripts/tests/template_fidelity.py` — all take `--base-url`.
- The claim flow, catalog, evidence-bundle conventions, lint/manifest
  hooks, and the qualification bar in NEW-MODEL-PLAYBOOK.md §7.
- Encoders under `vendor/official-encoding/encoding/` are pure Python.

## 5. What stays per-backend

- Engine builds (pins, flags, isolation) and their build manifests.
- The production switch. `scripts/52_engine_switch.sh` is the cuda/GB10
  switch; a new backend writes its own transactional equivalent (stop
  verified → verify hashes → launch → verify serving → commit/rollback)
  before any profile on it can be called production-grade.
- Privilege plumbing (sudoers on Linux; whatever is idiomatic and
  narrowly scoped elsewhere) and the guard service.

## 6. Qualification on a non-reference backend

Identical bar (NEW-MODEL-PLAYBOOK.md §7): strict speed cells that are
`suite_valid`, fidelity suites through the holdout ledger, evidence
bundles committed, README row citing exactly those numbers. Numbers from
different backends are never merged into one row — a model qualified on
`cuda` shows a dash for `apple-silicon` until someone qualifies it
there.
