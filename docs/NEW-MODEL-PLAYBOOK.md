# New-Model Playbook

How to bring a new model into this repo on the single DGX Spark (GB10,
128 GB unified memory, ~273 GB/s LPDDR5x, one 3.7 TB NVMe). This is the lean
procedure set by the owner directive of 2026-08-15
(`docs/owner-directive-2026-08-15.md`); it replaces the heavier drand/20-arm
reviewer ceremony for new-model work. Distilled from the GLM-5.2 effort.

## 0. Before anything: what stays true always

- **One large model process at a time.** Stop the running engine, then wait
  for memory release (`python3 scripts/03_memory_guard.py --required-gib 110
  --stable-samples 3 --timeout-seconds 180`) before loading anything new.
  110 GiB is the pre-load release gate, not a residency budget: derive the
  unit's `MemoryHigh`/`MemoryMax` from a measured preflight covering weights,
  cache arena, AND engine RSS, and preserve the whole-system kill floor
  (AGENTS.md UMA/OOM section).
- Launch models only through the cgroup outer launcher
  (`results/glm52-gates/harness/glm_cgroup_run.sh` pattern): it takes the
  inference lock and creates the fresh transient systemd user unit with finite
  `MemoryHigh`/`MemoryMax`, `MemorySwapMax=0`, `OOMPolicy=kill`, group kill.
  (`glm_safe_run.sh` is the inner supervisor, not the containment.)
- **`--no-mmap` is mandatory for llama.cpp engines** on this host (GB10
  mmap-fault pathology). The ds4 engine sidesteps mmap by design
  (O_DIRECT staged reads) and does not take this flag.
- No interactive sudo (one-time `scripts/dev/grant_access_once.sh` grants
  cover the sanctioned commands), no reboots without explicit owner
  authorization.
- Never overwrite a running or retained harness script.
- Preserve negative/null evidence; never delete an inconvenient result.
- Production telemetry/diagnostics default-off; evidence flags are exact
  startup switches resolved once at init.

## 1. Claim, goal, catalog

1. Follow the AGENTS.md model-claim flow: choose the backend, create the
   `claim-model/<slug>/<backend>` branch and draft claim PR, and set a
   persistent model/backend goal with the agent harness's built-in goal tool
   before any engine work.
2. Add the model to `models/catalog.json`: `slug`, `ollama_tag` (or `null`
   with a `source` URL), `context`, `parameters`, `modalities`,
   `repo_status` (`available` → `active` → `qualified`), optional `license`.
   The catalog schema allows no other fields — record artifact SHA-256/size
   in a `weights/<slug>/manifest.json` (see `weights/unsloth-ud-q2_k_xl/` for
   the pattern), not in the catalog.
3. Update `scripts/tests/test_model_claim_catalog.py` expectations and run it:
   `python3 -m unittest scripts.tests.test_model_claim_catalog`.
4. Add a README queue row. **README performance cells contain only
   diagnostics-off production fast-path numbers; use a dash until one
   exists.** Evidence-mode, smoke, or instrumented timings never go in a
   headline table.

## 2. Artifact acquisition and verification

1. **Disk budget first**: check `df` before downloading; the working set
   often needs 1.5-2x the artifact size (source + candidate + captures).
   Record the artifact's SHA-256 and size in `weights/<slug>/manifest.json`.
2. Store weights read-only (owner-writable bits cleared); harnesses verify
   the model by device:inode:size and digest at launch — keep files stable
   (no rewrites in place; hardlink where a second path is needed).
3. Sanity-check GGUF metadata against the publisher's card (tensor count,
   `n_expert`, tokenizer). Known pitfalls from GLM-5.2:
   - converters may duplicate shared tensors (e.g. DSA indexer) rather than
     implement sharing — loads, but is not bit-exact;
   - quantizers often strip MTP/NextN tensors — verify before planning any
     speculative decoding on them;
   - upstream llama.cpp support lags model releases; the ds4 fork is often
     the only working path. Check both before writing new engine code.

## 3. Engine support check

1. The engine source of record for *reading/auditing* is the ds4 checkout
   under `/home/dsv4/ds4-project/src/` (read-only ACLs; may be dirty —
   snapshot it with hashes before relying on it, as in
   `~/ds4-source-snapshot-2026-08-15/`). **`vendor/ds4/` is stale and never
   authority.** Engine *changes* happen in a dedicated frozen worktree under
   `~/.cache/` (the existing `glm52-*`/`dsv4-*` candidate dirs are the
   pattern), never in place under `/home/dsv4`.
2. Ask: does the current engine load this architecture at all? Does it fit
   resident, or does it need `--ssd-streaming` plus the persistent expert
   cache? Note the CUDA facts from the 2026-08-15 source audit: the cache is
   sized by `DS4_CUDA_EXPERT_CACHE_GB` (decimal GB, env), not by the CLI
   cache flag; a model whose uniform-slab experts all fit in that arena runs
   effectively resident after warmup. Budget with the measured-preflight rule
   from section 0, not a fixed number.
3. All engine changes go through codex: implementation at
   `codex exec -m gpt-5.6-sol -c model_reasoning_effort=medium "<task>"`,
   review at `model_reasoning_effort=high "<review task>"`. Build with
   ≤2 jobs, never while a model is loaded. Record the binary SHA-256.

## 4. Profile and serving

1. Create a profile config binding: binary SHA-256, model digest, context
   cap, and the exact environment allowlist. Profiles are campaign-specific
   JSON (see `configs/glm52-lossless-plateau-profile.json` for a worked
   example) — copy an existing one and adapt; there is no generic schema, and
   runtime-config byte-comparison is implemented per-harness, not globally.
2. Serve on 127.0.0.1 only; poll `/v1/models` for readiness. Production
   promotion additionally requires `scripts/52_engine_switch.sh` support for
   the new engine (it currently recognizes only `dsv4`/`glm52`) and the
   AGENTS.md switch/rollback/endpoint-preservation tests.

## 5. The lean verification loop (per candidate/change)

1. **Speed**: `python3 scripts/30_bench_speed.py
   --base-url http://127.0.0.1:<port> --stack-label <label>
   --out <result.json> --reps 5` (context levels 0/4K/16K/28K). On the ds4
   engine add `--token-timing-log <server.log> --prompt-count-format ds4` so
   decode rates come from true per-token timestamps (≥128 strictly
   increasing); pass the model's tokenizer options for non-DeepSeek models so
   context slicing uses the right tokenizer.
2. **Fidelity**: a fixed, committed NLL suite for the model (GLM-5.2 uses
   the 100-case `glm52-openrouter-100` manifest via the engine's
   `score_official` tool; reference 0.4515 NLL / 0.834 top-1). Build the
   equivalent once per model and never change it mid-campaign. A ~30-case
   paired subset is fine for iteration; adoption requires the full suite.
   **Adoption rule**: lossless changes must be byte-identical (or leave the
   suite unchanged) with decode ratio ≥ 1.0; any lossy delta is reported in
   full and adopted only by explicit owner decision — never self-adopted.
3. **Review**: sol-high review of every diff before it lands.
4. **Commit**: one honest commit per accepted change — the patch, the result
   JSON under `results/glm52-gates/`, and a status line in the plan doc.
   Report regressions and null results with the same prominence as wins.

## 6. Evidence you keep

For any campaign-grade measurement, commit the three-file bundle under
`results/glm52-gates/<topic>-<verdict>/`:

- `manifest.json` — digests binding source, binary, scorer, model, config,
  and the bundle's own `raw.jsonl`/`summary.json`;
- `raw.jsonl` — the raw per-arm records;
- `summary.json` — the fixed scorer's exact output (formulas + verdict).

Gate evidence is deliberately git-tracked (`.gitignore` force-includes
`results/glm52-gates/`). `scripts/57_build_matched32k_bundle.py` shows the
pattern.

## 7. Qualification statement

A model becomes `qualified` in the catalog when: it serves through the
production profile with diagnostics off; its speed and fidelity numbers are
committed as a bundle; the README row cites exactly those numbers; **and** the
AGENTS.md production-promotion requirements hold — largest-useful-context
qualification, authenticated endpoint preserved, safe one-command switching
with rollback (including stale-PID / wrong-model / startup-death tests), and a
passing review of the evidence. Until then it stays `active`, and the README
shows a dash.
