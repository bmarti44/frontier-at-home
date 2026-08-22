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

## 8. Mechanics that repeatedly bit us (Qwen3.8/Laguna campaigns, 2026-08)

Start here for a new model: `scripts/90_scaffold_model.sh` generates the
build script, serve script, and encoder/test stubs from the newest reference
implementation and prints the manual-steps checklist (`--backend <name>`
targets a non-cuda backend). Serving routes through the backend registry:
`scripts/91_serve.sh --model <slug> --backend <name> start|stop|status`
(configs/backends.json). Only `cuda` is implemented on this host; other
backends fail closed with a pointer to **docs/BACKEND-CONTRACT.md**, the
one-page surface a new architecture must implement to inherit every gate
and suite unchanged.

**Sol (codex) workflow.** Implementation:
`codex exec -m gpt-5.6-sol -c model_reasoning_effort=medium -s workspace-write "<task>" </dev/null`
— the `</dev/null` is mandatory (codex hangs waiting on stdin otherwise) and
there is no `--full-auto` flag; use `-s workspace-write` (or `-s read-only`
for reviews at `model_reasoning_effort=high`). Never let the implementer be
the only author of its acceptance tests: sol's first Laguna encoder passed
9/9 of its own tests while diverging from the official template on three
byte-level inputs; only the adversarial sol-high review caught it. Route
every encoder through the shared matrix in
`scripts/tests/template_fidelity.py` (see
`scripts/tests/test_template_fidelity_laguna.py` for the wiring), and have
sol-high review every deliverable before it lands.

**Git traps.**
- `vendor/` is gitignored with negation patterns carved out for
  `vendor/official-encoding/encoding/encoding_*.py`.
  `scripts/tests/test_encoder_registration.py` fails if a registered encoder
  is neither tracked nor covered by the DSV4 official-encoding pin — run it
  after adding an encoder.
- `models/` is gitignored but `models/catalog.json` is tracked: `git add`
  by directory prints an ignore warning and exits 1, killing `&&` chains.
  Add the file path explicitly.
- Never pipe a commit (`git commit ... | tail`) — the pipe masks lint-hook
  failures. Run commits unpiped or check `PIPESTATUS`.
- The pre-commit hook now verifies `verification/MANIFEST.sha256` against
  the working tree: if you edit a manifested harness file (e.g.
  `31_bench_accuracy.py`, `lint_secrets.sh`), refresh its line in the same
  commit (`sha256sum <file>`, replace the line).
- `scripts/lint_secrets.sh` blocks commits on new digest-bearing paths;
  budget for allowlist entries for `results/<slug>-gates/`,
  `weights/<slug>/manifest.json`, profile configs, and any new script that
  prints public digests.

**Downloads.** Weight downloads saturate the uplink and starve `git push` /
`gh` calls: push branches and open the claim PR *before* starting a big
download, and run pushes in the background with generous timeouts during
one. Fetch only the primary quant; start ladder quants (the bigger/smaller
fallbacks) only when a gate actually asks for them.

**Live-script edits.** Never edit a script in place while any process may
be running (or queued to run) it — a running bash keeps reading from its
open inode, and in-place writes corrupt it mid-parse. Install changes by
editing a copy and atomically `mv`-ing it over the original; the running
process keeps the old inode untouched. If an invocation is *queued*
(e.g. blocked on a lock), remember it will execute its original text
when it unblocks — kill and re-issue it after the fix lands. If the
switch ever hangs, see docs/RUNBOOK-stuck-switch.md.

**Gate windows.** Don't hand-roll window boilerplate — source
`scripts/lib/gate_window.sh` (`gate_window_open` → `gate_serve_cycle` →
probes → `gate_window_close`; the EXIT trap restores production even on
mid-probe death, and `capture_json` refuses multi-line evidence files that
the lint hook would later reject).

**Harness runtime contracts** (discoverable only from source/old results —
read this before burning a window):
- `31_bench_accuracy.py` **holdout** runs REQUIRE `--config-evidence
  <files>` (convention: build manifest + weights manifest + the speed
  result binding the serving config) and a `--config-hash` string; dev
  runs don't. Run a **dev-split truncation probe first** — holdout rowsets
  are one-shot per ledger namespace (`DSV4_LEDGER_NAMESPACE` to re-spend,
  owner-authorized only).
- Accuracy result JSON top-level keys: `n`, `correct`, `accuracy`,
  `invalid_count` (truncation proxy), `config_digest`, `ledger_namespace`.
  Per-item detail lives in the transcripts dir, not the result file.
- `30_bench_speed.py` strict cells need `--ignore-eos-supported`,
  `--output-tokenizer-path` + `--output-tokenizer-sha256` for non-DSV4
  models, `--request-timeout ≤2700`; a cell is README-quotable only if
  `suite_valid` is true. Verify per-token streaming first if a
  speculative decoder is on (G2-style granularity check) — block
  streaming invalidates the timestamp pipeline.
- Evidence files must each be a single valid JSON document (the lint hook
  parses exempted JSON); use `capture_json`, never `tee` a mixed stream.
- When giving the owner a command to run, use **absolute paths** — they
  won't be sitting in the repo directory.

**Engine builds beside live production.** The build scripts refuse to run
uncontained when less than 110 GiB is available; wrap them in a capped user
unit (`systemd-run --user --collect -p MemoryMax=11G -p MemorySwapMax=0
-p OOMPolicy=kill`). Remember the GB10 rule: cgroup accounting is blind to
CUDA unified memory, so systemd caps are backstops and the 8 GiB
MemAvailable watchdog floor is the real guard.
