# Owner directive — 2026-08-15 (supersedes prior process where stated)

Recorded verbatim in intent by the orchestrating agent from the owner's
2026-08-15 instructions. Future agent sessions must follow this directive; it
overrides the heavier qualification ceremony described in AGENTS.md and
RUNG-PLAN.md for the work it covers.

**Scope**: the owner explicitly extended the lean process to future new-model
implementation work in this repo ("make sure … there are clear directions for
future agents on how they should implement new models"), not only to GLM-5.2.
`docs/NEW-MODEL-PLAYBOOK.md` is that direction. AGENTS.md invariants that the
directive does not name (UMA/OOM safety, evidence honesty, claim/goal
mechanics, endpoint preservation, switch/rollback qualification before
production promotion) remain binding.

## Goal

Get GLM-5.2 implemented and verified on this single DGX Spark, as close to
production DeepSeek-V4-Flash **decode and prefill parity** as achievable
(DSV4 reference: ~18 tok/s decode, ~330 tok/s prefill at 32K). Fidelity may
be spent, but **only where no lossless path to the speed exists**; keep as
much fidelity as possible at the chosen speed.

## Authorizations

- **Rung 3 / new model artifact: authorized now.** Pruned, bit-reallocated,
  and (if later approved) healed GLM artifacts may be built from the local
  211 GB IQ2_XXS GGUF. Rented compute or large downloads still require an
  explicit owner decision.
- **Lean verification loop** for all new speed work, replacing the
  drand/20-arm/Nash-Singer ceremony:
  1. bench: `python3 scripts/30_bench_speed.py --base-url http://127.0.0.1:<port>
     --stack-label <label> --out <result.json> --reps 5` against the served
     candidate (add `--token-timing-log <server.log> --prompt-count-format ds4`
     on the ds4 engine for true per-token timestamps, and the model's tokenizer
     options for non-DeepSeek models);
  2. fidelity: the fixed 100-case `glm52-openrouter-100` NLL suite vs the
     0.4515 NLL / 0.834 top-1 reference (30-case paired subset acceptable for
     iteration; full 100 for adoption);
  3. sol-high code review of every diff
     (`codex exec -m gpt-5.6-sol -c model_reasoning_effort=high "<review task>"`);
  4. one honest commit per accepted change (patch + result JSON under
     `results/glm52-gates/` + a RUNG-PLAN status line).
- **Adoption rules**: lossless changes require byte-identical output (or
  unchanged NLL suite results) plus a decode ratio ≥ 1.0 on the bench. Lossy
  artifact adoption is an **owner decision on the reported fidelity delta** —
  the agent reports the full NLL/top-1 delta and never self-adopts a lossy
  candidate. The fixed AGENTS.md loss thresholds apply only where this
  directive does not supersede them.
- Implementation work runs at sol medium
  (`codex exec -m gpt-5.6-sol -c model_reasoning_effort=medium "<task>"`).
- The matched-32K candidate-15 campaign (`p15-r6379759`) receives a
  **minimal freeze**: frozen-scorer verification of raw.jsonl, a committed
  manifest/raw/summary bundle, and the pending RUNG-PLAN.md corrections —
  no mutation tests, no reviewer rounds.
- **Interactive sudo is eliminated** via the one-time
  `scripts/dev/grant_access_once.sh` run (read ACLs on `/home/dsv4`, scoped
  passwordless sudoers rules for the sanctioned service commands). Nothing may
  be designed to require an interactive sudo prompt; the sanctioned
  `sudo <command>` invocations now run without one.

## Unchanged invariants

- Never two large model processes; memory guard before loads; cgroup
  wrappers; preserve negative/null evidence; never overwrite a running
  harness; README/headline metrics come only from the diagnostics-off
  production fast path (dash otherwise); production telemetry default-off.
- The native codex goal (`dcf05ead-…`) is not renamed or reset.

## Standing plan (self-contained — do not rely on any session state)

Staged artifact strategy: v0 prune-only residency speed probe (any keep-list;
speed does not depend on which experts survive) → lossless kernel push →
v1 saliency-pruned + sensitivity-quantized fidelity candidate →
(owner-gated) v2 healing. Fall back to the lossless streaming plateau with an
honest ceiling report if residency physics fail. Supporting artifacts, all in
this repo: `results/glm52-gates/source-audit-2026-08-15.md` (engine facts),
`patches/glm-dynamic-expert-count.patch` (pruned-GGUF loader support),
`scripts/58_prune_glm_experts.py` (artifact surgery + verify),
`scripts/57_build_matched32k_bundle.py` (campaign evidence freeze).
