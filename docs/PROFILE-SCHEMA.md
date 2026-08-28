# Profile schema (normative)

Serving configuration is declarative, keyed by (model, backend, RAM tier).
The launch truth for every profile lives in JSON under `configs/profiles/`;
launchers render it through `scripts/lib/profile_resolver.py` and never carry
their own copy. `scripts/92_resolve_profile.py` is the read-only CLI
(`render` / `check` / `list`).

## Layout

```
configs/
  hardware-matrix.json          host classes, tiers, usable-memory formulas,
                                engine-pin applicability, infeasible cells
  hosts/<host-id>.json          machine-local paths/ports/ram (committed for
                                spark-aba1; other machines add their own or
                                use ~/.config/frontier/host.json)
  profiles/<catalog-slug>/
    model.json                  artifacts + digests, engines, backend_support
    _base-*.json                optional overlay bases ("partial": true)
    <backend>-<class|tier>[-variant].json   one file per profile
```

Host selection order: `FRONTIER_HOST` env → `configs/hosts/<hostname>.json` →
`~/.config/frontier/host.json` → fail closed.

## Profile file (`schema_version: 4`)

Required: `profile_id`, `model`, `backend`, `hardware_class`
(`spark|mac|dgpu|strix|any`), `ram_tier_gib` (minimum-match; VRAM for dgpu,
plus `min_system_ram_gib`), `status`, `launch`, `memory_model`,
`context_cap`, `port_role`.

- `status.state` — `qualified` (requires `evidence`), `estimated` (requires
  `basis`; feasibility computed, promoted only by gates on real hardware), or
  `unsupported` (requires `reason`).
- `engine` + `artifact_roles` resolve against `model.json`; digests live
  there (host-independent), paths use `{model_root}`/`{cache_root}`
  placeholders. A GGUF may carry `identity` (sampled-hash form, GLM-5.2)
  instead of `sha256`.
- `launch.mechanism` — how the process is started:
  - `systemd-run`: transient unit with `containment` properties + flock
    (qwen38, qwen38-1m, laguna).
  - `setsid-memwatch`: memory_guard → memwatch arm → `setsid env -i`
    (glm52). `memory_guard.required_gib` = `safety.minimum_start_gib`;
    `memwatch.threshold_gib` = `safety.kill_floor_gib`.
  - `delegated-launcher`: `runuser` + `env -i <env> <delegate> {verb}`;
    the delegate owns admission and safety (dsv4 via
    `scripts/21_serve_llamacpp.sh`).
  - `setsid-watchdog-portable`: non-Linux hosts; setsid + recorded PGID +
    the platform watchdog (macOS: `scripts/06_memwatch_macos.sh`).
- `launch.args` / `launch.env` accept placeholders
  `{model} {mmproj} {draft_model} {binary} {port} {verb} {repo}
  {model_root} {cache_root} {state_root}`. Unknown or unresolved
  placeholders are hard errors; arrays replace whole under `extends`.
- `memory_model` maps 1:1 onto `scripts/02_membudget.py`
  (`kv_bytes_per_token`, `overhead_gib`, `extra_gib`, `floor_gib`).
  `resident_weights_gib` overrides the artifact sum for streaming engines
  (GLM-5.2 charges its expert-cache arena via `overhead_gib`, not weights).
  Values come from measured sources recorded in
  `configs/hardware-matrix.json` `kv_bytes_per_token`.
- `bench.stack_label` — the `--stack-label` identity for the eval harness.
  New profiles use `<model>-<backend>-<class>-<tier>g[-variant]`. Labels
  already spent in `results/holdout-ledger.json` are frozen forever;
  migrated Spark profiles keep theirs, with ancestry in
  `bench.historical_stack_labels`. Same-host knob tweaks must NOT change
  the label — the holdout rowset guard exists to block "same stack,
  tweaked config" re-spends. Re-running a spent rowset requires an
  owner-authorized `DSV4_LEDGER_NAMESPACE`; the profile system never mints
  namespaces.
- `switch_alias` — reserved for the five production aliases pinned by the
  AGENTS.md operator CLI (`dsv4|glm52|qwen38|qwen38-1m|laguna`); exactly one
  profile may carry each.

## `extends` merge rules (deliberately crude)

Scalars override; objects merge one level; **arrays replace whole**; exactly
one level (`_base` files set `"partial": true` and may not extend). The
resolver always emits the fully-resolved snapshot, so inheritance mistakes
cannot hide from fixture comparison.

## Conformance guarantees

`scripts/tests/fixtures/profile-conformance/<alias>.json` captures the exact
production launch commands (argv, env, systemd properties, safety
parameters) as they were before the switch refactor.
`test_profile_render_conformance.py` proves rendered profiles == fixtures;
`test_engine_switch_next.py` proves the switch's assembled commands ==
fixtures (via its test-only `render` verb). Any edit to a production profile that
changes launch behavior must update the fixture in the same commit and then
follow docs/runbook.md "Changing serving knobs or profiles" (unittest →
agent-gate → 30-min soak when memory-relevant).

## Promotion (estimated → qualified)

On the target hardware: `scripts/04_host_facts.py` →
`scripts/92_resolve_profile.py check` → serve via
`scripts/93_profile_serve.sh` → the BACKEND-CONTRACT gate set (speed,
accuracy **dev split only** — holdout stays owner-run —, template fidelity,
soak with the platform watchdog armed) → evidence bundle under
`results/<model>-gates/<profile-id-slug>-<date>/` → one reviewed commit
flips `status` with the evidence pointer. Community-qualified rows carry the
"Qualified (community)" attestation; numbers are never merged across
backends. See docs/QUALIFY-OFFHOST.md.
