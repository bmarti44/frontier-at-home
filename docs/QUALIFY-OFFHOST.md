# Qualifying a profile on community hardware

Profiles under `configs/profiles/` with `status.state: "estimated"` were
authored from computed feasibility (weights sizes + measured KV rates,
`configs/hardware-matrix.json`) and have never run on their target
hardware. This is the procedure that promotes one to `qualified`. The
reference Spark cannot verify your run — it verifies your evidence bundle's
internal consistency — so community-qualified rows always carry the
"Qualified (community)" attestation and are never merged with Spark-measured
numbers (docs/BACKEND-CONTRACT.md section 6).

## 0. Prerequisites

- A host file: copy `configs/hosts/example-macbook-32g.json`, replace
  `{HOME}` and the sizes with your machine's values, and save it as
  `~/.config/frontier/host.json` (or `configs/hosts/<your-host>.json` in
  your branch).
- `scripts/00_preflight.sh` reports `host_lock: not-applicable
  (community host ...)` — the versions.lock equality assertions apply only
  to the reference Spark.

## 1. Describe the host

```
scripts/04_host_facts.py --out host-facts.json
```

Facts must satisfy the profile's `backend`, `hardware_class`, and
`ram_tier_gib`. Detection never gates; the resolver decides fit.

## 2. Build the engine for your host class

```
scripts/12_build_qwen38_llamacpp.sh --host-class metal      # example
scripts/13_build_llamacpp.sh --host-class cpu               # DSV4 base pin
scripts/13_build_laguna_llamacpp.sh --host-class metal      # verify-on-hardware
scripts/11_build_ds4.sh --host-class metal                  # ds4 Makefile default
```

Classes: `cuda-generic | metal | rocm | cpu` (`--cuda-arch N`,
`--rocm-arch gfxNNNN` where applicable). Off-Spark builds write a
class-suffixed build manifest and never overwrite the committed Spark one.
The manifest records the toolchain and the observed architecture assertion.

## 3. Fetch weights and check fit

Fetch the profile's artifacts with `scripts/12_fetch_gguf.sh` (digests in
`configs/profiles/<model>/model.json` / `weights/<model>/manifest.json`),
then:

```
scripts/92_resolve_profile.py check --profile <model>/<profile-file>
```

`check` fails closed on host mismatch, computed infeasibility, or missing
artifacts.

## 4. Serve and run the gates

```
scripts/93_profile_serve.sh --profile <model>/<profile-file> start
```

On macOS the watchdog is `scripts/06_memwatch_macos.sh` (vm_stat
availability; floors are conservative and themselves verify-on-hardware).
All gates take `--base-url http://127.0.0.1:<port>` and are inherited
unchanged (docs/BACKEND-CONTRACT.md section 4):

- **G1 smoke** — `/v1/models` identity, `scripts/tests/template_fidelity.py`.
- **G2 speed** — `scripts/30_bench_speed.py` strict cells; if a Metal
  MTP/DFlash draft emits block streams, strict cells are invalid — record
  raw cells + wall-clock per the contract's dspark precedent.
- **G3 accuracy** — `scripts/31_bench_accuracy.py --split dev` with
  `--profile-id <profile_id>` and the profile's `bench.stack_label`.
  **Dev split only: the holdout ledger stays owner-run.** Community bundles
  cannot mint holdout numbers.
- **G4 soak** — `scripts/35_soak.py` (30 min) with the watchdog armed;
  record steady-state free memory and the measured KV-bytes/token against
  the profile's estimate (measured beats arithmetic — update
  `memory_model` if they differ).

## 5. Submit the evidence bundle

Bundle under `results/<model>-gates/<backend>-<tier>-<date>/`:
`host-facts.json`, the class-suffixed build manifest, the finalized profile
(engine digest filled in `model.json` or the class manifest), raw gate
JSONs, watchdog + soak logs, and a `sha256sums.txt` over all of it.

The promotion PR flips the profile's `status`:

```json
"status": {
  "state": "qualified",
  "qualified_at": "...",
  "evidence": "results/<model>-gates/<backend>-<tier>-<date>/",
  "attestation": "community-hardware; unreproduced on reference host"
}
```

Backend `implemented` in `configs/backends.json` flips to `true` with the
first qualified profile on that backend, not before. Estimated profiles
never flip it.

## Ledger rules (correctness-critical)

- `bench.stack_label` identifies the stack:
  `<model>-<backend>-<class>-<tier>g`. Different hardware tier = different
  stack = legitimately separate spend. Same-host knob tweaks must NOT
  change the label — the rowset guard exists to block exactly that
  re-spend.
- Re-running a spent rowset requires an owner-authorized
  `DSV4_LEDGER_NAMESPACE`. Nothing in the profile system mints namespaces.
