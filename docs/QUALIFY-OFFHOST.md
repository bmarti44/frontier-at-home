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

- A host file: `scripts/04_host_facts.py --host-file ~/.config/frontier/host.json`
  writes one from the detected class, backends, and memory tier. It handles
  Mac, Strix Halo, discrete CUDA (tier = VRAM), and CPU hosts, and it never
  overwrites an existing file. Commit it as `configs/hosts/<hostname>.json` to
  share it. `configs/hosts/example-macbook-32g.json` shows the schema.
- macOS: `brew install bash cmake flock` and put Homebrew's bin first in
  `PATH`. The scripts need bash 4+ (`mapfile`; `/bin/bash` is 3.2) and
  `flock(1)`. `setsid(1)` is not needed: `93_profile_serve.sh` falls back
  to `scripts/lib/setsid.py`. Fetch weights with the `hf` CLI
  (`uv tool install huggingface_hub`).
- HumanEval (accuracy cell) runs each sample in a pinned Docker image
  (`configs/pins/humaneval-runtime.json`). On macOS without Docker Desktop:
  `brew install colima docker`, then
  `colima start --arch aarch64 --vm-type vz --cpu 2 --memory 2` (without
  `--arch aarch64` Colima may ask for qemu to emulate x86_64), then
  `docker pull python@sha256:<repo_digest>` and
  `docker tag python@sha256:<repo_digest> python:3.12-slim`.
  `colima stop` when the run ends.
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

Every llama.cpp engine is built from a pin in `configs/pins/llamacpp.json`.
Off-Spark profiles on upstream llama.cpp (Qwen3.8-27B, DeepSeek-V4-Flash)
use the registry's `portable` pin. Laguna keeps the poolside fork pin.

```
P=$(python3 -c 'import json;print(json.load(open("configs/pins/llamacpp.json"))["portable"])')
scripts/14_build_llamacpp_pin.sh --pin "$P" --host-class metal                    # Apple Silicon
scripts/14_build_llamacpp_pin.sh --pin "$P" --host-class rocm --rocm-arch gfx1151 # Strix Halo (HIP)
scripts/14_build_llamacpp_pin.sh --pin "$P" --host-class vulkan                   # Strix Halo / other GPUs (Vulkan)
scripts/14_build_llamacpp_pin.sh --pin "$P" --host-class cuda-generic --cuda-arch 89
scripts/14_build_llamacpp_pin.sh --pin "$P" --host-class cpu
scripts/14_build_llamacpp_pin.sh --pin llamacpp-laguna-06f8cebd --host-class metal  # verify-on-hardware
scripts/11_build_ds4.sh --host-class metal                                          # ds4 Makefile default
```

Add `--dry-run` to print the resolved repository, commit, paths, and cmake
flags without building. Classes: `cuda-spark | cuda-generic | metal | rocm |
vulkan | cpu`. Off-Spark builds write a class-suffixed build manifest
(`configs/build-manifests/<pin>-<class>.json`) and never overwrite the
committed Spark one. The manifest records the toolchain flags and the
observed architecture assertion.

To try a newer llama.cpp, register a pin, point `portable` at it, rebuild,
and re-run the gates on the same fixtures. Spark-qualified engines
(`llamacpp-qwen38`, `llamacpp-laguna`, `llamacpp-fusion`) keep their pin
until the Spark itself requalifies on the new one.

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
The context and soak cells sample the same vm_stat availability
(`scripts/lib/host_memory.py`), so their floors mean the same thing as the
watchdog's. Nothing starts a model on its own: the whole bundle is one
owner-run command, and it stops the server when it finishes:

```
scripts/94_qualify_profile.py --profile <model>/<profile-file> \
    --accuracy-split dev --with-soak
```

Cells can run in separate sittings into the same `--out` (`--cells
speed,toolcall,vision`, later `--cells accuracy`, ...); finished accuracy
suites are kept, not re-run.
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
