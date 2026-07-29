# Frontier at Home

This repository builds reproducible, safe ways to run frontier-level models on
consumer-accessible hardware. It starts with CUDA on an NVIDIA GB10 DGX Spark
and with DeepSeek V4 Flash and GLM-5.2, but its scope is deliberately broader:
additional accelerators, inference architectures, model families, compression
methods, and storage tiers belong here when they come with honest measurements
and a dependable operator path.

The goal is not a collection of one-off demos. A contributed profile should be
something another person can build, qualify at its largest useful context,
switch to with one command, recover from safely, and audit from preserved raw
evidence.

## Current CUDA status

Status below is current as of 2026-07-29.

| Model | Current result on one DGX Spark |
| --- | --- |
| **DeepSeek V4 Flash** | Direct 1M qualification passed with a `1,048,576` cap and `1,000,044` actual prompt tokens. Deterministic retrieval, negative control, completed generation, and resource-safety checks passed with more than 14 GiB available at the measured low point. This is the qualified default profile; it may be intentionally stopped while a contained GLM experiment owns the machine. |
| **GLM-5.2** | Active qualification. Real production tensors were captured and the F16 cache path passed its initial checks. Block-scaled E4M3 and symmetric-int8 both failed the fixed 100-case confidence bounds and remain preserved negative results. Affine-int8 looked better offline, but its first integrated run was invalidated because the default-off binary changed baseline output; a clean default-path identity repair is now gated against the trusted parent before any new quality run. GLM has **not** yet passed the direct 1M gate or replaced DeepSeek as the default. |

DeepSeek’s older frozen ≤28K engine comparison selected `entrpi/ds4-on-spark`
over upstream llama.cpp on composite accuracy and speed. The product profile
uses llama.cpp because long context is the priority; the older benchmark remains
unchanged in [results/DECISION.md](results/DECISION.md), with the rationale in
[results/DECISION-OVERRIDE.md](results/DECISION-OVERRIDE.md).

Production traffic follows
`Tailscale Serve → Caddy :8010 → authenticated streaming helper :8014 → llama.cpp :8011`.
Listeners are loopback-only, Funnel is forbidden, credentials are stripped
before the engine, and a watchdog protects unified CPU/GPU memory from a
whole-system freeze.

## Beyond CUDA

Apple Silicon/macOS backends and model profiles are open to pull requests.
Useful contributions include MLX, Metal, llama.cpp Metal, model-specific cache
or MoE work, and reproducible qualification on Mac hardware. The same evidence,
largest-context, safety, authentication, switching, and rollback expectations
apply, adapted to the platform’s memory and service controls.

Other Linux accelerators and CPU/offload architectures are also in scope. Start
with a measured baseline and roofline; do not assume that a CUDA-specific
optimization or DGX Spark memory threshold transfers to another machine.

## Reproduce and operate

- [REPRODUCING.md](REPRODUCING.md) gives the pinned host, build, benchmark, audit, and
  `llamacpp` production-install sequence.
- [docs/runbook.md](docs/runbook.md) covers day-2 operation and incidents.
- [PROTOCOL.md](PROTOCOL.md) defines the frozen evaluation versions.
- [docs/threat-model.md](docs/threat-model.md) states what the evidence does and does not
  prove.

## Contributing models and optimizations

Read [AGENTS.md](AGENTS.md) before changing an engine, model profile, benchmark,
or service. It is the working contract for both human and agent contributors.

The short version:

1. Reproduce inherited claims from clean source and independently hashed
   artifacts.
2. Write and commit a production-path test, demonstrate RED, then implement the
   smallest default-off diagnostic arm.
3. Clean-build only after safely unloading the active large model and recovering
   at least 110 GiB available memory.
4. Freeze source/binary/scorer/fixture/configuration hashes, obtain public
   randomness after the freeze, and run equal-fixture contained arms.
5. Use fixed scorers and preserve `manifest.json`, `raw.jsonl`, and
   `summary.json` for every outcome—including failures.
6. For context qualification, test the largest requested context directly.
   Smaller prompts are useful fidelity falsifiers, not context-capability
   evidence.
7. Never load two large models together. Experimental GLM runs use hard cgroup
   limits, disabled swap, continuous memory sampling, and an emergency kill
   floor.
8. Keep the authenticated endpoint and rollback behavior unchanged; DeepSeek
   remains the default until another profile passes all quality, safety,
   direct-1M, switching, and review gates.

The autonomous goal controller is:

```bash
scripts/glm52_goal.py run
scripts/glm52_goal.py resume
scripts/glm52_goal.py status --json
```

The stable operator interface is:

```bash
scripts/52_engine_switch.sh status --json
sudo scripts/52_engine_switch.sh glm52
sudo scripts/52_engine_switch.sh dsv4
```

Avoid routine reboots and repeated interactive privilege requests. Use the
installed delegated controls for exact, identity-verified operations; request
new authority only when no safe in-scope path exists.
