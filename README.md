# Frontier at Home

This repository builds reproducible, safe ways to run frontier-level models on
consumer-accessible hardware. It is CUDA-first today—not CUDA-only. DeepSeek V4
Flash and GLM-5.2 on an NVIDIA GB10 DGX Spark are the initial implementations,
not the boundary of the project. Additional models, accelerators, inference
architectures, compression methods, and storage tiers belong here when they
come with honest measurements and a dependable operator path.

The goal is not a collection of one-off demos. A contributed profile should be
something another person can build, qualify at its largest useful context,
switch to with one command, recover from safely, and audit from preserved raw
evidence.

## Project direction

The near-term path is to finish the CUDA profiles for DeepSeek V4 Flash and
GLM-5.2, then apply the same reproducible workflow to other frontier-class model
families and consumer-accessible systems. A backend need not copy the CUDA
implementation: platform-native engines and memory strategies are encouraged
when they preserve the same standards for correctness, safety, evidence, and
repeatable operation.

## Current model status and measurements

Status below is current as of 2026-07-29. A dash means that this repository
does not yet contain a qualifying measurement; it does not mean zero. Context
size materially changes TTFT and prefill, so every number includes its measured
prompt size. These are single-user measurements, not concurrency throughput.

### CUDA

| Model | Hardware / format | Largest context result | TTFT | Prefill | Decode | Warm / short-prompt TTFT | Current result, limitations, and caveats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **DeepSeek V4 Flash** | NVIDIA GB10 DGX Spark; UD-Q2_K_XL; llama.cpp | **1,000,044 tokens processed** with a `1,048,576` cap | **14.268 s @ 4K**; **57.789 s @ 16K**; **104.526 s @ 28K** | **290.790 tok/s @ 4K**; **284.412 tok/s @ 16K**; **274.812 tok/s @ 28K** | **13.882 tok/s @ 4K**; **13.512 tok/s @ 16K**; **13.147 tok/s @ 28K** | **0.554 s @ 52-token prompt**; 30-minute soak median **14.037 decode tok/s** | Qualified CUDA default. Direct 1M retrieval, negative control, generation, and safety checks passed with more than 14 GiB available at the low point. The displayed latency/throughput measurements are the preserved ≤28K suite, not a 1M speed claim. The 52-token result is a short-prompt baseline, not proof of a restored 1M prefix. |
| **GLM-5.2** | NVIDIA GB10 DGX Spark; routed IQ2_XXS experimental engine | Not yet qualified | — | — | — | — | Active qualification. Real production tensors were captured and F16 passed initial checks. Block-scaled E4M3 and symmetric-int8 failed the fixed 100-case confidence bounds. Affine-int8 still needs a root-authoritative paired campaign. No end-to-end TTFT, prefill, decode, or direct-1M number is accepted yet; research projections are deliberately excluded from this table. |

DeepSeek performance values come from the five-repetition
[speed suite](results/speed-llamacpp.json); the sustained decode value comes
from the 96-request [soak](results/soak-llamacpp.json). The 1M capability result
and the ≤28K performance suite answer different questions and must not be
combined into an implied 1M throughput figure.

### Apple Silicon

| Model | Hardware / backend | Largest context result | TTFT | Prefill | Decode | Warm TTFT | Current result, limitations, and caveats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **DeepSeek V4 Flash** | Apple Silicon; MLX, Metal, or llama.cpp Metal | — | — | — | — | — | Open to pull requests. No Apple Silicon profile or repository-qualified measurement has been submitted. |
| **GLM-5.2** | Apple Silicon; MLX, Metal, or llama.cpp Metal | — | — | — | — | — | Open to pull requests. No Apple Silicon profile or repository-qualified measurement has been submitted. Model-specific cache, MoE, and storage-tier implementations are welcome when accompanied by reproducible evidence and safe operating instructions. |

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

Apple Silicon/macOS hardware and model profiles are explicitly open to pull
requests. The same evidence, largest-context, safety, authentication, switching,
and rollback expectations apply, adapted to the platform’s memory and service
controls.

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
