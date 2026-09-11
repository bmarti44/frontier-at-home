# Frontier at Home

Run frontier-scale models on hardware you can actually buy — with receipts.

This repository builds reproducible, safe ways to operate frontier-level models
on consumer-accessible hardware. Serving configuration is declarative: every
model is described by profiles keyed by (model, backend, RAM tier) under
[`configs/profiles/`](configs/profiles/), so the same model can be set up on a
128 GB DGX Spark, a 32 GB MacBook, a 16 GB discrete GPU, or a CPU-only box —
each with its own quantization, context cap, and memory budget. The NVIDIA GB10
DGX Spark (CUDA) is the qualified reference host today; profiles for Apple
Silicon, AMD ROCm, discrete CUDA GPUs, and CPU ship as computed estimates until
someone qualifies them on real hardware.

This is not a collection of one-off demos. A contributed profile should be
something another person can build, qualify at its largest useful context,
switch to with one command, recover from safely, and audit from preserved raw
evidence.

## Project direction

Grow the set of qualified (model × backend × RAM tier) profiles: more frontier
model families, more consumer hardware, one workflow. A backend need not copy
the CUDA implementation: platform-native engines and memory strategies are
encouraged when they preserve the same standards for correctness, safety,
evidence, and repeatable operation.

## Serving profiles

Launch truth lives in [`configs/profiles/<catalog-slug>/`](configs/profiles/):
a shared `model.json` (artifact digests, engines, backend support) plus one
profile per backend and RAM tier. [`configs/hardware-matrix.json`](configs/hardware-matrix.json)
records the host classes, memory tiers, usable-memory formulas, and every
infeasible cell with its reason. The schema and rules are in
[`docs/PROFILE-SCHEMA.md`](docs/PROFILE-SCHEMA.md).

```bash
scripts/04_host_facts.py                                  # describe this machine
scripts/92_resolve_profile.py list                        # what this host can serve
scripts/92_resolve_profile.py check --profile <model>/<profile>   # fit + digests
scripts/93_profile_serve.sh --profile <model>/<profile> start     # dev serving
```

Profiles carry a status: `qualified` (measured, evidence linked) or
`estimated` (feasibility computed from weight sizes and measured KV rates —
never a performance claim). Estimated profiles are promoted by running the
gate suite on the target hardware; the procedure is
[`docs/QUALIFY-OFFHOST.md`](docs/QUALIFY-OFFHOST.md). Production switching on
the reference host renders from the same profiles
(`scripts/52_engine_switch.sh`).

## Model integration queue

The queue below is maintained manually (last updated 2026-08-27). A listing
here is a discovery reference, not the artifact this repository has qualified.
A contributor must independently identify public local weights, verify the
license, hash every model/tokenizer artifact, and publish measured evidence.

To claim an integration:

1. Fork the repository and create exactly
   `claim-model/<catalog-slug>/<backend>`, using a slug and backend from
   [`models/catalog.json`](models/catalog.json).
2. Immediately open a **draft** pull request with the
   [model-integration template](https://github.com/bmarti44/frontier-at-home/compare?expand=1&template=model-integration-claim.md),
   before substantial implementation work.
3. Before changing an engine, use the agent harness's built-in goal tool to set
   a persistent goal for that model integration.
4. The safe `pull_request_target` workflow reads only the base repository's
   catalog—never fork code—and labels the PR with the model, backend, and
   `status:self-declared`.
5. Click a model's status badge to see its open, self-declared claims. A claim
   communicates intent and links the work in progress; it is not independent
   proof of activity, progress, or eventual qualification. Parallel claims are
   allowed when the hardware/backend differs or the approaches are genuinely
   independent.

Supported backend slugs are `cuda`, `apple-silicon`, `rocm`, `vulkan`,
`intel-xe`, `qualcomm`, `tenstorrent`, and `cpu`. Opening a claim PR reserves no
exclusive rights and does not lower the evidence requirements in
[`AGENTS.md`](AGENTS.md).

### Architecture claim mapping

The branch names combine one model from the queue with the primary backend for
the target architecture:

| Target architecture | Branch backend |
| --- | --- |
| NVIDIA DGX, GeForce, RTX/Blackwell, or Jetson using CUDA | `cuda` |
| Apple Silicon using MLX or Metal | `apple-silicon` |
| AMD Strix Halo, Radeon, Radeon Pro, or Instinct using HIP | `rocm` |
| Cross-vendor GPU implementation using Vulkan | `vulkan` |
| Intel Arc/Xe using oneAPI, Level Zero, or SYCL | `intel-xe` |
| Qualcomm Snapdragon/Adreno/Hexagon native implementation | `qualcomm` |
| Tenstorrent Tensix using TT-Metalium, TT-NN, or TT-Forge | `tenstorrent` |
| CPU-first or CPU/offload implementation | `cpu` |

For example, GLM-5.2 on Apple Silicon is
`claim-model/glm-5.2/apple-silicon`; Kimi K3 on Strix Halo through HIP is
`claim-model/kimi-k3/rocm`. Choose `vulkan` only when Vulkan is the integration
being qualified rather than a secondary fallback.

### Queue

| Model | Listed context | Parameters / modalities | Open claims |
| --- | ---: | --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 976K | 756B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aglm-5.2?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aglm-5.2) |
| [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 1M | 320B total / 18B active; text, image, video | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aglm-5.3-flash?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aglm-5.3-flash) |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 1M | 2.81T; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Akimi-k3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Akimi-k3) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | 256K | E2B, E4B, 12B, 26B, 31B; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Agemma4?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Agemma4) |
| [Qwen 3.8 Max](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 256K | 2.4T total / 95B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aqwen3.8-max?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aqwen3.8-max) |
| [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 256K | 28B dense; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aqwen3.8-27b?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aqwen3.8-27b) |
| [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 262K native (1M via YaRN) | 125B total / 6B active; text, image, video | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aqwen3.8-flash-next?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aqwen3.8-flash-next) |
| [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) | 1M | 118B MoE / 8B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Alaguna-s-2.1?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Alaguna-s-2.1) |
| [MiniMax M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | 512K served | Not listed; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aminimax-m3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aminimax-m3) |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 256K | 120B / 12B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-super?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-super) |
| [Kimi K2.7 Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 256K | 1.04T; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Akimi-k2.7-code?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Akimi-k2.7-code) |
| [DeepSeek V4 Pro 0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | 1M | 1.6T total / 49B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Adeepseek-v4-pro?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Adeepseek-v4-pro) |
| [DeepSeek V4 Flash 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 1M | 284B total / 13B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Adeepseek-v4-flash?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Adeepseek-v4-flash) |
| [Nemotron 3 Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 256K served | 550B / 55B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-ultra?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-ultra) |
| [GPT-OSS](https://huggingface.co/openai/gpt-oss-120b) | 128K | 20B, 120B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Agpt-oss?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Agpt-oss) |
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1M | 4B, 30B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-nano?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-nano) |
| [Mistral Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | 256K | 675B; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Amistral-large-3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Amistral-large-3) |

## Current model status and measurements

Five models are qualified on the CUDA reference host (NVIDIA GB10 DGX Spark,
128 GB unified memory). `qwen38-1m` is the serving default (owner decision
2026-08-21); the others are switchable engines. Every other model/backend
combination is N/A until someone qualifies it. A dash means this repository
does not yet contain a qualifying measurement, not zero. Every number carries
its measured prompt size because context changes TTFT and prefill; all cells
are single-user, diagnostics-off, fastest measured production path. One
command reproduces a row: `scripts/94_qualify_profile.py --profile <model>/<profile>`
([docs/QUALIFY-PROFILE.md](docs/QUALIFY-PROFILE.md)).

### Speed

Prefill and TTFT are at a 28,672-token prompt; decode is tokens per second at
an empty context and at 28K; short-prompt TTFT is the warm, ~50-token case.

| Model | Format · engine | Context served | Prefill tok/s @28K | Decode tok/s @0 / @28K | TTFT @28K | Short-prompt TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Qwen 3.8 27B** (default) | Q4_K_M GGUF + mmproj · llama.cpp b10488, MTP draft | 1,048,576 as four 262K slots | **698.7** | **17.46** / **26.71** | 49.75 s | **0.39 s** |
| **GLM-5.3 Flash** | EXL3 2-bit experts, fp8 KV · vLLM fork + vllm-exl3 | 1,048,576 as four 262K slots (1,000,560 filled, 4/4 needles) | 512.5 cold (~3,400 warm prefix) | **20.06** / 19.91 | 59.05 s | 5.06 s |
| **Laguna S 2.1** | UD-Q4_K_XL GGUF + DFlash draft · poolside llama.cpp | 393,216 as four 98K slots | 622.4 | **25.55** / **27.52** | 57.16 s | 0.60 s |
| **DeepSeek V4 Flash** | UD-Q2_K_XL GGUF · llama.cpp | 1,048,576 as two 512K slots (1,000,044 processed single-slot) | 445.5 (485.0 @4K) | 18.6 @4K / 17.31 | 64.48 s | 0.42 s |
| **GLM-5.2** | dense Q4_0 + routed IQ2_XXS · patched ds4 CUDA (Spark-only) | 32,768 (1M not qualified) | — (28K cell not strict-valid; ~41 measured) | 3.28 shallow / 2.3–2.8 (not strict) | — | 21.5 s |

### Accuracy and fidelity

Task suites use the repo holdouts at reasoning effort low with a 16,384-token
budget; vision is MMMU-val-100 in chat mode; the tool-call probe is 20 cases.
Teacher fidelity is teacher-forced NLL against a reference, a diagnostic that is
not task accuracy.

| Model | GSM8K holdout | MMLU-Pro holdout | HumanEval | Vision MMMU-100 | Tool-call | Teacher fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen 3.8 27B** (default) | **98.00%** (98/100) | **85.02%** (210/247) | 79.27% (130/164) | 64% | 19/20 | — |
| **GLM-5.3 Flash** | not run | not run | not run | **73%** | **20/20** | dNLL **0.079** (upper-95 0.106), top-1 −1.53 pp vs BF16 teacher logits over 25 × 2,047 positions; fails the 0.01 gate |
| **Laguna S 2.1** | 86.00% (86/100) | 63.56% (157/247) | **89.63%** (147/164) | — | 14/20 | — |
| **DeepSeek V4 Flash** | 97.00% (97/100) | 74.09% (183/247) | 73.78% (121/164) | — | — | — |
| **GLM-5.2** | — | — | — | — | — | mean NLL 0.5139, top-1 82.9% vs hosted FP8 over 100 cases (full-Q8 reference: 0.4672 / 83.4%) |

### Status and caveats

- **Qwen 3.8 27B** — serving default (`sudo scripts/52_engine_switch.sh qwen38-1m`). All cells strict-valid ([speed](results/qwen38-gates/speed-2026-08-18/), [tune](results/qwen38-gates/tune-2026-08-19/summary.md), [accuracy](results/qwen38-gates/accuracy-2026-08-18/summary.md), [vision](results/qwen38-gates/vision-2026-08-19/summary.md)); the kit reproduces the row ([kit check](results/qwen38-gates/qualify-2026-09-11-kit-check/SUMMARY.md)). MTP is byte-identical under greedy; 28K decode beats 0-ctx because draft acceptance rises on fixture continuations.
- **GLM-5.3 Flash** — optional engine (`sudo scripts/52_engine_switch.sh glm53-1m`), not the default. Decode at 0-ctx beats Qwen; decode at 28K, prefill, and short-prompt TTFT are below it because the MTP layer cannot run on this stack (no SM120 sparse-MLA decode kernel shape for it). No 2-bit pack of a 320B model meets the 0.01 dNLL gate in 120 GiB. GSM8K/MMLU-Pro/HumanEval need a GLM chat encoder, not yet registered. 30-minute soak passed (136 requests, 0 errors). Evidence: [qualify-2026-09-11](results/glm53-flash-gates/qualify-2026-09-11/SUMMARY.md), [STATUS.md](results/glm53-flash-gates/STATUS.md).
- **Laguna S 2.1** — switchable engine (`sudo scripts/52_engine_switch.sh laguna`). Thinking `max` is the model default and self-budgets, so math/knowledge suites are truncation-sensitive at 16,384 tokens (64 of 90 MMLU-Pro misses); code strength is the qualification case. 1M native declined: 52.8 KiB/token f16 KV does not fit beside 73.4 GB of weights. Evidence: results/laguna-gates/ (G1-G5).
- **DeepSeek V4 Flash** — the switch's safe fallback (`sudo scripts/52_engine_switch.sh dsv4`). The endpoint has served the [0731 release](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) since 2026-08-09; the numbers above are the audited pre-0731 baseline ([results/DECISION.md](results/DECISION.md)), and 0731 itself is not re-qualified for speed, soak, holdout, or context ([staging records](results/dsv4-0731-staging/)). Direct 1M retrieval and safety checks passed on the single-slot profile; the installed default caps one request at 512K.
- **GLM-5.2** — switchable candidate (`sudo scripts/52_engine_switch.sh glm52`), Spark-only patched engine. Decode went 2.33 → 3.28 tok/s with direct-slot dispatch and Q4_0 dense; parity with DeepSeek is recorded as not achievable on this hardware ([qualification bundle](results/glm52-gates/fullq4-qualification-2026-08-18/summary.md), fidelity [G4-bench](results/glm52-gates/G4-bench.json)).

Production traffic follows
`Tailscale Serve → Caddy :8010 → authenticated streaming helper :8014 → engine :8013`.
The engine port is set by `scripts/52_engine_switch.sh` (`PORT=8013`). Listeners
are loopback-only, Funnel is forbidden, credentials are stripped before the
engine, and a watchdog protects unified CPU/GPU memory from a whole-system
freeze.

### Other backends

No other backend has a repository-qualified measurement yet — every row is open
to pull requests. Estimated profiles for Apple Silicon, discrete CUDA GPUs,
Strix Halo, and CPU already exist under [`configs/profiles/`](configs/profiles/)
with computed memory budgets and recommended quantizations per RAM tier;
qualifying one on real hardware follows
[`docs/QUALIFY-OFFHOST.md`](docs/QUALIFY-OFFHOST.md). The same evidence,
largest-context, safety, authentication, switching, and rollback expectations
apply, adapted to each platform's memory and service controls.

| Backend | Hardware notes | Status |
| --- | --- | --- |
| Apple Silicon | MLX, Metal, or llama.cpp Metal. Estimated profiles cover 32-192 GB unified memory; 16 GB needs a smaller quant artifact (see the matrix). | Estimated profiles — open to pull requests |
| AMD Strix Halo | Zen 5 + RDNA 3.5 iGPU (`gfx1151`) via ROCm/HIP, up to 128 GB shared LPDDR5X at 256 GB/s. See AMD's [processor specifications](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html) and [ROCm system guidance](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html). | Estimated profiles — open to pull requests |
| AMD discrete ROCm | Radeon, Radeon Pro, and Instinct with dedicated VRAM; verify against AMD's [compatibility matrix](https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html). Multi-card and host-RAM offload setups need their own profiles. | N/A — open to pull requests |
| NVIDIA discrete CUDA | GeForce/RTX with 8-32 GB VRAM; estimated profiles use computed layer offload (small models) or MoE-on-CPU (large sparse models, 128 GB system RAM). | Estimated profiles — open to pull requests |
| NVIDIA Jetson Thor | AGX Thor T5000: ARM64 Blackwell, 128 GB unified LPDDR5X at 273 GB/s, CUDA-X/JetPack. See NVIDIA's [specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/). | N/A — open to pull requests |
| Intel Xe | Arc Pro B-series via oneAPI/Level Zero, SYCL, or Vulkan; the [Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) has 24 GB GDDR6 at 456 GB/s and is multi-GPU Linux ready. | N/A — open to pull requests |
| Qualcomm Snapdragon X | X2 Elite: ARM64 SoC with Adreno GPU, Hexagon NPU, up to 128+ GB shared LPDDR5X at 228 GB/s. Practical large-model path may be CPU or Vulkan before the NPU is usable by an open engine. See the [product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf). | N/A — open to pull requests |
| Tenstorrent Tensix | Blackhole PCIe cards and QuietBox 2 with the open-source TT-Metalium/TT-NN stack; memory is distributed per device, not unified. See the [card overview](https://tenstorrent.com/en/hardware/cards) and [QuietBox 2 docs](https://docs.tenstorrent.com/tt-quietbox2-guide/first-timer/01-what-just-arrived/). | N/A — open to pull requests |
| CPU / other Linux accelerators | Start with a measured baseline and roofline; do not assume a CUDA-specific optimization or DGX Spark memory threshold transfers to another machine. | Estimated profiles — open to pull requests |

Contributions must record the exact hardware (device, per-card memory, PCIe or
interconnect topology), OS/kernel and driver or toolkit versions, backend,
power mode where relevant, model format, host-RAM offload, and whether reported
throughput includes inter-device transfers.

## Reproduce and operate

- [REPRODUCING.md](REPRODUCING.md) gives the pinned host, build, benchmark, audit, and
  `llamacpp` production-install sequence.
- [docs/PROFILE-SCHEMA.md](docs/PROFILE-SCHEMA.md) is the normative profile
  schema; [docs/QUALIFY-OFFHOST.md](docs/QUALIFY-OFFHOST.md) is the
  community-hardware qualification procedure.
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
8. Keep the authenticated endpoint and rollback behavior unchanged; the
   serving default only changes after a profile passes all quality, safety,
   direct-1M, switching, and review gates and the owner promotes it.

At the start of claimed work, agents set a persistent goal for the chosen model
and backend with the goal tool already provided by their harness. No separate
repository-specific goal system is required.

The stable operator interface is:

```bash
scripts/52_engine_switch.sh status --json
sudo scripts/52_engine_switch.sh glm53-1m
sudo scripts/52_engine_switch.sh qwen38-1m
```

Avoid routine reboots and repeated interactive privilege requests. Use the
installed delegated controls for exact, identity-verified operations; request
new authority only when no safe in-scope path exists.
