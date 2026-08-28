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

Status below is current as of 2026-08-19. Four models are qualified on the
CUDA reference host: Qwen 3.8 27B (`qwen38-1m` is the current serving default,
owner decision 2026-08-21), DeepSeek V4 Flash (the safe fallback the switch
restores to), and Laguna S 2.1 and GLM-5.2 as switchable engines. Every other
model/backend combination is N/A until someone qualifies it. A dash means this repository
does not yet contain a qualifying measurement — it does not mean zero. Context
size materially changes TTFT and prefill, so every number includes its
measured prompt size. These are single-user measurements, not concurrency
throughput. Performance cells use the fastest measured production path with
diagnostics disabled. Evidence-mode, control-configuration, instrumented,
smoke, and one-token diagnostic timings are kept in the evidence archive but
never substituted for headline model speed.

### Claim progress

| Model | Hardware / format | Context | Prefill t/s | Decode t/s | TTFT | Warm / short-prompt TTFT | Accuracy / fidelity | Current result, limitations, and caveats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **DeepSeek V4 Flash** | NVIDIA GB10 DGX Spark; UD-Q2_K_XL; llama.cpp | **1,000,044 tokens processed** with a `1,048,576` cap (qualified single-slot profile); installed default serves a `1,048,576` cap split across two 512k slots | **484.989 tok/s @ 4K**; **472.834 tok/s @ 16K**; **445.501 tok/s @ 28K** | **18.615 tok/s @ 4K**; **18.043 tok/s @ 16K**; **17.306 tok/s @ 28K** | **8.555 s @ 4K**; **34.761 s @ 16K**; **64.478 s @ 28K** | **0.421 s @ 52-token prompt**; agent-shaped cached turns process ~17 tokens ([agent-gate](results/agent-gate-2026-08-01.json)) | GSM8K holdout **97.00%** (97/100); MMLU-Pro holdout **74.09%** (183/247); HumanEval **73.78%** (121/164); composite **81.62%** | Qualified CUDA default; measurements predate the 2026-08-09 weights swap to the 0731 release — see the [DeepSeek notes](#deepseek-v4-flash-notes) below. Speed values are the 2026-08-01 five-rep suite on the installed 1M-fast profile (ub/b=2048, two 512k slots, owner-accepted 8 GiB watchdog floor). Direct 1M retrieval, negative control, generation, and safety checks passed on the single-slot ub=256 profile, which remains available via the engine switch; the installed default caps a single request at 512k tokens. The displayed latency/throughput measurements are the ≤28K suite, not a 1M speed claim. |
| **GLM-5.2** | NVIDIA GB10 DGX Spark; full 256-expert model, dense Q4_0 + routed IQ2_XXS, direct-slot expert cache (owner-accepted candidate 2026-08-18) | Fast profile configured for **32,768 tokens**; direct 1M not yet qualified | — (28K bench cell not strict-valid: GLM THINKING token accounting; measured ~41 tok/s recorded in [the qualification bundle](results/glm52-gates/fullq4-qualification-2026-08-18/summary.md)) | **3.28 tok/s** (qualified diagnostics-off bench, shallow context); 28K cell — (same caveat; measured 2.27–2.77) | — | **21.5 s** warm short-prompt (second rep; residual cache warming in rep 1) | Full 100-case suite: mean NLL **0.5139**, hosted-reference top-1 **82.9%** (full-Q8 reference: 0.4672 / 83.4%) | Switchable production candidate (`scripts/52_engine_switch.sh glm52`), not the serving default. Decode improved 2.33→3.28 (shallow) via direct-slot dispatch + Q4_0 dense; fidelity delta owner-accepted. Expert prune, prefetch, GPU directory, and two kernel widenings were measured and rejected/neutral (see results/glm52-gates/). Remaining levers are multi-day kernel projects; parity with DSV4 is recorded as not achievable on this hardware. Spark-only: the engine is a repo-locally patched ds4 CUDA binary with no portable equivalent. |
| **Qwen 3.8 27B** | NVIDIA GB10 DGX Spark; Q4_K_M GGUF + mmproj-f16; mainline llama.cpp b10488 | Fast profile configured for **32,768 tokens**; 1M profile (`qwen38-1m`) serves 1,048,576 as four native 262K slots | **698.7 tok/s @ 28K** | **17.46 tok/s @ 0-ctx**; **26.71 tok/s @ 28K** (production MTP profile n-max 8, p-min 0.6 — code-tuned, greedy-exact-validated) | **49.75 s @ 28K** | **0.39 s** short prompt | GSM8K holdout **98.00%** (98/100); MMLU-Pro holdout **85.02%** (210/247); HumanEval **79.27%** (130/164); MMMU-val-100 vision **64%** (0 transport errors) — reasoning effort low, 16384-token budget | Qualified switchable engine (`scripts/52_engine_switch.sh qwen38` / `qwen38-1m`). All cells strict-valid diagnostics-off ([speed](results/qwen38-gates/speed-2026-08-18/), [tune](results/qwen38-gates/tune-2026-08-19/summary.md), [accuracy](results/qwen38-gates/accuracy-2026-08-18/summary.md), [vision](results/qwen38-gates/vision-2026-08-19/summary.md)). MTP is byte-identical under greedy; deep-context decode exceeds shallow because draft acceptance rises on fixture continuations. |
| **Laguna S 2.1** | NVIDIA GB10 DGX Spark; UD-Q4_K_XL GGUF (3 shards) + DFlash BF16 draft; poolside llama.cpp fork `laguna` @ 06f8cebd | Qualified profile serves **393,216 tokens** as four native 98,304-token slots (1M native declined: measured 52.8 KiB/token f16 KV does not fit beside 73.4 GB weights; 524,288 breached the 8 GiB watchdog floor under sustained load) | **622.4 tok/s @28K** | **25.55 tok/s @0**; **27.52 tok/s @28K** (strict cells on the switch-launched production shape, DFlash n-max 4; raw decode without the draft is 20.97 @28K; code probes 28-45 tok/s acceptance-dependent) | **57.16 s @28K** | **0.60 s** short-prompt | GSM8K holdout **86.00%** (86/100); MMLU-Pro holdout **63.56%** (157/247; 64 of 90 misses are 16,384-token max-thinking truncations); HumanEval **89.63%** (147/164 — best on this host, +10.4 over the qwen default); tool-call probe 14/20 vs qwen38-1m 19/20 on the same harness | Qualified CUDA **switchable engine — NOT the serving default** (owner decision 2026-08-21; qwen38-1m remains default). Switch in with `sudo scripts/52_engine_switch.sh laguna`, back with `... qwen38-1m`. Thinking `max` is the model default and self-budgets: math/knowledge suites are truncation-sensitive at the repo's 16,384-token budget; code strength is the qualification case. Evidence: results/laguna-gates/ (G1-G5). |

Production traffic follows
`Tailscale Serve → Caddy :8010 → authenticated streaming helper :8014 → engine :8013`.
The engine port is set by `scripts/52_engine_switch.sh` (`PORT=8013`). Listeners
are loopback-only, Funnel is forbidden, credentials are stripped before the
engine, and a watchdog protects unified CPU/GPU memory from a whole-system
freeze.

### DeepSeek V4 Flash notes

The serving endpoint has loaded the
[0731 release](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(unsloth UD-Q2_K_XL, revision `fbbb5b93`) since 2026-08-09; every measurement in
the table above predates that swap and is the incumbent baseline, not a 0731
result. 0731 is the installed default **and** it is not qualified — both are
true: bring-up, weight integrity, token parity, golden correctness (11/11), and
dev-split accuracy re-runs are recorded, but speed, soak, holdout, and context
qualification have not been re-run. Key records:

- [Bring-up + accuracy comparison](results/dsv4-0731-staging/bringup-llamacpp-2026-08-09.json)
  — GSM8K dev at parity; MMLU-Pro dev showed a point-estimate regression under
  the non-thinking contract this reasoning model is not meant to run in.
- Thinking is the 0731 serving contract: the endpoint emits `reasoning_content`
  unconditionally, so `scripts/31_bench_accuracy.py` defaults to
  `--thinking-mode thinking`; reproducing a pre-0731 baseline requires
  `--thinking-mode chat` explicitly.
- **No verified local rollback path exists.** The pre-0731 anchors were deleted
  at owner instruction; re-fetch digests live in the git history of
  `weights/unsloth-ud-q2_k_xl/manifest.json` — see
  [the accounting record](results/dsv4-0731-staging/thinking-default-and-disk-2026-08-09.json).
- A separate 0731 evaluation on the **ds4** engine arm (the fast ≤28K
  alternative, not the serving path) is preserved in
  [results/dsv4-0731-staging/comparison-2026-08-09.json](results/dsv4-0731-staging/comparison-2026-08-09.json);
  the historical engine decision and its override are
  [results/DECISION.md](results/DECISION.md) and
  [results/DECISION-OVERRIDE.md](results/DECISION-OVERRIDE.md).

DeepSeek task accuracy is the audited llama.cpp result in
[results/DECISION.md](results/DECISION.md). GLM fidelity is the teacher-forced
comparison with a hosted FP8 reference in
[results/glm52-gates/G4-bench.json](results/glm52-gates/G4-bench.json) —
diagnostic fidelity, not task accuracy or qualification.

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
8. Keep the authenticated endpoint and rollback behavior unchanged; DeepSeek
   remains the default until another profile passes all quality, safety,
   direct-1M, switching, and review gates.

At the start of claimed work, agents set a persistent goal for the chosen model
and backend with the goal tool already provided by their harness. No separate
repository-specific goal system is required.

The stable operator interface is:

```bash
scripts/52_engine_switch.sh status --json
sudo scripts/52_engine_switch.sh glm52
sudo scripts/52_engine_switch.sh dsv4
```

Avoid routine reboots and repeated interactive privilege requests. Use the
installed delegated controls for exact, identity-verified operations; request
new authority only when no safe in-scope path exists.
