# Frontier at Home

Run frontier-scale models on hardware you can actually buy — with receipts.

This repository builds reproducible, safe ways to operate frontier-level models
on consumer-accessible hardware. It is CUDA-first today, not CUDA-only:
DeepSeek V4 Flash and GLM-5.2 on an NVIDIA GB10 DGX Spark are the first
implementations, not the boundary. Additional models, accelerators, inference
architectures, compression methods, and storage tiers all belong here — as long
as they arrive with honest measurements and a dependable operator path.

This is not a collection of one-off demos. A contributed profile should be
something another person can build, qualify at its largest useful context,
switch to with one command, recover from safely, and audit from preserved raw
evidence.

## Project direction

Finish the CUDA profiles for DeepSeek V4 Flash and GLM-5.2, then apply the same
reproducible workflow to other frontier-class model families and
consumer-accessible systems. A backend need not copy the CUDA implementation:
platform-native engines and memory strategies are encouraged when they preserve
the same standards for correctness, safety, evidence, and repeatable operation.

## Model integration queue

The queue below was refreshed from Ollama's
[model catalog](https://ollama.com/search?c=cloud) on 2026-07-29. The catalog
listing is a discovery reference, not the artifact this repository has
qualified. A contributor must independently identify public local weights,
verify the license, hash every model/tokenizer artifact, and publish measured
evidence.

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

| Model | Ollama-listed context | Parameters / modalities | Open claims |
| --- | ---: | --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 976K | 756B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aglm-5.2?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aglm-5.2) |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 1M | 2.81T; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Akimi-k3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Akimi-k3) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | 256K | E2B, E4B, 12B, 26B, 31B; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Agemma4?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Agemma4) |
| [Qwen 3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) | 256K | 0.8B–397B family; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aqwen3.5?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aqwen3.5) |
| [MiniMax M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | 512K served | Not listed; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Aminimax-m3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Aminimax-m3) |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 256K | 120B / 12B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-super?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-super) |
| [Kimi K2.7 Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 256K | 1.04T; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Akimi-k2.7-code?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Akimi-k2.7-code) |
| [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 1M | 1.6T; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Adeepseek-v4-pro?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Adeepseek-v4-pro) |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 1M | 284B total / 13B active (Ollama page displays 158B); text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Adeepseek-v4-flash?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Adeepseek-v4-flash) |
| [Nemotron 3 Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 256K served | 550B / 55B active; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-ultra?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-ultra) |
| [GPT-OSS](https://huggingface.co/openai/gpt-oss-120b) | 128K | 20B, 120B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Agpt-oss?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Agpt-oss) |
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1M | 4B, 30B; text | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Anemotron-3-nano?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Anemotron-3-nano) |
| [Mistral Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | 256K | 675B; text, image | [![open self-declared claims](https://img.shields.io/github/issues-pr/bmarti44/frontier-at-home/claim%3Amistral-large-3?label=open%20claims)](https://github.com/bmarti44/frontier-at-home/pulls?q=is%3Apr+is%3Aopen+label%3Aclaim%3Amistral-large-3) |

## Current model status and measurements

> **Weights changed 2026-08-09.** The serving endpoint now loads
> [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
> (unsloth UD-Q2_K_XL, revision `fbbb5b93`). Every measurement in the table below
> predates that swap and was taken on the previous release — treat them as the
> incumbent baseline, not as 0731 results. What has been verified for 0731 is
> bring-up only: weight integrity including full SHA-256, memory admission,
> health, two slots at 524,288 tokens each, and golden correctness 10/10
> ([bring-up record](results/dsv4-0731-staging/bringup-llamacpp-2026-08-09.json)).
> Token parity passes (`exact-ids`) and golden correctness is 11/11. Accuracy has
> been re-run under the baselines' own non-thinking contract on the serving arm:
> GSM8K dev 97/100 (parity with the 97/100 baseline) and MMLU-Pro dev 188/253
> against 197/253 — nine items down, with overlapping Wilson intervals and
> comparable invalid counts (15 vs 16), so it is a point-estimate regression under
> a contract this model is not meant to run in. Speed, soak, holdout, and context
> qualification have **not** been re-run, so 0731 is **not qualified**.
>
> 0731 emits reasoning content by default, which changes the generation contract
> these baselines were measured under; golden checks had to be made reasoning-aware
> before they would score it correctly, and thinking is now the harness default.
>
> **There is no local rollback path.** The `*.gguf.pre0731` anchors were deleted at
> owner instruction on 2026-08-09. Reverting requires re-fetching
> `unsloth/DeepSeek-V4-Flash-GGUF` at revision `e3aa0d6a`; the shard digests needed
> to verify that fetch are in the git history of
> `weights/unsloth-ud-q2_k_xl/manifest.json` at `72d1db7^`. No rollback has been
> executed or verified end-to-end. The two pre-0731 copies under
> `/var/lib/dsv4-context/models/` are not a rollback path — see
> [the accounting record](results/dsv4-0731-staging/thinking-default-and-disk-2026-08-09.json).

Status below is current as of 2026-08-15. Only DeepSeek V4 Flash and GLM-5.2 on
CUDA are actively worked on; every other model/backend combination is N/A until
someone qualifies it. A dash means this repository does not yet contain a
qualifying measurement — it does not mean zero. Context size materially changes
TTFT and prefill, so every number includes its measured prompt size. These are
single-user measurements, not concurrency throughput. Performance cells use the
fastest measured production path with diagnostics disabled. Evidence-mode,
control-configuration, instrumented, smoke, and one-token diagnostic timings are
kept in the evidence archive but never substituted for headline model speed.

### Claim progress

| Model | Hardware / format | Context | Prefill t/s | Decode t/s | TTFT | Warm / short-prompt TTFT | Accuracy / fidelity | Current result, limitations, and caveats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **DeepSeek V4 Flash** | NVIDIA GB10 DGX Spark; UD-Q2_K_XL; llama.cpp | **1,000,044 tokens processed** with a `1,048,576` cap (qualified single-slot profile); installed default serves a `1,048,576` cap split across two 512k slots | **484.989 tok/s @ 4K**; **472.834 tok/s @ 16K**; **445.501 tok/s @ 28K** | **18.615 tok/s @ 4K**; **18.043 tok/s @ 16K**; **17.306 tok/s @ 28K** | **8.555 s @ 4K**; **34.761 s @ 16K**; **64.478 s @ 28K** | **0.421 s @ 52-token prompt**; agent-shaped cached turns process ~17 tokens ([agent-gate](results/agent-gate-2026-08-01.json)) | GSM8K holdout **97.00%** (97/100); MMLU-Pro holdout **74.09%** (183/247); HumanEval **73.78%** (121/164); composite **81.62%** | Qualified CUDA default. Speed values are the 2026-08-01 five-rep suite on the installed 1M-fast profile (ub/b=2048, two 512k slots, owner-accepted 8 GiB watchdog floor — see the unit file history). Direct 1M retrieval, negative control, generation, and safety checks passed on the single-slot ub=256 profile (>14 GiB at the low point), which remains available via the engine switch; the installed default caps a single request at 512k tokens. The displayed latency/throughput measurements are the ≤28K suite, not a 1M speed claim. The 52-token result is a short-prompt baseline, not proof of a restored 1M prefix. |
| **GLM-5.2** | NVIDIA GB10 DGX Spark; routed IQ2_XXS streaming engine | Fast profile configured for **32,768 tokens**; **11,648 tokens** have production-path functional evidence; direct 1M not yet qualified | **~23–32 tok/s** on production-path prompt ingestion | **2.33 tok/s** on the fastest measured production streaming path | **147–165 s @ 5,047-token cold prompt** | **1.755 s** on the third exact replay with disk-KV checkpointing | Fixed reference: mean NLL **0.4515**; hosted-reference top-1 agreement **83.4%**; target-logprob MAE **0.386 nat** | Active qualification. These are the fastest established production-path values, not timings from the slower instrumented matched campaign. Warm TTFT is an exact-replay result and does not represent an appended agent turn. The current 32K matched campaign and direct larger-context qualification are unfinished; no parity, direct-1M, or switching verdict exists yet. Byte-identical levers are being exhausted before any owner-gated fidelity spend. |

DeepSeek performance values come from the five-repetition
[speed suite](results/speed-llamacpp.json), re-run 2026-08-01 on the installed
1M-fast profile. Results from superseded slower profiles remain in their raw
evidence files and are intentionally omitted here. The 1M capability result and
the ≤28K performance suite answer different questions and must not be combined
into an implied 1M throughput figure.

**0731 on the ds4 arm (not the serving path).** The banner above covers the
llama.cpp endpoint, which is what this box actually serves per
[DECISION-OVERRIDE](results/DECISION-OVERRIDE.md). The rest of this section
records the separate 0731 evaluation on the **ds4** engine, which is retained as
the fast small-context alternative (≤~28K prompt tokens) and is not the serving
path. Its findings are what established that 0731 requires thinking enabled.

The [0731 release](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
was fetched for both arms — the ds4 lineage
([pin](configs/pins/antirez-imatrix-0731.json)) and the llama.cpp lineage
([pin](configs/pins/unsloth-ud-q2_k_xl-0731.json)) — every file verified
against its published SHA-256, and the ds4 lineage
[serves and generates](results/dsv4-0731-staging/bringup-2026-08-09.json) on the
`mtp` profile. A partial qualification run
([comparison](results/dsv4-0731-staging/comparison-2026-08-09.json)) measured it
against the v0.4.2 `dspark` baseline under each baseline's own recorded
generation contract:

| Check | 0731 | v0.4.2 baseline |
| --- | ---: | ---: |
| Golden correctness | 10 / 10 | 10 / 10 |
| Decode @ 52-token prompt | **21.630 tok/s** | 19.153 tok/s |
| Decode @ 4K | **19.742 tok/s** | 18.739 tok/s |
| Decode @ 16K | **19.172 tok/s** | 16.175 tok/s |
| GSM8K dev | 97 / 100 | 98 / 100 |
| MMLU-Pro dev | 178 / 253 | 192 / 253 |
| HumanEval | not run | 147 / 164 |

The speed suite is `suite_valid=true` across all fifteen reps and is faster at
every context. Accuracy was measured with `enable_thinking: false`, matching the
baselines' recorded `extra_body`; an earlier run that left it unset scored GSM8K
94/100 purely from that mismatch, so the contract is now asserted before any
delta is reported. GSM8K is parity (overlapping Wilson intervals). **MMLU-Pro is
a genuine 5.5-point regression**: a failure-mode breakdown attributes ten of the
fourteen lost items to incorrect answers rather than parse failures, and output
lengths are nearly identical (median 56 vs 60 tokens), so it is not a formatting
or truncation artifact. Forcing non-thinking on a release whose stated
improvements are in reasoning may itself be the wrong contract for 0731; 0731
with thinking enabled has not been measured against anything.

**0731 is the installed default as of 2026-08-09, and it is not qualified.** Those
are separate statements and both are true: the owner directed the swap on a box
carrying no traffic, so 0731 is what the endpoint serves, while the evidence
required by `scripts/34_decision.py` has not been produced for it. The table above
is the **ds4** arm under the non-thinking contract and predates the swap.

Thinking is the serving contract for 0731. The endpoint emits `reasoning_content`
with no request flag on both engines, so `scripts/31_bench_accuracy.py` defaults to
`--thinking-mode thinking` as of 2026-08-09; reproducing any pre-0731 baseline now
requires passing `--thinking-mode chat` explicitly. HumanEval
has not run (its harness pins a Docker-image runtime digest and the account
lacks docker group membership), and no holdout, token-parity, soak, or
agent-gate evidence exists. Three further findings are recorded: the MTP weights
are byte-identical across the release; no 0731 DSpark drafter exists in any
published repository, so the `dspark` profile cannot start; and one GSM8K item
returned `completion_tokens=0` with `finish_reason=stop` on a 69-token prompt,
reproduced by the harness fallback retry.

DeepSeek task accuracy is the audited llama.cpp result in
[results/DECISION.md](results/DECISION.md). GLM fidelity is the teacher-forced
comparison with a hosted FP8 reference in
[results/glm52-gates/G4-bench.json](results/glm52-gates/G4-bench.json). These
measure different things: GLM's top-1 agreement and log-probability error are
diagnostic fidelity measurements, not task accuracy or qualification. Live
campaign values are excluded until the fixed scorer publishes a complete
result.

DeepSeek's older frozen ≤28K engine comparison selected `entrpi/ds4-on-spark`
over upstream llama.cpp on composite accuracy and speed. The product profile
uses llama.cpp because long context is the priority; the older benchmark remains
unchanged in [results/DECISION.md](results/DECISION.md), with the rationale in
[results/DECISION-OVERRIDE.md](results/DECISION-OVERRIDE.md).

Production traffic follows
`Tailscale Serve → Caddy :8010 → authenticated streaming helper :8014 → llama.cpp :8013`.
The engine port is set by `scripts/52_engine_switch.sh` (`PORT=8013`), which overrides
the launcher's own `DSV4_PORT` default of 8011; 8013 is what is actually listening.
Listeners are loopback-only, Funnel is forbidden, credentials are stripped
before the engine, and a watchdog protects unified CPU/GPU memory from a
whole-system freeze.

### Other backends

No other backend has a repository-qualified measurement yet — every cell that
would appear below is N/A, and every row is open to pull requests. The same
evidence, largest-context, safety, authentication, switching, and rollback
expectations apply, adapted to each platform's memory and service controls.

| Backend | Hardware notes | Status |
| --- | --- | --- |
| Apple Silicon | MLX, Metal, or llama.cpp Metal. | N/A — open to pull requests |
| AMD Strix Halo | Zen 5 + RDNA 3.5 iGPU (`gfx1151`) via ROCm/HIP, up to 128 GB shared LPDDR5X at 256 GB/s. See AMD's [processor specifications](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html) and [ROCm system guidance](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html). | N/A — open to pull requests |
| AMD discrete ROCm | Radeon, Radeon Pro, and Instinct with dedicated VRAM; verify against AMD's [compatibility matrix](https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html). Multi-card and host-RAM offload setups need their own profiles. | N/A — open to pull requests |
| NVIDIA Jetson Thor | AGX Thor T5000: ARM64 Blackwell, 128 GB unified LPDDR5X at 273 GB/s, CUDA-X/JetPack. See NVIDIA's [specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/). | N/A — open to pull requests |
| Intel Xe | Arc Pro B-series via oneAPI/Level Zero, SYCL, or Vulkan; the [Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) has 24 GB GDDR6 at 456 GB/s and is multi-GPU Linux ready. | N/A — open to pull requests |
| Qualcomm Snapdragon X | X2 Elite: ARM64 SoC with Adreno GPU, Hexagon NPU, up to 128+ GB shared LPDDR5X at 228 GB/s. Practical large-model path may be CPU or Vulkan before the NPU is usable by an open engine. See the [product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf). | N/A — open to pull requests |
| Tenstorrent Tensix | Blackhole PCIe cards and QuietBox 2 with the open-source TT-Metalium/TT-NN stack; memory is distributed per device, not unified. See the [card overview](https://tenstorrent.com/en/hardware/cards) and [QuietBox 2 docs](https://docs.tenstorrent.com/tt-quietbox2-guide/first-timer/01-what-just-arrived/). | N/A — open to pull requests |
| CPU / other Linux accelerators | Start with a measured baseline and roofline; do not assume a CUDA-specific optimization or DGX Spark memory threshold transfers to another machine. | N/A — open to pull requests |

Contributions must record the exact hardware (device, per-card memory, PCIe or
interconnect topology), OS/kernel and driver or toolkit versions, backend,
power mode where relevant, model format, host-RAM offload, and whether reported
throughput includes inter-device transfers.

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
