# inkling2 vLLM patch series (native-runtime-001)

These patches go on top of the prebuilt vLLM nightly `0.34.0rc1.dev1133+g647a76866` (commit `647a76866b`, aarch64 cu130 wheel), which ships the Inkling 2 model code under `vllm/models/inkling2/`.

Each patch is Python only, so no CUDA build is needed. They apply in order with `git apply`, run in `runtime/lib/python3.12/site-packages`.

**Where things are pinned:** `configs/build-manifests/inkling-2-mini-runtime.json` pins each patch's sha256, the wheel, and (once the runtime is installed) the patched tree's digest.

| Patch | Source | What it does | Switch |
|---|---|---|---|
| 0001 | vLLM #69288 @ `2202ea70`, `vllm/` hunk only (the PR's `tests/` hunk is left out) | On GPUs without TRT-LLM-gen kernels (Hopper, SM12x such as GB10), `expert_plan` rounds the `max_ctas` upper bound up to a power of two. The Triton MoE kernels take `MAX_CTAS` as a constexpr, so batch sizes then share compiled kernels instead of compiling one per size | via 0002 |
| 0002 | local | Puts 0001 behind `INKLING2_MAX_CTAS_POW2`. It is read once at import, accepts only `0` or `1`, defaults to off (the exact pre-#69288 bound), and logs one startup line when on | `INKLING2_MAX_CTAS_POW2=1` |
| 0003 | local | Off SM100, runs the vision encoder's non-causal varlen attention through vLLM's Triton prefill kernel (`context_attention_fwd`) instead of `vllm_flash_attn` FA2. The wheel's FA2 ships sm_80 SASS plus PTX ISA 9.2 only, and the r580 (CUDA 13.0) driver cannot JIT that PTX on GB10: smoke run1 died with `cudaErrorUnsupportedPtxVersion`. Read once at import; only `fa` (default, upstream) or `triton` | `INKLING2_VISION_ATTN=triton` |
| 0004 | local | Lets the profile set the side of the square image vLLM profiles the vision encoder with. That image sizes the encoder cache and compute budget, so any larger image is refused with HTTP 400. The model card accepts up to 4096 px a side (103 x 103 = 10,609 tokens), but upstream profiles at 2048 px (2,704 tokens): smoke run2 refused two MMMU images of 3,456 and 2,752 tokens. Read once at import; an integer in [40, 8192], default 2048 (upstream) | `INKLING2_MAX_IMAGE_SIDE=4096` |

**With the switch off, the runtime runs the unpatched expert plan.** The rounded bound only changes which compiled kernel variant runs, not the routed rows, so output should be unchanged with it on; a screen records whether it is.
