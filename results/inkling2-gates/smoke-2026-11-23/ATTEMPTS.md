# Smoke attempts (DIAGNOSTIC)

## run1 — FAILED_ATTEMPT (launch did not reach ready), 2026-11-23 03:23–03:26Z

Head `6d65b4fa`. Production `qwen38fn` was stopped through the switch at 03:23:17Z.

**Preflight**
- Page cache was released under the Inkling and Flash-Next trees.
- MemAvailable was 115.54 GiB (3 stable samples) and MemFree 115.55 GiB.

**GPU self-test** passed: torch 2.14.0+cu130, GB10 cc 12.1, the matmul was ok, `pageable_uses_host_page_tables=1`, and `inkling2_use_sm100_kernels=false`.

**Engine**
- Weights loaded: 18.19 GiB in 90.1 s.
- It then died in the vision encoder's startup profile (`profile_encoder_cache` → `Inkling2VisionAttention` → `vllm_flash_attn` FA2 `varlen_fwd`) with `CUDA error: the provided PTX was compiled with an unsupported toolchain` (`cudaErrorUnsupportedPtxVersion`).

**Cause**
- `_vllm_fa2_C.abi3.so` in the wheel has only sm_80 SASS plus sm_80 PTX at ISA `.version 9.2` (CUDA 13.2). The r580.159.03 driver (CUDA 13.0) cannot JIT that PTX for sm_121.
- `vision.py` hard-codes FA2 for the SM12x family.
- The text attention is unaffected: off SM100 it runs Inkling's own Triton kernels (`ops/attention_sm90.py`).
- The DSpark draft head is affected the same way: it declares `FlashAttentionBackend`. It must be addressed before the DSpark lever arm.

**Safety**
- No kernel lines during the window.
- MemAvailable stayed at or above 79.98 GiB, with no swap growth.
- memwatch did not fire.
- The scope was empty after the stop.
- Production was restored by `window.sh` (switch `restore`).

**Fix:** patch 0003 (`def3eba4`) routes the encoder attention through vLLM's Triton prefill kernel (`context_attention_fwd`, non-causal varlen) behind `INKLING2_VISION_ATTN=triton`, which the smoke profile sets. The self-test now also compares that kernel with exact fp32 attention at the encoder's shape before the server loads.

## run2 — COMPLETED (DIAGNOSTIC), 2026-11-23 03:38–03:48Z; production restored 03:59Z

Head `b451e1c3` (patches 0001–0003; `INKLING2_VISION_ATTN=triton`). Production `qwen38fn` was stopped through the switch before the run.

**Preflight**
- Page cache was released under the Inkling and Flash-Next trees.
- MemAvailable was 115.61 GiB (3 stable samples) and MemFree 114.87 GiB.

**GPU self-test** passed, including the Triton vision kernel against fp32 attention.

**Engine**
- Ready after 222 s. vLLM loaded 18.19 GiB of weights in 107.6 s.
- KV: 374,903 tokens in the fixed 24e9-byte arena, 2.86x at 131,072. There are 12 cache groups (block sizes 16 ×6 and 64 ×6, lcm 64).
- Memory: idle MemAvailable was 70.54 GiB. The low was 69.39 GiB (03:43:44Z), so the peak used 46.22 GiB. MemFree's low was 47.93 GiB, which leaves about 21 GiB of checkpoint page cache. SwapFree did not change.
- No kernel lines, and memwatch did not fire. The scope was empty after the stop.

**Results (diagnostic only; none is a headline number)**

| Check | Result |
|---|---|
| Greedy probes | 3/3 coherent; hashes `170c63d11040`, `c7fe1a930bcb`, `217aaabaf808`. SSE carried about 1.05–1.15 tokens per chunk. |
| Cold-length prefill | 6 unseen lengths, 3,428–42,585 tokens, 2,161 tok/s aggregate. The first request (3,428 tokens) took 7.6 s because it compiled. |
| Speed bench (2 reps + warmup) | decode 34.51 tok/s @0 and 29.41 @28,672; median TTFT 1.02 s @0 and 11.85 s @28,672. |
| Tool-call | 20/20 |
| MMMU (10 cases) | 3/10, with 7 invalid. Five answers ran out of the bench's 256-token default inside the reasoning. Two images (3,456 and 2,752 tokens) were refused with HTTP 400, because vLLM sizes the encoder budget from a 2048 px profiling image (2,704 tokens) and the model card accepts up to 4096 px. |

**Fixes after run2**
- `dace235e` / `27b52f53`:
  - patch 0004 (`INKLING2_MAX_IMAGE_SIDE=4096`, 10,609 tokens per image);
  - the kit's `qualification_options.vision_max_tokens` (MiMo's `ff4d6c0a`), with the Inkling profiles declaring 4096.
- KV accounting for the 1M profile:
  - The reported capacity charges every sliding-window group as if it held `max_model_len`, which makes it a conservative bound.
  - Decode held 5.3% of the pool for one request of about 29K tokens.
  - Each prefill held about 18% of the pool regardless of prompt length.
  - The 1M profile (`f7de2dd6`) sizes its arena at 68e9 bytes from the bound.
