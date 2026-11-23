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
