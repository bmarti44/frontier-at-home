# Diagnostic: why Phase 2 start 1 died at startup (DIAGNOSTIC, not acceptance)

These runs name the cause of the candidate-3 start-1 `FAIL`
(`../qualify-2026-10-11/RESULT.md`) and test a fix. None of them is
qualification evidence or a performance number.

Each arm loaded `qwen3.8-flash-next/cuda-spark-128g-1m` unchanged, with the
`run-phase2.sh` containment and a 112 GiB start guard on a quiet host. Arms
differ only in the hook reached through a scratch `PYTHONPATH`
(`run-diag.sh`), the stop level, and `dropcache.py`.

| Arm | Change | Outcome | Startup low (MemAvailable) | NVRM lines | Swap growth |
|---|---|---|---|---|---|
| a-1m | none, stack dumps only | Drop reproduced; SIGKILL at the 15 GiB stop | 13.86 GiB | 108 | — |
| b-1m-nodg | `VLLM_MOE_USE_DEEP_GEMM=0` | Past the drafter step; KV 1,079,117 tokens at 262,144; stopped at 15 GiB during the arena commit | 14.95 GiB | 124 | — |
| c-1m-nodg-drop | as b, plus checkpoint page-cache eviction during startup, stop at 11 GiB | **Ready**. KV 1,079,117 tokens (4.12×). Idle after 60 s: 13.91 GiB | 13.59 GiB (MemFree 12.48) | **0** | 0.00 GiB |

## Findings

1. **The start-1 drop is the drafter's DeepGEMM UE8M0 requant.**
   - Every stack dump during arm A's drop is in `fp8_utils.requant_weight_ue8m0_inplace`, line 1143, at `w_dq = w_q.to(torch.float32) * s_exp`.
   - That line is in the per-expert fp32 dequant/requant loop called by `prepare_fp8_moe_layer_for_deepgemm`, reached from the MTP drafter's `process_weights_after_loading`.
   - Memory grows by about 4 GiB/s and is still growing at the stop.
   - The smoke ran the same step with no growth. Why is not known; the per-block cast is `torch.compile(dynamic=True)`, and the cache state differs between those runs.
2. **Taking the drafter's FP8 MoE off DeepGEMM removes the step.**
   - With `VLLM_MOE_USE_DEEP_GEMM=0` the backend becomes `TRITON`.
   - The target's routed experts are NVFP4 on `FLASHINFER_CUTLASS` and do not use this path.
   - Greedy output is still the target's: the drafter only proposes tokens.
3. **The NVRM lines depend on MemFree, not MemAvailable.**
   - Every `NVRM: nvCheckOkFailedNoLog ... NV_ERR_NO_MEMORY ... mem_desc.c:1359` line lands during a large GPU allocation while MemFree is 1–3 GiB and up to about 23 GiB of checkpoint page cache is still being reclaimed.
   - In start 1 that was the target weights, the drafter and the requant. In arm B it was the drafter and the 17.9 GiB arena commit.
   - Keeping the checkpoint's unmapped page cache out of the way with unprivileged `posix_fadvise(DONTNEED)` kept MemFree at 12.5 GiB or more and produced none.
   - DONTNEED skips mapped pages, so the in-place PLE mapping is untouched.
4. **The measured idle is above the 11.9 GiB projection.**
   - The full 19.22 GB arena idles at 13.9 GiB MemAvailable on a 115.2 GiB quiet start.
   - The engine used 101.3 GiB against the memory model's 102.2 GiB (76 + 8.2 + 18.0).

## Next

Candidate 4 makes both changes on the production path, test-first:
- the 1m profile sets `VLLM_MOE_USE_DEEP_GEMM=0`;
- a worker class evicts the checkpoint's page cache from weight load through warmup, and stops before serving.

Candidate 4 gets a new qualification directory and preregistration. Candidate 3's directory stays frozen.
