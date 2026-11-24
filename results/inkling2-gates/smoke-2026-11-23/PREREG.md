# Inkling 2 Mini first-light smoke (DIAGNOSTIC)

Purpose: prove the native vLLM runtime loads and serves Inkling 2 Mini on this Spark, and measure what
the 1M profile and the lever screen need. **This is not a capability, speed or accuracy result**, and no
number from it may appear in the README or any headline table.

## Stack
- Weights: `~/models/inkling-2-mini` = `thinkingmachines/Inkling-2-Mini` @ `b85c3cfa`. Every file was
  checked against the HF sha256 / blob ids. Pinned in `weights/inkling-2-mini/manifest.json`.
- Runtime: `~/.cache/inkling-2-mini/native-runtime-001`, vLLM `0.34.0rc1.dev1133+g647a76866`
  (wheel sha256 `dfba4679…`), patches 0001+0002 applied. The 0002 gate is off: the profile does not set
  `INKLING2_MAX_CTAS_POW2`.
  - `vllm_tree` before patching is `ba2f72c8…` (6,525 files); after patching it is `9e93b5d8…`.
- Profile: `inkling-2-mini/cuda-spark-128g-smoke` on :8018. Two slots of 131,072, BF16 KV in a fixed
  24 GB arena, `--max-num-batched-tokens 2048`, no speculative decoding, reasoning effort `low`.

## Procedure (`run-smoke.sh <run-id>`, production stopped via `52_engine_switch.sh stop`)
1. Preflight before any directory is created: a clean tracked tree; the inference lock taken; the page
   cache released under the Inkling and Flash-Next trees; MemAvailable ≥ 110 GiB for 3 stable samples;
   MemFree ≥ 95 GiB.
2. Monitoring: a 1 Hz meminfo sampler and kernel-log capture throughout the run.
3. GPU self-test (`gpu_selftest.py`) in an 8 GiB swapless scope. It checks the bf16 matmul, cc, the
   pageable-memory attribute, and that vLLM picks the non-SM100 Inkling 2 MoE path.
4. Launch through `93_profile_serve.sh` in a user scope (`MemorySwapMax=0`, `OOMPolicy=kill`,
   `RuntimeMaxSec=7200`). memwatch kills at 15 GiB.
5. Record the KV pool (tokens), the attention block size, and the load and compile lines.
6. `probe.py`: 3 greedy prompts (sha256 of reasoning+content) plus SSE granularity.
7. `cold_ttft.py`: 6 prompts of unseen lengths (≈2–28K words), each with a unique prefix so the prefix
   cache cannot hit. It exposes per-length MoE recompiles (vLLM PR #69288).
8. `30_bench_speed.py` at 0 and 28,672 tokens: 2 reps plus 1 warmup.
9. `39_bench_toolcall.py`: 20 cases at temperature 0.
10. `38_bench_vision.py`: 10 MMMU cases in chat mode, as an eyeball check that images are not misread
    (vLLM #69046).
11. Stop. Then restore production through the switch, with output to a file, and check the kernel log.

## Failure conditions
The run counts as a failed attempt on any of: launch never reaching ready, OOM, any NVRM/Xid line,
memwatch kill, swap growth, or a scope surviving the stop. Failed attempts are kept in their own run
directory, and a rerun gets a new id.

## What it feeds
- The engine-reported KV bytes per token sizes the 1M profile's fixed arena: 4×262,144 slots plus
  transient blocks, with a projected low point ≥ 12 GiB.
- The measured peak and idle memory set `overhead_gib` / `minimum_start_gib`.
- The cold-length TTFT pattern decides whether the #69288 arm is worth screening.
