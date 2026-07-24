
- `DS4_SERVER_COALESCE_MAX` bounds bank count.
- `DS4_CONT_PREFILL_CHUNK` bounds `raw_cap`.
- `DS4_SERVER_COALESCE_MAX_TOKENS` bounds both prefill scratch and, when smaller than the chunk, raw sizing.

A better code fix is stable-VA, per-bank VMM allocation: reserve every bank virtually, map the full raw bank only on admission, and unmap it when retired. At one active bank this saves approximately **1.78 GiB** versus six eager banks without changing pointer identity or graph keys.

### Graph-owned single-sequence caches: 0.95–1.50 GiB

These are required for a serial graph, but not for the persistent continuous graph after slab repointing. The batch allocator currently creates them because it reuses the general graph allocator.

A `batch_external_cache` allocation mode should omit the original raw/comp/index/DSpark rings and make `ds4_batch_slabs_alloc` use explicit storage-feature flags rather than testing `saved_* != NULL`. This also removes much of the unexpected context dependence.

### Index scores/mask: 264/520 MiB

The formula is `2 × comp_cap × prefill_cap × f32` at [ds4.c:10084](/tmp/ds4-v042-review/ds4.c:10084). It is a full worst-case score matrix. Token-tiling the indexer could reduce it, but this is higher-risk kernel work; lowering prefill capacity is safer.

### Full-vocabulary staging

These are real overestimates, but not the 11 GiB cause:

- `batch_logits` uses compile-time `DS4_MULTISEQ_MAX_SEQ=128`, not runtime `max_seq=6`: 63.1 MiB instead of 3.0 MiB ([ds4.c:10532](/tmp/ds4-v042-review/ds4.c:10532)). The comment still says 32, another sign it drifted.
- DSpark verification allocates `MS × block_width = 6×5=30` vocabulary rows even though `DS4_DSPARK_MAX_NLIVE` defaults to one ([ds4.c:34357](/tmp/ds4-v042-review/ds4.c:34357)). Host and device staging total about 30 MiB.
- FP16 logits would save only tens of MiB and require conversion before sampling. Runtime-row sizing is safer and smaller in scope.

## 3. Existing free paths and an idle-release design

Existing paths:

- `ds4_batch_ctx_destroy` frees slabs and the graph at [ds4.c:33013](/tmp/ds4-v042-review/ds4.c:33013).
- `ds4_session_free` frees the lazy serial graph at [ds4.c:37719](/tmp/ds4-v042-review/ds4.c:37719).
- Continuous temporary logits and DSpark tensors are freed at [ds4.c:35937](/tmp/ds4-v042-review/ds4.c:35937).
- `ds4_gpu_cleanup` frees F16 caches, CUDA temp, and token-tile scratch at [ds4_cuda.cu:2816](/tmp/ds4-v042-review/ds4_cuda.cu:2816).
- `ds4_gpu_invalidate_captured_graphs` destroys cached execs at [ds4_cuda.cu:16317](/tmp/ds4-v042-review/ds4_cuda.cu:16317).

There is no usable idle teardown. Gaps include:

- Derepack buffers are function-local statics and cannot be freed externally.
- `g_fp8_predecode_scratch` is not freed by `ds4_gpu_cleanup`.
- Graph execs are not explicitly destroyed by `ds4_gpu_cleanup`.
- The MMQ singleton at [ds4_mmq.cu:307](/tmp/ds4-v042-review/cuda/mmq/ds4_mmq.cu:307) is never deleted.
- No `cudaMemPoolTrimTo` or `cudaDeviceGraphMemTrim` is called.
- The capture-warm flag is a local static. Freeing scratch without resetting it would allow the next capture to skip the eager warm pass and attempt forbidden allocation during capture.

A safe soft idle hook should run only on the GPU worker, between jobs:

1. `cudaDeviceSynchronize()`.
2. Destroy all layer/continuous/dense/MoE graph execs.
3. Reset every graph `warmed` flag and the global capture-warm flag.
4. Free the attention F16 cache, CUDA temp, token-tile scratch, FP8 predecode scratch, and refactored derepack buffers.
5. Synchronize the async pool, then call `cudaMemPoolTrimTo(default_pool, 0)`.
6. Call `cudaDeviceGraphMemTrim(device)`.
7. Leave the batch graph/slabs allocated, preserving warm KV banks.

That releases roughly 3 GiB in the default path while avoiding multi-GiB graph reconstruction. A hard idle mode can additionally destroy the batch context and serial graph, but it must invalidate warm-bank metadata and serial checkpoints. Use a long hysteresis interval because repeated multi-GiB `cudaFree`/`cudaMalloc` cycles are a fragmentation risk on GB10 unified memory.

## 4. Recommended patch set, ranked

| Rank | Change | Saving | Risk |
|---|---|---:|---|
| 1, conditional | If `session graph allocated lazily` appears, add `DS4_SERVER_SERIAL_FALLBACK=0` and return 503 rather than allocate it | 5.47–6.28 GiB | Low CUDA risk; changes availability semantics |
| 2 | Deploy `DS4_SERVER_COALESCE_MAX_TOKENS=2048` | ~3.4–3.6 GiB | No correctness risk; prefill throughput cost |
| 3 | Deploy `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`, or cap it | 2.69 GiB | Low correctness risk; benchmark prefill |
| 4 | Add per-bank VMM raw/DSpark rings in `ds4.c:ds4_batch_slabs_alloc` and `ds4_dspark_slabs_alloc` | Up to ~1.78 GiB at one active bank | Moderate; admission/fork/reset must ensure mappings before access |
| 5 | Add batch-only graph storage mode to `metal_graph_alloc_raw_cap` | 0.95–1.50 GiB | Moderate; audit every path for repoint-before-use |
| 6 | Add soft idle cache release in `ds4_cuda.cu`, MMQ trim in `cuda/mmq`, timed worker hook in `ds4_server.c:dequeue` | ~3 GiB, overlapping rank 3 | Moderate; capture-warm reset is mandatory |
| 7 | Pass runtime `max_seq` into graph allocation and size `batch_logits` accordingly | ~60 MiB | Low |
| 8 | Size DSpark verify logits from `max(MS, B×spec_live_cap)` | ~25 MiB | Low–moderate |
| 9 | Conditionalize the unconditional 32 MiB FP8 predecode warm allocation | 32 MiB | Low |

Before changing allocation logic, I would add a one-shot census around batch-context creation and first generation reporting:

- `g_q8_f16_bytes`
- `g_cuda_tmp_bytes`, token-tile and FP8-predecode sizes
- default CUDA pool used/reserved bytes
- CUDA graph used/reserved bytes
- mapped VMM slab bytes
- whether the serial graph and derepack buffers exist

That instrumentation will distinguish a source-level allocation from UMA first-touch or CUDA driver retention and should settle the “identical at 32K/64K” observation immediately.
