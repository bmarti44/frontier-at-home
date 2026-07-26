# sol xhigh re-review — hardened patches (binary ac70a19ced45)
Date: 2026-07-26. Verdicts: Change 1 (skip-load) CORRECT. Change 2 (flush fix) still BROKEN.

Change 2's remaining hole: same-process close/reopen. ds4_gpu_cleanup resets
g_model_fd but left the shadow tuple and the persistent cache intact, so a
reopened model that reuses address + size + first 4 KiB + fd number would not
bump the generation and would hit the previous model's cached expert slabs.
Fixed by cuda_model_gen_invalidate() on teardown and on fd clear (gen3).

sol also rejected my statistical framing; see the correction in the ledger.


codex
Change 1 — CORRECT

At [ds4.c:40239](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40239), ID collapse is now conditional on a successful synchronous weights copy. On a clean weights-write failure, IDs remain genuine and weights remain original: the step is merely un-truncated and loads eight experts, not mathematically wrong. Behavior after a poisoned CUDA context is unverifiable. A failed/partial ID write is also correctness-safe because unchanged slots retain their true IDs while changed slots have zero weights; it can only lose some byte reduction. Downstream correctness consumers are the selected-expert loader and routed-MoE dispatch, and both receive matched IDs/weights. Other consumers found are profiling/debugging only.

Highest-value test: inject a clean failure before the router-weights copy, then assert IDs remain unchanged, every load has `unique=8`, and logits exactly match keep-8.

Change 2 — BROKEN

The predicate at [ds4_cuda.cu:164](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:164) still admits a collision without breaking FNV:

1. Clone model A to B, preserving size, header/layout and offsets; change only expert payload bytes after 4 KiB.
2. Open A at `P`, fd `d`, populate the persistent cache, then `ds4_engine_close`.
3. Cleanup resets `g_model_fd` but neither the shadow tuple nor persistent cache state ([ds4_cuda.cu:2309](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:2309)).
4. Reopen B in-process. The fd number normally repeats because the lock/model descriptors are reopened in the same order; mmap can reuse `P`.
5. `(P,S,fp,d)` is identical, so generation does not bump. Identical tensor offsets let cached A slabs hit during B inference. The old prefetch fd is likewise dangerous if prefetch is enabled.

Main-model ordering is correct: multi-tier sets fd at [ds4.c:56200](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56200) before registration; single-tier does so at [ds4.c:56317](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56317). No in-place main reload exists in `ds4.c`. The support-model setter at [ds4.c:56499](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56499) does run before installing its fd, but its simultaneously-live mapping has a different pointer and therefore bumps. Hypothetically setter-first with identical `P/S/fp` remains unsafe.

The 4-KiB read is bounds-safe at normal call sites because it uses `min(size,4096)` after a full valid mmap; concurrent unmap/truncation can still fault and is unsynchronized. Removing the outer bump leaves no zero-bump success path: registration bumps before both its early return and argument failure at [ds4_cuda.cu:3310](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3310). First-call `last_fd=-2` works; close/reopen state is the failure.

Highest-value test: same-process A-close/B-reopen using same-size, same-first-4-KiB GGUFs with one known expert payload changed; ensure fd/address reuse and assert the first B load misses or produces B-reference logits.

The +13.0% arithmetic is defensible as this harness’s observed one-prompt result, not yet as a general throughput claim. The duplicated arms are a useful negative-control observation, but one 1.6% contrast is not a noise-floor estimate, so “8× noise” is overstated. Moreover, the four values are two measurements within each of only two server runs, and keep-6 generates a different token/routing trajectory; thus cache locality is not controlled. That single-prompt trajectory confound is the strongest remaining objection. The reported CI method is undocumented and therefore unverifiable.
tokens used
178,751
Change 1 — CORRECT

At [ds4.c:40239](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40239), ID collapse is now conditional on a successful synchronous weights copy. On a clean weights-write failure, IDs remain genuine and weights remain original: the step is merely un-truncated and loads eight experts, not mathematically wrong. Behavior after a poisoned CUDA context is unverifiable. A failed/partial ID write is also correctness-safe because unchanged slots retain their true IDs while changed slots have zero weights; it can only lose some byte reduction. Downstream correctness consumers are the selected-expert loader and routed-MoE dispatch, and both receive matched IDs/weights. Other consumers found are profiling/debugging only.

Highest-value test: inject a clean failure before the router-weights copy, then assert IDs remain unchanged, every load has `unique=8`, and logits exactly match keep-8.

Change 2 — BROKEN

The predicate at [ds4_cuda.cu:164](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:164) still admits a collision without breaking FNV:

1. Clone model A to B, preserving size, header/layout and offsets; change only expert payload bytes after 4 KiB.
2. Open A at `P`, fd `d`, populate the persistent cache, then `ds4_engine_close`.
3. Cleanup resets `g_model_fd` but neither the shadow tuple nor persistent cache state ([ds4_cuda.cu:2309](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:2309)).
4. Reopen B in-process. The fd number normally repeats because the lock/model descriptors are reopened in the same order; mmap can reuse `P`.
5. `(P,S,fp,d)` is identical, so generation does not bump. Identical tensor offsets let cached A slabs hit during B inference. The old prefetch fd is likewise dangerous if prefetch is enabled.

Main-model ordering is correct: multi-tier sets fd at [ds4.c:56200](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56200) before registration; single-tier does so at [ds4.c:56317](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56317). No in-place main reload exists in `ds4.c`. The support-model setter at [ds4.c:56499](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:56499) does run before installing its fd, but its simultaneously-live mapping has a different pointer and therefore bumps. Hypothetically setter-first with identical `P/S/fp` remains unsafe.

The 4-KiB read is bounds-safe at normal call sites because it uses `min(size,4096)` after a full valid mmap; concurrent unmap/truncation can still fault and is unsynchronized. Removing the outer bump leaves no zero-bump success path: registration bumps before both its early return and argument failure at [ds4_cuda.cu:3310](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3310). First-call `last_fd=-2` works; close/reopen state is the failure.

Highest-value test: same-process A-close/B-reopen using same-size, same-first-4-KiB GGUFs with one known expert payload changed; ensure fd/address reuse and assert the first B load misses or produces B-reference logits.

The +13.0% arithmetic is defensible as this harness’s observed one-prompt result, not yet as a general throughput claim. The duplicated arms are a useful negative-control observation, but one 1.6% contrast is not a noise-floor estimate, so “8× noise” is overstated. Moreover, the four values are two measurements within each of only two server runs, and keep-6 generates a different token/routing trajectory; thus cache locality is not controlled. That single-prompt trajectory confound is the strongest remaining objection. The reported CI method is undocumented and therefore unverifiable.
