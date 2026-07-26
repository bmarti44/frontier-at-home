# sol xhigh adversarial review — DS4_GLM_TOPK_SKIP_LOAD + model-generation flush fix
Date: 2026-07-26. Reviewer: codex/sol, xhigh, read-only sandbox.
Source reviewed: verbatim copies of the COMPILED ds4.c / ds4_cuda.cu (the repo's
vendor/ds4 tree is a stale Jul-16 snapshot and misled an earlier review).

Verdicts: Change 1 CORRECT-BUT-RISKY, Change 2 BROKEN.


codex
Bottom line: Change 1 is sound on the intended finite IQ2 decode path but has observability and failure-atomicity defects. Change 2 fixes the measured storm, but its pointer/size identity rule reintroduces stale-model correctness failures.

### Change 1 — CORRECT-BUT-RISKY

Strict “true top-`keep` for every input” answer: no, because ties can retain more than `keep`. The ID-collapse itself is equivalent to the implemented weight truncation for finite values.

Drivers:

- Dropped weights are zeroed and optionally renormalized at [ds4.c:40215](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40215); IDs are collapsed at [ds4.c:40247](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40247).
- The loader really deduplicates IDs and remaps every slot to a compact index at [ds4_cuda.cu:24292](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24292) and [ds4_cuda.cu:24688](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24688).
- IQ2 gate/up multiplies each slot by its own router weight at [ds4_cuda.cu:17134](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:17134) and [ds4_cuda.cu:17160](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:17160). IQ2 down is slot-local, and the final kernel simply sums eight slots at [ds4_cuda.cu:18138](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:18138) and [ds4_cuda.cu:20997](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:20997). No division, renormalization, unique-count use, or nonzero-slot count exists there.
- `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` is irrelevant for one-token decode: sorting requires `n_tokens > 1`, so atomic down is already false at [ds4_cuda.cu:21397](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:21397).

Numerics:

- Router weights are nonnegative and normally sum to 2.5 at [ds4_cuda.cu:27987](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:27987).
- For eight nonnegative weights, retained top-`K` mass is at least `K/8` of total, hence `total/kept <= 8/K`: at keep-6, at most 1.333× and retained mass normally at least 1.875. Thus `0 * scale` remains exact zero; realistic finite routing cannot create Inf/NaN here.
- Upstream NaN/Inf is not safely handled, but inference is already invalid in that case.
- Ties at [ds4.c:40219](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40219) retain more than `keep`. That only reduces byte savings; buffer sizing and kernels accept the larger compact count.

Downstream damage:

- Expert profiling runs after mutation at [ds4.c:40277](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40277), so it records fake duplicated IDs.
- Debug dumps also expose fake IDs at [ds4.c:46258](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:46258). `XTRACE` likewise sees collapsed IDs.
- Cross-layer prefetch happens before mutation and uses separate batch scratch. No model-output MTP or TP consumer requiring the original IDs was found. Streaming TP is explicitly rejected.
- This block is only in `glm_graph_encode_sparse_ffn_one`; batch prefill and MTP routing are unaffected.

Failing sequence: the router-weight write return is ignored at [ds4.c:40234](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40234). If that CUDA copy fails but the subsequent ID write succeeds, old nonzero weights are paired with duplicated keeper IDs, repeating the keeper contribution incorrectly.

Highest-value test: on the actual IQ2 streaming path, run each routed layer with identical input/cache state using (A) zeroed weights plus true IDs and (B) identical weights plus collapsed IDs. Require bitwise-equal routed output and `compact_count == number_of_nonzero_slots`; include ordinary keep-6, threshold ties, and zero-weight cases.

### Change 2 — BROKEN

The performance diagnosis is correct; the replacement invalidation predicate is not a safe model-generation definition.

Drivers:

- Generation now changes only on pointer/size inequality at [ds4_cuda.cu:119](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:119).
- Expert-cache validity depends entirely on that generation at [ds4_cuda.cu:24026](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24026).
- Cache lookup checks only `(layer, expert)` and unchanged offsets, not model identity/content, at [ds4_cuda.cu:24187](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24187).
- The prefetch pool’s FD and slots also rely on generation at [ds4_cuda.cu:23863](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23863), [ds4_cuda.cu:23878](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23878), and [ds4_cuda.cu:23949](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23949).
- Changing the model FD does not bump generation at [ds4_cuda.cu:3766](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3766).

Failing sequence:

1. Register model A at pointer `P`, size `S`; load expert `(L,E)` into the persistent cache and initialize prefetch from fd A.
2. Replace the bytes backing `P` with same-sized/layout-compatible model B—an in-place rewritten mapping or same-address remap—and install fd B.
3. Call the span/map setter again with `P,S`.
4. The equality early return suppresses generation change. Cache lookup returns A’s slab because keys and offsets match; the prefetch pool can still serve bytes from its private fd A.

That is stale-weight inference. The old unconditional bump flushed the expert cache and disabled the old-FD prefetch pool. Whether the deployed server ever hot-reloads without full cleanup is unverifiable. Ordinary `ds4_engine_close` does reset the map state, and simultaneously live main/MTP maps normally have different pointers.

The actual source also contradicts the stated range behavior: `ds4_gpu_set_model_map_range` calls the helper at [ds4_cuda.cu:3238](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3238), then `register_model_map_no_copy` calls it again at [ds4_cuda.cu:3257](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3257). A real change increments twice; a no-op increments zero times. Double increment is redundant but otherwise harmless.

Highest-value test: cache an expert from model A, replace content with model B at the identical pointer and size, swap FD, re-register, then demand the same expert. Require B’s bytes/output and rejection of every A-generation prefetch slot. Current code should fail.

### Speed prediction

`74.2 MiB × 75 = 5,565 MiB = 5.435 GiB = 5.835 GB` requested per token. At 10.7 GB/s, an all-miss I/O floor is about 0.545 s/token, or 1.83 tok/s. Therefore 1.4–1.8 tok/s is entirely consistent with an almost-I/O-bound cold cache.

At 20% byte hits, NVMe time is still about `0.8 × 5.835 / 10.7 = 0.436 s/token`. Raising hits to 40–50% saves roughly 0.11–0.16 s/token, predicting about a 20–40% decode-rate improvement across the stated baseline range.

Prediction on record: **large, tens-of-percent speedup—roughly 20–40%, potentially more for repeated short requests; not reliably 2×**. For a long static-map decode, the gain may concentrate in early tokens because the old path eventually refills the cache after its last flush.
tokens used
224,976
Bottom line: Change 1 is sound on the intended finite IQ2 decode path but has observability and failure-atomicity defects. Change 2 fixes the measured storm, but its pointer/size identity rule reintroduces stale-model correctness failures.

### Change 1 — CORRECT-BUT-RISKY

Strict “true top-`keep` for every input” answer: no, because ties can retain more than `keep`. The ID-collapse itself is equivalent to the implemented weight truncation for finite values.

Drivers:

- Dropped weights are zeroed and optionally renormalized at [ds4.c:40215](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40215); IDs are collapsed at [ds4.c:40247](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40247).
- The loader really deduplicates IDs and remaps every slot to a compact index at [ds4_cuda.cu:24292](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24292) and [ds4_cuda.cu:24688](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24688).
- IQ2 gate/up multiplies each slot by its own router weight at [ds4_cuda.cu:17134](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:17134) and [ds4_cuda.cu:17160](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:17160). IQ2 down is slot-local, and the final kernel simply sums eight slots at [ds4_cuda.cu:18138](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:18138) and [ds4_cuda.cu:20997](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:20997). No division, renormalization, unique-count use, or nonzero-slot count exists there.
- `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` is irrelevant for one-token decode: sorting requires `n_tokens > 1`, so atomic down is already false at [ds4_cuda.cu:21397](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:21397).

Numerics:

- Router weights are nonnegative and normally sum to 2.5 at [ds4_cuda.cu:27987](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:27987).
- For eight nonnegative weights, retained top-`K` mass is at least `K/8` of total, hence `total/kept <= 8/K`: at keep-6, at most 1.333× and retained mass normally at least 1.875. Thus `0 * scale` remains exact zero; realistic finite routing cannot create Inf/NaN here.
- Upstream NaN/Inf is not safely handled, but inference is already invalid in that case.
- Ties at [ds4.c:40219](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40219) retain more than `keep`. That only reduces byte savings; buffer sizing and kernels accept the larger compact count.

Downstream damage:

- Expert profiling runs after mutation at [ds4.c:40277](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40277), so it records fake duplicated IDs.
- Debug dumps also expose fake IDs at [ds4.c:46258](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:46258). `XTRACE` likewise sees collapsed IDs.
- Cross-layer prefetch happens before mutation and uses separate batch scratch. No model-output MTP or TP consumer requiring the original IDs was found. Streaming TP is explicitly rejected.
- This block is only in `glm_graph_encode_sparse_ffn_one`; batch prefill and MTP routing are unaffected.

Failing sequence: the router-weight write return is ignored at [ds4.c:40234](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:40234). If that CUDA copy fails but the subsequent ID write succeeds, old nonzero weights are paired with duplicated keeper IDs, repeating the keeper contribution incorrectly.

Highest-value test: on the actual IQ2 streaming path, run each routed layer with identical input/cache state using (A) zeroed weights plus true IDs and (B) identical weights plus collapsed IDs. Require bitwise-equal routed output and `compact_count == number_of_nonzero_slots`; include ordinary keep-6, threshold ties, and zero-weight cases.

### Change 2 — BROKEN

The performance diagnosis is correct; the replacement invalidation predicate is not a safe model-generation definition.

Drivers:

- Generation now changes only on pointer/size inequality at [ds4_cuda.cu:119](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:119).
- Expert-cache validity depends entirely on that generation at [ds4_cuda.cu:24026](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24026).
- Cache lookup checks only `(layer, expert)` and unchanged offsets, not model identity/content, at [ds4_cuda.cu:24187](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:24187).
- The prefetch pool’s FD and slots also rely on generation at [ds4_cuda.cu:23863](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23863), [ds4_cuda.cu:23878](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23878), and [ds4_cuda.cu:23949](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:23949).
- Changing the model FD does not bump generation at [ds4_cuda.cu:3766](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3766).

Failing sequence:

1. Register model A at pointer `P`, size `S`; load expert `(L,E)` into the persistent cache and initialize prefetch from fd A.
2. Replace the bytes backing `P` with same-sized/layout-compatible model B—an in-place rewritten mapping or same-address remap—and install fd B.
3. Call the span/map setter again with `P,S`.
4. The equality early return suppresses generation change. Cache lookup returns A’s slab because keys and offsets match; the prefetch pool can still serve bytes from its private fd A.

That is stale-weight inference. The old unconditional bump flushed the expert cache and disabled the old-FD prefetch pool. Whether the deployed server ever hot-reloads without full cleanup is unverifiable. Ordinary `ds4_engine_close` does reset the map state, and simultaneously live main/MTP maps normally have different pointers.

The actual source also contradicts the stated range behavior: `ds4_gpu_set_model_map_range` calls the helper at [ds4_cuda.cu:3238](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3238), then `register_model_map_no_copy` calls it again at [ds4_cuda.cu:3257](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4_cuda.cu:3257). A real change increments twice; a no-op increments zero times. Double increment is redundant but otherwise harmless.

Highest-value test: cache an expert from model A, replace content with model B at the identical pointer and size, swap FD, re-register, then demand the same expert. Require B’s bytes/output and rejection of every A-generation prefetch slot. Current code should fail.

### Speed prediction

`74.2 MiB × 75 = 5,565 MiB = 5.435 GiB = 5.835 GB` requested per token. At 10.7 GB/s, an all-miss I/O floor is about 0.545 s/token, or 1.83 tok/s. Therefore 1.4–1.8 tok/s is entirely consistent with an almost-I/O-bound cold cache.

At 20% byte hits, NVMe time is still about `0.8 × 5.835 / 10.7 = 0.436 s/token`. Raising hits to 40–50% saves roughly 0.11–0.16 s/token, predicting about a 20–40% decode-rate improvement across the stated baseline range.

Prediction on record: **large, tens-of-percent speedup—roughly 20–40%, potentially more for repeated short requests; not reliably 2×**. For a long static-map decode, the gain may concentrate in early tokens because the old path eventually refills the cache after its last flush.
