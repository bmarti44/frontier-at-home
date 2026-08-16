# DS4 CUDA source audit — 2026-08-15

## Scope and method

This is a read-only source audit of `/home/bmarti44/ds4-source-snapshot-2026-08-15/`. The snapshot records upstream commit `e0ae64873e2844a2e82e26803a0f0541c0319a06` plus modifications to `ds4.c` and `ds4_cuda.cu` (`git-state.txt:1-3`), and its own digest inventory identifies those two files as SHA-256 `ad3087…a877e` and `7e6aaa…f1794` (`SHA256SUMS:1-2`). A read-only digest comparison found the live checkout copies byte-identical to those two snapshot files. The supplied production binary was also checked read-only and matched SHA-256 `eec10ca8aae5…`. No compilation, model process, CUDA process, GPU query, or source-tree modification was performed.

Source references below are to the immutable snapshot unless an absolute path is shown. An exhaustive exact-string search covered the snapshot source, headers, Makefile, and `uncommitted.diff`. Absence findings necessarily have no matching source line; the snapshot identity above makes their search scope explicit.

## 1. R-K.1 and CUDA toolkit

### R-K.1 verdict

`DS4_CUDA_IQ2_DOWN_REFERENCE` does **not** exist anywhere in this source snapshot, its headers, Makefile, or `uncommitted.diff`. Consequently it gates no branches, there is no `1`/`0` behavior to classify, and **R-K.1 is vacuous for this candidate**. The CUDA file does contain IQ2/DP4A implementation code, but not that diagnostic switch; for example the IQ2 helper is unconditional source at `ds4_cuda.cu:16259-16283`.

### Toolkit/build definition

The snapshot Makefile defaults `CUDA_HOME` to `/usr/local/cuda`, derives `NVCC` as `$(CUDA_HOME)/bin/nvcc`, and uses `-O3 -g -lineinfo --use_fast_math`, optional `-arch=$(CUDA_ARCH)`, plus host `-march=native -pthread` (`Makefile:30-37`). The `cuda-spark` target deliberately invokes the build with an empty `CUDA_ARCH`, whereas `cuda-generic` uses `native` (`Makefile:103-120`). CUDA linking also goes through nvcc with CUDA runtime and cuBLAS (`Makefile:40-46`).

On the audited host `/usr/local/cuda` resolves to `/usr/local/cuda-13.0`; its installed version record says CUDA SDK `13.0.3` and nvcc `13.0.88` (`/usr/local/cuda/version.json:2-4`, `/usr/local/cuda/version.json:42-44`). The repository's reproducible build wrapper likewise pins `/usr/local/cuda/bin/nvcc` (`scripts/11_build_glm52_repro.sh:7-10`) and supplies `-O3 -g -lineinfo --use_fast_math -arch=native`, a fixed random seed, retained intermediates, and the same host flags (`scripts/11_build_glm52_repro.sh:120-137`). Therefore the checked build definition/current toolchain is **CUDA 13.0, not CUDA 13.2**. There is no build-provenance sidecar beside this particular production binary, so source inspection alone cannot prove the compiler version embedded in that exact binary; it can prove the Makefile, pinned repository builder, and current `/usr/local/cuda` selection.

## 2. CUDA expert-cache residency

### Two different controls must not be conflated

There are two cache mechanisms in the interface, but only one is a persistent CUDA host arena in this source.

1. The engine parses/plans `--ssd-streaming-cache-experts` as a count or byte budget. Auto-planning uses `DS4_N_LAYER * DS4_N_EXPERT` as its maximum expert population (`ds4.c:53816-53860`); an explicit byte budget first subtracts two routed-prefill layers of headroom (`ds4.c:54123-54144`) and then converts the remaining dynamic bytes to an expert count (`ds4.c:54235-54259`). Startup passes that count and the uniform slab size to the GPU backend (`ds4.c:56258-56269`). **On CUDA those backend budget setters are no-ops**, the budget query returns zero, and the reported current count is only the transient selected-expert staging count (`ds4_cuda.cu:28978-28983`, `ds4_cuda.cu:29098-29112`). Thus the CLI option participates in engine memory accounting/guards, but it does not size the persistent CUDA arena.

2. The actual persistent CUDA cache is enabled and sized by `DS4_CUDA_EXPERT_CACHE_GB`. Its source contract is a fixed host arena allocated once, keyed by `(layer << 32) | expert`, with hit, miss, bypass, and fail-open behavior (`ds4_cuda.cu:23628-23647`). `DS4_CUDA_EXPERT_CACHE_PIN` attempts to register that arena as pinned host memory, and `DS4_CUDA_EXPERT_CACHE_SLRU` selects segmented LRU (`ds4_cuda.cu:24205-24211`, `ds4_cuda.cu:24223-24251`). These environment switches are tested by presence, so even a textual value such as `0` enables PIN or SLRU (`ds4_cuda.cu:24206-24207`, `ds4_cuda.cu:24223-24223`).

### (a) Persistence and SLRU

**Yes: this cache is persistent across tokens and requests.** Initialization is one-shot (`ds4_cuda.cu:24167-24170`), selected-load startup invalidates only the per-batch staging object (`ds4_cuda.cu:24329-24337`), and the persistent arena is flushed only when the model-load generation changes (`ds4_cuda.cu:24100-24107`). The public header also explicitly says prompt-local hotness reset leaves the resident SSD cache warm across sessions (`ds4_gpu.h:167-170`).

With SLRU enabled, a hit moves a probationary entry to a protected MRU list and demotes the protected LRU if the protected segment exceeds its cap (`ds4_cuda.cu:24147-24165`). The protected cap is 60% of slots (`ds4_cuda.cu:24202-24211`). Without SLRU, a hit simply refreshes the ordinary LRU; with SLRU it promotes/refreshes the protected segment (`ds4_cuda.cu:24261-24284`).

### (b) Limits and a 90–95 GiB budget

No index-width or bookkeeping cap is close to a roughly 9,800-slab population. Keys use 64 bits with 32 bits each for layer and expert; slot/list indices and the stored map value use `uint32_t` (`ds4_cuda.cu:23649-23656`, `ds4_cuda.cu:23659-23684`). Initialization caps a single slab below 2^32 bytes and the number of slots at 4,194,304, requires at least 64 slots, and accepts a requested value no greater than 1024 GB (`ds4_cuda.cu:24171-24182`). Per-layer routing metadata uses dynamic vectors sized from `table->n_total_expert`, validates every selected ID, and uses 64-bit byte-overflow checks (`ds4_cuda.cu:24342-24390`). The table itself uses `uint32_t` for layer/expert count and 64-bit offsets/sizes (`ds4_gpu.h:155-165`). None of these structural limits blocks about 9,800 global `(layer, expert)` slabs.

The practical risks are allocation and units. `DS4_CUDA_EXPERT_CACHE_GB` is converted as `gb * 1e9`, so it is **decimal GB**, not GiB (`ds4_cuda.cu:24171-24180`). Values `90` and `95` therefore request only about 83.82 and 88.48 GiB; requesting 90–95 GiB would require approximately `96.64`–`102.01` in this variable. The arena and metadata are each one monolithic `malloc`, followed by a fully reserved `unordered_map`; failure disables the cache (`ds4_cuda.cu:24183-24195`). PIN then calls one monolithic `cudaHostRegister` over the entire arena; registration failure leaves the identical cache active but pageable (`ds4_cuda.cu:24223-24251`). There is no chunking, partial-arena fallback, or retry in this path. On unified-memory GB10, that one-shot host allocation/registration and whole-system headroom—not index width—is the material feasibility risk.

The cache is one slab-size class: `slot_bytes = 2*gate_bytes + down_bytes`; a layer/model with different gate/down sizes bypasses it (`ds4_cuda.cu:23639-23646`, `ds4_cuda.cu:24175-24179`, `ds4_cuda.cu:24461-24463`). Therefore “all experts fit” must mean all relevant experts in the uniform slab class, plus adequate slots after decimal-GB conversion.

### (c) Fully resident behavior and 100%-hit decode cost

Once all uniform-class experts are warmed and offsets/model generation remain unchanged, the actual storage miss path disappears: lookup validates key and model offsets (`ds4_cuda.cu:24261-24284`), while disk/device streaming occurs only after a miss (`ds4_cuda.cu:24518-24523`, `ds4_cuda.cu:24585-24613`). Prefetch also skips an already resident key (`ds4_cuda.cu:23979-23991`). Different slab classes and a model-generation flush remain explicit exceptions (`ds4_cuda.cu:23643-23646`, `ds4_cuda.cu:24100-24107`).

**A 100% cache hit is not a zero-cost/resident-device path.** Every decode layer still synchronously copies selected expert IDs device-to-host (`ds4_cuda.cu:28408-28432`), allocates/builds host vectors and de-duplicates/validates IDs (`ds4_cuda.cu:24354-24387`), updates an unconditional SHA-256 access-stream record and byte counters for every expert (`ds4_cuda.cu:24464-24485`), performs hash-map lookup plus LRU/SLRU list mutation (`ds4_cuda.cu:24261-24284`), and issues three host-to-device copies—gate, up, and down—from the arena into the transient selected-expert device buffers (`ds4_cuda.cu:24489-24516`). It finally copies the remapped selected IDs host-to-device (`ds4_cuda.cu:24762-24770`). The source comment reports the historical pageable hit-copy cost as about 92 ms/token and motivates PIN for asynchronous DMA (`ds4_cuda.cu:24223-24227`); that number is a source comment, not a measurement made by this audit.

If `DS4_GLM_PREFETCH` is enabled, decode additionally runs a next-layer router prediction and closes GPU commands to read its IDs (`ds4.c:40058-40093`), although the host prefetch queue then skips resident entries (`ds4_cuda.cu:23983-23991`). With prefetch disabled, that extra predictor cost is absent.

## 3. Cheapest saliency hook

The routed decode control point is `glm_graph_encode_sparse_ffn_one`: it materializes router logits, selected expert IDs, normalized router weights, and probabilities in `g->router_*` tensors (`ds4.c:40013-40057`), then passes `g->router_selected` and `g->router_weights` to `glm_graph_routed_moe_one_dispatch` (`ds4.c:40462-40479`). The dispatch forwards those tensors to `ds4_gpu_glm_routed_moe_one_tensor` for the GLM path (`ds4.c:39534-39615`), whose CUDA implementation delegates to the batch implementation with one token (`ds4_cuda.cu:27960-27996`). Thus gate weights and expert IDs are already materialized and reusable.

For the normal one-token fast path, the cheapest exact hook is in/adjacent to `glm_routed_moe_down_warp_kernel`, selected by `ds4_gpu_glm_routed_moe_batch_tensor` when `DS4_GLM_MOE_SCALAR` is absent (`ds4_cuda.cu:27933-27953`). This down kernel already reads the expert ID and gate weight and computes down-projection dot products (`ds4_cuda.cu:27269-27281`). However, it fuses all selected experts into one weighted accumulator before the warp reduction/output store (`ds4_cuda.cu:27271-27286`), so **the per-expert output vector and its norm are not materialized anywhere reusable**. An exact `[layer, expert] += gate_weight * ||expert_output||_2` hook needs a default-off diagnostic specialization that preserves/reduces each slot's `s^2` across output rows before applying the already available weight. Hooking here reuses the expensive down-projection values before they disappear; computing after `out` would be impossible because experts have already been summed.

The scalar diagnostic kernel is the simplest reference implementation: `glm_routed_moe_batch_q2K_down_kernel` has a complete per-expert scalar output coordinate `s` and its weight immediately before `acc += w*s` (`ds4_cuda.cu:27046-27077`), but it runs only with `DS4_GLM_MOE_SCALAR` (`ds4_cuda.cu:27906-27913`, `ds4_cuda.cu:27933-27941`). The expert-major prefill kernel likewise has per-expert `s` and `weights[pair]` before its atomic output add (`ds4_cuda.cu:27461-27499`), but that path requires at least 16 tokens and `DS4_GLM_MOE_EXPERT_MAJOR` (`ds4_cuda.cu:27722-27725`, `ds4_cuda.cu:27865-27887`), so it is not the normal decode hook.

The existing imatrix collector is not a reusable saliency-output source. It documents/materializes FFN-normalized input, selected IDs, and routed SwiGLU mid activations (`ds4.c:32384-32397`), and reads only those tensors (`ds4.c:32461-32470`); it accumulates squared mid inputs to the down matrices, not norms of each expert's down-projection output (`ds4.c:32475-32500`).

## 4. Pruned-GGUF loading

**No: this source will not load the described 256→128 expert GGUF without source changes.** GLM-5.2's compiled shape fixes `n_expert = 256` and `n_expert_used = 8` (`ds4.c:611-629`). GGUF metadata loading reads `glm-dsa.expert_count` and hard-validates it against that compiled value (`ds4.c:5650-5663`, `ds4.c:5674-5689`). Therefore metadata changed to 128 is rejected.

Keeping metadata at 256 does not help: GLM layout validation requires the router matrix's expert dimension, optional router bias, and each routed expert tensor's third dimension to equal `DS4_N_EXPERT` (`ds4.c:5000-5005`). An earlier GLM layout variant has the same fixed expectations (`ds4.c:4883-4894`). Thus sliced router rows and 128-expert `ffn_*_exps` tensors fail layout validation before routing.

Several routing/cache pieces are already count-aware: CUDA router selection accepts `n_expert` through 384 and uses its fast kernel for counts through 256 (`ds4_cuda.cu:28120-28157`); streaming validates IDs against `table->n_total_expert` and computes slab offsets as `expert * per_expert_bytes` (`ds4_cuda.cu:24366-24381`, `ds4_cuda.cu:24464-24475`); and range validation uses 64-bit `n_total_expert * slab_bytes` arithmetic (`ds4_cuda.cu:23589-23611`). These pieces would handle 128 after the model shape is made dynamic.

The specialized GLM Q2_K MoE kernel still contains additional literal-256 assumptions: it discards `n_total_expert` and resolves each gate/up/down tensor as exactly `256 * expert_bytes` (`ds4_cuda.cu:27629-27661`). Its expert-major/tile scratch allocates counts and lists for 256 and invokes map/tile builders with 256 (`ds4_cuda.cu:27722-27764`). For a bypassed 128-expert model, the scratch/map sizing would be wastefully oversized and selected IDs below 128 would keep the kernels from indexing experts 128–255, but the resolved weight span would no longer describe the actual tensor and could extend into following mapped data. Expert IDs themselves are 32-bit and are range-checked, so ID width is not the blocker (`ds4_cuda.cu:28125-28135`, `ds4_cuda.cu:28418-28432`). Required work is at least dynamic GLM shape/metadata/layout support plus parameterizing the literal-256 tensor-span and expert-map constants.

## 5. `cache_f16` and CUDA stubs

### Constant and implemented F16 consumers

`DS4_GPU_ATTN_COMP_CACHE_F16` is `1` only on Apple and `0` everywhere else; the GLM compact-cache alias is the same constant (`ds4.c:14812-14827`). The graph converts that constant to 2- versus 4-byte cache elements and a boolean (`ds4.c:34408-34414`). Therefore this CUDA build uses **F32 compact KV/indexer caches**.

Much of the CUDA compact/indexed implementation does honor a hypothetical `cache_f16=true`: compact KV stores branch to `__half` (`ds4_cuda.cu:28178-28259`); indexer-K stores size and write F16 correctly (`ds4_cuda.cu:28347-28405`); indexer scoring instantiates either `__half` or `float` consumers (`ds4_cuda.cu:26431-26475`); indexed prefill's causal/selected attention-LORA functions instantiate F16 consumers (`ds4_cuda.cu:25254-25309`, `ds4_cuda.cu:25312-25376`); indexed decode sizes the cache at two bytes and launches F16 weight/LORA kernels (`ds4_cuda.cu:25855-25910`, `ds4_cuda.cu:25992-26075`); and fused Q/KV normalization writes half cache rows when requested (`ds4_cuda.cu:26807-26818`, `ds4_cuda.cu:26821-26875`).

### End-to-end verdict

**No, CUDA does not safely honor `cache_f16=true` end-to-end.** The split-group8 predicate becomes true for more than 512 selected rows when the fixed GLM dimensions match and compact cache is F16 (`ds4.c:37837-37853`). GLM-5.2 fixes indexer top-k at 2048 (`ds4.c:611-635`), and normal indexed decode calls `ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor` when that predicate is true (`ds4.c:46092-46118`). That CUDA function is a stub returning zero (`ds4_cuda.cu:25410-25442`). With the current F32 constant, the predicate is false and decode uses the implemented `ds4_gpu_glm_attention_indexed_decode_typed_tensor` instead (`ds4.c:46133-46158`). Simply flipping the constant to F16 would therefore break ordinary indexed decode rather than transparently use all implemented F16 kernels.

### All 12 `CUDA stub called` functions and reachability

| # | CUDA stub | Source | Configuration/path that reaches it in this tree |
|---:|---|---|---|
| 1 | `ds4_gpu_glm_attention_flash_staged_tensor` | `ds4_cuda.cu:25046-25060` | Expanded/full-KV batch prefill, flash enabled, first whole-cache chunk (`ds4.c:43524-43527`, `ds4.c:43579-43592`). Expanded KV is currently hard-disabled (`ds4.c:34416-34418`), so this is latent in the normal graph. |
| 2 | `ds4_gpu_glm_attention_flash_tensor` | `ds4_cuda.cu:25063-25077` | Expanded/full-KV batch prefill with flash and non-staged KV (`ds4.c:43593-43605`), or the layer-0 attention diagnostic when flash is enabled (`ds4.c:52383-52395`). |
| 3 | `ds4_gpu_glm_attention_full_tensor` | `ds4_cuda.cu:25080-25094` | Expanded/full-KV batch with flash disabled or fewer than 24 tokens (`ds4.c:40793-40803`, `ds4.c:43607-43619`), non-indexed decode (`ds4.c:46205-46216`), or the layer-0 diagnostic with flash disabled (`ds4.c:52397-52408`). Normal GLM decode is indexed because expanded KV is disabled and compact cache is present (`ds4.c:40822-40829`). |
| 4 | `ds4_gpu_glm_attention_indexed_batch_typed_tensor` | `ds4_cuda.cu:25379-25407` | Indexed prefill only when the split value-projection path is unavailable (`ds4.c:44691-44724`, `ds4.c:44806-44833`). A successfully initialized indexed graph allocates `batch_attn_lora`, making split projection the normal path (`ds4.c:39150-39157`, `ds4.c:44060-44064`). |
| 5 | `ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor` | `ds4_cuda.cu:25410-25442` | Indexed decode with F16 compact cache and >512 selected rows under GLM's fixed dimensions (`ds4.c:37841-37853`, `ds4.c:46098-46118`). This is the direct blocker to changing the CUDA cache constant to F16. |
| 6 | `ds4_gpu_glm_build_kv_cache_flash_tensor` | `ds4_cuda.cu:26127-26151` | Expanded/full-KV staged-flash prefill (`ds4.c:43524-43551`); latent because expanded KV is hard-disabled (`ds4.c:34416-34418`). |
| 7 | `ds4_gpu_glm_build_kv_cache_tensor` | `ds4_cuda.cu:26154-26178` | Expanded/full-KV non-staged prefill (`ds4.c:43552-43574`), non-indexed decode (`ds4.c:46182-46203`), or the layer-0 diagnostic (`ds4.c:52360-52381`). |
| 8 | `ds4_gpu_glm_k_b_project_typed_tensor` | `ds4_cuda.cu:26510-26522` | Expanded/full-KV batch or non-indexed decode before stubs 6/7 (`ds4.c:43506-43515`, `ds4.c:46161-46170`). The compatibility wrapper also delegates to it (`ds4_cuda.cu:29193-29198`). |
| 9 | `ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor` | `ds4_cuda.cu:26972-27001` | The `direct_scalar_q4` batch-dispatch arm (`ds4.c:39742-39770`). Indexed prefill passes that arm when grouped MoE is disabled; grouped MoE is exactly `!g->quality`, so CUDA indexed prefill in quality mode reaches this stub (`ds4.c:41166-41170`, `ds4.c:41593-41615`). |
| 10 | `ds4_gpu_matmul_quant_rows_scalar_tensor` | `ds4_cuda.cu:28600-28611` | Scalar-row Q8 fallback helper first attempts it and then falls back to per-row matmul on zero (`ds4.c:40971-41020`). Current indexed-prefill batch feature selectors are hardcoded true (`ds4.c:40855-40860`, `ds4.c:40871-40896`), so it is principally a scalar/fallback path and is non-fatal when reached. |
| 11 | `ds4_gpu_tp_big_gate_encode` | `ds4_cuda.cu:29033-29038` | Two-rank CUDA tensor-parallel batch attention/FFN exchange, including the FFN combine (`ds4.c:39657-39680`). Single-GPU CUDA does not enter this path. |
| 12 | `ds4_gpu_tp_gate_encode` | `ds4_cuda.cu:29041-29043` | Two-rank CUDA tensor-parallel per-layer attention or routed-FFN exchange (`ds4.c:22597-22615`, `ds4.c:40481-40487`). Single-GPU CUDA does not enter this path. |

The important distinction is that current F32 compact/indexed single-GPU production avoids the fatal attention stubs, while F16 would activate stub #5. Several other stubs are dormant behind currently hard-disabled full-KV paths, diagnostics, quality/scalar fallbacks, or two-rank tensor parallelism; their presence still means the CUDA compatibility surface is incomplete (`ds4_cuda.cu:29056-29058`).

## 6. NVIDIA Spark MoE uplift

**Unknown.** This tree contains no source attribution, commit reference, or identifiable port naming the 2026 NVIDIA `llama.cpp` GB10 MoE work. The only `llama.cpp` mention is the legacy imatrix output format (`ds4.c:32395-32397`); the Makefile's only GB10-specific surface is the `cuda-spark` label and an `sm_120` example (`Makefile:103-107`, `Makefile:122-128`). Source searches found no Blackwell-specific `cp.async`, TMA, WGMMA, or cuBLASLt implementation.

There are generic techniques that may overlap conceptually: CUDA DP4A dot products (`ds4_cuda.cu:4281-4291`), WMMA indexer score kernels (`ds4_cuda.cu:26308-26387`), expert-major/tiled routed-MoE paths (`ds4_cuda.cu:27289-27505`, `ds4_cuda.cu:27722-27887`), and the persistent pinned host expert cache described above (`ds4_cuda.cu:23628-23647`, `ds4_cuda.cu:24223-24251`). None establishes provenance from, or technical equivalence to, NVIDIA's 2026 llama.cpp uplift. Without a source-to-source comparison against a pinned NVIDIA/llama.cpp revision, incorporation/equivalence must remain **unknown**, not inferred.

## Executive summary

1. `DS4_CUDA_IQ2_DOWN_REFERENCE` is absent; R-K.1 is vacuous and gates no source branch (`git-state.txt:1-3`; exhaustive snapshot search).
2. The Makefile/current pinned toolchain selects CUDA 13.0 (SDK 13.0.3, nvcc 13.0.88), not 13.2 (`Makefile:30-37`; `/usr/local/cuda/version.json:2-4,42-44`).
3. `--ssd-streaming-cache-experts` does not size a CUDA resident arena here; its CUDA budget setters/query are no-ops (`ds4_cuda.cu:28978-28983,29098-29112`).
4. `DS4_CUDA_EXPERT_CACHE_GB` creates the real persistent cross-token/request host cache, with optional PIN and SLRU (`ds4_cuda.cu:23628-23647,24167-24251`).
5. About 9,800 slabs fit the index structures, but decimal-GB sizing and one giant malloc/HostRegister are the 90–95 GiB risks (`ds4_cuda.cu:24171-24195,24223-24251`).
6. Full residency removes disk misses, not the hit hot path: ID readback, hashing/LRU, three host→device slab copies, and remap copy remain (`ds4_cuda.cu:24261-24284,24464-24516,24762-24770,28408-28432`).
7. The cheapest exact saliency hook is the routed-MoE down kernel; router weights exist, but per-expert output norms do not (`ds4.c:40040-40057`; `ds4_cuda.cu:27269-27286`).
8. A 128-expert pruned GGUF is rejected by fixed metadata/layout checks; CUDA also retains literal-256 spans/maps that need parameterization (`ds4.c:626-627,5688-5689,5001-5005`; `ds4_cuda.cu:27653-27661,27729-27764`).
9. CUDA compact KV is currently F32; flipping it to F16 routes normal 2048-row indexed decode into stub #5 (`ds4.c:14821-14827,37841-37853,46098-46118`; `ds4_cuda.cu:25410-25442`).
10. Incorporation/equivalence of NVIDIA's 2026 GB10 llama.cpp MoE uplift is unknown; generic DP4A/WMMA/tiled-MoE code is not provenance evidence (`ds4_cuda.cu:4281-4291,26308-26387,27289-27505`).
