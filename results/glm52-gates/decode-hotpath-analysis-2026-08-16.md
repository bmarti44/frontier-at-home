# GLM-5.2 CUDA decode hot-path analysis at context ~=30K

Date: 2026-08-16  
Scope: read-only source analysis; no build, model load, or benchmark was run.  
Source of truth:

- `/home/bmarti44/.cache/glm52-dynexp-patched/ds4.c`, SHA-256 `61ae99d19866…`
- `/home/bmarti44/.cache/glm52-dynexp-patched/ds4_cuda.cu`, SHA-256 `f51f55ca8690…`

The patched tree was supplied as the tree currently serving. The measured
2.33 tok/s, 1.08 tok/s cache-off result, 81.95 GiB/9,042-slot cache, warm
residency, and blk.78 behavior are inputs to this analysis, not measurements
reproduced here. All component times below that are not already present in a
committed timing artifact are explicitly estimates.

## Bottom line

The large cache is working, but a cache **hit is not a resident-weight compute
path**. It removes the file read and then still does all of the following for
each selected normal-layer expert:

1. CPU hash-table lookup and SLRU list mutation;
2. unconditional CPU SHA-256 access-stream accounting;
3. three host-arena-to-device-staging copies;
4. execution of the same bandwidth-bound expert kernels over the transient
   staging buffers.

At one decode token, that is 75 routed layers x 8 experts = 600 lookups and
hits, 1,800 `cudaMemcpyAsync` calls, 9,600 access-stream bytes hashed, and
5.839 GB (5.438 GiB) copied into transient buffers. The copies use the legacy
default stream and are drained before that layer's MoE dispatch; they do not
overlap the same layer's expert compute.

The cache-off/cache-on delta is consistent with precisely this diagnosis:

```
cache off: 1 / 1.08 = 925.93 ms/token
cache on : 1 / 2.33 = 429.18 ms/token
delta    :            496.75 ms/token
```

Thus hits remove about 497 ms/token of storage/fault service on this fixture.
Increasing residency after that does not remove the remaining 429 ms of GPU
weight reads, copies, synchronization dependencies, attention, dense/shared
GEMVs, or launch/control overhead. Pruning the router from 256 to 128 experts
also does not reduce the eight experts actually evaluated per routed layer.

The source-derived roofline is sobering. Normal selected expert bytes are
5.839 GB/token. The main model's Q8-equivalent non-routed projection/shared/
output bytes are approximately 18.62 GB/token from the declared shapes. Adding
the warm blk.78 selected Q2_K experts gives about 24.56 GB of unique weight
bytes/token. At GB10's nominal 273 GB/s, weight traffic alone has an ideal
floor near 90.0 ms/token, before indexed attention, cache control, launches,
or MTP bookkeeping. Therefore 10 tok/s (100 ms/token) is close to an
all-kernel roofline; no collection of small host patches defensibly gets there.

## Reconstructed 429 ms/token critical-path budget

This is a constrained estimate, not a profiler trace. The table deliberately
sums to the observed 429 ms/token. The 107 ms routed-MoE row is anchored by
`loadprof-2026-07-25.json`; the other rows allocate the previously unresolved
attention/dense and `other` buckets using source byte/operation counts and the
completed-time W3 result. Ranges express analysis uncertainty, not confidence
intervals.

| Component | Central ms/token | Plausible range | Basis |
| --- | ---: | ---: | --- |
| Normal routed-expert IQ2 GEMVs, layers 3..77 | 107 | 90-125 | Historical completed kernel bucket was 107.1 ms: 70.6 gate/up + 35.5 down. Eight active experts and their 5.84 GB do not change when the total router width is pruned. |
| Arena-to-transient expert copies and their drain | 43 | 30-65 | 5.839 GB copied, but UMA copy traffic is approximately one arena read plus one staging write = 11.679 GB; `11.679 / 273 = 42.8 ms` is the nominal bandwidth floor. W3 direct-slot serving saved 15.87 ms/token net because direct mapped reads themselves were slower and other work overlaps at system level. |
| Selected-ID D2H, remap H2D, and dependency latency | 10 | 5-25 | 75 blocking 32-byte D2H reads and 75 blocking 32-byte H2D writes. Payload is trivial; stream drain/host round-trip is the cost. |
| CPU vector construction, 600 hash lookups, SLRU mutation, SHA | 4 | 1-8 | 75 sets of temporary vectors, 600 `unordered_map::find` operations/list edits, 1,800 SHA update calls/150 SHA compression blocks. |
| blk.78's eight warm Q2_K experts, direct mapped views | 2 | 1-5 warm | 94.5 MiB selected Q2_K bytes; 0.36 ms nominal DRAM floor, about 1.8 ms at the historical 55 GB/s expert-kernel rate. Excludes cold faults. |
| DSA indexer score/top-2048 plus compact indexed attention | 70 | 45-95 | 21 full indexers, 78 selected-attention layers, about 0.691 GB unique cache data, but about 11.1 billion score/weighted-sum FMAs and substantial per-head rereads. Detailed arithmetic below. |
| Main non-routed attention projections, shared experts, three dense FFNs, output head | 165 | 135-205 | About 18.62 GB Q8-equivalent unique weights/token from declared dimensions; 68.2 ms nominal DRAM floor, with current small-GEMV efficiency substantially below peak. |
| Remaining blk.78/MTP attention, shared expert, projections, head | 10 | 5-20 | One extra full nextn block and another shared output-head evaluation when MTP is invoked. Source path is separate from the 78 normal layers. |
| Routers, norms, residuals, activation quantization, kernel launches, logits read, uncategorized | 18 | 8-30 | Residual needed to close the measured wall; includes thousands of hot `getenv` calls and final synchronization/readback. |
| **Total** | **429** | **not additively bounded** | Matches `1 / 2.33 = 429.18 ms/token`. |

Do not add the old 44.1 ms `loader_other_ms` to this table. Its profiler timed
asynchronous enqueue sites, so completed copy work migrated into later blocking
calls and the `other` bucket. The audit in `audit-2026-07-26/F08-pinned-arena.md`
already showed why 4.62 GB completing in the reported 2.6 ms would imply an
impossible 1.78 TB/s. The 43 ms copy row above is the source/roofline attribution
of that deferred work, not an additional charge on top of the old 44.1 ms.

## 1. Exact normal expert-hit path

### Count and slab arithmetic

The model declares 79 blocks, one nextn block, three leading dense blocks, and
top-8 routing (`ds4.c:611-640`). The executable target pass has 78 normal
layers (`ds4.c:34484-34489`); routed normal layers are therefore 3 through 77,
or 75 layers. The cache log's slab size is exactly consistent with three
IQ2_XXS matrices:

```
IQ2_XXS block       = 66 bytes / 256 weights       (ds4.c:794-805)
one 6144x2048 matrix= 6144 * 2048 * 66 / 256
                    = 3,244,032 bytes = 3.09375 MiB
one expert slab     = 3 * 3.09375 = 9.28125 MiB
cache arena         = 9,042 * 9.28125 / 1,024
                    = 81.9542 GiB
uniform population  = 75 * 128 = 9,600 experts
slot coverage       = 9,042 / 9,600 = 94.1875%
per-token hit bytes = 75 * 8 * 9.28125 MiB
                    = 5,839,257,600 bytes
                    = 5.839 GB = 5.438 GiB
```

The persistent cache is a fixed `malloc` arena plus metadata and a reserved
`unordered_map` (`ds4_cuda.cu:24167-24222`). The pin option registers it with
`cudaHostRegisterPortable`, not `cudaHostRegisterMapped`
(`ds4_cuda.cu:24223-24240`).

For every selected expert, the source performs:

- a map lookup keyed by `(layer << 32) | expert`, verifies model offsets, then
  unlinks and promotes/pushes the SLRU node (`ds4_cuda.cu:24261-24284`);
- three unconditional SHA update calls over layer u32, expert u32, and slab
  bytes u64 (`ds4_cuda.cu:24474-24485`);
- three `cudaMemcpyAsync(..., cudaStreamLegacy)` calls for gate, up, and down
  (`ds4_cuda.cu:24487-24516`).

The access stream is 16 bytes/selected expert, hence `600 * 16 = 9,600`
bytes/token. Because the code calls `ec_sha256_update` three times per record,
that is 1,800 update calls. The stream is continuous and processes one 64-byte
compression block per four records, or 150 compression blocks/token. Digest
finalization is not per token. There is no production/evidence gate around the
updates (`ds4_cuda.cu:23686-23761,24476-24485`), so they are unconditional.

The host also constructs and destroys `expert_to_slot`, `compact_ids`, and
`slot_ids` vectors for every layer (`ds4_cuda.cu:24329-24382`) and reserves a
miss vector even on a hit-only layer (`ds4_cuda.cu:24444-24451`). These are
small compared with 5.8 GB of traffic, but they are unnecessary decode-path
allocator/control work.

### Stream ordering and overlap

All three expert copies explicitly target `cudaStreamLegacy`
(`ds4_cuda.cu:24496-24508`). The routed kernels are launched with no stream
argument and therefore also use stream 0; the normal decode graph likewise
uses the default CUDA command surface. After all expert copies, the code uses
a **blocking** `cudaMemcpy` to upload the remapped IDs
(`ds4_cuda.cu:24762-24770`). The MoE dispatch occurs only after that function
returns (`ds4.c:40377-40444,40505 onward`).

Consequences:

- copies for different experts can enqueue while the CPU continues its lookup
  loop;
- they cannot overlap the same layer's routed-expert compute, because both use
  the legacy/default stream and the blocking remap occurs before dispatch;
- the next layer cannot select IDs until the current layer has computed its
  hidden state/router, so there is no cross-layer GPU overlap in this path;
- there is no explicit per-layer `cudaStreamSynchronize` in the cache-hit
  function, but the blocking D2H/H2D operations create the dependency and drain
  points that matter.

## 2. Selected-ID D2H and remapped-ID H2D

For the default CUDA path, `glm_graph_use_streaming_selected_async_load()` is
false unless the oddly named
`DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD` variable is set
(`ds4.c:39864-39877`). Therefore the normal path calls
`ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor()` once per routed
layer (`ds4.c:40389-40438`). That function:

1. allocates an eight-element host vector;
2. issues blocking `cudaMemcpyDeviceToHost` for the selected IDs;
3. performs CPU compaction/cache lookup;
4. issues blocking `cudaMemcpyHostToDevice` for the eight remapped IDs.

The D2H source is `ds4_cuda.cu:28426-28450`; the H2D source is
`ds4_cuda.cu:24762-24770`. This is 75 D2H and 75 H2D calls/token. Only 32 bytes
move in each direction per layer, so bandwidth is irrelevant. The D2H must wait
for router selection and all earlier stream-0 work; the H2D comes after the
legacy-stream expert copies and before the MoE kernels. The estimated intrinsic
host/dependency contribution is 5-25 ms/token. Time spent waiting for actual
GPU kernels/copies is attributed to those components rather than double-counted
here.

Changing only these calls to asynchronous copies does not remove the dependency:
the CPU cannot look up selected experts before the IDs arrive, and the kernel
cannot consume remapped IDs before CPU lookup finishes. A useful fix must move
the cache directory/remap decision to the GPU or decouple it by prediction;
merely changing API spelling changes the scope of the barrier, not the data
dependency.

## 3. blk.78 bypass

The MTP function explicitly selects block
`DS4_N_LAYER - DS4_N_NEXTN_PREDICT`, which is block 78
(`ds4.c:42382-42408`). Its streaming selected-expert load is compiled only for
ROCm (`ds4.c:42673-42689`). CUDA falls through to the routed-MoE dispatch with
`force_resident=false` (`ds4.c:42690-42711`), so this block does not enter the
persistent selected-expert cache.

The CUDA model mapping itself is registered with
`cudaHostRegisterMapped | cudaHostRegisterReadOnly` and resolved with
`cudaHostGetDevicePointer` (`ds4_cuda.cu:3284-3293` and
`ds4_cuda.cu:3361-3371`). The weight resolver returns registered mapped views
when available (`ds4_cuda.cu:640-665`). Thus steady-state blk.78 expert
execution does **not** issue `read()`/`pread()` calls. `pread` exists in the
selected-expert miss/staging machinery, not in this mapped-view kernel path.

For three Q2_K matrices, using the declared 84-byte/256-weight block
(`ds4.c:759-764,800`):

```
one blk.78 expert = 3 * 6144 * 2048 * 84 / 256
                  = 11.8125 MiB
eight experts     = 94.5 MiB = 99.09 MB/token
273 GB/s floor    = 0.363 ms/token
55 GB/s observed expert-kernel rate ~= 1.80 ms/token
```

The warm estimate is therefore about 1-5 ms/token for the selected experts.
The whole MTP block costs more because it also runs attention, shared expert,
projections, another output head, a device synchronization, and logits readback
(`ds4.c:42348-42357,42742-42766`).

Cold behavior is different. The registered pointer is backed by the model's
file `mmap`; a nonresident page may be serviced through Linux page cache and
GPU ATS/HMM fault handling. There is no explicit read at the call site, so
latency depends on page residency/fault coalescing. A crude sequential storage
lower bound for 99 MB at 7-12.5 GB/s is 8-14 ms, but many GPU-visible page
faults can make a truly cold first touch much slower. After two prefills and
repeated per-token MTP use, recurring steady-state major faults are not
expected. This source-only audit cannot verify `mincore` residency or fault
counters.

## 4. DSA indexed attention at depth 30K

The normal target pass has 78 layers. Full indexer selection runs on layers
0, 1, 2 and 6, 10, ..., 74: 3 + 18 = 21 full selections/token
(`ds4.c:34479-34482`). Each full selection runs indexer Q, indexer weights,
score over all visible rows, and exact top-2048
(`ds4.c:46048-46126`). The selected set is reused on intervening layers, but
all 78 layers still execute qk-low and selected attention
(`ds4.c:46128-46216`).

The current compact cache is F32 (`ds4.c:34466-34472`). Source byte counts at
30,000 visible rows are:

```
indexer K useful bytes = 21 * 30,000 * 128 * 4
                       = 322.56 MB/token
selected compact KV    = 78 * 2,048 * (512 + 64) * 4
                       = 368.05 MB/token (351 MiB)
unique total           = 690.61 MB/token
273 GB/s ideal floor   = 2.53 ms/token
```

The 2.53 ms figure is not a runtime estimate because the attention kernels
reread selected data per head and do substantial work. The staged decode path
is selected at top-2048 and launches weights, lora accumulation, and value
projection kernels per layer (`ds4_cuda.cu:25965-26096`). With 64 attention
heads, its score and weighted-value loops perform approximately:

```
78 * 64 * 2,048 * ((512 + 64) + 512)
= 11.12 billion multiply-add terms/token
```

plus softmax, qk-low, value projection, indexer projections, and 234 staged
attention launches. Reuse in cache/L2 means this is not 64 times the unique
DRAM byte count, but it explains why the 2.53 ms useful-byte floor is not
attainable.

The current exact top-2048 path divides scores into 4,096-row chunks and tree
merges them (`ds4_cuda.cu:11881-11957`). The existing W4 microgate measured the
old path at about 4.92 ms for eight queries over 1,048,576 rows and the exact
candidate at about 1.61 ms. Linear scaling is unsafe at 30K because launches
dominate, but it bounds top-k itself to roughly 1-3 ms/token across 21 layers,
not tens of milliseconds. Indexer scoring is estimated at roughly 4-12 ms;
the 78 selected-attention layers are estimated at roughly 40-80 ms. A central
DSA total of 70 ms/token, range 45-95 ms, is defensible but remains an estimate
until the already-present decode-stage boundaries are measured with CUDA
events rather than host enqueue timestamps.

### Non-routed weight-byte cross-check

The 165 ms non-routed row uses a Q8-equivalent byte count, not a claim that the
source alone proves every tensor in the live artifact is Q8_0. The loader
accepts Q8_0, Q4_K, or Q4_0 for GLM dense tensors (`ds4.c:4269-4296`); exact
artifact types would require reading its GGUF metadata. Q8_0 is 34 bytes per
32 weights (`ds4.c:2008-2017`), and the decode functions select typed/Q8 paths.
Using the declared shapes at `ds4.c:611-642` and tensor dimensions at
`ds4.c:4901-4939` gives the following reproducible upper-side traffic model:

```
per normal attention layer, in millions of Q8 bytes:
  q_a       6144*2048              *34/32 =  13.369 MB
  q_b       2048*(64*256)          *34/32 =  35.652 MB
  kv_a      6144*576               *34/32 =   3.760 MB
  k_b       (256-64)*512*64        *34/32 =   6.685 MB
  v_b       512*256*64             *34/32 =   8.913 MB
  attn_out  (64*256)*6144          *34/32 = 106.955 MB
  subtotal/layer                            = 175.333 MB
  78 layers                                =  13.676 GB

21 full-indexer q/k projections            =   0.205 GB
75 shared experts, three 6144x2048 matrices=   3.008 GB
3 dense FFNs, three 6144x12288 matrices    =   0.722 GB
one 6144x154880 output head                =   1.011 GB
total Q8-equivalent unique weight bytes    =  18.622 GB/token
```

The small F32 indexer projection, norms, biases, and routers add traffic but do
not change this estimate materially. If GGUF metadata shows Q4 for a component,
that component's byte term should be reduced; the table's wide 135-205 ms range
is intended to cover that unresolved artifact detail and small-GEMV efficiency.

## 5. What remains of the old 44.1 ms `other` bucket

The earlier R-K.2 plan expected roughly 80-90 device-wide barriers/token. That
count is **not present on this CUDA source path**:

- CUDA enables the static decode map by default (`ds4.c:17422-17432`).
- `streaming_decode_sync_each_layer` is true only when streaming and the static
  map is disabled (`ds4.c:45757-45767`).
- Consequently the per-layer `ds4_gpu_end_commands()` at
  `ds4.c:46363-46366` is not called on the default CUDA static-map path.
- The four CUDA compatibility functions do call `cudaDeviceSynchronize`
  (`ds4_cuda.cu:28983-28985,29031-29048,29068-29081`), but they belong to the
  optional async-selected-load path. That path is off by default on CUDA, so
  their normal decode-path count is zero.
- The target forward pass performs one output `end_commands()`
  (`ds4.c:46369-46380`). With `DS4_CUDA_END_STREAM_SYNC` absent, that is one
  `cudaDeviceSynchronize`; with the flag present it is one stream-0 sync
  (`ds4_cuda.cu:3216-3224`). An MTP step adds one more explicit
  `end_commands()` and a blocking logits read (`ds4.c:42759-42766`), and a
  tier handoff may add another (`ds4.c:42410-42427`).

So the source count for a normal target token is approximately **one explicit
device-wide synchronization**, not 80-90, plus **150 blocking selected-ID/remap
copies** that serialize the dependency chain without being written as
`cudaDeviceSynchronize`. If blk.78 MTP runs, add one (possibly two with a tier
handoff) explicit syncs.

This count assumes the source-default CUDA static map and the repository's
recorded absence of `DS4_CUDA_END_STREAM_SYNC`; the sandbox did not expose the
live serving process environment. If a live profile explicitly sets a Metal/
ROCm static-map-disable alias, the per-layer `end_commands()` path must be
recounted from that resolved environment. Source alone proves that the earlier
80-90 count is not the CUDA default, not that an unobserved operator override
is impossible.

This explains the historical 44.1 ms bucket better than the stale sync count:
it contains completion of legacy-stream copies that the hit-site profiler only
timed as enqueue, repeated tiny blocking transfers, host allocations/cache
control, and launch/runtime overhead. A pure event replacement for the four
dormant stubs has negligible expected value on the stated serving path.

Another real but small item is dynamic configuration lookup. The generic
`routed_moe_launch` contains 51 `getenv()` calls in one invocation
(`ds4_cuda.cu:21308-21569`). The 75 normal routed layers therefore execute
3,825 such calls/token; including one generic blk.78 invocation makes 3,876.
Hoisting these flags is appropriate production hygiene, but not a path from
429 to 100 ms.

## 6. Can the GPU read the pinned arena directly?

Yes in principle, and the repository has already demonstrated the mechanism,
but the current source is not wired for it.

GB10 is coherent unified memory, and CUDA kernels can consume mapped pinned
host allocations. Current model weights already use that mechanism. The expert
arena does not: it is registered only `cudaHostRegisterPortable`
(`ds4_cuda.cu:24223-24240`), while model mappings use
`cudaHostRegisterMapped | cudaHostRegisterReadOnly` followed by
`cudaHostGetDevicePointer` (`ds4_cuda.cu:3284-3293`). The first mechanical
requirement is therefore mapped registration and a retained device alias.

Alignment is not the principal blocker. The arena registration is page-wide,
the 9,732,096-byte uniform slot is 256-byte aligned in size, and each tensor
starts at a quant-block-compatible offset. The blockers are representation and
lifetime:

- current kernels receive three compact contiguous bases (`gate_w`, `up_w`,
  `down_w`) and small remapped IDs; a cache slot is one interleaved slab, and
  selected experts may occupy arbitrary slots;
- kernels need either per-expert pointer tables or `(arena_base, slot_id,
  tensor_offset)` indirection;
- selected IDs are currently learned on the CPU only after a synchronous D2H
  read;
- SLRU may evict/reuse a slot immediately after lookup. Copying makes this safe
  because the kernel consumes a snapshot; direct reads require pinning/refcount
  or completion-event ownership until the last consuming kernel finishes;
- the 5.8 GB transient copy can disappear, but the expert kernel must still
  read 5.8 GB of source weights. Zero-copy removes the extra arena-read/staging-
  write pass; it does not remove expert bandwidth.

The prior W3 evidence is particularly useful. Its hit-only microgate measured
about 14-15% direct-slot kernel headroom, while the end-to-end five-block
campaign reduced 128-token completed time from 59.0415 to 57.0099 seconds:
`2.03166 / 128 = 15.87 ms/token`, a real 3.44% improvement that failed the
pre-registered 5% adoption bar. This proves feasibility and supplies a more
credible savings estimate than the nominal 42.8 ms copy-traffic floor. A
post-hot-path W3a rescreen is reasonable, but claiming 43 ms savings from
zero-copy alone would contradict the completed-time evidence.

## Ranked bounded lossless changes

Savings are predicted critical-path reductions at the current 429 ms baseline.
They are not additive without a new matched decomposition because several
changes attack the same memory traffic.

| Rank | Change | Class | Predicted saving | Assessment |
| ---: | --- | --- | ---: | --- |
| 1 | GPU-resident expert directory + direct-slot dispatch, with miss bitmap/fallback and completion-event slot ownership | Major kernel/runtime work | **25-60 ms/token** including copy/readback/control removal | This is the complete form of W3: eliminate 75 D2H/remap round trips, 600 CPU lookups on the hit path, and 1,800 staging copies. It must retain fail-closed miss handling and cannot evict an in-flight slot. Direct-slot W3 alone measured only 15.87 ms; the upper part of this range requires moving directory/remap control to the GPU too. |
| 2 | Optimize the single-token IQ2_XXS routed MoE kernels without changing reductions | Kernel work | **40-65 ms/token** | The measured 107 ms bucket reads 5.84 GB at only about 55 GB/s. Reaching 100-140 GB/s while preserving the exact quant decode and accumulation order would cut it toward 42-65 ms. Pruning total experts cannot do this because top-8 work is unchanged. |
| 3 | Optimize/fuse mapped Q8 small-GEMV paths for attention projections, shared experts, dense FFNs, and output | Broad kernel work | **60-100 ms/token** | Largest remaining bucket: about 18.62 GB Q8-equivalent weights/token. Requires systematic decode-GEMV work, not cache sizing. Fusion can remove intermediate activation traffic and launches, but byte identity constrains reassociation. |
| 4 | Rework staged indexed attention to reuse selected KV across heads/warps and reduce its three-launch-per-layer structure | Kernel work | **20-45 ms/token** | Targets the estimated 70 ms DSA bucket. Preserve row selection, softmax order, and value accumulation. Exact top-k alone is much smaller at 30K. |
| 5 | Make MTP verification reproduce the single-token reduction order exactly, then use accepted drafts | Large algorithm/kernel work | **1.2-1.4x throughput multiplier**, conditional | Potential final multiplier, not a subtractive patch. Current source performs an extra blk.78 pass and its historical batched verification changed floating-point order/output. It is lossless only after byte-identical target logits/tokens are demonstrated. |
| 6 | Re-screen direct-slot W3a alone against the new profile | Medium kernel patch | **14-22 ms/token** | Bounded by the prior measured 15.87 ms/token end-to-end saving. Useful independently, but subsumed by rank 1. |
| 7 | Gate access-stream SHA behind an evidence-only startup flag, hoist all MoE `getenv` decisions, retain/reuse host vectors | Small patch | **1-4 ms/token** | Correct hot-path hygiene: removes 9.6 KB hashed, 150 SHA blocks, 3,825-3,876 `getenv` calls, and 75 vector allocation sets per token. It cannot explain the plateau. |
| 8 | Replace blocking ID copies with persistent pinned 32-byte buffers and stream events, without a GPU directory | Small/medium patch | **0-5 ms/token** | Can narrow synchronization scope but cannot remove the router-to-CPU-to-kernel dependency. Treat any larger forecast as double-counting GPU work waited on by the D2H. |
| 9 | Exact W4 top-2048 candidate at 30K | Small dispatch/integration patch after prior gate resolution | **~1-3 ms/token** | Strong 1M/prefill lever, modest 30K single-token lever. The current source's top-k tree is not the dominant attention cost here. |
| 10 | `DS4_CUDA_END_STREAM_SYNC=1` / four-stub event conversion alone | Small patch | **~0-2 ms/token on this path** | R-K.2's 80-90-barrier premise is stale under CUDA static decode mapping. Keep as correctness/containment cleanup, not a headline decode lever. |

### Ordered path toward 10 tok/s

A defensible order is rank 7 first as a cheap cleanup/falsifier; W3a or the
full rank-1 directory next; then isolated routed-MoE, non-routed Q8, and DSA
kernel campaigns; exact MTP only after the single-token kernels are stable.
Every candidate needs completed CUDA-event/client-wall timing because enqueue
timing already produced one false conclusion in this codebase.

Using central estimates, ranks 1-4 might reduce 429 ms to roughly 200-250 ms,
or 4-5 tok/s. A proven exact MTP multiplier might then reach roughly 5-7 tok/s.
Only the optimistic edge of every range (about 60 + 65 + 100 + 45 ms saved,
plus launch cleanup, followed by about 1.4x exact MTP) approaches 100 ms/token.
That is consistent with the 90 ms unique-weight roofline and leaves almost no
margin for attention/control. Therefore:

- **2.33 -> 4-6 tok/s** is a defensible lossless kernel/dispatch target;
- **7-10 tok/s** is an optimistic full-program range requiring several
  successful kernel campaigns and exact MTP;
- **10+ tok/s is not supported as the predicted result of bounded small
  patches**. It requires near-roofline behavior across essentially the whole
  decode stack, a byte-identical multi-token multiplier, or a new residency/
  representation decision.

## What to measure next

The smallest decisive read-only diagnosis has now been exhausted. The next
evidence-mode binary should put CUDA events (default off) around exactly five
completed groups: selected-ID/router-to-remap, arena copies, routed MoE,
indexer score/top-k, and selected attention/non-routed GEMVs. It should also
record major/minor faults for blk.78 and count actual calls to the four async
stubs. That one decomposition would replace the widest estimated rows above.
It must not time only the hit-site enqueue calls, and its disabled production
path must satisfy the repository's no-hot-path-tax gate before promotion.
