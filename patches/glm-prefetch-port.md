# GLM cross-layer expert prefetch port

## Scope and provenance

- Source feature: ds4 commit `846dca421797199bb8b928f448326898614efdb0`
  (`cuda: v4.10 — cross-layer expert prefetch worker`).
- Port target: the supplied current tree at
  `/home/bmarti44/.cache/glm52-dynexp2-patched`, described by the owner as
  `4d54edd` plus expert-count, direct-slot, Q4_0 kernels, and stage timing.
- Target-tree source hashes before the port:
  - `ds4.c`: `4c377f3e661d…`
  - `ds4_cuda.cu`: `672ef6e09083…`
  - `ds4_gpu.h`: `0e93217e5db0…`
- Deliverable patch: `patches/glm-prefetch-port.patch` (466 additions,
  no deletions; SHA-256
  `c915fafd88a4…`).
- Neither source tree was modified. No commit, model load, inference, or GPU run
  was performed.

## Ported behavior

`DS4_GLM_PREFETCH` remains unset/off by default. `ds4.c` resolves it once into
a function-local cached integer; disabled decode pays only the cached boolean
test. The CUDA-side initializer is reachable only from that enabled hint path;
OFF selected-loads do not call it, allocate buffers, open a descriptor,
register memory, or create threads. Demand misses add only the cached
`g_pf.enabled` boolean check.

When enabled for single-token CUDA streaming decode, the current layer's
selected expert load is completed first. If that load used the refactored async
service-thread path, the diagnostic joins it early. The next layer's router is
then evaluated against the current `ffn_norm` hidden state using the batch router
scratch tensors. The predicted ids are read back and submitted to a bounded
worker pool. Current-layer routed work is encoded after the hint, giving the
private file reads useful overlap without clobbering the live router tensors.

The workers use a private descriptor cloned from the loader descriptor and
private aligned staging. They read the exact gate, up, and down source spans
into fixed READY slabs. Requests for resident keys and duplicate in-flight keys
are skipped. Exact source offsets plus the model-load generation are checked
again when a demand load claims a slab.

## Refactor interactions

### Direct-slot protection bitmap

The worker never chooses or writes a resident-cache slot. A prefetched slab is
claimed by the selected-load thread and admitted through the existing
`cuda_expert_cache_insert()` path. That path walks backward past every
`protected_slots[]` entry, so a prefetch-served miss cannot evict an expert
whose address is still referenced by the current token. After a successful
prefetch fill in direct-slot mode, the returned slot is placed in
`arena_slots[]` and marked protected immediately, exactly like a demand fill.
If insertion or fill fails, direct-slot use is disabled for that layer and the
existing staging fallback remains valid.

### Shared SLRU/LRU insertion and thread safety

The historical worker lock protected only its bounded request/READY-slab state;
the original design deliberately kept cache mutation on the demand thread. The
port preserves that division. Worker threads never touch the resident map,
metadata, links, arena, or protection bitmap. The existing shared insertion
routine is called only by the selected-load thread after it claims a READY
slab. In the refactored async-load path, the graph thread joins the current
selected load before issuing the next-layer hint, so the hint's read-only
resident-map check cannot race an insert. No new SLRU lock or worker-side cache
mutation is introduced.

### Byte-identical slab contents

Each prefetch request records the same gate/up/down offsets and byte counts used
by demand loading. A claim requires key, all three offsets, and model generation
to match. The selected-load thread performs blocking copies from the slab into
the normal device staging ranges, then populates the resident arena from those
same staged ranges through the existing demand-fill layout:
`gate | up | down`. Thus a prefetch hit supplies the same source bytes and slab
layout as a demand fill. The worker never transforms, quantizes, or reorders
weight data; Q4_0 and other expert tensor types remain opaque byte spans.

## Verification

- Patch dry run: **PASS**. `patch --dry-run -p1` checked `ds4.c`,
  `ds4_cuda.cu`, and `ds4_gpu.h` cleanly against a fresh copy of the supplied
  target tree. Applying the patch in that copy produced files byte-identical to
  the independently staged port.
- `/tmp` build (`make cuda-spark -j2`): **PENDING SAFETY FLOOR**. At verification
  time `MemAvailable` was about 87.5 GiB while the owner's measurement gate was
  resident, below this repository's mandatory 110 GiB clean-CUDA-build floor.
  The running workload was not stopped or disturbed. This status must be
  replaced with the actual build result after memory naturally returns above
  the floor.
