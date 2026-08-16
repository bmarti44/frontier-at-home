# Expert direct-slot dispatch

## Scope

This incremental patch targets the supplied `ds4` 4d54edd source plus the
dynamic expert-count and decode-stage timing patches. It introduces the
presence-tested `DS4_CUDA_EXPERT_DIRECT_SLOT` diagnostic switch. The switch is
read exactly once, when the persistent expert arena is initialized, and is OFF
when absent.

The source directory was read-only to this agent's filesystem sandbox, so the
patch was applied and built in the byte-for-byte working copy
`/tmp/glm52-dynexp2-direct-slot`. The requested source tree itself was not
modified. The patch applies directly to its `ds4_cuda.cu`.

One discrepancy was found between the supplied source and the prerequisite
analysis: this exact `ds4_cuda.cu` has no `cudaHostRegisterPortable` call for
the persistent arena. Accordingly, the patch preserves the supplied OFF path
(an unregistered arena) and performs mapped registration only when the new
flag is present. It does not silently add pinning to the OFF path.

## Initialization and fail-open behavior

When the flag is present and the persistent arena is successfully allocated,
initialization first queries `cudaDevAttrPageableMemoryAccess`. On a coherent
pageable-memory device, the host arena base is used directly in device pointer
table entries, without host registration or an alias lookup. Otherwise the
arena is registered with `cudaHostRegisterMapped | cudaHostRegisterReadOnly`,
then resolved once with `cudaHostGetDevicePointer`; the retained alias is used
in device pointer-table entries.

### GB10 coherent pageable access

GB10 reports `cudaDevAttrPageableMemoryAccess == 1` even though
`cudaHostRegister` returns `cudaErrorNotSupported`. Its ATS/HMM coherent
addressing allows kernels to dereference pageable host pointers directly. The
direct-slot initialization therefore uses the arena host pointer as its
device-visible base and logs `direct-slot: coherent pageable access, using host
arena pointer directly`. Devices without coherent pageable access retain the
mapped-registration attempt and the existing fail-open staging fallback.

If registration, alias resolution, or the small protected-slot bitmap
allocation fails, direct-slot mode is disabled, one diagnostic is printed, the
sticky CUDA error is cleared, and selected experts use the existing staging
copy path. No model or serving operation depends on direct-slot availability.

## Payload layout

The existing remapped-ID device allocation is widened only for an eligible
single-token top-8 layer. The single existing H2D upload contains:

```text
offset  size                 contents
0       8 * sizeof(int32_t)  selected slots remapped to compact IDs 0..7
32      8 * 3 * uint64_t     device addresses, indexed [compact_id][tensor]
```

Tensor index 0 is gate, 1 is up, and 2 is down. Each address is computed as:

```text
arena_device + arena_slot * slot_bytes + tensor_offset
```

where the tensor offsets are `0`, `gate_expert_bytes`, and
`2 * gate_expert_bytes`. The generic formula aligns the pointer table to eight
bytes, although the top-8 payload naturally begins at byte 32. No additional
H2D copy or blocking transfer is introduced.

Direct mode is restricted to eight distinct selected experts and equal gate/
down slab sizes. That is the normal three-IQ2-matrix decode layout described by
the campaign; mixed IQ2/Q2 layouts and blk.78 keep staging unchanged.

## Kernel and output identity

The single-token generic routed-MoE decode kernels receive an optional pointer
table. Both the normal and LUT IQ2 gate/up kernels, the IQ2 down kernel, and
their float-activation fallback kernels select each expert tensor base from
the table when it is non-null. With a null table they retain the original
`compact_base + compact_id * expert_bytes` calculation.

The direct path changes only the base address of each weight load. Arena slots
contain the same gate/up/down bytes used to populate the staging buffers. Row
offsets, quantized block decoding, lane assignments, dot-product loop order,
warp reductions, activation arithmetic, expert weighting, and final reduction
order are untouched. Therefore the ON-path arithmetic and output are
bit-identical to the copy path. This campaign did not run a model or GPU test,
as explicitly prohibited; byte identity remains a source-level invariant to
confirm in the later paired runtime gate.

## Miss and fallback behavior

Hits on an eligible direct layer skip all three arena-to-staging copies. Misses
still use the existing disk/model-map-to-staging machinery, insert and fill an
arena slot from those staged bytes, and only then contribute addresses to the
pointer table. If insertion or fill metadata cannot produce a complete table,
the layer falls back to staging; any earlier hit copies that were skipped are
performed before dispatch.

## Eviction safety

A one-byte-per-arena-slot protection bitmap is allocated only for successfully
enabled direct mode. A hit is protected immediately when its slot is found,
before later selected misses can insert. A newly filled miss slot is protected
as soon as its three tensors have been copied into the arena. SLRU insertion
walks backward from the tail and skips protected victims.

Protection is deliberately stronger than a per-layer event: all direct slots
selected during one token remain protected through that token. Normal decode
visits routed layers in increasing order, so a layer-number wrap identifies
the next token and clears the bitmap. By the time the next token's selected-ID
readback begins, the legacy/default CUDA stream dependency has completed the
previous token's MoE dispatch. Thus no slot referenced by a published pointer
table can be evicted or overwritten while a consuming kernel is in flight.
Model-generation flushes also clear the bitmap.

## OFF-path behavior

With `DS4_CUDA_EXPERT_DIRECT_SLOT` absent:

- no protection bitmap is allocated;
- no mapped registration or device alias lookup is attempted;
- the remap upload remains exactly `slot_count * sizeof(int32_t)`;
- every cache hit performs the same three staging copies as before;
- kernels receive a null pointer table and use their original compact staging
  bases;
- miss fill, blk.78 handling, SLRU ordering, kernel arithmetic, and reductions
  remain unchanged.

The only shared code-shape changes are an optional slot-output argument on the
cache lookup and a protected-victim predicate whose feature state is resolved
at initialization; neither changes OFF-path data or control outcomes.

## Build

`make cuda-spark -j2` completed successfully in
`/tmp/glm52-dynexp2-direct-slot` on 2026-08-16. `nvcc` compiled
`ds4_cuda.cu`, and `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
`ds4-agent` all linked successfully. No model or GPU run was performed.
