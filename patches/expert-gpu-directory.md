# Expert GPU-resident directory

## Scope

This incremental patch targets the supplied direct-slot source at
`/home/bmarti44/.cache/glm52-dynexp2-patched`. It adds the presence-tested
`DS4_CUDA_EXPERT_GPU_DIR` switch. The environment is read once and the feature
is OFF when absent. Requesting GPU-directory mode also requests the prerequisite
direct-slot arena mapping; direct-slot can still be requested independently.

The sandbox did not allow writes outside the repository and `/tmp`, so the
source was changed and built in the byte-identical copy
`/tmp/glm52-gpu-dir.AULM87/src`. The requested source tree was not modified.
`patches/expert-gpu-directory.patch` is an incremental diff against its supplied
`ds4_cuda.cu` and applies directly there.

## Directory and update ordering

The directory is a flat 76-by-256 `uint32_t` device array indexed as
`layer * 256 + expert`; `UINT32_MAX` means nonresident. A pinned host mirror is
initialized to the same value. Successful arena fills publish their slot with a
four-byte asynchronous H2D update on CUDA stream 0.

Eviction and stale-offset removal drain an older publication, enqueue an
invalid entry, and drain that invalidation before the arena slot is reused.
Publication happens only after all three arena tensors have been filled
successfully. Model-generation flushes clear both the host map and the complete
device directory. A CUDA failure disables only GPU-directory mode; the existing
direct-slot or staging path remains available.

The miss-only synchronization around reuse is deliberate. It keeps the pinned
mirror cell stable until its asynchronous copy is consumed and establishes the
required `invalidate -> arena overwrite -> publish` ordering without adding
work to an all-hit layer.

## Decode translation

For an eligible single-token top-8 IQ2 layer, a one-block kernel consumes the
router's device-resident selected IDs and writes the direct-slot payload already
used by the MoE kernels:

```text
offset  size                 contents
0       8 * sizeof(int32_t)  remapped IDs 0..7
32      8 * 3 * uint64_t     arena addresses [expert][gate, up, down]
```

Eligibility retains the direct-slot constraints: eight distinct selections,
three IQ2 tensors, equal gate/down slab sizes, one GPU, layer below 76, and at
most 256 experts. Duplicate, invalid, or nonresident selections set the miss
word and cause fallback.

The kernel also marks each resolved arena slot in the existing protection
bitmap. GPU-directory mode allocates that bitmap as mapped pinned memory, so
the host eviction walk and device translation share the same token-lifetime
state. At the layer-number wrap, the translate kernel clears protection on
stream 0 before protecting the new token's selections. On a miss, the event has
completed before the unchanged host fallback clears/reconstructs protection.

## Hit and miss behavior

The translate kernel writes a mapped pinned miss word and records one reusable
event. The host waits for that event to make the word visible. On an all-hit
layer it publishes the already-device-resident remap and pointer-table metadata
without selected-ID D2H, CPU compaction/hash lookup, or remap/pointer-table H2D.

The event is still a host dependency check; this patch removes the two blocking
memcpy round-trips and CPU directory work, but does not claim a fully
host-unsynchronized dispatch. A later runtime campaign should measure whether
event completion remains material and, if so, move miss recovery into a
separately scheduled continuation.

When the miss word is nonzero, the original blocking selected-ID D2H and
host remap/fill path runs for that layer. Its successful fills publish new
directory entries for subsequent layers/tokens. No miss streaming behavior was
reimplemented in the device path.

## Output identity and OFF path

For all-hit decode, the selected experts are required to be distinct, so the
existing CPU compactor would assign compact IDs 0 through 7 in selection order.
The translate kernel writes those exact IDs and computes each pointer with the
same formula as direct-slot:

```text
arena_device + arena_slot * slot_bytes + {0, expert_bytes, 2*expert_bytes}
```

The consuming gate/up/down kernels and all arithmetic are unchanged. Thus
ON+all-hit produces the same pointer values and kernel execution as direct-slot
ON. Runtime byte identity remains to be verified in the later paired GPU gate;
no model or GPU execution was performed in this campaign.

With `DS4_CUDA_EXPERT_GPU_DIR` absent, no directory, event, mapped miss word, or
mapped protection bitmap is allocated. The legacy selected-ID readback and
host loader remain the active path. The cached presence check adds no
environment lookup after its first call.

## Build

`make cuda-spark -j2` completed successfully in
`/tmp/glm52-gpu-dir.AULM87/src` on 2026-08-17. `nvcc` compiled
`ds4_cuda.cu`, and `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
`ds4-agent` all linked successfully. No model or GPU run was performed.
