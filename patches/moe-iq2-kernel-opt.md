# Routed-MoE IQ2_XXS decode kernel optimization

Date: 2026-08-16  
Scope: CUDA single-token routed-MoE IQ2_XXS gate/up and down kernels only  
Patch: `patches/moe-iq2-kernel-opt.patch`

## Result

The patch replaces the strided scalar global reads of each quarter-warp's
eight IQ2_XXS blocks with a cooperative, contiguous 528-byte read-only load.
The 528 bytes are exactly eight 66-byte IQ2_XXS blocks and are issued as 33
`uint4` loads into a private shared-memory row for that quarter-warp. Each lane
then passes its original block, byte for byte, to the unchanged dot helper.

The requested cache checkout was read-only in this sandbox, so it was not
modified. The patch was prepared against this final source baseline:

```
/home/bmarti44/.cache/glm52-dynexp2-patched/ds4_cuda.cu
SHA-256 7c8ced1dd096…
```

The source changed concurrently once during the work. The kernel change was
rebased onto the hash above, rebuilt, and the delivered patch was verified by
`patch --dry-run -p1` against that exact source. The patch contains none of the
concurrent direct-slot edits.

## Active-path audit

Although the older fallback kernels are named `glm_routed_moe_gateup_*` and
`glm_routed_moe_down_*`, they operate on Q2_K blocks. GLM-5.2's IQ2_XXS sparse
layers satisfy `glm_graph_layer_uses_generic_routed_moe()` and dispatch through
`ds4_gpu_routed_moe_one_tensor()` to `routed_moe_launch()`. At `n_tokens == 1`
with the recorded 6144/2048 dimensions, the active IQ2_XXS kernels are:

- `moe_gate_up_mid_qwarp32_kernel` (`xq_blocks == 24`);
- `moe_down_iq2_xxs_qwarp32_kernel` (`midq_blocks == 8`).

Both compact staging and the newly added direct-slot path reach these same
kernels. `moe_expert_weight_base()` chooses the base pointer before the changed
load loop, so the optimization covers both sources without changing dispatch.

## Changes and byte-identity argument

### 1. Eight-block read-only vector staging

`moe_iq2_stage_wave8()` has eight lanes cooperatively copy 33 aligned `uint4`
values through `__ldg` into shared memory. Compile-time assertions pin the
assumptions that one IQ2_XXS block is 66 bytes and eight blocks are exactly 528
bytes, an integral number of 16-byte vectors.

Why accumulation order is unchanged: this helper performs no arithmetic and
changes no ownership. It copies a contiguous byte range only. Lane `l` still
consumes block `b0 + l`; the fact that lanes cooperate to fetch the range does
not change which lane decodes or accumulates any block.

### 2. Gate/up wave staging

The gate/up loop still visits, for each lane, `b = lane, lane + 8, lane + 16`
for the GLM-5.2 shape. For each wave it stages gate bytes, calls the existing
`dev_dot_iq2_xxs_q8_K_block()` once for the same gate block, stages up bytes,
then calls the same helper once for the same up block.

Why accumulation order is unchanged:

- the `(thread, row, b)` mapping is identical;
- each lane's gate additions occur in the same ascending `b` sequence;
- each lane's up additions occur in the same ascending `b` sequence;
- gate is still updated before up for each `b`;
- the dot helper, integer partial sums, scale operations, and activation
  expression are untouched;
- `quarter_warp_sum_f32()` is called at the same point with the same lanes and
  is unchanged.

The added quarter-warp synchronizations only protect the shared staging row.
Every eight-lane group has a distinct row, so there is no cross-row sharing.

### 3. Down wave staging

The down loop applies the same cooperative copy to the eight blocks already
owned by a quarter-warp.

Why accumulation order is unchanged: lane `l` still evaluates block `l` for
the GLM-5.2 `midq_blocks == 8` shape, using the unchanged dot helper. The local
`acc += dot` remains in the same position, and the same
`quarter_warp_sum_f32()` reduction tree follows it. Expert summation and output
writes are untouched.

### 4. Alignment and tail fallback

Vector staging is entered only for a complete eight-block wave whose source is
16-byte aligned. Any incomplete or unaligned wave executes the original scalar
statement.

Why accumulation order is unchanged: the fallback is the original dot call,
and the loop exposes the same per-lane ascending `b` sequence on either arm.
For the target GLM-5.2 shape the fast arm is expected for every wave:

```
gate/up row:    24 * 66 = 1,584 bytes (16-byte aligned stride)
down row:        8 * 66 =   528 bytes (16-byte aligned stride)
one matrix: 2,048 * 1,584 = 3,244,032 bytes
           6,144 *   528 = 3,244,032 bytes
one slot:   3 * 3,244,032 = 9,732,096 bytes
```

All row, tensor, and slot strides are divisible by 16. CUDA allocations are
more strongly aligned, and the direct-slot arena originates from `malloc`,
which supplies `max_align_t` alignment on this platform. The runtime check
still fails closed to the scalar load if a different layout violates this.

## Flag decision

No `DS4_CUDA_MOE_IQ2_FAST` flag was added. The patch does not select a new
arithmetic implementation: it copies the same bytes to shared memory and then
invokes the same dot function with the same thread/block assignment and the
same local and warp accumulation order. The alignment/tail guard preserves the
old load path for layouts outside the proved vector case.

This is a source-level identity proof, not a substitute for the requested
byte-identity A/B. Runtime identity and performance remain to be measured by
the repository owner.

## Build and object inspection

The final rebased candidate was built in an isolated writable copy with:

```
make cuda-spark -j2
```

Status: **PASS** (exit 0). All five targets (`ds4`, `ds4-server`, `ds4-bench`,
`ds4-eval`, and `ds4-agent`) compiled and linked. No model was loaded and no
GPU/kernel execution was performed.

Candidate source SHA-256:

```
12402b786afd…
```

`cuobjdump` confirms that nvcc retained the intended vector operations as
`LDG.E.128.CONSTANT.SYS` followed by `STS.128`; the quant payload is then read
from shared memory. Resource usage is:

| Kernel | Baseline registers/shared/local | Candidate registers/shared/local |
| --- | --- | --- |
| `moe_gate_up_mid_qwarp32_kernel` | 64 / 0 B / 0 B | 62 / 16,896 B / 0 B |
| `moe_down_iq2_xxs_qwarp32_kernel` | 51 / 0 B / 0 B | 51 / 16,896 B / 0 B |

There is no register spill (`LOCAL:0`). The 16,896-byte allocation is exactly
32 quarter-warp rows times 528 bytes. Occupancy and realized bandwidth were not
measured because GPU runs were explicitly excluded.

## Applying

From the engine source root corresponding to the baseline above:

```
patch -p1 < /home/bmarti44/spark-deepseek-v4-flash/patches/moe-iq2-kernel-opt.patch
make cuda-spark -j2
```

Patch SHA-256:

```
b3ad418834e7…
```

The performance target (100--140+ GB/s, or roughly 40--60 ms/token for this
bucket) is deliberately not claimed before the owner's stage-timing A/B.
