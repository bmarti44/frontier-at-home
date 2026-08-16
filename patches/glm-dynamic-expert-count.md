# GLM dynamic routed-expert count patch

## Scope and source identity

`glm-dynamic-expert-count.patch` is an authored, unapplied unified diff against
the immutable snapshot at
`/home/bmarti44/ds4-source-snapshot-2026-08-15/`:

- `ds4.c`: `ad3087878988…`
- `ds4_cuda.cu`: `7e6aaabb8771…`

The patch was not applied, compiled, or run. No model, CUDA, or GPU process was
started or stopped while authoring it.

## Runtime count flow

The patch deliberately adds no global and no struct member. GLM metadata is
validated in `config_validate_glm_dsa_model`, then stored in the existing
`g_ds4_shape.n_expert`. Consequently the existing `DS4_N_EXPERT` macro carries
the runtime value through graph sizing, router dimensions, layout validation,
cache planning, and `graph_stream_expert_table_make`. That table already copies
`DS4_N_EXPERT` into `ds4_gpu_stream_expert_table.n_total_expert`; the CUDA GLM
MoE call already receives that field as `n_total_expert`.

`DS4_SHAPE_GLM52.n_expert` remains `256`. It is both the unchanged default for
ordinary GLM-5.2 metadata and the upper bound for pruned models. The selected
expert count (`n_expert_used`) remains fixed at `8`.

## Hunk-by-hunk explanation

### `ds4.c`: explicit router/routed-tensor count cross-check

Adds `tensor_expect_glm_expert_count`. For each routed tensor it requires:

- router rows equal the metadata-derived `DS4_N_EXPERT`;
- routed tensor dimension 2 be exactly divisible by the metadata count;
- the quotient be exactly one, so dimension 2 is neither a multiple containing
  concatenated expert sets nor a partial set; and
- routed tensor dimension 2 equal the router row count.

The ordinary layout validators still run first and still require the exact
dimensions and types. Therefore this hunk is defense in depth: for a valid
256-expert model it cannot change accepted layout or execution. For malformed
models the existing validators reject first, so it also does not reorder the
existing failure path.

### `ds4.c`: invoke the cross-check for gate, up, and down experts

After the existing exact routed-expert layout checks, invokes the new check for
all three routed tensors using `ffn_gate_inp` as the router matrix. This proves
that the metadata count, router rows, and every routed tensor's third dimension
agree. The call site is GLM-only; DeepSeek V4 layouts are untouched.

At `expert_count == 256`, this hunk has no successful-path behavior change: all
values checked are already required to equal 256 by the immediately preceding
validators.

### `ds4.c`: cross-check the dedicated MTP weights structure

The distinct `ds4_mtp_weights` validator already checks the MTP router matrix,
required router bias, and the exact routed gate/up/down layouts. Immediately
after each routed layout check, this hunk reuses
`tensor_expect_glm_expert_count` to tie the router row count and each routed
tensor's `dim[2]` to `DS4_N_EXPERT`. This covers the separately named
`mtp.0.*` tensors at the point where `mtp_weights_bind` has made them available.

At 256 experts, each new call compares values that the immediately adjacent
layout checks already require to be 256. It therefore cannot change a valid
256-expert layout or any execution behavior.

### `ds4.c`: accept and publish the metadata expert count

Replaces the old equality check against compiled `256` with a bounded GLM-only
check. `expert_used_count` is still hard-validated against the profile value
`8`, after which `expert_count` must satisfy:

```text
n_expert_used < expert_count <= DS4_SHAPE_GLM52.n_expert (256)
```

Only then is the count assigned to the existing runtime shape. Because the
function begins by resetting `g_ds4_shape` from `DS4_SHAPE_GLM52`, missing or
ordinary metadata still starts from—and, for ordinary files, assigns—256.

At `expert_count == 256`, the assignment writes the value already present in
the shape. The only removed operation is an equality validator that would have
accepted 256; all other metadata checks and the resulting value are unchanged.

### `ds4.c`: validate the bound GLM next-token/MTP layer range

`weights_bind` excludes `DS4_N_NEXTN_PREDICT` layers from its normal executable
range, then binds those layers for the GLM drafter. The original validation call
still ends at the executable range. For a full GLM binding, this hunk invokes
`weights_validate_glm_dsa_layout` a second time over exactly the bound trailing
range. Thus the next-token layer receives the same router-matrix, required
router-bias, `tensor_expect_routed_expert`, and
`tensor_expect_glm_expert_count` checks as every normal sparse GLM layer. Slice
loads that do not cover the complete executable range do not bind that trailing
range and do not take the new validation path.

At 256 experts, the added validation only rechecks already required MTP tensor
layouts against the unchanged value 256; it does not alter binding, graph
construction, or execution.

### `ds4_cuda.cu`: bound direct Q2_K expert IDs

Five tightly related diff hunks protect the scalar/direct Q2_K path:

- the gate/up kernel receives `n_total_expert` and treats a negative or
  `expert >= n_total_expert` selection exactly like the existing negative-ID
  skip, before either expert blob is indexed;
- the down kernel receives the same bound, and its loop skips an out-of-range
  selection before reading its router weight or indexing the down blob;
- the separate gate/up and down launch hunks pass the already validated
  `n_total_expert` argument through without changing launch geometry.

At 256 experts, each signature/launch hunk only threads the existing value 256,
the gate/up predicate remains false for every previously valid ID in
`[0, 255]`, and the down predicate likewise remains false for every previously
valid ID. Therefore all five hunks leave valid 256-expert kernel work,
weighting, and accumulation unchanged.

### `ds4_cuda.cu`: consume and validate `n_total_expert`

Stops discarding the already-threaded `n_total_expert` argument and rejects a
total count that is not greater than the selected slot count or exceeds 256.
Metadata currently guarantees `8 < n_total_expert <= 256`; this boundary check
also protects direct or future callers.

For the existing 256-expert call, the new predicate is false and execution is
unchanged.

### `ds4_cuda.cu`: resolve exact routed tensor spans

Changes gate, up, and down mapped-weight span lengths from literal
`256 * expert_bytes` to `n_total_expert * expert_bytes`. This is the safety-
critical fix for pruned GGUFs: resolution no longer claims bytes beyond the real
routed tensor.

For `n_total_expert == 256`, each multiplication has identical operands and
produces the identical span. Pointer resolution and kernel math are otherwise
untouched.

### `ds4_cuda.cu`: size expert map and tile scratch from the runtime total

Sizes `counts`, expert lists, and the worst-case extra tile entries from
`n_total_expert`. The `255`/`256` values used for 256-byte alignment are left
literal because they are byte-layout constants, not expert counts.

At 256 experts every size and offset is arithmetically identical to the old
code.

### `ds4_cuda.cu`: build the initial map with the runtime total

Passes `n_total_expert` to `glm_moe_expert_map_kernel` and
`glm_moe_build_expert_tiles8_kernel`. The CUDA launch block size remains 256
threads. This changes only the number of expert buckets the kernels consider.

At 256 experts, kernel arguments and launch geometry are byte-for-byte the same
values as before.

### `ds4_cuda.cu`: rebuild chunked tile maps with the runtime total

The exact-down tile path rebuilds its map for chunks larger than 512 tokens.
This hunk uses `n_total_expert` for the chunk tile bound and for both builder
calls, matching the initial map. Launch widths and token chunking are unchanged.

At 256 experts all computed values and arguments are identical to the old code.

### `ds4_cuda.cu`: bound expert-major grids by the runtime total

Changes the Y dimensions of the non-tile expert-major gate/up and down grids
from literal 256 to `n_total_expert`. These kernels index the `counts` and
`lists` arrays by that Y coordinate, so the grids must match their runtime-sized
arrays. Their 256-thread block dimensions remain unchanged.

At 256 experts the grid dimensions are identical to the old launch geometry.

### `ds4_cuda.cu`: guard inactive lanes in the 256-thread router

After each shared-memory argmax reduction, thread 0 now validates the winning
index and winning score before using the index to read `probs`. A winner at or
above `n_expert`, or a winner whose score is the inactive-lane sentinel
`-1e30f`, is deterministically replaced with expert 0. The shared arrays and
the 256-thread launch remain unchanged; only a reduction result that cannot be
a valid active expert is repaired.

At `n_expert == 256` there are no inactive lanes: all 256 shared entries are
filled from active logits. With the fixed `n_expert_used == 8`, a valid active
candidate remains available in every top-k round, so neither the index bound
nor inactive-sentinel condition fires. The selected IDs, probability reads,
weights, normalization, and tie-breaking are therefore unchanged for the
256-expert path.

## Literal-256 audit

The snapshot was searched for literal `256` occurrences near `expert`, `moe`,
`router`, `gate`, `ffn`, and GLM symbols in both requested source files. The
following categories were checked:

- GLM profile/default: `ds4.c:626` remains 256 intentionally as the default and
  maximum. The 256-expert DeepSeek Flash defaults at `ds4.c:550,670` are outside
  the GLM path and remain unchanged.
- GLM Q2_K mapped spans: `ds4_cuda.cu:27654,27657,27660` are expert-count
  assumptions and are parameterized.
- GLM expert-map storage/building: `ds4_cuda.cu:27729-27734,27754,27764`, plus
  the chunk rebuild at `27809,27815,27818`, are expert-count assumptions and are
  parameterized.
- GLM expert-major grids: `ds4_cuda.cu:27865,27878` use 256 as the expert-axis
  extent and are parameterized.
- GLM direct Q2_K kernels: the scalar gate/up and down kernels at
  `ds4_cuda.cu:27009-27078` now receive `n_total_expert` and reject selected IDs
  outside the mapped expert population before expert-blob pointer arithmetic.
- GLM router: `ds4_cuda.cu:28000-28157` keeps 256-entry shared arrays and a
  256-thread launch as capacity/launch constants. The fast-path predicate
  supports runtime counts up to 256, while the new post-reduction guard prevents
  an inactive shared lane from becoming an emitted expert ID and performs the
  `probs` read only after that bound check.
- QK_K dimensions: `ds4.c:757` and `ds4_cuda.cu:28`, together with
  `ds4_cuda.cu:27665-27666`, describe 256-element quantization blocks, not the
  number of experts, and stay fixed.
- CUDA launch widths and round-up expressions throughout MoE code (for example
  `(n + 255) / 256, 256`) describe threads per block, not expert population, and
  stay fixed.
- Byte-alignment expressions in map scratch (`+255`, `~255`) enforce 256-byte
  alignment and stay fixed.
- Generic DeepSeek router specializations at `ds4_cuda.cu:13179` and
  `16143-16220` require the DeepSeek Flash 256/6 shape. They are not used by the
  `glm-dsa` router and remain unchanged.
- The cache comment at `ds4_cuda.cu:24450` says `<=256`; it remains true under
  the new GLM bound and requires no code change.
- `ds4.c:62704` is a fixed 256-row `ffn_gate_inp` check in a separate diagnostic
  path outside the GLM serving graph; it was inspected and not changed by this
  smallest serving-path patch.

The count-aware paths identified by the source audit were also rechecked:
router selection uses its `n_expert` argument, streaming validates IDs against
`table->n_total_expert`, and mapped range arithmetic uses 64-bit checked
multiplication. Valid 256-expert kernel arithmetic, selection, weighting, and
accumulation order are unchanged; reduced-count invalid selections are now
rejected or deterministically clamped before dereference.

## Read-only dry run

From `/home/bmarti44/ds4-source-snapshot-2026-08-15/`, the patch was checked
with:

```text
git apply --check --verbose --whitespace=error-all \
  /home/bmarti44/spark-deepseek-v4-flash/patches/glm-dynamic-expert-count.patch
```

The command exited 0 and reported `Checking patch ds4.c...` and
`Checking patch ds4_cuda.cu...` with no offset, fuzz, or whitespace warning.
The snapshot files retained the source hashes recorded above. This was a
read-only check; the patch was not applied.

## Summary

The diff has five `ds4.c` hunks: add an explicit count-consistency validator,
apply it to normal GLM routed tensors, apply it to the dedicated MTP structure,
accept/publish a metadata count in `[n_expert_used + 1, 256]`, and validate the
bound GLM next-token/MTP layer range. Its `ds4_cuda.cu` hunks validate and use
the runtime total for direct Q2_K bounds, exact tensor spans, map/tile scratch
and builders, chunked map rebuilds, expert-axis grid bounds, and post-reduction
router safety. Every ordinary 256-expert input follows the pre-patch selections,
pointer spans, launch geometry, weighting, and accumulation path unchanged.
