# DSA staged indexed-attention decode: conservative LoRA warp tile

Date: 2026-08-17

## Scope and source facts

This candidate is based on `/home/bmarti44/.cache/glm52-dynexp2-patched`
with original `ds4_cuda.cu` SHA-256
`cd26980119ab…`.
It changes only `ds4_cuda.cu`; it does not modify the source tree in the
cache, run a model, launch a CUDA kernel, or commit anything.

Section 4 of the hot-path analysis establishes the target:

- `results/glm52-gates/decode-hotpath-analysis-2026-08-16.md:232-240`
  identifies 21 full selections but 78 selected-attention layers.
- `results/glm52-gates/decode-hotpath-analysis-2026-08-16.md:242-268`
  identifies the F32 compact cache, per-head rereads, 11.12 billion
  multiply-add terms/token, and 234 staged launches/token.
- `results/glm52-gates/decode-hotpath-analysis-2026-08-16.md:270-279`
  bounds exact top-k to about 1-3 ms/token and selected attention to about
  40-80 ms/token.
- `results/glm52-gates/decode-hotpath-analysis-2026-08-16.md:403-414`
  ranks selected-KV reuse and staged-launch reduction fourth, with a
  20-45 ms/token estimate for the complete change.

## Reread pattern found

The accepted staged implementation has three launches per layer:

1. The weights kernel maps one block to one head
   (`ds4_cuda.cu:26138-26150`), assigns selected rows to its 256 threads, and
   rereads the same selected compact-KV and rope row independently for every
   head (`ds4_cuda.cu:26164-26195`). Its reduction tree and per-thread row
   assignment define the exact softmax result (`ds4_cuda.cu:26198-26224`).
2. The LoRA kernel maps `blockIdx.y` to a head and each thread to one adjacent
   value pair (`ds4_cuda.cu:26238-26246`). Each `(head, pair)` thread walks
   `s = 0..row_count-1`, so all 64 heads reread the same compact-KV row
   (`ds4_cuda.cu:26247-26261`).
3. Value projection again maps blocks by head and uses the existing
   lane-to-block assignment and shuffle reduction
   (`ds4_cuda.cu:26265-26304`).

The single-token indexer score kernel already loads the 128-float index row
once into shared memory and reuses it across four head warps
(`ds4_cuda.cu:10495-10535`), and it is selected by the decode-one dispatch
(`ds4_cuda.cu:11770-11778`). No indexer or top-k code is changed.

## Change

`patches/dsa-attention-opt.patch` adds a diagnostic/default-off LoRA
accumulation kernel. Set `DS4_CUDA_GLM_ATTN_LORA_WARP_TILE=1` before CUDA
initialization to select it. Initialization resolves a launch-function pointer
once and logs the exact 8-head x 4-value-pair mode; the disabled decode path
does not test the environment or branch on the diagnostic flag per layer.

The new kernel maps each warp as:

```
lane:          0  1  2  3 | 4  5  6  7 | ... | 28 29 30 31
head in tile:  0  0  0  0 | 1  1  1  1 | ... |  7  7  7  7
value pair:    0  1  2  3 | 0  1  2  3 | ... |  0  1  2  3
```

For each selected-row iteration, lanes 0-3 load four adjacent compact-KV
pairs and broadcast them to eight heads. The first lane for each head loads
that head's softmax weight and broadcasts it to four value pairs. A 256-thread
block contains eight such warps. For the GLM shape (`n_head=64`,
`kv_lora_dim=512`), the launch has 8 x 8 = 64 blocks rather than 4 x 64 = 256
blocks, while retaining the same 16,384 output-owning threads.

The diagnostic launcher falls back to the original kernel for any head count
other than 64. Both F32 and F16 cache instantiations and the two-token range
instantiations remain covered.

## Why the output order is unchanged

For every output identified by `(token, head, j)`:

- exactly one thread owns `acc0` and `acc1`, as before;
- it visits selected positions in the identical increasing `s` order;
- it uses the identical selected row, softmax weight, and converted `float2`
  compact-KV values;
- warp shuffle only transports those value bits; it performs no arithmetic;
- the executed arithmetic remains `acc += w * v` for each `s`, followed by
  the same division by that head's denominator;
- no cross-thread reduction is introduced into LoRA accumulation;
- row selection, score calculation, maximum/sum softmax reduction trees, and
  value-projection shuffle trees are untouched.

The `__ldg` uses are limited to selected indices, staged F32 weights, and the
denominator. They change the read path, not the stored bits or arithmetic.

This is a source-order proof, not a completed byte-identity acceptance result.
The required matched GPU output gate was intentionally not run under the
campaign's prohibition on model/GPU runs. The flag therefore remains
diagnostic and default-off.

## What was deliberately left unchanged

The weights stage still rereads selected compact KV and rope per head. Sharing
there while preserving byte identity would require separating score production
from softmax or retaining 256 logical row lanes per head inside a multi-head
block; either approach changes the launch structure and risks the established
per-head reduction tree. The value-projection stage is also unchanged because
its warp reduction order is part of the identity contract. Consequently this
candidate does not fuse the three launches. Those are the remaining pieces of
the analysis's full rank-4 opportunity.

## Expected saving

At 78 layers, 64 heads, 2,048 rows, and 512 F32 LoRA values, the old LoRA
stage issues logically about 20.94 GB/token of compact-KV value reads:

```
78 * 64 * 2048 * 512 * 4 = 20,937,965,568 bytes/token
```

Eight-head reuse reduces that component to about 2.62 GB/token. Four-pair
weight reuse adds less favorable staged-weight transactions than the old
same-head warp broadcast; a simple transaction model still removes about
16 GB/token net from this stage. Actual reuse in L2 means this is not a DRAM
roofline prediction.

Expected end-to-end saving: **about 8-18 ms/token at 30K**, pending measurement.
That is intentionally below the analysis's 20-45 ms/token estimate for full
score/value reuse plus launch fusion.

## Verification status

- Patch dry-run in fresh `/tmp/dsa-attention-opt.bhY0h1/verify`: **PASS**
  (`patch --dry-run -p1 -i .../patches/dsa-attention-opt.patch`).
- Patch application in that copy: **PASS**.
- Applied `ds4_cuda.cu` SHA-256:
  `97575138d1c7…`;
  it matches the separately edited `/tmp` work copy.
- `make -n cuda-spark`: **PASS**; the target resolves to a forced build of all
  CUDA Spark binaries.
- `make cuda-spark -j2`: **NOT RUN / SAFETY-BLOCKED**. At the build decision,
  `scripts/52_engine_switch.sh status --json` reported recorded profile
  `dsv4`, `MemAvailable` was 12,789,076 kB, and swap was already nonzero
  (`SwapTotal` 16,777,212 kB; `SwapFree` 16,424,524 kB). Repository policy
  requires stopping the identity-verified large-model engine and waiting for
  at least 110 GiB available before this forced CUDA build. Stopping/restoring
  the production model would violate this campaign's no-model/GPU-run scope,
  so the build was not started.

No CUDA build, model execution, GPU kernel, service stop, or commit occurred.
