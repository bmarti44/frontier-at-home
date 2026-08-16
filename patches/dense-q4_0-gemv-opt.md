# Dense Q4_0 CUDA GEMV optimization

## Scope and safety

This is an incremental source patch against the already-patched engine tree at
`/home/bmarti44/.cache/glm52-dynexp2-patched`. The source tree was readable but
not writable in the agent sandbox, so the implementation and build were done in
the isolated copy `/tmp/glm52-q4-opt-stage.fuSm9G` and emitted as
`patches/dense-q4_0-gemv-opt.patch`.

The live tree and its binaries were not modified or relinked. No model, CUDA
kernel, GPU workload, scorer, or NLL run was performed. No commit was created.

## What changed

- Replaced the single-token Q4_0 dense GEMV's direct FP32 dot loop with the
  same high-level strategy used by the fast Q8_0 path: quantize the activation
  once to Q8_0, then reuse it across all output rows and perform the weight dot
  with DP4A.
- Added packed Q4 helpers that load four packed bytes, form four signed low
  nibbles and four signed high nibbles with one `__vsub4` per half, and consume
  them with two DP4A operations. A 32-element Q4 block therefore uses four
  packed-word decodes and eight DP4A operations instead of 32 scalar nibble
  extracts and 32 FP32 weight/activation multiplies per output row.
- Added an aligned two-block decode kernel. For normal GLM dense shapes, each
  lane handles one consecutive 36-byte pair and reads both 18-byte blocks as
  nine aligned read-only `uint32_t` words. Those words contain both fp16 scales
  and all 32 packed Q4 bytes, so no packed weight byte is loaded twice. The host
  checks base, row-stride, and even-block alignment before selecting this arm;
  a deterministic unaligned-safe read-only fallback handles other shapes.
- Applied the prequantized DP4A decode path to both the common dense dispatcher
  and the Q4_0 k-slice dispatcher used by split attention-output projection.
  This covers dense attention projections, leading dense FFNs, shared-expert
  split projections, attention output, and the output head through their
  existing Q4 dispatch routes.
- Reduced duplicate packed-byte work in the existing token-8 Q4_0 batch/prefill
  kernel: each four-byte packed word is now loaded once and supplies both its
  low- and high-nibble DP4A operations for every token in the tile.
- Removed the superseded direct-FP32 Q4 decode and k-slice kernels.

The new decode arithmetic is deterministic for a fixed launch and input: Q8
activation quantization, per-lane block ownership, DP4A accumulation, and warp
reduction order are all fixed. It is not byte-equivalent to the original Q4
FP32-activation kernel. This is intentional for the new Q4 path and must be
judged by the repository's paired NLL suite before adoption.

## Expected bandwidth reasoning

The measured Q4 path read about 9.3 GB/token in 253 ms, only about 36.8 GB/s,
while Q8 read about 18.6 GB/token in 187 ms, about 99.5 GB/s. Meeting the
150 ms target requires only about 62.0 GB/s. Matching the measured Q8 effective
bandwidth would put the Q4 weight-read component near 93.5 ms/token before
fixed launch and activation-quantization overhead.

The old Q4 kernel was not bandwidth-bound: for every output row and 32-weight
block it performed scalar unpack/address work and 32 FP32 multiplies while
loading the same 16 packed bytes through 32 indexed byte operations. The new
kernel reduces that work to four packed-word transforms and eight DP4A
instructions. The aligned pair arm covers each contiguous 36-byte pair with
nine non-overlapping 32-bit `__ldg` loads, matching the warp-per-row streaming
shape of the successful Q8 kernel while retaining Q4's half-sized weight
traffic. Activation quantization traffic is proportional to one input vector,
not the full output matrix, and is negligible compared with roughly 9.3 GB of
weight reads per token.

This is a roofline expectation, not a performance result; only a contained
model/GPU measurement can establish whether `attn_dense <= 150 ms/token`.

## Occupancy inspection

Static `cuobjdump --dump-resource-usage` results from the copied build:

- aligned decode pair kernel: 30 registers/thread, no stack/local/shared use;
- unaligned-safe decode kernel: 34 registers/thread, no stack/local/shared use;
- token-8 batch kernel: 96 registers/thread, no stack/local/shared use;
- existing Q4 value-project tile-16 kernel: 101 registers/thread.

Both new decode kernels use `__launch_bounds__(256, 2)`. At 256 threads/block,
their register counts leave substantial occupancy headroom. The batch change
raises its compiled register count but does not introduce stack/local spills;
the specialized value-project tile was inspected and left unchanged to avoid
increasing its already-higher register footprint. Decode remains the priority.

## Build and patch verification

`make cuda-spark -j2` completed successfully in
`/tmp/glm52-q4-opt-stage.fuSm9G` on 2026-08-16. It rebuilt and linked `ds4`,
`ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent` without warnings or
errors on the final build.

The emitted patch passes `patch --dry-run -p1` against the requested engine
tree. Applying it to a fresh copied `ds4_cuda.cu` produced a file byte-identical
to the built staged source.

- Patch SHA-256: `745d1ce81ac5…`
- Patched `ds4_cuda.cu` SHA-256:
  `672ef6e09083…`
