# Dense Q4_0 CUDA batch/prefill support

## Result

The requested engine tree at
`/home/bmarti44/.cache/glm52-dynexp2-patched` is readable but not writable in
the agent sandbox. The incremental patch against that already-patched tree is
`patches/dense-q4_0-batch.patch`. It was applied and compiled in the exact
staged copy `/tmp/codex-dense-q4-batch-stage-20260816/build`.

No model or GPU workload was run, so the reported 5.8 tok/s has not been
remeasured. No commit was created.

Q4_0 block decoding remains 32 values per 18-byte block: fp16 scale followed
by 16 packed bytes. Elements 0..15 use low nibbles, elements 16..31 use high
nibbles, and each decoded value is `scale * (nibble - 8)`.

## Fallback traced from the 5.8 tok/s path

The common GLM prefill helper `glm_graph_matmul_q8_0_tensor` resolves the real
weight type and calls `ds4_gpu_matmul_quant_tensor`. For Q4_0 that dispatcher
previously selected `cuda_matmul_q4_0_tensor`, whose only kernel was
`glm_matmul_q4_0_warp8_kernel`. Despite accepting `n_tok`, it assigned a
separate warp to every `(token, output-row)` pair, read and decoded the same
weight row again for each token, consumed F32 activations directly, and had no
Q8-style token tile or GEMM arm. This is the primary scalar/per-row device
fallback reached by attention projections, leading dense FFNs, shared-expert
split projections, attention output, and the non-TP batched output head.

There was also a host-side per-token fallback in
`glm_graph_encode_sparse_ffn_indexed_batch_routed_moe`: its batched shared
expert arm required exact Q8_0 gate/up weights. Q4_0 therefore entered the
`for (t = 0; t < n_tokens; t++)` loop, created row views, and separately
launched gate, up, SwiGLU, and down operations for each token. The patch adds
a Q4_0-only batch arm before the unchanged Q8_0 arm and loop.

`glm_graph_matmul_q8_0_rows_scalar` first calls
`ds4_gpu_matmul_quant_rows_scalar_tensor`; the previous Q4_0 decode patch made
that CUDA entry point delegate to `ds4_gpu_matmul_quant_tensor`. It therefore
avoids the helper's host loop, but before this patch still landed in the
untiled Q4_0 device kernel described above.

## Kernels and dispatchers changed

- `q4_0_dequant_f16_kernel`: Q4_0 analogue of
  `q8_0_dequant_f16_kernel`; two threads decode the low/high 16-value halves
  and write contiguous FP16 values.
- `glm_matmul_q4_0_preq_batch_warp8_tiled_kernel<8>`: mirrors the Q8_0
  warp-8/token-8 batch structure. Activations are quantized once with the
  existing `quantize_q8_0_f32_kernel`; one warp owns an output row, decodes an
  18-byte Q4_0 block once, reuses it across eight token accumulators, and uses
  DP4A when enabled.
- `cuda_matmul_q4_0_tensor`: preserves the existing F32 single-token kernel.
  For `n_tok > 1` it uses the new prequantized token-8 kernel. At 128 or more
  tokens with the existing GLM dequant-GEMM switch enabled, it mirrors the Q8
  large-batch policy by dequantizing Q4_0 weights to FP16 and calling the same
  cuBLAS GEMM shape; failure falls through to the native token tile.
- `ds4_gpu_matmul_quant_tensor`: unchanged in this incremental patch, but its
  existing Q4_0 case now reaches the new batch logic above. This is the common
  dispatcher for `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, full-indexer
  `indexer_attn_k`/`indexer_attn_q_b`, the ordinary attention-output matrix,
  shared-expert gate/up/down, leading dense-FFN gate/up/down, and generic dense
  output projections.
- `ds4_gpu_matmul_quant_rows_scalar_tensor`: unchanged here and continues to
  delegate to `ds4_gpu_matmul_quant_tensor`; Q4_0 calls with `n_tok > 1` now
  inherit the same batch kernels instead of the untiled device fallback.
- `ds4_gpu_glm_qk_lowrank_typed_batch_tensor`: extends the existing
  128-token dequantized cuBLAS arm to Q4_0 using the Q4 block decoder. Smaller
  Q4 batches retain the already-added typed batch kernel. This covers
  `attn_k_b` during indexed prefill.
- `glm_value_project_q4_0_batch_heads_tiled_kernel<16>` and
  `ds4_gpu_glm_value_project_typed_batch_heads_tensor`: add the Q4_0 analogue
  of the Q8_0 token-16 weight-row reuse kernel for `attn_v_b`. The existing
  single-token/small-batch Q4 kernel remains the tail path.
- `glm_graph_encode_sparse_ffn_indexed_batch_routed_moe`: adds the Q4_0
  batch shared-expert sequence (gate, up, whole-batch SwiGLU, down) and bypasses
  its former per-token loop.
- `metal_graph_output_logits_head_matmul`: accepts Q4_0 in the batched
  vocabulary TP dispatcher, uses 18-byte row offsets, and calls the typed
  quant dispatcher for Q4_0 shards and the non-TP Q4_0 head. Existing Q8_0
  calls remain in their original branches.

## Prefill consumer coverage

- Attention input projections: Q4_0 `q_a`, `q_b`, `kv_a`, and full-indexer
  projections use the common typed batch dispatcher.
- MLA low-rank attention: Q4_0 `k_b` uses the typed qk-lowrank batch
  dispatcher; Q4_0 `v_b` uses the typed batch-head value dispatcher and its
  token-16 tile.
- Attention output: the GLM dense `attn_output` matrix uses the common typed
  batch dispatcher. The previously added Q4_0 k-slice support remains intact
  for split attention-output paths.
- Shared experts: both `glm_graph_encode_ffn_batch` and the indexed sparse-FFN
  path issue whole-batch Q4_0 gate/up/down projections. The Q8-only fused
  gate/up helper remains guarded by the exact Q8 predicate.
- Leading dense FFNs: gate, up, and down use the common typed batch
  dispatcher; their Q8-only fused helper remains unchanged.
- Output head at `n_tokens > 1`: the speculative/batched vocabulary head now
  selects Q4_0 for both a local head and TP shards, including correct 18-byte
  row addressing.

## Existing-path preservation

All new kernels and branches are selected only for weight type Q4_0. Q8_0
continues to call its existing native, MMA, dequant-GEMM, fused shared-expert,
value-projection, qk-lowrank, and output-head code. F16 dispatch is unchanged.
The Q4_0 single-token dense kernel is also unchanged. Consequently the patch
does not modify Q8_0 or F16 arithmetic or launch selection.

The Q4_0 batch common-matmul path intentionally adopts the existing Q8_0
activation policy: F32 activations are blockwise quantized to Q8_0 before the
native DP4A tile, while the 128-token large-batch arm stages weights and
activations as FP16 for cuBLAS. This is a Q4_0-only numerical-path change and
requires the repository's normal paired quality gate before production
promotion.

## Build and checks

- `make cuda-spark -j2`: **PASS** in the staged copy on 2026-08-16. It cleanly
  rebuilt and linked `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
  `ds4-agent` with no compile/link errors or warnings.
- `patch --dry-run -p1`: **PASS** against
  `/home/bmarti44/.cache/glm52-dynexp2-patched` (the tool only noted that the
  target source files are read-only).
- `git diff --no-index --check`: **PASS** for both changed source files.
- Patch SHA-256:
  `9921bd54f782…`.

## Final dispatcher list

1. `glm_graph_matmul_q8_0_tensor` -> `ds4_gpu_matmul_quant_tensor` ->
   `cuda_matmul_q4_0_tensor` (Q4_0 GEMM/token-8 batch arms).
2. `glm_graph_matmul_q8_0_rows_scalar` ->
   `ds4_gpu_matmul_quant_rows_scalar_tensor` ->
   `ds4_gpu_matmul_quant_tensor` (same Q4_0 batch arms).
3. `ds4_gpu_glm_qk_lowrank_typed_batch_tensor` (Q4_0 `attn_k_b`).
4. `ds4_gpu_glm_value_project_typed_batch_heads_tensor` (Q4_0 `attn_v_b`).
5. `glm_graph_encode_sparse_ffn_indexed_batch_routed_moe` (Q4_0 shared
   gate/up/SwiGLU/down batch selection).
6. `metal_graph_output_logits_head_matmul` (Q4_0 batched local/TP vocabulary
   head selection).

Build status: **PASS**. Runtime/performance status: **not measured by request**.
