# Dense Q4_0 CUDA support for the GLM serving path

## Result

The engine cache tree at `/home/bmarti44/.cache/glm52-dynexp2-patched` was
readable but not writable from the agent sandbox. The patch was therefore
applied and built in the exact staged copy
`/tmp/glm52-q4-build.hHBt5y`; the incremental diff against the requested
engine tree is `patches/dense-q4_0-gemv.patch`.

Q4_0 is decoded as 32 values per 18-byte block: fp16 scale followed by 16
packed bytes. Elements 0..15 use the low nibbles and elements 16..31 use the
high nibbles, with each value computed in fp32 as
`scale * (nibble - 8)`.

No model or GPU run was performed. No commit was created.

## Dispatchers extended

- `ds4_gpu_matmul_quant_tensor`: added type `2u` and a separate Q4_0
  warp-8 kernel. This is the common GLM dense dispatcher used by attention
  `q_a`, `q_b`, `kv_a`, ordinary attention/output projections, shared-expert
  split gate/up/down, leading dense FFN gate/up/down, and the output head.
- `ds4_gpu_matmul_quant_decode_mpp_model_view_tensor`: already delegates to
  `ds4_gpu_matmul_quant_tensor`; it now inherits Q4_0 without changing its
  Q8_0/F16 behavior.
- `ds4_gpu_matmul_quant_rows_scalar_tensor`: added a Q4_0-only delegation to
  the common typed dispatcher, covering scalar-row prefill and shared-expert
  fallback consumers. Non-Q4_0 behavior remains the existing stub/fallback so
  Q8_0 and F16 dispatch behavior is unchanged.
- `ds4_gpu_matmul_quant_kslice_tensor`: added a separate Q4_0 k-slice kernel
  for the split/tensor-parallel `attn_output_b` path.
- `ds4_gpu_glm_qk_lowrank_typed_batch_tensor` (and therefore the single-row
  `ds4_gpu_glm_qk_lowrank_typed_tensor` wrapper): added Q4_0 row sizing and a
  separate Q4_0 kernel for the `attn_k_b` q/k low-rank projection. The Q8_0
  dequantized cuBLAS arm remains Q8_0-only.
- `ds4_gpu_glm_value_project_typed_batch_heads_tensor`: added Q4_0 row sizing
  and a separate Q4_0 value-projection kernel for `attn_v_b` prefill/batch
  use. Existing Q8_0 tiled and scalar kernels are unchanged.
- `ds4_gpu_glm_attention_indexed_decode_typed_tensor`: accepts Q4_0
  `attn_v_b`, uses 18-byte blocks, and sends Q4_0 through the existing staged
  score/lora pipeline followed by a separate Q4_0 value kernel. The fused
  Q8_0 decode kernels remain unchanged.

## Consumer coverage

- Attention `q_a`, `q_b`, and `kv_a`: common typed dense dispatcher.
- Attention `k_b`: typed q/k-low-rank dispatcher.
- Attention `v_b`: indexed-decode typed dispatcher and typed batch-head value
  dispatcher.
- Attention output: common typed dense dispatcher for the first stage and the
  typed k-slice dispatcher for split `out_b`.
- Shared experts: Q4_0 takes the existing split gate/up, SwiGLU, and down path
  through the common typed dispatcher. The Q8_0-only fused pair kernels are
  selected only after `glm_graph_weights_are_q8_0`, so they are not Q4_0
  consumers.
- Leading dense FFNs: common typed dense dispatcher for gate/up/down. Q8_0
  fused gate/up remains guarded by the Q8_0 type predicate.
- Output head: common typed dense dispatcher. Token embeddings remain Q8_0 as
  requested.

## Q4_0 intentionally absent or unreachable

- `ds4_gpu_embed_token_quant_tensor` and
  `ds4_gpu_embed_tokens_quant_tensor`: intentionally remain Q8_0-only because
  embeddings stay Q8_0.
- Routed-expert IQ2/Q2K/Q4K kernels and dispatchers: not dense tensor
  consumers and intentionally untouched.
- Q8_0 fused shared-expert and attention-output helpers: their graph call sites
  are guarded by exact Q8_0 predicates; Q4_0 uses the typed split paths above.
- `ds4_gpu_glm_attention_indexed_batch_typed_tensor`,
  `ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor`, and
  `ds4_gpu_glm_k_b_project_typed_tensor`: existing CUDA stubs that do not read
  any weight format. The active indexed serving path uses the implemented
  qk-lowrank and indexed-decode dispatchers listed above. The split-group8
  stub also lacks Q8_0 behavior, so it is not a Q4_0 dense consumer.
- CPU/F32 reference and special first-token diagnostic paths with hard-coded
  Q8_0 checks are outside the GLM CUDA serving path and were not changed.

## Preservation of existing paths

All Q4_0 decode and launch code is separate. Existing Q8_0 and F16 kernels,
arithmetic, and dispatch calls were not modified; type switches only gained
new Q4_0 cases or Q4_0-only branches.

## Build status

`make cuda-spark -j2` completed successfully in the staged tree on 2026-08-16.
It rebuilt and linked `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
`ds4-agent` with no compile or link errors.

The patch also passes a `patch --dry-run -p1` check against the requested
engine tree. Patch SHA-256:
`48ea0809b827…`.
