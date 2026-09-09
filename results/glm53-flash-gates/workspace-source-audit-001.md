# Remaining workspace source audit

Read-only preparatory audit by the persistent gap reviewer against pinned
build-source-005/vllm/vllm. This is source-derived sizing, not measured allocation,
correctness, memory fit or a context result. No source edits or GPU calls.

The actual GLM indexer path is SparseAttnIndexerKpool/sparse_attn_indexer_kpool
(model_executor/layers/sparse_attn_indexer_kpool.py:261,950), with metadata from
DeepseekV32IndexerMetadataBuilder.build(v1/attention/backends/mla/indexer.py:954).
Logical8704-token cache-manager blocks become64-entry pooled kernel pages
(models/glm5next/nvidia/attention.py:123). Use the qualified cache derivation.

Shared gather arena:40*262144 rows, FP8[rows,128]+uint8[rows,4], exactly
1,384,120,320 bytes. Profiling reserves another1 MiB radix workspace in that
arena. WorkspaceManager shares one arena per ubatch/lane across layers; extra
lanes multiply it. Model-wide top-k output[2048,2048]INT32 is16 MiB.
Each of11 indexer layers caches512 KiB FP32 head-gate weights[4096,32].

Endpoint prefill:four requests*512 query rows or one request*2048, ending262143.
Four full requests each have65536 pools. The real512 MiB chunk splitter uses
two M1024/N131072 chunks; assignment of the next logits tensor happens before
the previous iteration's tensor loses its reference(kpool.py:482,542). Thus
measure two-logit overlap, not just the nominal512 MiB chunk cap. Together with
the gather arena this can exceed2.28 GiB before inputs/outputs/runtime overhead.

Actual indexer inputs:q_quant[T,32,128]FP8,q_scale=None,weights[T,32]FP32;
k and gate_score[T,128]BF16;compress_ape[4,128]FP32;
tailcache[blocks,2,4,128]BF16 with distinct request blocks;topk2048,index_kpool4,
head_dim128,max_model_len262144,total_seq_lens10485760. Decode batches1–4,
next_n1, positions262140–262143 exercise incomplete and completed pools.
Execution includes compression/tail, cache gather, DeepGEMM logits, top-k and
pool expansion. The checkpoint has32 indexer heads; ignore stale source comments.

Convolution: actual GLM calls causal_conv1d_fn and causal_conv1d_update
(models/glm5next/nvidia/kda.py:449–478;definitions layers/mamba/ops/causal_conv1d.py:481,1096).
Allocate BF16[T,24896], then take[:,:24576] to preserve real fused-projection
strides. Weights FP32[24576,4] are384 KiB per KDA layer (34=>12.75 MiB).
Prefill output[24576,2048]BF16 is96 MiB. State[5,24576,3]BF16 is720 KiB,
including protected null state plus four active states, already counted in the
qualified cache. Use compute_causal_conv1d_metadata(backends/utils.py:1042),
retain its buffers and pass real metadata rather than selecting a different
CPU-copy path with None. Absolute position is absent from the convolution kernel;
a valid continuation state is not evidence of processing262144 tokens.

No single allocation above independently falsifies the remaining9.2 GiB
metadata overhead envelope. Proposed next work is independent full-geometry
convolution/indexer preparation and sealed replay, with full output/state checks.
Incremental model-loader overhead remains separate and unmeasured.

Follow-up layout check: mamba_utils.get_conv_state_layout defaults to SD, and the
qualified cache backing has shape [state_blocks,3,24576]. GLM transposes the last
two dimensions before convolution (models/glm5next/nvidia/kda.py:375). Therefore
construct backing [5,3,24576] and pass its [5,24576,3] view with strides
[73728,1,24576], rather than substituting contiguous [5,24576,3] state. This
changes neither the 720 KiB byte count nor the scope of the proposed probe.
