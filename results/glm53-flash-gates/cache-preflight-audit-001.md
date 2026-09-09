# GLM cache allocation seams

Source-only audit of prepared004 and the pinned K2 checkpoint configuration.
The persistent gap reviewer independently verified config bytes against the
pinned Hub API receipt. No GPU, model or allocator measurement occurred.
Processor/tokenizer bytes are independently verified in processor-metadata-002;
processor-metadata-001 preserves the failed initial LFS-pointer comparison.

For four262144-token slots at TP1, the raw persistent payload lower bound is:

| State | Formula | Bytes |
| --- | --- | ---: |
| MLA,11layers | `11*4*262144*656` | 7566524416 |
| Index,pooled4 | `11*4*(262144/4)*132` | 380633088 |
| KDA,34layers | `34*4*(64*128*128*4+3*64*128*3*2)` | 590479360 |
| Index tails | `11*4*2*4*128*2` | 90112 |
| Total | | 8537726976 |

This is approximately7.9514GiB, not a memory-fit claim or an admission budget.
It excludes allocator padding/reserve, workspaces, graphs, weights, weight-load
peaks, vision and retained scheduler/encoder outputs. Keep the conservative
estimated profile until measured preflight supports a new frozen envelope.

The real model-free allocation seam is `v1.worker.utils.allocate_kv_cache`.
Build specs using the normalized configuration and the model's
`get_mamba_state_shape_from_config`/`get_mamba_state_dtype_from_config`, then use
`get_kv_cache_groups` and `get_kv_cache_config_from_groups`. The GLM grouping
aliases MLA/Mamba and index/tails. Count unique backing storage once; descriptor
sizes refer to the same shared buffer. Resolve the real manager block size,
including KDA state and SM121 index alignment256, and account for a null/reserved
block plus admission capacity for four complete requests.

Run actual `FlashInferMLASparseSM120Impl.forward_mqa` with populated packed KV
via `do_kv_cache_update`, real metadata/request IDs/block tables, top-k2048 and
one-layer BF16 queries `[T,64,512]`. Cover1–4 decode tokens and prefill through
2048 plus the final allocated block. Logical rope dimension0 is padded by the
selected implementation. Sparse MLA shared workspace defaults to394MiB.

`SparseAttnIndexerKpool.forward_cuda` must receive real cache/tail metadata:
its profiling branch allocates scratch and returns before the logits kernels.
Measure gather buffer132bytes/entry, radix top-k1MiB, actual DeepGEMM logits,
compression/tails and shared workspace once. Initial default logits profiling
reservation512MiB is not a measured peak.

KDA uses `chunk_kda_with_fused_gate`/`fused_recurrent_kda`, BF16 Q/K/V and raw
gate `[1,T,64,128]`, beta `[1,T,64]`, FP32 states `[4,64,128,128]`. Exercise
2048-token prefill with four boundaries, retained-state continuation, decode,
gather/scatter and causal convolution. Metadata-free warmup returns early and
cannot prove readiness. Explicit `@torch.compile` leaves exist even when vLLM
compilation mode0 is selected; they need separate compilation closure.

Pinned processor defaults allow8000 image tokens and240000 video tokens before
processor-specific clamps. Do not treat vision `image_size=448` as the admitted
maximum. Derive shapes through `_pixel_budget`/`smart_resize` and the patched
16-frame sampler. Cover four images or one16-frame video per request and four
simultaneous requests. One representative vision block can measure scratch,
while real vision weights and retained encoder outputs remain separate costs.

Source roots: prepared004 `vllm/vllm/`; cache layouts in
`model_executor/layers/attention/mla_attention.py`,
`model_executor/models/glm5next/nvidia/attention.py`,
`model_executor/layers/mamba/mamba_utils.py`, `v1/core/kv_cache_utils.py`;
kernels in `v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`,
`model_executor/layers/sparse_attn_indexer_kpool.py` and
`third_party/flash_linear_attention/ops/kda.py`.
