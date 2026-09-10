# Native BF16 reference: bounded storage and layer audit

The owner has no reference dataset or second machine. The official
[GLM-5.3-Flash-BF16 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/a5b45eb41df6402735dedc900be14a42e8d5e538)
contains 642,646,653,816 tensor bytes across 120 shards. It cannot be loaded
whole into this host's available memory or copied into its current free disk
space. The measured host snapshot and exact API/index/configuration bytes are
retained. No weight payload was downloaded.

The repository's existing bounded HTTP range reader validated all 120 tensor
headers and their complete shape/offset coverage. The largest forward layer is
14,825,277,272 bytes; the larger MTP layer 45 is 14,865,185,408 bytes. The largest
checkpoint shard is 5,368,754,192 bytes. A conservative one-layer-plus-one-shard
sum is 20,233,939,600 bytes. That is tensor geometry only, excluding activations,
pinned staging, copies, attention workspaces and allocator overhead.

A reference-only layer-streaming approach therefore merits a bounded correctness
falsifier. It must preserve stock Transformers computation, checkpoint conversions,
FP32 exceptions, all four hyperconnection streams and cross-layer index state.
The native KDA fallback has substantial sequence-dependent FP32 workspace; the
final serialized prompt cap and measured containment must account for it.
Whole-shard hashes must be verified before tensors are used, and new device-copy
paths need persistent pinned staging. Casting EXL3 weights to BF16 cannot recover
the original reference weights.

No streaming implementation, real layer execution, equivalence, native logits,
fidelity or speed result is established here. The archive retains complete
metadata and the exact reused header reader. Prior reference-availability
failures remain unchanged.
