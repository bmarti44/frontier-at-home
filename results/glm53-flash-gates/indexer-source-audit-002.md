# Indexer fixture and metadata source audit

Read-only follow-up by the persistent gap reviewer after convolution campaign
round 44. Sources are the pinned build-source-005 tree and inventoried runtime
headers. No GPU run, model result or measured workspace claim follows.

The physical indexer cache is uint8 `[pages,64,132]`, but each page is planar:
8,192 FP8 key bytes followed by 256 bytes containing 64 FP32 scales. Use the
actual page stride. Four requests need different seed-derived logical rank
permutations, as well as permuted physical block tables: identical logical
scores would hide wrong-request reads despite changed physical addresses.

With FP8 query/key channel zero equal to one, other channels zero, and 32 FP32
head weights equal to 1/32, each historical pool's logit equals its FP32 scale.
The SM120 DeepGEMM headers accept arbitrary FP32 scales, including the exact
dyadic values `(rank+1)/65536`. These are synthetic cache inputs, not a claim
that the selected compressor emits arbitrary scales. Sources:
`deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh:246–288` and
`sm120_fp8_paged_mqa_logits.cuh:274–320` in the packaged runtime.

The top-k op returns 512 pools without a rank-order guarantee. GLM slices the
first 511, expands those into 2,044 tokens, then appends only incomplete-tail
tokens. A correct oracle must accept any 511 distinct groups from the uniquely
ranked eligible historical top-512 set, require all four consecutive token IDs
in each group, and require exact tail/padding. At positions 262140, 262141 and
262142, append respectively one, two and three tokens beginning at 262140.
At 262143 there is no incomplete tail: the final four columns stay -1.
The newly completed pool is not automatically appended. Source:
`sparse_attn_indexer_kpool.py:589–607,863–894`; unsorted atomic output positions
are explicit in `csrc/libtorch_stable/persistent_topk.cuh:822–852`.

Use small phase-varying dyadic raw keys, zero gate scores and zero APE. Then
pooling is an exact four-way mean. The independent cache-byte reference is:
mean → BF16 rounding → normalized Hadamard-128 → BF16 rounding →
absmax clamped to at least 1e-4 → power-of-two scale rounding → FP8 E4M3 rounding.
Constant values across channels simplify the Hadamard to its DC component;
phase variation still distinguishes stale tail keys. All modified cache bytes
and unchanged history/padding must be checked. Prefill retains the last four
raw keys/gates per request; decode replaces one ring slot and compresses at
phase three (`ops/kpool_compress.py:137–255,368–409,495–609`).

Create indexer metadata through actual `AttentionGroup.create_metadata_builders`
with the cache probe's spec and `kernel_block_size=64`. The helper copies the
spec to storage block size 256, yielding 64 pooled states; the builder then
coarsens common 64-token block tables by four. Passing the raw 8,704-token
manager spec directly would bypass this worker adaptation. Use a separate
four-token tail spec/builder and distinct request tail blocks. Common metadata
stays token-granular with retained pinned CPU/device starts and lengths, exact
positions, block tables, slots and CPU length bounds. Sources:
`v1/worker/utils.py:267–308` and `v1/attention/backends/mla/indexer.py:983–1014,1142–1163`.

Old-history pool counts are 65,408 for four × 512 prefill, 65,024 for one × 2,048,
and 65,535 before decode begins at 262140. Independently rank only eligible
history and ensure inserted pools remain below its top-512 cutoff. These
full-address-space fixtures process no actual model context tokens.

Compilation spans Triton metadata/compression/tail/expansion, compiled vLLM
CUDA gather/top-k, and DeepGEMM native JIT logits/scheduling. Existing Triton
sealing does not cover DeepGEMM. Post-projection calls omit Inductor projection
leaves and do not qualify those operations. Python metadata uses device-side
construction rather than another bulk pageable H2D copy; supplied tensors must
use retained pinned staging. This does not establish pinning for every internal
DeepGEMM/compiler transfer. First execution must remain preparation-only.
