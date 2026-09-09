# Runtime compilation audit

This is a source/control-flow checkpoint, not a serving qualification. The
vLLM source build continues independently. No GLM weights have been loaded.

Both persistent reviewers approved the Triton-only component at `994c6666`
with zero critical/high findings. Eight synthetic cache tests and five tests
of installed Triton control flow passed independently. The low-priority
retained-original-loader-alias regression was then committed at `e63aea84`;
all six installed-source tests pass. No real compiler/GPU kernel runs in these
tests. The successful-cache case mocks `CompiledKernel`, so native artifact
loading still needs a real fresh-process gate.

`scripts/lib/glm53_runtime_jit.py` remains uninstalled and unselected. Its
startup selector defaults off. When selected it binds a verified closed cache,
rejects force-compile/override/dump/remote-cache/compiler-hook settings, seals
the dangerous knobs, preserves native load errors and blocks native compiler
fallback. The lifecycle must activate it in every spawned worker before
device/helper initialization and make cache contents immutable to that worker.

## Remaining compilation paths

| Path | Pinned-source observation | Required closure before a large load |
| --- | --- | --- |
| FlashInfer sparse MLA | `flashinfer/jit/core.py` honors `FLASHINFER_DISABLE_JIT`, but nvcc cache lookup accepts the AOT layout, not an ordinary cached JIT `.so` | Prebuild `gen_sparse_mla_sm120_module`, stage with version-matched `copy_built_kernels` into a new AOT directory, freeze, and test fresh-process load with JIT disabled |
| FlashInfer CuTe | Direct `cute.compile` sites bypass that switch | Prove actual selected paths avoid them, or add a separately reviewed cache-miss guard |
| DeepGEMM | K-pool prefill/decode indexers call its compiler-backed functions; cache misses can use NVCC or in-process NVRTC | Prewarm actual specialization buckets and enforce failure before either compiler on a cache miss |
| ExLlamaV3 | Missing native extension can fall back to `torch.utils.cpp_extension.load` | Require native extension import and immutable inventory before weight allocation |
| Torch Inductor | Independent compilation path | Initial mode0 is a bounded baseline option; the intended compiled fast path needs separate cache/guard closure |

DeepGEMM's `Compiler.build` creates a new temporary directory before NVCC or
NVRTC. An enforced read-only cache mount would reject misses early. Two local
user-systemd probes (`readonly-namespace-001` and `002`) did not enforce that
mount: the sentinel remained writable, including with `PrivateUsers=yes`.
Their failed attempts are retained. Thus that user-manager mount route is
`NO_RESULT` for cache sealing; a property declaration cannot authorize loading.
Alternatives are a verified production namespace/ownership boundary or a
startup-selected native cache-only guard. Do not broaden delegated controls.

Prewarm must cover sparse MLA; K-pool compression/tails/index conversion;
vendored FLA KDA, convolution and gated norm; DeepGEMM contiguous/paged indexers;
the actual vision backend; shared normalization/RoPE/sampling; and native
driver helpers. Include slots1–4, admitted media geometry, scheduler chunks
through2048 and all context/page/stride/constexpr buckets through262144.
This list identifies families, not already-known complete specialization keys.

A fresh-process warm-cache success must be paired with deliberate unknown,
missing, corrupt and unloadable artifacts. Compiler stages, C compiler,
NVCC, NVRTC and CuTe compilation must remain uncalled on rejection. A later
cache miss fails that model attempt: unload, extend prewarm without a resident
model, freeze a new candidate and retry.

Source identities: `configs/build-manifests/glm53-runtime-dependencies.json`
(Triton3.7.1, FlashInfer0.6.18rc10, CUTLASS DSL4.6.2), source-preparation-003,
and pinned DeepGEMM commit from vLLM's CMake dependency. Exact seams:
`triton/compiler/compiler.py:265–294`, `triton/runtime/build.py:78–93`,
`flashinfer/jit/core.py:299–319,396–410`, `flashinfer/aot.py:868–883`,
`vllm/model_executor/layers/sparse_attn_indexer_kpool.py:539,802`, and
DeepGEMM `csrc/jit/compiler.hpp:100–120`.
