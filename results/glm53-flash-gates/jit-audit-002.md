# Model-free native cache confirmation seams

Read-only follow-up by the persistent adversarial reviewer. No GPU, compiler or
model execution; this is a plan for native falsifiers, not compilation closure.
Source pins remain those recorded by source-preparation-004/runtime dependency
receipts and DeepGEMM8b1392b9 (verified in prepared003 CMake dependency).

FlashInfer: `flashinfer.jit.mla.gen_sparse_mla_sm120_module().build_and_load()`
builds the monolithic SM12x module; the public sparse-MLA module getter registers
its callable ops. Exercise real GLM backend decode/prefill for T1,4,2048,
empty/short/full top-k and actual FP8 packing. Generic import is insufficient.
`JitSpecNvcc.try_load` uses AOT layout only: cached_ops output still routes through
ninja and fails with DISABLE_JIT. Stage the module into final package
`flashinfer/data/aot/<name>/<name>.so` using version-matched AOT tooling, or use a
separately reviewed startup cache-only loader. There is no AOT-directory env
switch. Module import opens flashinfer_jit.log, and miss takes FileLock before
checking DISABLE_JIT; separate writable logs/lock policy from immutable code.

DeepGEMM: use real vLLM `fp8_fp4_mqa_logits`,
`get_paged_mqa_logits_metadata`, and `fp8_fp4_paged_mqa_logits` wrappers with
checkpoint index dimensions and actual clean_logits choices. Paged FP8 KV is
planar inside each64-token page: all value bytes followed by float scales,
not per-token interleaving. Use the pinned test cast helper. Signature coverage
includes prefill Q/KV grid heuristics, paged next_n/head/dim/page/varlen and
metadata align(batch,32)/numSMs/varlen. Enumerate actual signatures before
claiming a small warmup covers the admitted scheduler domain.

A native patch is avoidable for the narrower requirement that no CUDA compile
occurs with weights loaded: preinitialize DeepGEMM's compiler before weights
(including nvcc --version), then enforce builder-owned cache/ancestors that the
actual dsv4 service cannot write, with no inherited writable descriptors. Every
miss must create cache/tmp/<new UUID> before either NVCC or NVRTC compile.
Omit tmp from a fresh sealed cache; an actual dsv4 missing-kernel probe must fail
at mkdir. Pair with a native warm hit and observe native nvrtcCompileProgram as
well as NVCC compile execution. Python subprocess monkeypatching is insufficient.

CuTe: inspected SM121 sparse MLA and DeepGEMM use native NVCC, but other selected
families must be audited. Direct compile paths can run in memory before cache
writes, bypassing FLASHINFER_DISABLE_JIT. Seal native object loading or reject
compile entrypoints before imports. Fresh-process object load and missing,
corrupt, unloadable, disabled-cache and unreadable-source controls must all
prove no compiler fallback. Read-only files alone do not establish this.

Prewarm after final runtime-path packaging: DeepGEMM compiler signatures and
CuTe source keys include absolute paths. Moving a warm runtime invalidates the
assumption that its cache covers serving. A later miss fails the attempt; unload,
prewarm the missing specialization, freeze anew and obtain a new public seed.
