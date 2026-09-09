# DeepGEMM indexer compilation audit

Read-only inspection by the persistent adversarial reviewer after convolution
round 44. No GPU execution, compilation or edits. This is a source-derived
preparation plan, not proof of a sealed DeepGEMM runtime.

The source is the clean `build-source-003/vllm/.deps/deepgemm-src` worktree at
`8b1392b978f5a03c828dd1711090d7fb50958b8a`, matching vLLM's CMake pin. The installed
extension is `vllm/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so` in
native-runtime-002, already bound by the complete runtime inventory.

DeepGEMM has its own C++ JIT. `DG_JIT_USE_NVRTC=0` selects NVCC by default;
there is no native reject-cache-miss startup switch. Set `DG_JIT_CACHE_DIR`
explicitly before vLLM lazy initialization. Otherwise vLLM selects its cache
root's deep_gemm directory; the native fallback is `$HOME/.deep_gemm`.
Existing Triton sealing does not cover this compiler.

`csrc/jit/compiler.hpp:101` uses `<DG_JIT_CACHE_DIR>/cache/kernel.<name>.<digest>/`.
Required files are `kernel.cu` and `kernel.cubin`; diagnostics may add PTX/SASS.
Native validity checks existence, so an independent closed hash inventory is
mandatory. The custom digest covers name, compiler signature, flags and generated
code; code incorporates a recursive deep_gemm header digest. That parser does
not independently bind all CUTLASS/CUDA headers. Freeze complete installed
includes and toolchain inputs before authoritative preparation. Flags contain
absolute include paths: moving the runtime/toolchain can change cache keys even
when relocating only the cache root would preserve them.

NVCC initialization executes `nvcc --version` before cache lookup. On this CUDA
13 configuration, SM121 maps to `120f`; compilation uses compute_120f/sm_120f.
NVCC source/output handling is at compiler.hpp:219; NVRTC handling is at
291–321; compiler selection is at361. Freeze compiler selection, override,
C++ standard and debug/PTXAS/line-info settings. The extension links NVRTC even
when NVCC is selected.

Required kernel families are `sm120_fp8_mqa_logits`,
`sm120_paged_mqa_logits_metadata` and `sm120_fp8_paged_mqa_logits`
(`csrc/jit_kernels/impls/sm120_mqa_logits.hpp:152,379,512`). Production uses
`clean_logits=False`. Keys bind geometry, tiles, SM count and relevant decode
modes; sequence lengths are largely runtime inputs. Warm the exact selected
specializations instead of assuming every input shape creates a new binary.

The prospective sealing seam is before compiler execution: after a cache miss,
Compiler::build must create `<cache-root>/tmp/<uuid>`. A builder-owned closed
cache denying writes to the actual service identity, with tmp absent, can stop
that path. A verified immutable regular-file tmp sentinel can additionally force
ENOTDIR. Owner-controlled chmod alone does not establish the service boundary.
Use the reviewed access/credential checks, verify ancestors/ACLs/descriptors,
and prove missing-key behavior in fresh processes. This allows the constructor's
version query; it does not claim zero compiler processes. Test missing kernel
keys/cubins, unwarmed specializations and changed cubin bytes before adoption.

No internal tensor H2D copy was found in these SM120 logits/metadata paths.
Metadata is produced on device; TMA descriptors are host launch arguments;
paged from_blob views do not copy device storage. Keep every supplied tensor in
persistent pinned staging and preserve this limited scope. Paged inputs require
FP32 weights, no query scale, contiguous context lengths `[B,next_n]`, 64-state
pages, and scheduling metadata `[SM_count+1,2]`. Packed pages contain all values
before all scales, as recorded in indexer-source-audit-002.md.

Native lazy device initialization also allocates a **32 MiB cuBLASLt workspace**,
which belongs in the actual measured indexer envelope. The source worktree and
installed includes are reproduction inputs; do not remove them as build debris.
