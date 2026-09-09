# Indexer preparation 003 — NO_RESULT

All six full-address-space synthetic cases and the complete host/identity checks
passed. The combined result remains NO_RESULT because this attempt generated
kernels after freeze; separate sealed confirmation is required. This is not
model fidelity, actual context-token processing, full-model fit or serving speed.

The independent scorer checked 268,434,423 valid FP32 logits byte-for-byte,
selected token groups and exact tail padding, and all 41,648,640 cache bytes plus
296,960 tail bytes per case. Every comparison passed. Masked/uninitialized logit
columns are absent from the artifacts. Cases cover one through four decode
requests and both 2,048-row prefill shapes (one request and four × 512).

Source 00a2fcf3, freeze 1788974266.0447047, drand round 6451442 published at
1788974280, seed 7631521019026542407. The 61,353-file runtime and all frozen CUDA
compiler, metadata, code and configuration inputs match before and after.
Generated state has 65 files / 1,204,234 bytes: 57 Triton files, six DeepGEMM
source/cubin files, one model-info JSON and one zero-byte Humming lock.

Minimum available memory was 114,051,640 KiB, with 386 memory samples and 420
identity samples. Whole-system swap-in/out deltas and cgroup swap were zero.
The controller verified successful completion and empty containment. Cgroup
peak was 2,335,010,816 bytes; it is below peak live CUDA allocation, confirming
that cgroup accounting alone is insufficient for this UMA host.

The actual profiling arena is 1,385,168,896 bytes. Four-request prefill observed
3,179,186,176 peak Torch CUDA allocation bytes, including two overlapping
512 MiB logit buffers. These are component evidence-process observations, not
an additive full-model memory estimate or a production performance result.

## Reconstruct the exact captured files

The three largest gzip captures are additionally compressed with XZ so every
repository file stays below hosting limits. This preserves the original gzip
bytes and hashes; it does not regenerate scores or alter raw evidence.
Run `python3 restore_artifacts.py /absolute/fresh/output-directory` from this
archive directory. The helper checks every source and reconstructed digest.
`archive.json` records original and archive sizes/hashes. Generated compiler
state is preserved separately in the verified `state.tar.gz` bundle.
The original complete attempt remains at its recorded local path. Failed
attempts 001 and 002 remain unchanged.
