# Tiny native loading/parity preparation 002: FAIL

The deterministic loading-info serialization repair passed. All synthetic loaded
parameters and buffers matched the predeclared BF16/FP32 bytes exactly. The first
forward then failed because the installed FLA package selected a Triton kernel on
a CPU-only process, despite hub kernels being disabled. All original artifacts
and the traceback remain in the archive. No forward-equivalence verdict or native
GLM reference result exists.

The next bounded falsifier may explicitly select the pinned stock Torch fallback
functions in both tiny arms. It must bind their original source, demonstrate the
selector reaches those functions, and keep the same loading/conversion, per-stage
meta residency, tensor equality and sequence-length assertions. This selector is
confined to the evidence script; no serving files or installed runtime change.
