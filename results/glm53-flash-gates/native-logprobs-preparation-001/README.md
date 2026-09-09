# Native probability check prepared

One non-final `fit-0000` window from the pinned native BF16 teacher archive was
fully downloaded and hash-verified. Its 2,048 integer input tokens and 2,047
full-vocabulary FP32 output rows align. A CPU-only reduction computes target
NLL and top-1 correctness, retaining every position and the extraction source.
The 1.27 GB reference remains local; its immutable URL and complete hash are
preserved for independent reproduction.

The prepared capture uses one ordinary `/v1/completions` request and the
existing `glm53_prompt_metrics` reducer. It checks that the server is idle
before submitting. No native candidate probability request has been sent;
the four-slot context test is still running. Focused review checked alignment
and arithmetic, and the optimized-Python bypass was corrected before use.

This is a plumbing diagnostic on a non-final window, not quality acceptance.
The native public reference currently supplies only 25 final windows, whereas
the unchanged gate requires 100. The alternate shared-head data has sufficient
window count but a known nonzero native/replay discrepancy. Neither reference
currently establishes the required 100-case native qualification; do not split
windows or mix calibration roles to manufacture coverage.
