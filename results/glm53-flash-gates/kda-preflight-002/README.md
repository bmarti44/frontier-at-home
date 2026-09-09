# KDA preparation 002 — NO_RESULT

All five analytic cases completed: one through four decode rows and four
512-row prefill sequences. Every output and recurrent-state element passed the
fixed absolute-plus-relative tolerance; none was nonfinite or out of bounds.
The prefill maximum absolute difference was 0.01615762710571289 and decode
maximum was 0.0015413761138916016. These are synthetic analytic comparisons,
not model fidelity results or an adopted quantization delta.

The complete frozen runtime/fixture/seed and host/identity checks passed.
Lowest MemAvailable was 118,751,200 KiB; cgroup peak was 1,612,734,464 bytes.
There were 186 memory samples and 204 identity samples, no new swap, and
verified process-group/cgroup cleanup. All outputs and generated state are
retained. The earlier metadata failure remains unchanged as FAIL.

The combined verdict is NO_RESULT because 865 generated cache files were
inventoried after execution. Compiled kernels still require separately frozen
replay with a new public seed. No model weights were loaded and zero actual
input tokens were processed. Convolution, general KDA fidelity, full model
workspace, serving and long-context capability remain unqualified; performance
is not measured. The pinned-staging receipt covers the probe-owned buffers
only; upstream's bounded pageable chunk-index copy remains an explicit
large-serving prerequisite.
