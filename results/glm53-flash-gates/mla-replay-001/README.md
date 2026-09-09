# Frozen MLA replay 001

PASS for the model-free frozen constant-cache replay only. All five query-row
cases (1, 2, 3, 4 and 2,048) completed; every captured BF16 element matched the
analytic reference exactly. Complete runtime and compiled-kernel inventories
verified before and after execution, with a fresh BLS-verified post-freeze seed.
The selection receipt binds the preloaded FlashInfer module and sealed Triton
cache to the frozen bundle. The MLA001 preparation remains NO_RESULT.

Host and continuous identity checks passed. Lowest MemAvailable was
117,539,580 KiB; cgroup peak was 2,056,151,040 bytes. There were 78 memory
samples and 82 identity samples, no new whole-system swap, and verified empty
process-group/cgroup cleanup. Raw outputs and generated state are retained.

No model weights were loaded and zero actual input tokens were processed.
This does not qualify general attention fidelity, full-model memory, production
cache immutability, serving, context capability or performance. Diagnostic
kernel timings remain in raw evidence and are not headline performance values.
