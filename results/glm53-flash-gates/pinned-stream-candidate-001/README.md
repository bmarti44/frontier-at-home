# Persistent pinned stream candidate 1 / campaign round 39

Fixed acceptance: configs/decision-specs/glm53-pinned-stream.json. Initial test
0604fdde, genuine RED aab43772, implementation66f91572. Six focused tests and
207 scoped CPU tests pass on the actual packaged interpreter. Real safetensors
and stock lazy iterator run in the selection tests; CUDA allocation/event/pin
operations are mocked only in CPU byte-contract tests. CUDA remains uninitialized.

The helper is uninstalled and default-off. It uses a local copy of the stock
iterator globals to filter selected exact names before get_tensor; shared
loader globals are unchanged. Full-file inventory hashing is explicit I/O and
selection does not claim zero readahead. One pinned byte buffer/event is reused,
with completion before each overwrite, full byte digests and synchronous consume.
The callback contract forbids retaining the CUDA temporary or its views; actual
module tests must verify the resulting storage and destination-copy overlap.

This candidate does not qualify GPU DMA, full model loading, module finalization,
internal pointer-table pinning, memory fit, fidelity or production performance.
Those require a separately frozen contained actual loading probe. Frozen KDA/MLA
orchestration and evidence are outside this review scope.
