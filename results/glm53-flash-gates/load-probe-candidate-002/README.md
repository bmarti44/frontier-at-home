# Component-load candidate2 / campaign41

Implementation0ec85be9 closes only review40 H1/H2. Genuine correction RED09786db2
retains both reproduced acceptance failures and one small test-fixture no-op
mutation, subsequently corrected to mutate upload_chunks to0.

H1: phase CUDA current/peak minima are derived from unique constructor/final
backing storage, simultaneous transfer plus diagnostic contiguous-copy overlap,
and finalizer copy overlap. Exact native GTensorCache geometry is mandatory.
Read-only CUDA driver attribute16 reports48 SMs on this GB10 (no context or
allocation); startup checks48 SMs and SM121. Pinned source defines8 SMs/expert,
so MoE concurrency must be6, with four exact shared scratch buffers. Pointer,
cache and scratch storage cannot overlap or alias parameter storage. RSS/PSS
must be positive and ordered. Two simultaneously live pinned buffers must be
represented in the observed pinned peak.

H2: the external scorer independently reconstructs the complete canonical
safetensors file hash/length, header and every selected/excluded payload from the
frozen generator and public seed. The actual retained file must match it. CPU
mutation tests now use a real, small canonical fixture with substituted geometry,
not a fake file plus mocked expected digests. Full geometry remains covered by
the original pinned-header fixture tests and actual meta constructor checks.

No GPU attempt, new runtime helper, production change or broader scope. Frozen
pinned-stream, fixture-generator and replay helper bytes remain unchanged.
Both persistent reviewers' final round40 assessments agreed on exactly two HIGH
findings and identified no additional loader/API high issue.
