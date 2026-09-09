# MLA falsifier candidate 1 review

Candidate 1 / campaign round 31; implementation `89ed9c48` with probe `de870bdb`.
The adversarial reviewer found no high/critical defects in the bounded analytic
falsifier and independently checked final-element corruption, backend shapes,
and MAX_JOBS=2 / FlashInfer NVCC default1. Eight focused CPU tests passed.

The gap reviewer identified H1/high: an authoritative combined MLA PASS would
violate the frozen-kernel-binary contract. The probe calls Triton JIT remapping
and FlashInfer build_and_load; newly generated cache files are outside the
pre-freeze runtime inventory. A declared sealed-JIT exclusion does not establish
binary provenance. Source/runtime hashes are insufficient for that claim.

Required narrow correction: the first invocation is explicitly preparatory,
with NO_RESULT for kernel binary qualification even when analytic and host
checks pass. Failed checks still produce FAIL. Preserve generated cache/build
inventories after cleanup. Authoritative confirmation requires a separate
prewarm/freeze/sealed-cache candidate and a new public seed. No GPU run occurred.
