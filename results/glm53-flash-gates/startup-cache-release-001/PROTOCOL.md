# Startup cache-release falsifier

Observed defect: soak-startup-001 was rejected by the existing host wrapper with
exit16 after a kernel NV_ERR_NO_MEMORY record in dummy profiling. API readiness
succeeded, but no durability fixture or timed request was admitted. Driver logs
do not establish which allocation failed or prove fragmentation.

Candidate1 changes only startup allocation cleanup: when the exact startup flag
GLM53_RELEASE_LOAD_CACHE=1 is selected, collect unreachable objects and release
unused accelerator allocator blocks before the existing parent profiling call.
Retain the existing post-profile cleanup. Resolve the flag once during module
initialization; default off. No live tensor is freed or rewritten, no budget is
changed, and no token/layer/expert code is added. Select the flag only in the
experimental1M profile and log it through the existing launch environment.
Agent-fast and production profiles remain off. No default/switch promotion.

CPU acceptance: the real worker method fails a simulated retained-load-allocation
profiling precondition on unchanged code; enabling the candidate closes that
condition, preserves the returned budget and live object, and retains cleanup
after profiling. Default-off order remains identical. Environment is read at
initialization and cannot change behavior during later calls.

Actual acceptance: clean source and reviewed candidate; unchanged model/runtime,
context(4x262144), KV budget, prepared kernels, scorer and durability protocol;
fresh source/configuration freeze and verified later public seed. Start through
the named profile in fresh existing containment. Require exact logged flag,
authenticated READY, a correct4224-token retrieval startup fixture, no new kernel
OOM/Xid, no swap, external memory>=18GiB, and compiled inputs unchanged. Any
failure stops the attempt and remains preserved. If startup passes, run the
already fixed30-minute native durability protocol unchanged. Require clean
identity-verified stop, empty cgroup, memory>=110GiB and default/rollback unchanged.
No speed claim or fidelity adoption follows from this startup-only falsifier.

If candidate1 is falsified, the second bounded hypothesis is advice to release
verified model-file page cache after hashing and before loading, without root or
global cache changes. Do not implement allocator fractions or garbage-collection
thresholds without measured allocation/reservation evidence. Neither hypothesis
is a diagnosed cause or guaranteed fix. Preserve NO_RESULT if neither works.
