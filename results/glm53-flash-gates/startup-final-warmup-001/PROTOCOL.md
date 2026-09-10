# Bounded reclamation after final engine warmup

The retained replay 006 reached authenticated READY but failed the host gate:
three pages swapped out during startup, before any test workload was admitted.
All 2,484 prepared kernel inputs stayed unchanged. The observed interval overlaps
API multimodal processor warmup, after final engine warmup completed. A separate
six-page swap-out occurred later during unloaded artifact verification. These
observations support testing transient memory reclamation; they do not attribute
all swap to the model or authorize a weaker host gate.

First bounded alternative: release unreachable objects and unused CUDA allocator
blocks immediately after the parent `compile_or_warm_up_model` returns, before
the API processor warmup. Select it only through the exact startup flag
`GLM53_RELEASE_WARMUP_CACHE=1` in the experimental full-context profile. Resolve
that choice at module initialization. Disabled mode inherits the parent method
unchanged; it performs no extra diagnostic calls. Preserve the parent's return
object and propagate failures. The enabled startup method emits one observation
of reserved bytes before and after reclamation. External timestamped memory,
process identity, swap and cgroup samples remain the host acceptance evidence.
No method on the serving token/layer/expert path changes.

The production-path CPU test must first fail on unchanged code because the unused
warmup allocation is not released before return. Require exact disabled method
inheritance, exact flag selection at initialization, preserved parent return and
failure propagation. Commit genuine RED before the smallest implementation.
Keep all prior worker cleanup behavior and closed scorer files byte-identical.

Keep 512/128 scheduling, four 262,144-token slots, aggregate 1,048,576 context,
weights/tokenizer/precision, KV reservation, media settings, containment, timeouts,
inference lock, host floor and existing startup flags. Do not change the agent
profile, production profile, default, global VM settings or privileges.

After clean commit/review, freeze all inputs and obtain a later verified public
seed. Require startup no-swap admission, unchanged prepared caches, authenticated
correctness and the fixed necessary window before the unchanged full-duration
gate. Any new host swap, failed correctness, timeout, missing identity, memory
floor violation or changed compiled input remains a failed or null attempt.
A reserved-byte decrease alone is not model qualification or a speed result.
Direct context, native paired fidelity and promotion remain separate gates.

If this bounded alternative does not close the observed host finding, preserve
its measured result. The second bounded alternative is process-local libc heap
trimming after existing startup garbage collection, with its own prospective
production-path test, exact default-off flag and separate freeze/seed. Do not
combine unresolved variants or change model mathematics. If neither helps, record
this startup-reclamation branch as NO_RESULT and retain the exact open finding.
