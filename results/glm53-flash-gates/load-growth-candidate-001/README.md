# Two-layer growth candidate 1 / campaign round 42

Implementation:3965efa8. Fixed acceptance:configs/decision-specs/glm53-load-growth.json.
Probe RED:dd046700; controller RED:ea15939c. All225 scoped CPU tests pass with
the packaged interpreter and -I -B. Diff whitespace and commit secret scans pass.
The diff was inspected for canned results, prompt detection, disabled assertions,
timing manipulation and production activation. No such branches were introduced.

Scope:43_probe_glm53_growth.py and tests; growth-only39_freeze/39_run routing;
fixed growth decision. The proven42 component API, fixture generator, pinned
stream and replay helpers remain byte-identical to their signed-off versions.

Two independent full MoE modules remain owned in one process, with one Capture
readback buffer and the same canonical public-seed input. Verify all bytes at
transfer/loaded/final stages, then re-read module one after module two finalizes.
Parameter owners and distinct Python/native handles are checked; nine pointer
tables per module target their own handles. Both modules share precisely the
same four scratch and two GTensorCache allocations with stable global key coverage.
Known memory is a union, not a duplicated shared allocation. Phase counters must
cover the previous module, new constructor and simultaneous upload temporary.

Four focused CPU tests include full metadata geometry and semantic evidence
mutations. Only the already-qualified canonical-byte generator is mocked in
that CPU census test; it is explicitly required to be called. The actual scorer
always independently regenerates the complete canonical input inventory through
the frozen42 API and verifies retained bytes. No CUDA or payload claim follows
from CPU fixtures.

Report observed incremental RSS/PSS and CUDA allocation deltas only. No linear
full-model extrapolation, cold I/O, forward, fidelity, context or performance
claim. No additional GPU run has occurred yet. Model admission remains closed.
