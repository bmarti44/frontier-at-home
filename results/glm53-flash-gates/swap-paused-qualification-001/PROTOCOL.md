# Current-profile qualification after the owner swap pause

Resume the current four-slot profile at 512/128 with all weights, cache budgets,
startup flags, authentication/default and lifecycle safeguards unchanged. Require
the native probe's process/cgroup cleanup and at least 110 GiB available before
either full-model load. The owner has paused the existing swap unit; require zero
SwapTotal and no active swap entries. Never run two large models together.

The freezer reuses the latest containerd-isolation recipe with a fresh output,
explicit swap-off admission and the reviewed current-scheduler direct adapter.
It verifies the closed model/runtime inventories and prepared cache, keeps the
existing two-GiB disk admission, and freezes all direct/durability source bindings.
Do not reuse consumed soak-freeze-014 or its failed preflight. Capture a new broad
host baseline and the existing external observer, freeze clean source, obtain a
later verified public seed and prepare inputs while unloaded.

Attempt direct context first, using context-scheduler-003/adapter.py with the
frozen directory for every action. Prepare the unchanged explicit instructions;
bind actual launch to planned arguments; run the 4,224-token startup correctness
check; then run four concurrent 250,128-token inputs directly. All existing
retrieval/negative-control, actual token, stream-overlap and no-preemption checks
remain mandatory. Host, identity, kernel, compiled-input and terminal checks are
separate required gates. A short check is never context capability.

Use a separate fresh server for durability, keeping the same declared profile.
Reuse soak-scheduler-002/candidate.py preparation, smoke, fixed 20-request window
and full 30-minute workload. Admit the full workload only after the window and
host/frozen-cache checks pass. Preserve every failed or null attempt.

Both use the existing named-profile launcher and hardened wrapper: 92/94 GiB
containment, zero cgroup swap, 18 GiB whole-host kill floor, continuous identity
and memory monitoring, and the existing 9,000-second server limit. No timeout,
memory, fidelity or acceptance limit changes. Restore the exact swap unit after
terminal and postartifact campaign observations as agreed with the owner.

Even passing these operational gates does not supply the missing 100-case native
BF16 reference, maximum-media confirmation, production switching/rollback or
qualified production-speed comparison. GLM remains optional and non-default.
