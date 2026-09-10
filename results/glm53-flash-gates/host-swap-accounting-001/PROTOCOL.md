# Passive cgroup swap accounting and unloaded control

Attempts 008 and 009 remain FAIL; the two startup reclamation alternatives remain
NO_RESULT. Current kernel counters cannot retrospectively attribute those events.

Read the installed kernel's `memory.stat` pswpin/pswpout counters at root,
init.scope, system.slice, user.slice, docker.service and snapd.service. Preserve
the entire raw stat file, cgroup device/inode, boot identity, observer identity,
source/configuration hashes and global before/after brackets for each sample.
No service queries or changes, model loads, pressure generation, swap-policy
changes, privileges or production imports. Run the explicit external tool in a
fresh 64/128 MiB zero-swap cgroup, with a wall timeout 30 seconds beyond capture.

First run a fixed 120-second unloaded control. Integrity requires all 121 ordered
samples and the terminal record, unchanged identities, finite timestamps,
nonnegative counters, and no resets or read errors. The quiet-interval check is
true only if every global swap-in/out and used-swap observation stays at its
initial value. Capture failure is FAIL. Even an intact quiet observation is
NO_RESULT for model qualification and causal attribution. Missing data is never
zero. Reject malformed, missing, duplicate and reset observations with CPU tests.

Report each group's endpoint deltas separately. Never sum nested groups or claim
the selected branches exhaust the hierarchy. Cgroup accounting ownership is
different from the process responsible for pressure; ownership can persist after
process migration. The reads are not atomic and residuals remain unknown. A quiet
preflight may admit a new frozen model attempt under the same broad baseline and
unchanged host gate; it cannot excuse any later paging or earlier failure.

Counter definitions and ownership limits: [Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)
and [memory ownership](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-ownership).
