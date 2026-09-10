# Docker-isolated replay 012: unloaded FAIL

The owner stopped Docker and its socket; their inactive state and old daemon's
absence were independently verified. The unchanged profile's runtime/model
freeze and freshly seeded input preparation passed. The host then read one page
from swap before GLM loaded. Swap-out and used swap stayed unchanged during this
preflight. The fixed verdict is FAIL: no model, smoke, necessary window or
full-duration workload was started.

The same observed interval records containerd PID 2120/start ticks 1002 losing
4 KiB of VmSwap and gaining one major fault. This correlation does not establish
complete attribution or causation and does not exempt the failure. Containerd
is a separate service left running after Docker stopped. Further isolation is
proposed separately; it has not occurred in this attempt.

The original external observer completed all 900 censuses and 902 raw rows.
It starts after the broad baseline and before freeze completion; retained phase
markers cover the earlier sampled counters. One additional swap-in page occurred
after the failed preflight, in a later interval that also includes read-only
diagnostics. That event and all process-read errors remain in the raw evidence.
The closed census validator accepts the original stream and rejects six malformed
copies. There is no continuous per-cgroup accounting or model measurement here.

All frozen file bindings and actual prepared fixtures were verified. Every archive
member was read back against its source and scanned against stored model API keys.
The manifest binds the freeze, verified post-freeze public randomness, prepared
inputs, raw census, phase markers, exact commands, terminal state, frozen source
and publication code. Default/proxy/guard state is unchanged; this does not claim
a running production model. Prior failed attempts remain unchanged. Docker stays
stopped, containerd stays active, and GLM stays stopped. No context, fidelity,
durability, media or production-speed qualification follows from this attempt.
