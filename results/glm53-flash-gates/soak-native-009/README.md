# Startup CPU heap trim 009: FAIL; reclamation branch NO_RESULT

The second bounded startup reclamation experiment released 8,552 KiB of anonymous RSS before profiling and 544 KiB afterwards. Authenticated startup, the actual worker's frozen libc mapping and a 4,224-input-token correctness smoke passed. All 2,486 prepared kernel files stayed unchanged.

The host read one page from swap during the smoke. Used swap and swap-out counters stayed unchanged; GLM cgroup swap and memory-limit events remained zero. The fixed verdict is FAIL, so no necessary window or full-duration workload was admitted. The external census retains a simultaneous dockerd VmSwap decrease of 4 KiB and major-fault increase; this correlation does not establish causation or exempt the run from its gate.

GLM stopped cleanly, memory recovered, and runtime/model, identity, default/proxy/guard checks passed. Neither bounded reclamation alternative reliably closed the host finding, so this startup-reclamation branch is NO_RESULT. No third allocator variant is proposed. Read-only daemon/timer observations after shutdown are diagnostic context, not qualification or changes to the services. The prior million-token result used a different configuration; this smoke is not a context, paired fidelity, durability or production performance result.

The census also recorded 17 swap-in pages after GLM had stopped, during the
retained systemctl/timer/journal/Docker status queries. PID 1 had a concurrent
VmSwap decrease and major faults; this does not establish complete attribution.
These later diagnostic observations are separate from the one-page loaded failure.
The existing legacy engine guard's circuit breaker was already open and prevented
restarts of the unloaded legacy engine. The checks above establish unchanged
default/proxy/guard state, not a running production model or a reset of that breaker.
