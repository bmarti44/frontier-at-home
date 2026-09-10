# Prepared-kernel replay 006: FAIL at host startup gate

The unchanged 512/128 profile reached authenticated READY and stopped cleanly. All 2,484 prepared kernel files stayed unchanged. No smoke, necessary window or full-duration workload was admitted: the host wrote three pages to swap during startup, increasing used swap by 12 KiB. GLM cgroup swap and memory-limit events remained zero.

The external census retained the startup event and a later six-page swap-out during artifact verification after GLM had stopped. Process metadata shows partial correlations with snapd, but does not establish full or causal attribution. Original no-swap requirements remain unchanged. Model/runtime verification, memory recovery, guard identity, clean descendants and unchanged default/proxy/guard checks passed. This is not a context, fidelity, durability or performance result.
