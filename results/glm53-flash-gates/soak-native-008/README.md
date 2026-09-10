# Prepared-kernel replay 008: FAIL at host gate

The unchanged 512/128 profile reached authenticated READY, passed a 4,224-input-token correctness smoke and stopped cleanly. All 2,486 prepared kernel files stayed unchanged. No necessary window or full-duration workload was admitted: the host wrote one page to swap, increasing used swap by 4 KiB. GLM cgroup swap and memory-limit events remained zero.

The external census retained a simultaneous snapd VmSwap increase of 4 KiB. This correlation does not establish causation and does not relax the host gate. Model/runtime verification, memory recovery, guard identity, clean descendants and unchanged default/proxy/guard checks passed. The short smoke is not a context, paired fidelity, durability or production performance result. The preceding million-token result used a different configuration.
