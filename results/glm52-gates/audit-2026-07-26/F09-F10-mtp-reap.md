# Adversarial falsification review: F09-F10-mtp-reap

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The measured MTP result is one 1.10x observation, not an established ceiling. The summary reports 1.61 versus 1.77 t/s, but the harness always runs OFF before ON, has no ABBA repetitions, and reports no variance or per-prompt timings.

2. The claimed “~1.5x unique experts” is contradicted by the committed counters. Baseline lookups are 474.3 + 166.4 = 640.7; MTP reports 618.5 + 213.6 = 832.1, a 1.30x ratio. Fetch time rises only 1.25x and total loader time 1.22x. No committed measurement gives 1.5x.

3. Those absolute profiling figures were subsequently proven wrong. sol-round2-review.txt:37-64 documents a hard-coded 79-layer denominator despite only 75 routed layers, producing the impossible 640.7 accesses/token when the maximum is 600. The ledger corrected later SLRU figures but never corrected or revalidated the MTP explanation built from the same profiling family.

4. The acceptance arithmetic leaves substantial cost unexplained. A 72.5% depth-one acceptance rate gives 1.725 tokens/cycle. Combining that with the observed 1.10x speedup implies an MTP cycle costs about 1.57x a baseline step. The committed expert-event ratio is only 1.30x, and only part of a step is expert work. Therefore expert diversity alone cannot account for the missing gain. Moreover, the 647/892 acceptance count covers both warmup and timed passes, while 434 seconds covers pass two only; the harness never measures pass-two acceptance separately.

5. The A/B workloads are not equivalent. All eight MTP outputs diverge from baseline early, so subsequent tokens, routes, cache hits, and eviction histories differ. Each configuration warms its cache using its own divergent trajectory. There is no access-stream digest, cache-occupancy record, cold-reset pairing, or randomized order. This can measure the two modes as operated, but cannot causally assign the difference to expert diversity.

6. The ON arm uses “--glm-mtp-timing,” not a plain production MTP flag. Any logging, synchronization, or timing overhead is unmeasured. The harness does use the same executable path and environment for both arms, so a different-binary confound is not demonstrated, but it also records no binary hash or startup command in the committed evidence.

7. The byte-identity observation survives: mtp-ab2-summary.txt shows 0/8 identical outputs. That supports the conservative non-adoption decision. It does not prove the ledger’s “near-tie FP-order flip” diagnosis: no first-divergence logits, margins, or tensor comparison were retained, so an MTP state or verifier defect remains an alternative explanation.

8. The REAP claim is internally false as written. G4-bench.json reports REAP decode at 1.91 t/s versus streaming at approximately 1.4 t/s in the same comparison, 1.61 t/s in the MTP OFF run, and 1.6–1.8 t/s in the final snapshot. REAP is therefore faster in decode by the ledger’s own numbers. Only prefill is slower: 4.37 versus 23.2 t/s.

9. The REAP prefill comparison is dominated by backend choice, not pruning. REAP ran llama.cpp with “--cpu-moe,” placing all routed-expert arithmetic on the Grace CPU with scattered mmap/page-cache faults. Faithful streaming ran CUDA MoE with an explicit pinned expert cache, parallel O_DIRECT fetching, and batch dispatch. This demonstrates that one CPU-MoE configuration is slow; it does not demonstrate that REAP itself is slower.

10. The matched REAP CUDA alternative was never tested. The committed ds4 patch shows the CUDA routed-MoE launcher accepting Q4_K/Q4_K or IQ2_XXS gate-up with Q2_K/IQ2_XXS down, while rejecting a uniform Q2_K gate type. Thus the REAP artifact needs additional gate/up kernels or requantization. That compatibility gap explains the missing experiment; it is not negative performance evidence.

11. REAP on ds4 CUDA could plausibly beat the tested CPU path because it would remove CPU expert math and gain explicit caching/batched I/O. It is not demonstrated likely to reach 18.4 t/s: top-8 still executes eight experts per routed layer, Q2_K moves more bytes per selected expert, the 139 GB artifact still exceeds physical memory, and fidelity was skipped. The defensible conclusion is only “REAP50 with llama.cpp --cpu-moe is not parity.” The broader “REAP is not the parity path” remains untested.

12. The REAP raw audit trail is absent from the repo. No REAP timings, server log, responses, memory trace, artifact manifest, or hashes exist under logs/loadprof1; only the harness and narrative G4 JSON are committed. Consequently, even the 1.91/4.37 measurements cannot be independently recomputed from the cited repository evidence.
