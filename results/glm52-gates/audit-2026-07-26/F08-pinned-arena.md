# Adversarial falsification review: F08-pinned-arena

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The 2.6 ms and 2.1 ms figures are enqueue times, not completed-copy times. The timers surround cudaMemcpyAsync submission in ds4-iq2xxs-down-cuda.patch:2706-2728 and 2910-2941. Completion occurs at a later blocking cudaMemcpy/synchronization point around lines 2968-2991. The ledger explicitly concedes this at loadprof-2026-07-25.json:64. Comparing pre-pin blocking time with post-pin enqueue time changes the measured quantity.

2. The byte arithmetic proves the timing artifact. 474.3 hits × 9.28 MiB is 4.30 GiB, or 4.62 GB, per reported token. Completing that in 2.6 ms requires 1.78 TB/s. Likewise, 166.4 fills × 9.28 MiB is 1.51 GiB; 2.1 ms implies 771 GB/s. Those cannot be completed-copy rates on this system. The transfers were queued, overlapped, and later synchronized—not eliminated.

3. The displaced cost appears directly in the ledger. Hit plus fill buckets apparently fall by 122.3 ms/token, but “other” rises from 0.9 to 44.1 ms and fetch rises from 214.9 to 227.5 ms. Thus 55.8 ms, or 45.6% of the apparent saving, reappears elsewhere. Total loader time falls only from 342.8 to 276.3 ms, a 66.5 ms improvement before correcting the aggregation error.

4. All quoted per-token numbers contain a known 6.8% denominator error. evidence.txt records 4,725 layer lines per request. With 75 routed layers, that is 63 evaluated tokens, but the original harness divided by a hard-coded 79 layers and reported 59. The impossible 474.3 + 166.4 = 640.7 accesses/token exceeds the top-8 maximum of 600. The repo’s later audit admits this in sol-round2-review.txt:37-64. Corrected F08 figures are approximately 86.6 to 2.44 ms for hit blocking/enqueue, 32.3 to 1.97 ms for fill blocking/enqueue, and 321.0 to 258.8 ms total loader time. The real loader saving is about 62.3 ms per evaluated token.

5. End-to-end decode evidence supports only a modest real gain. evidence.txt shows 34.556 to 30.411 seconds: 4.145 seconds, 12.0%, or 1.14×. The request contains 63 evaluated tokens, giving 65.8 ms saved per evaluated token, closely matching the corrected 62.3 ms loader reduction. This confirms that most of the purported 114.5 ms corrected hit/fill reduction was accounting or overlap. The ledger’s “~93 ms/token saved” is also arithmetically unsupported; with at most 32 output tokens, the wall delta is at least 129.5 ms per output token.

6. The cold-first-request result is real but mischaracterized. Committed raw evidence supports 507.996 to 54.383 seconds, or 9.34×, not 51 seconds. The 50.9-second revalidation appears only as an unaccompanied ledger summary. Earlier G4a runs independently show unpinned first requests around 518-542 seconds, so a large cold-start benefit is credible. But this measures an entire short completion, including the lazy 13.3-second cudaHostRegister operation, page prefaulting, disk reads, kernels, and generation—not isolated “arena fill time.”

7. The mechanism better fitting the cold result is bulk registration/prefaulting plus asynchronous overlap. The arena is allocated lazily and registered during the first selected-expert load, after server readiness. Registration replaces pathological repeated pageable staging/page-fault behavior with one bulk 13.3-second operation. That explains the roughly 10× cold improvement without implying that 1.5 GB of fills physically completed in 2.1 ms.

8. “Plain cudaMemcpyAsync without pinning gave no gain” is not independently auditable from the cited raw evidence. The ledger alone states 92.5 to 91.6 ms. evidence.txt contains no async-only timing, wall result, output, routing trace, or hash. Commit bc718cc also changes synchronous calls to cudaMemcpyAsync and adds cudaHostRegister together; the alleged intermediate A/B is not preserved in the repo evidence.

9. Pin-specific byte identity is likewise asserted rather than reproducible. evidence.txt contains no response JSON or hashes. G4a proves byte identity for the earlier cache-on/cache-off implementation, but not the three-way v4.6.2/async-only/pinned comparison. The pin result may well be deterministic, but “byte-identical throughout” is supported only by the ledger’s SHA summary for one repeated fixture.

10. The defensible restatement is: registering the arena enabled genuinely asynchronous transfers, reduced host blocking and permitted overlap, cutting measured loader critical-path time by roughly 62 ms per evaluated token and one repeated-request wall time by 12%, while reducing cold first-request latency from about 508 to 54 seconds on this fixture. The 2.6 ms and 2.1 ms values are submission overhead and must not be presented as completed copy times.
