# Adversarial falsification review: F07-slru-win

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The claim conflates two different experiments. SLRU’s corrected result is 155.8 → 136.4 misses/token, a 12.4% reduction. The 136.4 → 74.0 result belongs to the later prefetch experiment in loadprof-2026-07-25.json, which reports a 46% reduction and explicitly says prefetch was not adopted. Numerically, 136.4 → 74.0 is a 45.7% reduction, not 12.4%.

2. The layer correction itself is valid. GLM-5.2 has 75 routed layers, L3–L77. The old harness divided by 79, turning 4,725 records into 59 nominal tokens instead of 63. Applying 59/63 yields approximately:
   Legacy LRU: 444.2 hits, 155.8 misses/token.
   SLRU: 463.6 hits, 136.4 misses/token.
   Because the same scaling applies to both runs, the 74.0% → 77.3% hit-rate ratio and 12.4% relative miss reduction survive, conditional on equivalent complete windows.

3. “Per token” here includes the whole 63-token request trajectory, not just 32 decoded tokens. The fixture has 31 prompt tokens and 32 completion tokens. Calling this a decode-only miss rate is misleading, although it does not invalidate the hit-rate comparison.

4. I found no SLRU-specific counter inflation bug. cuda_expert_cache_promote only manipulates list membership. A LOADPROF hit is incremented after lookup returns a resident slab and the three cache-copy operations are accepted; misses become miss_jobs only after lookup fails. SLRU neither skips the miss branch nor double-counts promotion hits. The list transitions are internally consistent under the assumed single-threaded access.

5. The causal evidence is not properly committed. slru-ab.txt contains only four aggregate lines—no server logs, per-layer records, launch commands, response bodies, output hashes, initial-occupancy evidence, or access-stream digests. The harness does not hash responses or close each request with the available stream-digest endpoint. Identical generated text, even if the ledger’s unarchived SHA assertion is trusted, would not strictly prove identical internal expert routing.

6. Hit counts would be deterministic if both runs had identical access streams, empty starting caches, capacity, and binary. Under those conditions ordinary timing variance cannot create a 3.3-point hit-rate change. But those conditions are asserted rather than demonstrated. The generic harness also does not explicitly set the SLRU variable; it relies on inherited environment state, while the implementation treats even DS4_CUDA_EXPERT_CACHE_SLRU=0 as enabled.

7. The sample is highly unrepresentative: one short prompt is run three times, and the measured request follows two identical warmups. There are no repeated A/B pairs, randomized order, or varied serving traffic. A 3.3-point gain is plausible for SLRU on a repetitive scan—equivalently, it eliminates about 12.4% of baseline misses—but cannot establish a general serving hit rate. Its exact match to the earlier 77.3% model is not corroboration because that model used a different pin/LRU hybrid and trace.

8. SLRU is configured in scripts/52_engine_switch.sh, so the narrow “added to the serving profile” statement is true. However, the A/B harness defaults to a 68GB cache and 8K context, while the serving profile uses 72GB, 32K context, disabled streaming-token prefill, and disk KV settings. The claimed 77.3% rate was therefore not validated on the adopted profile. The later prefetch producing approximately 74 misses/token is explicitly absent from that profile.
