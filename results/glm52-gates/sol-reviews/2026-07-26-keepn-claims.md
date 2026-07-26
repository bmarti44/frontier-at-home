# sol xhigh adversarial review — keep-N, batchall, warm-TTFT claims
Date: 2026-07-26. Verdicts: Claim 1 FALSIFIED, Claim 2 WEAKENED, Claim 3 WEAKENED.

### Claim 1 — FALSIFIED

**Driver:** The auditable implementation is not physical expert skipping.

- Keep-N changes only `router_weights`: it zeros low weights and optionally rescales survivors in [ds4-iq2xxs-down-cuda.patch](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/ds4-iq2xxs-down-cuda.patch:203). It never compacts `router_selected` or changes `DS4_N_EXPERT_USED`.
- The loader still receives all selected IDs and count eight in [ds4.c](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4.c:14289) and iterates all of them in [ds4_cuda.cu](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:3117).
- CUDA launches work over `n_tokens * n_expert`; the gate weight is applied only after gate/up dot products in [ds4_cuda.cu](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:12173), and down projection still runs for every expert.

Bluntly: **keep-6 fetches and computes eight experts. It saves neither the claimed 25% bytes nor 25% expert compute.** This is a contribution-ablation pilot, irrelevant as evidence of the performance gain.

Additional defects:

- `DS4_GLM_TOPK_NORENORM` appears in [keepn_controlled.sh](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/keepn_controlled.sh:43), but not in the committed engine patch/source. Whether `keep6nr` actually disabled renormalization is unverifiable.
- Zero bad UTF-8 and word-level 3-gram repetition are gross-corruption checks, not fidelity. They miss factual errors, degraded hard reasoning, arithmetic/planning failures, instruction-following errors, and code/SQL that looks plausible but is incorrect. The repetition metric also misses character/subword loops and paraphrased semantic loops.
- The summary commits only one 60-character sample per arm. “All samples read as coherent” cannot be independently checked from [keepn-controlled.txt](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/logs/loadprof1/keepn-controlled.txt:1).
- Keep-7 and keep-6 both changing 8/8 outputs proves universal sequence drift. That is logically consistent with “no catastrophic collapse”—greedy decoding can avalanche after one near-tie—but it provides no evidence that semantic drift is small.

The minimum defensible fidelity gate is a paired keep-8/keep-6 run of the existing 100-case, 2,299-token teacher-forced suite in [g5_bench_glm.sh](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/g5_bench_glm.sh:45): token-weighted ΔNLL, median/p90/max per-case ΔNLL, top-1 agreement loss, and confidence intervals, with thresholds declared beforehand. The repo’s provisional thresholds—mean ΔNLL ≤0.01 nat/token and top-1 loss ≤0.5 percentage point—are reasonable. Shipping still needs task accuracy for reasoning and code.

**Most valuable next experiment:** implement true count-6 compaction before the loader and MoE launch, then run the paired 100-case NLL suite while recording actual expert bytes, misses, and decode time.

### Claim 2 — WEAKENED

**Driver:** Batchall is a genuine confound, but “entirely an artifact” is stronger than the committed evidence.

- The old keep-6 run explicitly enabled batchall in [keepn_ab.sh](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/keepn_ab.sh:19), while its external keep-8 baseline used a different 68GB/no-SLRU/8K configuration without batchall in [mtp_ab.sh](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/mtp_ab.sh:34). The original causal conclusion was invalid.
- Batchall alone is associated with degeneration in [batchall-fidelity-bug.txt](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/logs/loadprof1/batchall-fidelity-bug.txt:1), and commit `a33d021` removed it from [52_engine_switch.sh](/home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh:95).
- But the committed batchall evidence is only four truncated prefixes—not the raw responses, server logs, binary hash, or a complete factorial run. Thus “batchall can cause the symptoms” survives; “batchall was the sole cause” remains unproven.

Remaining keep-N confounds: fixed arm order, one run per arm, inherited environment not sanitized, no cache flush, and no proof that terminated servers fully exited. Fresh processes reset the in-process expert arena if shutdown succeeds, but storage/page/device state is not cold-equivalent across arms.

**Most valuable next experiment:** randomized, repeated 2×2 factorial: keep-8/keep-6 × batchall off/on, same hashed binary/config/prompts, with explicit `env -u`, HTTP validation, byte counters, and the 100-case NLL gate.

### Claim 3 — WEAKENED

**Driver:** 1.755 seconds is one third-request observation, not a reproducible warm-TTFT distribution.

[ttft-batchall-ab.txt](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/logs/loadprof1/ttft-batchall-ab.txt:1) records 147.610s / 2.378s / 1.755s and one-token text `"The"`. The harness fires them sequentially in [ttft_probe3.sh](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/ttft_probe3.sh:49).

The specific reproducibility threat is that **warm2 is the second exact replay after warm1 has additionally primed the KV checkpoint, expert cache, storage/page cache, and runtime state**. The 623ms warm1→warm2 gap demonstrates that the cache state is still changing. Restart, cache eviction, competing workload, or a non-exact appended turn can therefore lose 1.755s. The harness also does not enforce curl success/HTTP status, and an externally exported batchall variable is not explicitly removed.

**Most valuable next experiment:** at least 20 fresh-server cold/warm1/warm2 triples on the production profile, with explicit environment sanitization and HTTP checks; report warm1 and warm2 p50/p95 separately and include appended-turn fixtures.
