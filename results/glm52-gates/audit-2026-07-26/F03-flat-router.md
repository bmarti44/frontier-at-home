# Adversarial falsification review: F03-flat-router

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The normalization attack does not succeed. The selector computes sigmoid weights, selects eight experts, and normalizes those selected weights to expert_weight_scale. Thus the logged row already sums to 2.5, and the aggregation script’s division by its sum merely removes that scale. The reported fractions validly describe contribution mass among the selected eight experts.

2. They do not describe global router flatness. Unselected experts, selection-score bias, expert-selection frequency, and popularity across tokens are absent from the measurement. The same ledger later explicitly concedes that this measures “diffuseness AMONG the 8 selected,” not global expert or popularity flatness.

3. “14,175 routed-layer decode observations” is materially misleading. The harness submits one deterministic fixture three times, with identical input and output SHA. Each run contains 31 prompt tokens plus 32 generated tokens across 75 layers:
   3 x 31 x 75 = 6,975 prefill rows
   3 x 32 x 75 = 7,200 actual decode rows
   The data therefore represent one 63-position trajectory repeated three times, not 14,175 independent decode observations.

4. The claimed raw evidence is not present in results/glm52-gates/logs/loadprof1/. No committed GATEMASS rows permit independent recomputation, separation of runs, or stratification by layer and prompt/decode phase. The ledger means can be traced to an aggregation command, but cannot be reproduced from the cited repository evidence.

5. “Flat” is a poor description without qualification. Uniform mass over eight experts would give cumulative masses 0.125, 0.250, 0.375, 0.500, and 0.750. The observed values are 0.251, 0.421, 0.553, 0.663, and 0.848. Top-1 is 2.01 times uniform, and the implied mean rank masses are approximately 0.251, 0.170, 0.132, 0.110, 0.093, 0.093, 0.076, and 0.076. The leading expert therefore has about 3.3 times the average mass of either bottom-two expert. “Moderately concentrated” or “diffuse relative to a top-2-dominates assumption” fits better than “flat.”

6. “Unusually” is unsupported. The cited evidence contains no matched comparator for Mixtral, DeepSeek, another GLM prompt, another GLM checkpoint, or another quantization/runtime. A single arithmetic prompt cannot establish that this model is unusual.

7. Pooling all layers and phases can conceal exploitable regimes. Even the retained aggregate percentiles range substantially: top-1 p10/p90 is 0.171/0.346 and top-2 is 0.321/0.535. Layer-specific, prompt-versus-decode, and token-specific distributions were not reported. Adaptive skipping could target peaked layers or positions even when a global fixed keep-N policy is poor.

8. The tiering conclusion does not follow from within-token gate mass. Storage tiering and caching can exploit expert invocation frequency, temporal locality, layer-specific popularity, or stable hot experts without dropping their contributions. The probe measures none of those. It can at most challenge fixed contribution-based expert omission on this one trajectory.

9. The keep-6 follow-up is confounded. The patch zeros two experts and renormalizes the retained six by total/kept mass. With mean retained mass 0.848, that amplifies retained contributions by about 1/0.848 = 1.179. Result degradation therefore cannot distinguish damage from omitted experts from damage caused by an approximately 18% activation-scale increase. The ledger’s later accepted review acknowledges this and narrows the result to “keep-6-renormalized fails,” not “all skipping is dead.”

10. The assertion that a 90%-mass tier “requires approximately seven of eight” was not directly measured. The aggregation computed top-6 but not top-7, and a mean top-6 mass of 0.848 does not determine how many individual rows cross a 90% threshold.

11. The supplied system geometry is inconsistent with the audited engine configuration: the GLM source used for these tests specifies 256 routed experts with top-8, not 160. This does not change the selected-eight fractions, but it further undermines the provenance of the broader architectural claim.

12. The defensible result is much narrower: for one repeated 31-token prompt plus 32-token completion, the eight selected contributions were moderately diffuse on average, so unconditional keep-6 with renormalization is risky. That evidence does not establish unusual global router flatness or structural weakness of importance-based expert tiering/skipping.
