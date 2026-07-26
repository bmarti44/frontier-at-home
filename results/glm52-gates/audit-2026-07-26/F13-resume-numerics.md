# Adversarial falsification review: F13-resume-numerics

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The cited ledger rejects the claim. loadprof-2026-07-25.json:258-311 says corruption is “NOT excluded,” the same-lineage L40 discrepancy remains open, the replay closure is “NOT SUPPORTED,” and the strict guard must remain. Claim F13 silently restores the categorical conclusion that the final audit explicitly retracted.

2. The failure signature fits stale state better than ordinary floating-point amplification. append-probe-evidence.txt:1-6 and append-clean-evidence.txt:1-6 show resumed outputs diverging at character zero and repeatedly starting “oft...,” continuing the prior generation’s mid-word tail, while cold controls respond to the appended instruction. Different adversarial and clean appends also report identical resumed digests for ap2 and ap3. Numerical perturbations can flip an argmax and amplify, but they do not specifically predict suffix-insensitive continuation from the old frontier. Stale cache/indexer state being attended explains all three observations directly.

3. The repository’s own technical report still identifies a state bug. docs/ds4-glm-resume-frontier-bug-2026-07-26.md:7-12 describes the previous frontier being continued; lines 51-62 say graph state is not rewound to the checkpoint; and lines 69-77 say restoration leaves state that corrupts subsequent evaluation. No later experiment identifies an alternative mechanism that invalidates this explanation.

4. TOKDUMP does not establish complete input equivalence. It prints only positions 5040-5055. The failing resumed suffixes contain 22-29 tokens from position 5044, leaving 10-17 suffix positions unchecked. More fundamentally, identical token IDs exclude tokenization differences only in that window; they say nothing about cache contents, visibility bounds, indexer state, or which rows attention reads. Input equality strengthens rather than weakens the case for an internal state defect.

5. The replay experiment does not test a clean store/load round trip. rowdump_replay.sh compares dump 001 with dump 002, but dump 001 is post-load/pre-replay and dump 002 occurs at the later dummy request after replay evaluation. The log also shows a 5063-token live state being stored and a different 5044-token shard loaded. Evaluation therefore occurs between the dumps. The observed L40 changes could be produced during replay; the experiment neither proves benign serialization nor excludes corrupted append behavior.

6. Byte-identical replay output proves only identical greedy token choices in one short suffix-3 run. It provides no logit comparison, attention trace, or proof that the differing L40 coordinates were unread. The failing append path uses suffixes of 22-29 tokens. As sol-closure-review.txt correctly states, replay-inertness does not imply append-inertness. The claim relies on precisely that invalid implication.

7. The rowdump analysis contains a concrete slicing error. rowdump_probe.sh labels X.dump1[:half] versus Y.dump1[:half] as “RESTORED-LOWER,” but half of the 40,960-byte file is the entire L0 block, not the lower four rows of every tensor and layer. Its “first diff at byte 8192” establishes equality only for the first four assumed L0 kv_lora rows; it does not establish equality for lower k_rope, indexer, or L40 rows. The comparison itself returns False.

8. Dump association is unverified. Server X issued three HTTP requests but produced four dumps; server Y issued two requests but produced three. Nevertheless, the analyzer hard-codes X.dump2 and Y.dump1 as corresponding post-write states. The committed evidence omits the sync trace needed to explain the extra dumps. Thus the supposedly matched states cannot be independently reconstructed.

9. The producer fixes tensor order but does not verify the parser’s segment boundaries. ds4-iq2xxs-down-cuda.patch:723-743 computes strides dynamically, silently skips null/invalid tensors, ignores write errors, and writes no headers or metadata. rowdump_replay.sh then assumes fixed offsets 0, 16384, 18432, 20480, 36864, and 38912 without asserting tensor strides or segment presence. The 40,960-byte total is consistent with the assumed layout, but it is not an independent verification. The raw binaries are also absent, preventing coordinate-mask or boundary checks.

10. “L40 differs between ANY two evaluations with different chunkings” is an unsupported universal generalization. The committed record contains one prompt family, one checkpoint region, four rows, layers 0 and 40 only, aggregate byte counts, and no raw difference masks. The approximately 25% and 100% arithmetic is correct, but neither those counts nor L0 equality proves causation. A layer-local restoration, visibility, synchronization, or indexer defect could leave L0 exact and corrupt deeper layers.

11. The llama.cpp analogy is asserted, not demonstrated. There is no branch-matched llama.cpp experiment, logit comparison, or evidence that ordinary prefix reuse produces suffix-insensitive continuations of an old generation. Similar deep-layer numerical differences do not establish the same failure mechanism or make semantically stale output acceptable.

12. The strongest defensible conclusion is narrower: chunk-shape numerical path-dependence exists, but it has not been shown to cause the append failure. A genuine resume-state defect remains the better-fitting explanation, with the exact corrupted component still unresolved. That is sufficient to falsify the categorical “numerics, NOT state corruption” claim.
