# Adversarial falsification review: F04-keepn-collapse

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The repository itself retracts the broad conclusion. loadprof-2026-07-25.json:463 states that only “keep-6-renormalized decisively fails” is proven and “‘all skipping dead’ is not.” sol-round2-review.txt:152-165 reaches the same conclusion and requires additional controls.

2. The intervention is not literal expert skipping. ds4-iq2xxs-down-cuda.patch:219-227 zeroes weights and then multiplies every survivor by total/kept. With reported top-6 mass 0.848, this is roughly 1.179x amplification; the exact average is E[1/r], not 1/E[r]. That pervasive amplification is an untested alternative cause of collapse. Moreover, 15.2% gate-weight mass is not necessarily 15.2% of the expert-output vector norm.

3. The requested implementation-bug theories do not rescue the claim. Sorting a copy preserves the original ID-to-weight positional pairing. Ties are handled with gm >= threshold, so they can retain more than six experts, never accidentally drop more than two or all eight under finite nonnegative weights. The write occurs after routing but before downstream MoE use, so modifying weights after IDs are selected is intentional. However, router_selected remains eight IDs, meaning all eight experts are still loaded and executed; this is a contribution-ablation experiment, not an implementation of physical expert skipping.

4. There is no valid same-binary keep-8 control. keepn_ab.sh:19-24 runs keep-6 with a 72 GB SLRU cache, 32K context, and DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1. Its “keep-8” files come from mtp_ab.sh, whose server used 68 GB, no SLRU, 8K context, and no prefill override at lines 34-39. The harness never runs afc7378bb80f with DS4_GLM_TOPK_KEEP=8 through the same synchronization/readback path.

5. The committed evidence does not substantiate the reported eight-case result. keepn6-evidence.txt is only 462 bytes containing truncated prefixes for p0 and p5. It contains one demonstrable invalid UTF-8 sequence at byte 111, not committed proof that two of eight responses were unparseable. The other six responses, complete outputs, keep-8 controls, timings, server log, and summary are absent.

6. The experiment abandoned its own decision criterion. loadprof-2026-07-25.json:375 prescribed a 100-case NLL suite at keep-6 and keep-4 versus keep-8. Instead, keepn_ab.sh:42-57 performs subjective coherence inspection of eight greedy continuations. No NLL, perplexity, top-token agreement, repetitions, or representative corpus evaluation was performed.

7. The harness can report completion after failures. It uses only set -u, curl lacks --fail and HTTP-status validation, the readiness loop has no timeout failure branch, and Python parsing is piped to tee without pipefail. Invalid JSON or UTF-8 therefore does not prevent the final KEEPN_DONE message. This makes transport, serialization, and server failures insufficiently separated from model degradation.

8. The probe tests fixed keep-6 on every decode layer and token of one IQ2_XXS artifact. It does not test adaptive mass thresholds, layer sensitivity, selective skipping on only safe tokens/layers, non-renormalized skipping, or a less aggressively quantized GLM-5.2. The narrow pilot result is alarming and probably real: fixed keep-6 with renormalization collapses this configuration. The inference that AdapMoE-class skipping is dead does not follow.
