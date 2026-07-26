# Adversarial falsification review: F12-32k-context

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The repo’s own final audit rejects the claim. results/glm52-gates/loadprof-2026-07-25.json:298 calls “32K MET overstated,” and line 302 says only “11.6K functional” with the full-depth probe open. docs/glm52-final-status-2026-07-26.md:21 likewise labels it “partially validated.”

2. The successful prompt exercised only 11,648 of 32,768 tokens: 35.5% of the advertised window. The remaining 21,120 rows, including every capacity-edge condition, were untouched. Failures could still appear in compact-cache bounds, positional indexing, top-k workspace selection, or prompt-plus-generation room near 32K.

3. The passphrase test is narrow and unusually easy. It uses repetitive filler and a uniquely salient “IMPORTANT RECORD” marker. The secret is around row 9.7K but only roughly 2K tokens before the final question. This proves likely retrieval from one high absolute position; it does not prove reliable use of the whole 32K span, multiple depths, distributed evidence, or ordinary agent content. No indexer dump establishes that the secret’s exact rows were selected.

4. “Deterministically twice” overstates independence. The initial 12-output-token A test failed; only the 64-token retry passed. Its first successful run was cold and the second was an exact-prefix checkpoint replay on the same server, with trace start=11648 and suffix=1. That proves cold/resume equality for this one fixture, not two independent full prefills or restart-level determinism.

5. The committed “raw evidence” is actually a derived summary. ctx-regate-evidence.txt omits the generated fixture, response JSONs, complete server logs, tokenizer output locating the secret, binary hash, memory series, and checkpoint listing. The approximate row-10K placement appears only as a harness comment; only total prompt length was traced.

6. The 5.81 GiB cache calculation itself survives: 186 KiB/token × 32,768 = 5.8125 GiB. But the ledger’s memory wording double-counts it. docs/glm52-single-spark-2026-07-24.md:127-130 says the 5.81 GiB cache plus approximately 4.19 GiB scratch yields about 10.0–10.2 GiB total context buffers. The ledger instead says “10.2 GiB context buffers + 5.81 GiB compact DSA.” The claimed approximately 98 GiB footprint has no committed measurement trace.

7. The headroom is not a demonstrated safe margin. Even accepting 98 GiB, 119.7−98 leaves 21.7 GiB before other host usage. That is only about 9.7 GiB above the deployed 12 GiB kill floor, or 3.7 GiB above the safe-run default of 18 GiB. Worse, ctx_regate.sh launches the server directly without either memory watcher, so the 11.6K run supplies no full-depth MemAvailable high-water evidence.

8. The 16 GiB disk-cache claim is theoretical. Two 5.8125 GiB full checkpoints fit; three require at least 17.44 GiB. The successful re-gate checkpoint was only 11,648 tokens, approximately 2.07 GiB by the measured 186 KiB/token rate. Defaults create continued checkpoints every 10K tokens and cap cold checkpoints at 30K, while the ledger explicitly leaves multi-session eviction behavior open. No full-size checkpoint, eviction, or restore was tested.

9. The defensible result is narrower: the server accepted -c 32768, completed an HTTP-200 11.6K request, retrieved one likely post-8192 needle exactly, and reproduced it through exact-prefix resume. That is meaningful evidence for 11.6K indexed operation, but it does not establish that this box serves a functional 32K context window.
