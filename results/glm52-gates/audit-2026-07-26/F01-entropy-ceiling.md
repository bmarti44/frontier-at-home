# Adversarial falsification review: F01-entropy-ceiling

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED

1. The 66-byte block size is correct, but the purported stream separation is not. ggml-common.h:378-385 defines a 2-byte fp16 scale followed by 32 uint16 values. However, ggml-quants.c:2488-2511 shows each 8-byte subgroup contains four lattice-index bytes followed by four bytes packing four 7-bit sign selectors and a 4-bit scale. iq2_stream_entropy.py:46-49 instead partitions low/high bytes of alternating uint16 values. Each resulting “lane” therefore alternates lattice bytes with packed sign/scale bytes. It is not a field lane.

2. Consequently, the claimed lattice result does not measure the lattice-index stream. “even_lo” alternates subgroup bytes 0 and 4: one lattice index, then packed sign bits. Its H1 measures index-to-sign and sign-to-index transitions. Calling H0=7.970 and H1=7.966 evidence that lattice indices are memoryless is a category error.

3. Recomputing the actual fields on the exact retained 64 MiB sample produces materially different results. A model conditioned only on each symbol’s fixed position within the block requires 512.557 bits per 528-bit block, an ideal saving of 2.925%, not 1.12%. This uses ordinary static contexts—32 lattice positions, 32 sign positions, eight scale positions, and the two fp16 bytes—not an exotic compressor. Thus 1.12% is not an entropy ceiling.

4. The hidden structure is substantial. For the real lattice-index symbols, global H0 is 7.96986 bits and flat-stream H1 is 7.96679, but entropy conditioned on the index’s position within the block falls to 7.80637. For sign selectors it falls from H0=6.95260 to 6.73004 bits. Flat order-1 analysis misses this periodic block phase entirely.

5. The weighted-H0 arithmetic itself is correct, but it applies to the wrong streams. Using full precision, the harness yields 7.910348 bits/byte and 1.120648% nominal savings. There is no arithmetic slip; the failure is interpreting that number as a format-aware upper bound.

6. A correct field split without positional contexts actually gives only 0.585% ideal savings. This explains why merely relabeling the bytes is insufficient. The counterexample to the claimed ceiling is correct field extraction plus the obvious position-within-block context that the probe omitted.

7. The sample cannot establish a model-wide ceiling. It covers 67,108,800 parsed bytes, only 8.08% of one 830,472,192-byte blk.40.ffn_gate_exps tensor—approximately the first 20.7 of the tensor metadata’s 256 expert slabs. It samples one layer, one projection, and one contiguous prefix, with no stratification across layers, experts, or gate/up/down tensors.

8. The earlier evidence does not close this gap. The reported zstd-19 result came from a generic 512 MiB file-offset region rather than this field-aware sample, so it says little about a position-conditioned coder. The cited zero deduplication and cross-expert MI experiments were performed on 74-byte IQ2_XS sibling blocks, not the 66-byte IQ2_XXS layout under this claim.

9. A held-out adjacent-expert context test did not improve over the position-only model, so there is no evidence here for dramatic cross-expert compression. That weakens the practical upside, but it cannot restore the asserted 1.12% ceiling: a directly reproducible 2.925% format-aware bound already disproves it.

10. The repository also lacks a committed raw entropy log under results/glm52-gates/logs/loadprof1/. The ledger records the summarized values, but not the exact input hash or complete probe output. The retained temporary sample reproduces those values, yet the claimed evidentiary trail is not self-contained in the cited repository artifacts.
