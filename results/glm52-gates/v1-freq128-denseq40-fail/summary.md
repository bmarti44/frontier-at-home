# v1 artifact (freq-128 prune + dense Q4_0) — FAIL on fidelity, 2026-08-16

Artifact: glm52-v1-freq128-denseq40.gguf (107.05 GB) = 50% whole-expert prune
(top-128/layer by routing frequency traced on ~70K tokens of ONE document,
fixtures/ctx-32k.txt) + dense/attention/shared/output Q8_0→Q4_0 (embeddings
Q8_0; requant via scripts/59). Engine: 4d54edd + expert-count + direct-slot +
Q4_0 GEMV/batch + optimized Q4_0 decode GEMV.

## Speed (measured, warm, ctx≈30K, stage-timed)
- decode total 305 ms/token declining ≈ 3.3 tok/s (vs 2.33 production start;
  cumulative +41% with the committed direct-slot win)
- attn_dense 148 ms (Q8 baseline 187; naive Q4_0 253) — optimized Q4_0 GEMV
  beats Q8 as designed
- prefill ~45.8 tok/s (Q4_0 batch kernels)

## Fidelity (100-case glm52-openrouter-100, teacher-forced)
- token-weighted NLL **1.9292** vs reference **0.4515** (full model)
- mean top-1 rate **0.5633** vs reference **0.834**
- VERDICT: FAIL — not adoptable. Attribution between the frequency prune
  (fat routing tail: top-128 covers only 76.8–94.8% of routing mass; single-
  document calibration corpus) and the Q4_0 dense requant is not yet
  decomposed; prune-only (v1a) NLL measurement is the next discriminator.

Per the owner directive, lossy adoption requires the owner's decision on the
reported delta; this delta is reported as unacceptable by default.
