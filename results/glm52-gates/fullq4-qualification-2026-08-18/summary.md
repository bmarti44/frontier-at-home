# GLM-5.2 full-Q4 candidate qualification — 2026-08-18 (stopping point)

Configuration: glm52-full-denseq40.gguf (202.3 GB; original 256-expert model,
dense Q8_0→Q4_0, embeddings Q8_0, experts byte-identical) on the
glm52-dynexp2-patched engine (profile
configs/glm52-fullq4-production-profile.json, binary sha a093812a…),
DS4_CUDA_EXPERT_CACHE_GB=94, DIRECT_SLOT=1, diagnostics off.

## Fidelity — full 100-case glm52-openrouter-100 (q100-fullq4.tsv)
- token-weighted NLL **0.5139** (reference full-Q8: 0.4672)
- mean top-1 rate **0.8293** (reference: 0.8338)
- Owner accepted this delta (2026-08-17).

## Speed — diagnostics-off bench (bench5.json), GLM output tokenizer bound
- ctx≈0 cell VALID: **median decode 3.28 tok/s** (reps 3.46 / 3.09);
  warm TTFT 21.5 s (first rep included residual cache warming).
- ctx=28,672 cell: reps rejected by the bench's strict token-accounting check
  (160 SSE events vs 157 client-retokenized; GLM THINKING transitions not
  counted as field_transitions on this path). Measured-but-not-strict-valid
  values from those reps: decode 2.27/2.37 (SSE), prefill 41.0 tok/s; the
  server's own steady-state chunk rate reached 2.77 tok/s. Recorded here;
  headline tables show a dash for this cell per the diagnostics-off/strict
  rule.

## Status
Candidate is production-switchable (`sudo scripts/52_engine_switch.sh glm52`,
rollback `... dsv4`; switch tests 17/17). The live switch was intentionally
NOT performed — DSV4 remains the serving engine at this stopping point.
Remaining known decode levers (multi-day): MoE gather redesign, attention
restructure, exact MTP, prefetch reserve-fill-publish; bench-harness fix for
GLM THINKING token accounting.
