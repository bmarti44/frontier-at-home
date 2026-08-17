# Full-model Q4-dense candidate — 2026-08-17

Artifact: glm52-full-denseq40.gguf (202.3 GB) — the ORIGINAL 256-expert model
with dense/attention/shared/output Q8_0→Q4_0 (embeddings Q8_0; expert tensors
byte-identical to source, streaming machinery unaffected). Engine: 4d54edd +
expert-count + direct-slot + Q4_0 GEMV/batch (optimized) + stage timing.

## Fidelity (paired 30 cases vs bench3 full-Q8 reference)
- NLL 0.5083 → 0.6002 (+0.092); top-1 rate 0.8097 → 0.8097 (unchanged)
- Owner accepted this delta as the working candidate (2026-08-16/17),
  contingent on nothing being lost from MoE streaming (confirmed: expert
  bytes untouched; original GGUF preserved; requires the new engine build).

## Speed (warm, ctx≈30K, cache 94 decimal GB = 47% of experts, direct-slot)
- decode ~360-370 ms/token ≈ 2.7-2.8 tok/s (vs 2.33 production baseline)
- split: attn_dense ~148 (was 187 at Q8), MoE 102, router+miss ~100, idx 12
- prefill improved (Q4 batch kernels); misses stream — fidelity-free by design

## Prefetch A/B (ported 846dca4 worker, DS4_GLM_PREFETCH=1)
- Outputs byte-identical on/off; minflt drops (prefetch active) but decode
  NET-NEUTRAL (totals ~360-373 both arms) — consistent with the original
  "net-neutral pending reserve-fill-publish" caveat. Not adopted; port
  preserved at patches/glm-prefetch-port.patch. The identified gap: fills
  are not published/claimed in time with one-layer lookahead.

## Remaining decode levers on this candidate
router+miss ~100 ms (GPU directory / deeper lookahead / reserve-fill-publish),
MoE kernels ~102 ms, attention-at-depth ~80 ms, exact MTP multiplier.

## GPU-resident directory A/B (2026-08-17)
Byte-identical on/off; NET-NEUTRAL (router 96-106 off vs 107-129 on ms/token).
Cause: at 47% cache capacity only ~36% of layers are all-hit (0.88^8), so the
host fallback still dominates and the poll adds overhead. Patch preserved
(patches/expert-gpu-directory.patch); revisit only if per-layer all-hit rates
rise materially. Router-bucket floor at this capacity ≈ miss streaming +
selected-ID dependency; three approaches (prefetch, GPU dir, async spelling)
have now failed to move it — deprioritized in favor of attention and MoE
gather latency.
