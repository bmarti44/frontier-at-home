# v0 pruned-resident probe — 2026-08-16

Artifact: `glm52-pruned-v0-128exp.gguf` (115,823,173,696 bytes) — whole-expert
prune of the production GLM-5.2 IQ2_XXS GGUF from 256 → 128 experts across all
76 routed layers (keep-list: experts 0..127, arbitrary; v0 is a speed probe,
not a fidelity candidate). Built and verified by
`scripts/58_prune_glm_experts.py`: 48,640 per-expert SHA-256 checks PASS,
1,429 non-expert tensors byte-identical, expert-count metadata 256→128.

Engine: `glm-dynamic-expert-count.patch` applied to the ds4 working tree at
`e0ae648` + local diff (snapshot `~/ds4-source-snapshot-2026-08-15`), built
`make cuda-spark -j2`, CUDA 13.0. **Byte-identity A/B at 256 experts: PASS**
(two greedy prompts byte-identical between baseline and patched builds,
identical wall time).

Serving: `--ssd-streaming --ssd-streaming-cache-experts 40GB` +
`DS4_CUDA_EXPERT_CACHE_GB=88` (decimal → 81.95 GiB arena, 9,042 slots = 94.2%
of the 9,600 uniform-class experts), PIN ok, SLRU on. blk.78 (Q2_K slab class)
bypasses the cache by design, as predicted by the source audit.

## Measurements (server-side per-request timing, warm cache)

| Metric | Value |
|---|---|
| Decode @ ctx≈30K (warm, reps agree) | **2.33–2.34 tok/s** (~429 ms/token) |
| Decode @ shallow ctx (warm) | **3.09 tok/s** (~323 ms/token) |
| Prefill 30K | **~50.1–50.2 tok/s** (vs ~43 unpruned evidence-mode) |
| Evidence-mode reference (cache off, unpruned) | 1.08 decode |
| Production fast path (unpruned, cached) | 2.33 decode |

## Conclusion

Residency alone does not raise decode: v0 at 94% expert residency decodes at
the same 2.33 tok/s as the unpruned production path. The cache-hit path is not
a resident-compute path — per token it still performs ~600 CPU hash lookups,
an unconditional SHA-256 access-stream update, ~1,800 arena→staging
`cudaMemcpyAsync` copies (~5.8 GB), 75 blocking ID readbacks, then the same
bandwidth-bound kernels. Full decomposition and the ranked lossless fix list:
`decode-hotpath-analysis-2026-08-16.md`. Physics: ~24.6 GB unique weight
bytes/token (18.6 GB non-routed Q8) → ~11 tok/s roofline for this artifact;
lossless engineering target 4–6 tok/s; dense Q8→Q4 requant + exact MTP raise
the plausible landing zone to ~8–12.

Owner decision (2026-08-16): **full push** — kernel campaigns + v1 artifact
(saliency prune + dense-tensor Q4 requant, NLL-measured, owner approves the
delta) + exact MTP. 18 tok/s DSV4 parity is recorded as not achievable on this
hardware for this model.
