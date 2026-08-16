# Direct-slot expert dispatch A/B — 2026-08-16

Change: DS4_CUDA_EXPERT_DIRECT_SLOT (default off). Routed-MoE kernels read
expert weights directly from the persistent host arena via GB10 coherent
pageable access (cudaDevAttrPageableMemoryAccess=1; cudaHostRegister is
unsupported on GB10 and unnecessary). Pointer table (24 uint64) rides the
existing remap H2D payload; per-slot protection bitmap defers eviction of
selected slots until the next token (drain safety verified by review:
per-layer completion ds4.c:45444/46102, blocking logits readback ds4.c:46161).

Measured (v0 128-expert artifact, cache 94 decimal GB = full coverage,
DS4_DECODE_STAGE_TIMING, warm, ctx≈30K, greedy):

| Stage ms/token | OFF | ON |
|---|---|---|
| copies (arena→staging) | 112 | **0.00** |
| routed MoE kernels | 101.6 | 101.9 |
| attention+dense | 186 | 187 |
| indexer | 11.6 | 11.7 |
| router/remap (declining tail) | 48 | 39 |
| **total** | **459** | **339** |

Decode @30K: 2.33 → **2.95 tok/s (+27%)**. Outputs byte-identical ON vs OFF
(two greedy prompts). Sol-high adversarial review: no critical/high; mediums
(single-request concurrency assumption documented — server serializes via
inference_mu with -np 1; OFF-path adds an inlined null-check, needs the
matched-block gate before production promotion; tensor-type eligibility to be
asserted) queued as hardening.
