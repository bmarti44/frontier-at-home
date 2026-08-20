# Track C gate 3 — f16 KV at -np 4 / 1M — 2026-08-19 — ADOPTED

Same config as gate 1 but f16 KV (no -ctk/-ctv), caps 88G/95G.

- Ready at 16 GiB resident (lazy commit), 21 GiB after 28K bench;
  worst case ~83 GiB with 4 full f16 slots — inside caps.
- Sanity code decode 29.72 tok/s (vs 28.79 q8_0); MTP fine (58/83).
- Strict cells (bench-np4-kvf16.json, suite valid): decode 17.50 @0 /
  **26.68 @28K** — identical to the qualified 32K profile (26.71), vs
  q8_0's 23.06.

Decision: **f16 KV adopted** for qwen38-1m: decode faster at every measured
context; 28K median prefill slightly lower and TTFT slightly higher. It
fits the envelope and carries zero fidelity delta by
construction (same numerics as the qualified baseline), so no further
suite gate is required. q8_0 results remain documented (gate 2) as the
fallback if memory headroom is ever needed.
