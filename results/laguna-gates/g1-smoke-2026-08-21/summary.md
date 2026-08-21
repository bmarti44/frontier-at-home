# G1 smoke — Laguna S 2.1 UD-Q4_K_XL raw (no DFlash) — PASS

Window: 2026-08-21, gate script `g1_smoke_laguna.sh` (attempt 4; attempts
1–3 failed on infrastructure, not the model: sudo-grant gap for stopping
qwen production → `stop` verb added to 52_engine_switch.sh; switch.lock
fd-leak deadlock → fixed, see docs/RUNBOOK-stuck-switch.md; /run/dsv4
inaccessible to bmarti44 → serve state moved to /run/user/1000).
Engine: poolside llama.cpp `laguna` @ 06f8cebd (version 10010).
Production qwen38-1m stopped via `switch stop` and restored+verified via
`switch restore` on 8013 afterwards.

## Results

- **Load**: UD-Q4_K_XL (73.4 GB, 3 shards) + 65,536 ctx f16 KV loads
  clean; MemAvailable 42.5 GiB at idle-loaded (baseline ~115.7 free with
  production stopped → footprint ≈ 73 GiB incl. 3.4 GiB KV reserve).
- **KV cache**: 39.2 GiB avail @131,072 ctx vs 42.5 @65,536 →
  **≈52.8 KiB/token** (3.3 GiB per extra 64K). Implication: 1M ctx needs
  ≈52 GiB KV on top of 73.4 GiB weights — **does not fit** in 119.7 GiB.
  Practical f16-KV ceiling ≈ 500–600K total slot budget; revisit with
  q8_0 KV (would halve to ~26 KiB/token) as a gated experiment.
- **Greedy canary**: coherent ("A mutex protects against race conditions
  by ensuring…") — canary.json.
- **Code probe (raw, thinking=max, greedy, 512 tokens)**:
  **27.62 tok/s wall-clock incl. prefill** — code-probe.json. Community
  reference for raw Spark decode was 12.6 tok/s; already above the
  ≥20 tok/s promotion bar BEFORE the DFlash speed lever (G2).
- **Template byte-check**: server /apply-template output differs from the
  encoder ONLY by the leading `〈|EOS|〉` glyph (template-check.json).
  Root cause: GGUF `add_bos_token=true` with BOS==EOS==token 2 —
  llama.cpp strips the leading BOS text and adds the token at
  tokenization. Encoder gained real `add_default_bos_token` semantics and
  31_bench renders laguna with it False (ENCODER_EMIT_BOS_TEXT), so
  /v1/completions prompts carry exactly one leading token 2. Token-level
  single-BOS assertion scheduled into the G2 window.

## Verdict

G1 PASS. Proceed to G2 (DFlash greedy-equivalence + draft-depth code
probes + /tokenize single-BOS check).
