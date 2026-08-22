# G3 strict speed cells — Laguna S 2.1 UD-Q4_K_XL — PASS

Window: 2026-08-21, `g3_speed_laguna.sh`. 30_bench_speed strict cells,
5 measured reps + 1 warmup per cell, seed 42, ignore_eos, per-token SSE
timestamps (G2 verified per-token streaming with DFlash on), Laguna
output tokenizer pinned (sha 809240f7…, vocab 100,352), serve
-c 65536 single slot (dev shape; fixture renders 40,657 input-tokenizer
tokens — no truncation at this ctx). Both suites `suite_valid=true`.
Production stopped/restored via switch stop/restore; healthy after.

## Cells (median of 5 valid reps)

| config | ctx | decode tok/s | prefill tok/s | TTFT s |
|---|---|---|---|---|
| raw (no draft) | 0 | 26.56 | 195.1 | 0.51 |
| raw (no draft) | 28672 | 20.97 | 651.0 | 54.65 |
| DFlash n-max 4 | 0 | 26.98 | 170.0 | 0.60 |
| DFlash n-max 4 | 28672 | **26.90** | 622.4 | 57.16 |

- DFlash n4 holds decode flat at depth: 26.90 vs 20.97 raw @28K (+28%).
- Reference: qwen38-1m production README cell is 26.71 decode @28K —
  Laguna at the serving-candidate config is at parity on the strict
  fixture, with the G2 code probes (28-45 tok/s) suggesting a code-heavy
  advantage from DFlash acceptance.
- Prefill ~620-650 tok/s @28K, TTFT ~55-57 s @28K (qwen38-1m: 698.7 /
  ~64 s on its own fixture rendering — comparable class).

## Verdict

G3 PASS. Speed bar (≥20 code tok/s with DFlash) cleared with margin.
Proceed to G4 fidelity holdout on the production shape
(-c 524288 --parallel 4, DFlash n-max 4).
