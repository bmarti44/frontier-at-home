# Track C gate 2 — fidelity under q8_0 KV + deep retrieval — 2026-08-19

Same serve config as gate 1 (-c 1048576 -np 4, q8_0 KV, MTP n8p6, low
effort, 16384 budget). Holdout ledger namespace trackc-np4q8.

## Needle retrieval (code-flavored, planted at 35% depth)
- found at **158,598 tokens** (prefill 497 t/s at depth)
- found at **249,489 tokens** — 95% of the native 262K slot cap
  (prefill 336 t/s, measured concurrent with a running suite)
- two probes exceeded the per-slot cap due to a chars/token misestimate
  (400 exceed_context_size, recorded in needle.json) — sizing artifact,
  not a model failure.

## Accuracy suites under q8_0 KV (vs f16 baseline)
| suite | q8_0 | f16 | delta |
|---|---|---|---|
| GSM8K holdout | 98.0 | 98.0 | 0 |
| MMLU-Pro holdout | 84.21 (1 invalid) | 85.02 | -0.81 |
| HumanEval | 77.44 | 79.27 | -1.83 |

q8_0 KV is not free on this model: ~-1 to -2 points on MMLU/HumanEval
(3 HumanEval items). Gate 3 therefore measures f16 KV at the same np4
config — the memory math fits (~88 GB worst case) — to obtain a
fidelity-free 1M profile; q8_0 stays the documented fallback where
memory headroom matters.
