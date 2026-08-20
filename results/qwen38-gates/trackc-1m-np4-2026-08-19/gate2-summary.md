# Track C gate 2 — np4-1m-q8_0 quality comparison + deep retrieval — 2026-08-19

Same serve config as gate 1 (-c 1048576 -np 4, q8_0 KV, MTP n8p6, low
effort, 16384 budget). Holdout ledger namespace trackc-np4q8.

## Needle retrieval (code-flavored, planted at 35% depth)
- found at **158,598 tokens** (prefill 497 t/s at depth)
- found at **249,489 tokens** — 95% of the native 262K slot cap
  (prefill 336 t/s, measured concurrent with a running suite)
- two probes exceeded the per-slot cap due to a chars/token misestimate
  (400 exceed_context_size, recorded in needle.json) — sizing artifact,
  not a model failure.

## Accuracy suites for np4-1m-q8_0 (vs cited qualified baseline)
| suite | np4-1m-q8_0 config | cited baseline | delta |
|---|---|---|---|
| GSM8K holdout | 98.0 | 98.0 | 0 |
| MMLU-Pro holdout | 84.21 (1 invalid) | 85.02 | -0.81 |
| HumanEval | 77.44 | 79.27 | -1.83 |

The np4-1m-q8_0 configuration scored ~1 to 2 points lower on
MMLU-Pro/HumanEval (3 HumanEval items) than the cited f16 baseline. This
delta is not isolated to KV format: the arms also differ in parallelism,
total context, and MTP settings. Gate 3 therefore measures f16 KV at the
same np4 configuration — the memory math fits (~88 GB worst case) — for
the selected 1M profile; np4-1m-q8_0 remains the documented fallback
configuration where memory headroom matters.
