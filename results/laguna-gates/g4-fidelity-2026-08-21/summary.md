# G4 fidelity — Laguna S 2.1 UD-Q4_K_XL + DFlash n4 — PASS (code-profiled)

Windows: 2026-08-21/22. Serving config: the production shape (4 slots,
DFlash --spec-draft-n-max 4, thinking max, budget 16384, encoder laguna
with add_default_bos_token=False, --config-evidence = build manifest +
weights manifest + G3 dflash-n4 bench). DFlash is not greedy-exact (G2),
so all suites ran on the serving config per the DSpark rule.

## History (all preserved)

- Dev truncation probe (gsm8k dev, 100 items): accuracy 0.90,
  invalid 0 → 16384 budget cleared for spend.
- First window served -c 524288 -np 4: gsm8k holdout completed
  (86/100), then the **8 GiB memwatch floor breached at 7.95 GiB**
  ~2.5 h in (lazy KV + prompt-cache growth; steady state hovered
  ~9 GiB) and the watchdog killed the server mid-mmlu-pro. That run is
  archived as acc-mmlu-pro-holdout-CRASHVOID-524k.json (206/247 items
  were Connection refused — void, not model output). Production shape
  corrected to -c 393216 -np 4 (commit ed671e41); the re-run held
  ~15 GiB available throughout.
- mmlu-pro holdout re-spent under ledger namespace **laguna-v2**
  (crash-void justification; first-look items: 41 of 247 saw answers).
  gsm8k remains namespace laguna. HumanEval supports --split all only.

## Scores (vs qwen38-1m production references)

| suite | Laguna | qwen38-1m | note |
|---|---|---|---|
| HumanEval (all, 164) | **89.63** (147/164) | 79.27 | **+10.4 — best HumanEval on this host** |
| GSM8K holdout (100) | 86.00 (86/100) | 98.00 | invalid 0; real misses |
| MMLU-Pro holdout (247) | 63.56 (157/247) | 85.02 | 64 invalid ≈ all 16K-budget truncation of max-thinking; knowledge score is truncation-suppressed |

One HumanEval miss was a sandbox failure (exit 1), counted as a miss
fail-closed.

## Verdict

G4 PASS for qualification as a **code-focused switchable engine**:
decisive HumanEval win on the serving config, math/knowledge suites
below the qwen default (owner decision 2026-08-21: Laguna will NOT be
the serving default). Truncation sensitivity at 16384 with max thinking
is a recorded limitation; raising the budget was not exercised.
