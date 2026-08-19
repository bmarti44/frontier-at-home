# Qwen3.8-27B accuracy gate — 2026-08-18/19

Config: Q4_K_M + MTP (draft-mtp n=2, greedy-exact per smoke gate), encoder
qwen38, thinking mode, reasoning effort LOW (owner default), max_tokens
16384 (owner default), ctx 40960, b10488 (sha bcef273b…), port 8015.
Config hash: qwen38-q4km-b10488-mtp2-ctx40960-thinking-low-mt16384-v3.
Encoder cross-checked byte-equal against the GGUF template via
/apply-template before each window.

## Results (all zero invalid)

| Suite (split) | Qwen3.8-27B Q4_K_M | DSV4-flash reference |
|---|---|---|
| GSM8K (holdout, 100) | **98.0** | 97.0 |
| MMLU-Pro (holdout, 247) | **85.02** | 74.09 |
| HumanEval (all, 164) | **79.27** | 73.78 |

Q4_K_M beats the DSV4 reference on every suite at LOW reasoning effort.
Per the plan rule (Q6_K only if Q4 disappoints) the Q6_K accuracy pass is
skipped; Q4_K_M is the fidelity candidate.

## Ledger honesty trail

- v1 (max_tokens 512, template-default effort): GSM8K holdout 90/100 —
  8 of 10 misses were finish_reason=length truncations. Recorded in the
  holdout ledger under config …-thinking-v1; superseded, not hidden.
- v2 (8192): aborted before any holdout rows ran (ledger refused re-spend;
  owner then authorized).
- v3 re-spend authorized by owner 2026-08-18 under ledger namespace
  mt16k-low with the corrected budget; both entries remain in
  results/holdout-ledger.json.

Transcripts: scratchpad (session); combined sha256 digests per suite in
transcript-digests.json (gsm8k 100, mmlu-pro 247, humaneval 164 files).
