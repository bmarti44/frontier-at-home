# G2 DFlash — Laguna S 2.1 UD-Q4_K_XL — PASS (serving candidate: n-max 4)

Window: 2026-08-21, gate script `g2_dflash_laguna.sh`. Engine poolside
llama.cpp `laguna` @ 06f8cebd; draft `laguna-s-2.1-DFlash-BF16.gguf`
(`--spec-type draft-dflash`). Production stopped/restored via switch
stop/restore; healthy on 8013 after.

## Results

- **Single-BOS check** (G1 follow-up): encoder text rendered with
  `add_default_bos_token=False` + `/tokenize` `add_special=true` →
  tokens[0]=2, tokens[1]=97 — exactly one leading BOS/EOS token. PASS
  (single-bos-check.json).
- **Speed, 384-token greedy code probes** (tok/s wall incl. prefill):

  | config | p0 merge-intervals | p1 LRU cache | p2 bash | mean | floor |
  |---|---|---|---|---|---|
  | draft off | 25.18 | 26.09 | 26.01 | 25.76 | 25.18 |
  | n-max 4 | 32.65 | 45.16 | 28.34 | **35.38** | **28.34** |
  | n-max 7 | 26.39 | 46.60 | 22.39 | 31.79 | 22.39 |
  | n-max 10 | 24.58 | 52.63 | 21.76 | 32.99 | 21.76 |

  **n-max 4 wins on mean and floor** (deep drafts win peaks on
  high-acceptance prompts but pay overhead on low-acceptance ones).
- **Greedy equivalence: NOT exact** (equivalence.json): p1 identical in
  all four configs; p0 diverges @char837, p2 @char86, and the three
  draft depths also differ among themselves on those prompts →
  depth-dependent numeric nondeterminism, not a systematic draft bias.
  Consequence (DSpark rule): **G4 fidelity runs on the exact serving
  config (DFlash n-max 4)**, not on draft-off.
- **Streaming granularity**: mean 2.1 chars/event, max 10, 128 events for
  128 tokens → per-token streaming; **strict 30_bench cells are valid
  with DFlash on** (unlike SGLang DSpark block streaming).

## Verdict

G2 PASS. Serving candidate: UD-Q4_K_XL + DFlash `--spec-draft-n-max 4`,
thinking max. Proceed to G3 strict speed cells (draft-off and n4 cells)
and G4 fidelity holdout on the n4 config.
