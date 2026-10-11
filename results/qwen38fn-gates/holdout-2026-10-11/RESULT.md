# Holdout result: Flash-Next is more accurate than the default on text

The rule in `README.md` was committed before the draw. Under it, Flash-Next is
**more accurate on text**: 454 of 511 correct against the default's 438, with
no suite significantly worse.

## The run (2026-10-11, one fresh start, head `b196d4e4`, clean)

- **Start:**
  - Preload MemAvailable was 115.1 GiB.
  - The server was ready in 638 s.
  - vLLM's KV capacity was 1,079,117 tokens at 262,144.
  - Idle MemAvailable was 13.81 GiB.
- **Restart probe:** byte-identical to candidate 5 on all three hashes, so the
  holdout was drawn (08:40Z).
- **Safety:**
  - 0 kill lines (OOM/NVRM/Xid, memwatch breach);
  - external low point 12.74 GiB;
  - swap growth 0.00 GiB over 12,932 one-second samples;
  - 0 transport errors.
- **Ledger:** `results/holdout-ledger.json` has a `started` and a `completed`
  entry for each of GSM8K and MMLU-Pro. Stack
  `qwen3.8-flash-next-cuda-spark-128g-1m`, namespace `""`.

## Scores (thinking, effort low, 16,384-token budget, the default's contract)

| Suite | Flash-Next | Default (`qwen38-1m`) | Delta | z (one-sided worse if < −1.645) |
|---|---|---|---|---|
| GSM8K holdout | 98/100 (0 invalid) | 98/100 | +0 | +0.000 |
| MMLU-Pro holdout | **218/247** (88.26%; 3 invalid: truncated at the budget without an answer) | 210/247 (85.02%) | +8 | +1.058 |
| HumanEval (all) | **138/164** (84.15%) | 130/164 (79.27%) | +8 | +1.143 |
| **Sum** | **454/511** | 438/511 | +16 | — |

## Run-to-run variation (reported, not hidden)

Candidate 5's start 2 scored HumanEval 139/164 on the same configuration, a
few hours earlier. Exactly one problem differs: HumanEval/28.
- In candidate 5 it stopped after 4,161 tokens and passed.
- Here greedy decoding ran to the 16,384-token budget and failed in the sandbox.

The three short probes are byte-identical across all three starts. So
long-generation greedy output on this vLLM + MTP stack is not bit-stable
across fresh starts. Flash-Next's HumanEval is 138–139/164 over two runs. The
comparison uses this run's 138, as the README fixed.

## What this does not cover

- **Images.** The default serves images (MMMU-val-100 64/100). This
  Flash-Next profile is text-only, because images at 1M are `NO_GO`
  (`../diag-vision-2026-10-11/RESULT.md`).
- **Speed** is candidate 5's bundle (`../qualify-2026-10-11-c4/RESULT.md`).
  Short-prompt TTFT, 0.569 s against 0.39 s, is the one miss.
- **Promotion** is the owner's decision.
