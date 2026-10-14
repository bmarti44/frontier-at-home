# Holdout result: MiMo-V2.6-Flash is not more accurate on text

Under the rule in `README.md`, committed before the draw, MiMo is **not more
accurate on text**. All three suites completed. Two of them are
significantly worse than Flash-Next, the default:
- **MMLU-Pro** (z −2.357);
- **HumanEval** (z −2.906).

The sum is 412 of 511 against Flash-Next's 454.

The draw also exposed a defect: **MiMo's long generations break down** on
this launch. It is described below and is being diagnosed separately
(`../longgen-2026-10-14/`). It does not explain the gap. On the rows where
MiMo did not break down, it is still significantly worse on both suites.

## The run (2026-10-14, one fresh start, head `12a2e5a7`, clean at launch)

- **Start:**
  - The driver checked a clean tree at launch. The kit's manifest says
    `dirty: true` only because this run's own evidence files were already
    untracked in the tree.
  - The render was identical to candidate 3's (`config_hash
    …-eac45232288f`), and the code-identity refusals passed.
  - The start gate read MemFree 113.2 / MemAvailable 115.3 GiB.
  - The server was ready in 56 s, with idle MemAvailable at 12.50 GiB.
- **Restart probe:** byte-identical to candidate 3 on all three hashes, so the
  holdout was drawn (00:54:40Z).
- **Commits during the run:** five commits landed on the branch while it ran.
  They touched only the long-generation diagnostic, its two variant profiles,
  the shim's flag list and tests. Nothing a running process had loaded
  changed, and `run-holdout.sh` was not touched.
- **Safety:**
  - 0 kill lines (OOM/NVRM/Xid) and no memwatch breach;
  - external low point 10.22 GiB (03:37:13Z), over 25,554 one-second samples;
  - swap growth 0.00 GiB;
  - 0 transport errors.
- **End:**
  - The kit finished at 08:02:16Z, 7.1 h after the probe, inside the 11.5 h
    cut.
  - The `setsid` restore of `qwen38fn` returned rc 0 at 08:13:46Z, and the
    driver exited with rc 0.
- **Ledger:** `results/holdout-ledger.json` has a `started` and a `completed`
  entry for each of GSM8K and MMLU-Pro. Stack
  `mimo-v2.6-flash-cuda-spark-128g-262k`, namespace `""`.

## Scores

`breakdown_analysis.py` reproduces every number here from the transcripts
(`breakdown-analysis.json`). Each suite's `rowset_sha256` equals Flash-Next's,
so both models answered the same questions.

| Suite | MiMo | Flash-Next | Delta | z (one-sided; worse if < −1.645) |
|---|---|---|---|---|
| GSM8K holdout | 97/100 (0 invalid) | 98/100 | −1 | −0.453 |
| MMLU-Pro holdout | 199/247 (80.57%) | 218/247 (88.26%) | −19 | **−2.357**: significantly worse |
| HumanEval (all) | 116/164 (70.73%) | 138/164 (84.15%) | −22 | **−2.906**: significantly worse |
| **Sum** | **412/511** | 454/511 | −42 | — |

**MiMo's invalid answers:**
- **MMLU-Pro**, 18: 13 have no anchored answer and 5 were truncated at the
  budget without one. Flash-Next had 3.
- **HumanEval**, 1: a 10 s sandbox timeout. Its 47 other failures are the
  sandbox's `exit 1`.

**The protocols differ on GSM8K and MMLU-Pro**, as the README fixed:
- MiMo ran its template's default thinking through the `mimo26` encoder,
  which has no effort level.
- Flash-Next ran thinking at effort `low`.

HumanEval is the same raw completion for both. All three suites used the
same 16,384-token budget, 2,700 s timeout and scorer.

## Token use (reported)

| Suite | MiMo tokens | Flash-Next tokens | MiMo at the budget | Flash-Next at the budget |
|---|---|---|---|---|
| GSM8K | 14,334 | 23,945 | 0 | 0 |
| MMLU-Pro | 341,279 | 257,939 | 6 | 3 |
| HumanEval | 1,212,948 | 193,523 | 48 | 11 |

In raw completion, MiMo keeps writing after the function: tests, prints and
more functions. It reached the 16,384-token budget on 48 of 164 items, which
took HumanEval 4.7 h.

## Defect: long generations break down (DIAGNOSTIC, written after the draw)

**What it looks like.** Past a few thousand generated tokens, MiMo's output
on this launch turns into word repetition ("the the the…") and stray tokens
in scripts an English answer never uses (Devanagari, Thai, Hangul, Arabic).
- **Where it starts:** the estimated onset is near an absolute position
  that is a multiple of 2,048, at or above 4,096.
- **How often** (late breakdowns, with an onset at 3,000 or more generated
  tokens):

  | Suite | MiMo | Flash-Next | Note |
  |---|---|---|---|
  | GSM8K | 0 | 0 | no answer runs that long |
  | MMLU-Pro | 26 | 0 | 26 of MiMo's 30 answers over 3,000 tokens |
  | HumanEval | 37 | 0 | words and stray scripts only, since code legitimately holds long digit runs |

- **Not seen elsewhere:** the M3 gate's prefill (a direct 250K fill) and its
  vision answers did not break down. The gate never tested long decode.

**It does not explain the gap.** Paired per row (exact McNemar,
two-sided):

| Suite | Rows | MiMo | Flash-Next | Only MiMo right | Only Flash-Next right | p |
|---|---|---|---|---|---|---|
| MMLU-Pro, MiMo not broken | 221 | 190 | 204 | 5 | 19 | 0.0066 |
| MMLU-Pro, MiMo broken | 26 | 9 | 14 | 3 | 8 | 0.23 |
| HumanEval, MiMo not broken | 127 | 92 | 110 | 8 | 26 | 0.0029 |
| HumanEval, MiMo broken | 37 | 24 | 28 | 4 | 8 | 0.39 |

The broken rows are hard ones: Flash-Next also missed 12 of the 26 MMLU-Pro
rows. Even if MiMo had matched Flash-Next on every broken row, it would
still trail by 14 on MMLU-Pro and 18 on HumanEval, and both suites would
stay significantly worse (z −1.785 and −2.426).

This split is post hoc and diagnostic. It does not change the label above.

**What is next.** `../longgen-2026-10-14/PREREG.md` tests the leading
hypothesis: the breakdown comes from TabbyAPI's output chunking requeueing a
DFlash-drafted generation at exllamav3's 2,048-token recurrent checkpoint.
It uses three fresh starts on equal fixtures.

## What this does not cover

- **Speed and the gates:** see candidate 3's bundle
  (`../qualify-2026-10-13-c1/`).
- **Images:** MMMU-val-100 is in that bundle too.
- **A run with the defect fixed.** Any fix changes the launch, so it needs a
  new acceptance candidate. A new holdout draw would need the owner's
  authorization.
- **Promotion** is the owner's decision. MiMo stays a switchable alias, and
  `qwen38fn` stays the default.
