# MiMo-V2.6-Flash long-generation breakdown: diagnostic (preregistered)

**Label: DIAGNOSTIC.** This run locates a defect. Nothing here is headline,
README or acceptance evidence.

## Why

The holdout draw (`../holdout-2026-10-13/`) found MiMo's long answers
breaking down on the qualified launch:
- **MMLU-Pro:** 23 of 247 answers degenerated into repetition and garbage
  ("the the the…", stray tokens), at temperature 0. That is 23 of the 30
  answers longer than 3,000 generated tokens. Flash-Next had none.
  - 14 of MiMo's 18 invalid MMLU-Pro answers are among them.
- **Where it starts:** at an estimated **absolute position of about 4,096
  tokens** (prompt plus generated) in most cases, with a few near 8,192 and
  12,288. Estimated from each answer's mean characters per token.
- **HumanEval:** the first raw completions run about 4,950 tokens and
  degenerate the same way past about 4,100.
- **Prefill is not affected:** candidate 3's direct 250K fill retrieved its
  needle.

**Hypothesis.** TabbyAPI's output chunking is on by default, so
`max_rq_tokens` equals the 2,048-token chunk size.
- exllamav3 aligns each requeue to the next recurrent-checkpoint boundary
  (MiMo's interval is 2,048). For any prompt longer than one token, the
  first requeue therefore lands at absolute position 4,096.
- The requeued job resumes from the recurrent (sliding-window) checkpoint
  stashed there.
- exllamav3's own comment says a speculative (DFlash) window writes K/V past
  that boundary. A checkpoint that captures it would corrupt the
  sliding-window state from then on.

## What runs (`run-longgen.sh`, one window, three fresh starts)

The production stop, inference lock, identity check, start gate,
containment, observers and restore are the holdout driver's
(`../holdout-2026-10-13/run-holdout.sh`, reviewed). Containment:
- scope MemoryHigh 8G / MemoryMax 16G, MemorySwapMax 0, `OOMPolicy=kill`;
- the profile memwatch at 8 GiB;
- MemFree ≥ 108 / MemAvailable ≥ 114 GiB before each start;
- `RuntimeMaxSec` 1 h per arm.

The arms run in this fixed order on :8019:

| arm | profile | the one change |
|---|---|---|
| a | `mimo-v2.6-flash/cuda-spark-128g-262k` (qualified) | none |
| b | `…-262k-longgen-b` | `--output-chunking false`: the completion is allocated up front and never requeued |
| c | `…-262k-longgen-c` | no DFlash drafter |

The variant profiles are exact copies of the qualified one apart from that
change. They are `estimated`, have no switch alias, and use their own log
names and units.

Each start runs the restart probe (`../smoke-2026-10-13/probe.py`, recorded
only), then `longgen.py run`. The fixtures are fixed here, and none comes from
an evaluation set:

| fixture | request | budget | check |
|---|---|---|---|
| `count-raw` | raw completion of the lines `1`…`200` | 6,144 | the lines must continue `201, 202, …` exactly |
| `count-chat` | "Write the integers from 1 to 3000…", rendered by the `mimo26` encoder in thinking mode | 8,192 | after `</think>`, the lines must be `1, 2, …` exactly |
| `essay-chat` | a ≥ 6,000-word technical essay, same rendering | 8,192 | degeneration only |

Every request is a `/v1/completions` call at temperature 0, seed 42, the
holdout's path.

## Scoring (fixed; `longgen.py verdict`)

**Per fixture:**
- **reached:** prompt plus completion tokens ≥ 4,600, so the output passed
  the 4,096 boundary with margin.
- **broken** if either holds:
  - degeneration: one word repeated 8 or more times in a row, or one
    character 60 or more times in a row;
  - for a counting fixture, a line that is not the next integer. A last line
    cut by the budget may be a prefix of the expected one.
- The break position is reported as an approximate absolute token position.

**The verdict:**
- **`NO_RESULT`:** an arm has no record, or a malformed one.
- **`NOT_REPRODUCED`:** no fixture that reached the boundary broke in arm a.
- Otherwise, let *B* be the fixtures that broke in arm a. An arm is *clean*
  if every fixture in *B* reached the boundary in that arm without breaking.
  - **`OUTPUT_CHUNKING`:** b is clean and c is not.
  - **`DRAFTER`:** c is clean and b is not.
  - **`CHUNKING_AND_DRAFTER`:** both are clean; the breakdown needs requeueing
    and drafting together.
  - **`NEITHER`:** neither is clean.

**What each verdict leads to.** These are proposals to the owner, not
actions this run takes:
- **`OUTPUT_CHUNKING` or `CHUNKING_AND_DRAFTER`:** the smallest fix is
  `--output-chunking false` in the qualified profile. It changes the
  launch, so it needs a test-first change and a new acceptance candidate.
- **`DRAFTER`:** the fix is in the drafter path. Serving without DFlash
  costs decode speed.
- **`NEITHER` or `NOT_REPRODUCED`:** the next bounded alternative is a
  window that replays the holdout's already-scored failing prompts, and
  varies the recurrent checkpoint interval and the cache mode.

Each arm's decode rate is reported beside it, as diagnostic only.

## Safety

- A kill line or a memwatch breach ends the window: no further arm starts.
- Any kill line, a failed restore, or a low below the 8 GiB floor is
  reported to the owner.
- Production is down for about 1–1.5 h; GPU windows are pre-approved (owner,
  2026-10-12).
