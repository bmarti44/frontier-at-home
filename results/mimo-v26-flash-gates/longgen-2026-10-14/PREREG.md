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
- **Where it starts:** at an estimated absolute position (prompt plus
  generated) **near a multiple of 2,048, at or above 4,096**: 13 near 4,096,
  3 near 6,144, 2 near 8,192 and 2 near 12,288 (reviewer A's count). Each is
  estimated from the answer's mean characters per token.
- **HumanEval:** the first raw completions run about 4,950 tokens and
  degenerate the same way past about 4,100.
- **Prefill is not affected:** candidate 3's direct 250K fill retrieved its
  needle.

**Hypothesis.** TabbyAPI's output chunking is on by default, so
`max_rq_tokens` equals the 2,048-token chunk size.
- exllamav3 aligns each requeue to the next recurrent-checkpoint boundary
  (MiMo's interval is 2,048). For a prompt of 2 to 2,049 tokens, which
  covers every prompt here, the first requeue lands at absolute position
  4,096, and later requeues every 2,048 after it.
- The requeued job resumes from the recurrent (sliding-window) checkpoint
  stashed there.
- exllamav3's own comment says a speculative (DFlash) window writes K/V past
  that boundary. It tries to handle that: it rejects unconsumed drafts before
  stashing at a requeue. A gap in that handling would corrupt the
  sliding-window state from then on.
- Every arm still stashes recurrent checkpoints while decoding at each 2,048
  boundary. So `DRAFTER` or `NEITHER` would point at the stash combined with
  the drafter, not at the drafter alone.

## What runs (`run-longgen.sh`, one window, three fresh starts)

The production stop, inference lock, identity check, start gate,
containment, observers and restore are the holdout driver's
(`../holdout-2026-10-13/run-holdout.sh`, reviewed). Containment:
- scope MemoryHigh 8G / MemoryMax 16G, MemorySwapMax 0, `OOMPolicy=kill`;
- the profile memwatch at 8 GiB;
- MemFree ≥ 108 / MemAvailable ≥ 114 GiB before each start;
- `RuntimeMaxSec` 3,900 s per arm.

The arms run in this fixed order on :8019:

| arm | profile | the one change |
|---|---|---|
| a | `mimo-v2.6-flash/cuda-spark-128g-262k` (qualified) | none |
| b | `…-262k-longgen-b` | `--output-chunking false`: the completion is allocated up front and never requeued |
| c | `…-262k-longgen-c` | no DFlash drafter |

The variant profiles are exact copies of the qualified one apart from that
change. They are `estimated`, have no switch alias, and use their own log
names and units.

Before anything stops, the driver renders every arm and runs the TabbyAPI
shim's own flag check on it.

Each start runs the restart probe (`../smoke-2026-10-13/probe.py`, recorded
only), then `longgen.py run`. The fixtures are fixed here:

| fixture | request | budget | check | primary |
|---|---|---|---|---|
| `count-raw` | raw completion of the lines `1`…`200` | 6,144 | the lines must continue `201, 202, …` exactly | yes |
| `count-chat` | "Write the integers from 1 to 3000…", rendered by the `mimo26` encoder in thinking mode | 6,144 | after `</think>`, the lines must be `1, 2, …` exactly (a code fence around them is allowed) | yes |
| `holdout-06889` | the holdout's MMLU-Pro prompt 06889, re-rendered by the bench | 5,000 | degeneration | yes |
| `holdout-08349` | the holdout's MMLU-Pro prompt 08349, re-rendered by the bench | 5,000 | degeneration | yes |
| `essay-chat` | a ≥ 6,000-word technical essay, same rendering as `count-chat` | 6,144 | degeneration | no: reported only |

**The two holdout prompts are positive controls.** On the warm holdout
server they broke at about 4,199 and 4,195 tokens.
- They are replayed only as reproducers. Their answers were already scored
  and recorded, and nothing here is scored for accuracy or touches the
  ledger.
- `longgen.py` re-renders each one with the bench's `render_item`, and
  refuses unless it hashes to the prompt the holdout sent.
- Caveat: these arms start fresh, while the holdout had served about 350
  requests with a 512 MB checkpoint cache. If the breakdown needs that
  cache pressure, arm a may not reproduce it.

Every request is a `/v1/completions` call at temperature 0, seed 42, the
holdout's path.
- **Arms a and b are target-greedy:** DFlash accepts only drafts equal to
  the target's greedy token. They should match token for token until the
  first requeue, and the verdict reports where they diverge.
- **Arm c** runs single-token forwards, so its text may drift from a's.
- **The record is rewritten after each fixture**, so an arm cut by its
  timeout keeps what it finished.

## Scoring (fixed; `longgen.py verdict`)

**Break signals.** Each is located at an approximate absolute token
position, from the fixture's mean characters per token:
- one word repeated 5 or more times in a row;
- one character repeated 60 or more times in a row;
- a script an English answer never uses: Devanagari, Thai, Hangul or Arabic.
  CJK is excluded, because MiMo sometimes thinks in Chinese.
- For a counting fixture, the first line that is not the next integer. A
  leading or closing code fence and blank lines around the list are allowed,
  and so is a last line cut by the budget (a prefix of the expected one).

On the holdout, these signals flag 26 of MiMo's 247 MMLU-Pro answers and none
of Flash-Next's 247.

**Each fixture gets one state:**
- **broken:** its first signal is at or after 3,500 tokens;
- **early:** its first signal is before 3,500 tokens. This is reported, never
  attributed;
- **clean:** no signal (and, for a counting fixture, an exact sequence), and
  it ran to at least 4,600 tokens;
- **inconclusive:** a request error, missing usage, or no signal but too
  short.

**The verdict.** Let *B* be the primary fixtures that are broken in arm a.
For arms b and c, each fixture in *B* is:
- broken, if it is broken in that arm;
- clean, if it is clean and ran at least 500 tokens past arm a's break;
- otherwise inconclusive.

An arm is *broken* if any of its fixtures in *B* is broken, *clean* if all
are clean, and *inconclusive* otherwise.
- **`NO_RESULT`:** arm a has no record, or a record is malformed or missing.
- **`NOT_REPRODUCED`:** *B* is empty.
- **`INCONCLUSIVE`:** arm b or c is inconclusive.
- **`OUTPUT_CHUNKING`:** b is clean and c is broken.
- **`DRAFTER`:** c is clean and b is broken.
- **`CHUNKING_AND_DRAFTER`:** both are clean; the breakdown needs requeueing
  and drafting together.
- **`NEITHER`:** both are broken.

**What each verdict leads to.** These are proposals to the owner, not
actions this run takes:
- **`OUTPUT_CHUNKING` or `CHUNKING_AND_DRAFTER`:** the smallest fix is
  `--output-chunking false` in the qualified profile. It changes the
  launch, so it needs a test-first change and a new acceptance candidate.
- **`DRAFTER`:** the fix is in the drafter path. Serving without DFlash
  costs decode speed.
- **`NEITHER`, `NOT_REPRODUCED` or `INCONCLUSIVE`:** the next bounded
  alternative varies the recurrent checkpoint interval and the cache mode,
  and warms the server first.

Each arm's decode rate is reported beside it, as diagnostic only.

## Safety

- A kill line or a memwatch breach ends the window: no further arm starts.
- Any kill line, a failed restore, or a low below the 8 GiB floor is
  reported to the owner.
- Production is down for about 1–1.5 h; GPU windows are pre-approved (owner,
  2026-10-12).

## Review (one focused round, before the run)

**Reviewer B (run safety):**
- **LG-1 (high):** the TabbyAPI shim refused arm b's `--output-chunking`
  flag at server start, after production was down. Closed:
  - the shim allows the flag (RED f4d58c70, GREEN d551cb4e);
  - the driver now runs the shim's check on every arm before anything
    stops.
- **LG-2 (low):** the between-arms check now also scans the server log.
- **LG-3 (low):** the per-arm cap now leaves room for the arm's own
  timeouts.

**Reviewer A (measurement), no critical or high findings:**
- **M1:** too short or failed counted as broken. Closed by the fixture
  states and the `INCONCLUSIVE` verdict.
- **M2:** an early preface or fence forced `NEITHER`. Closed by the
  3,500-token threshold and the fence allowance.
- **M3:** the essay's regex missed some breakdowns. Closed by the stronger
  signals; the essay is reported only.
- **M4:** no known reproducer. Closed by the two holdout controls.
- **L1:** the record is rewritten after each fixture.
- **L2:** the a/b divergence is reported.
- **L3:** the hypothesis wording above.
