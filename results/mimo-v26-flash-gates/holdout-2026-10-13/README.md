# MiMo-V2.6-Flash holdout accuracy (owner-authorized spend)

On 2026-10-12 the owner authorized one holdout draw for MiMo-V2.6-Flash. It
is drawn on the configuration that M3 candidate 3 qualified
(`../qualify-2026-10-13-c1/`), so MiMo and the default, Flash-Next, are
compared on the same questions. The draw is **reported, not gating**: MiMo is
already qualified as a switchable alias, and promotion stays the owner's
decision.

This file and `run-holdout.sh` were committed before the draw. One focused
review round (the two persistent reviewers) found one high, one medium and
five lows; all are closed here before the draw (see "Review").

## What runs (`run-holdout.sh`, one fresh start)

**The machinery is candidate 3's driver**
(`../qualify-2026-10-13-c1/run-c1.sh`, reviewed over three rounds). This run
reuses it, so the following are unchanged:
- the refusals, the production stop and the inference lock;
- `verify_identity.py` (runtime sources, the wheel, every served weight file);
- the MemFree ≥ 108 / MemAvailable ≥ 114 GiB start gate;
- containment: scope MemoryHigh 8G / MemoryMax 16G, MemorySwapMax 0,
  `OOMPolicy=kill`, the profile memwatch at 8 GiB and the 1 s load-time
  page-cache release;
- the observers, the hardened cleanup and the `setsid` restore of `qwen38fn`.

**What differs from candidate 3:**
- The HumanEval sandbox is resolved with the bench's own resolver before
  anything stops. HumanEval fails closed at its suite start, after the
  holdout is drawn, so a broken sandbox must be caught first.
- **The launch must be candidate 3's.** M4 (`b5ff1dca`) changed the profile
  after the freeze: its status, its admission note and its memory model.
  - Before anything stops, the kit renders the launch at this head with
    `--dry-run`.
  - The render must equal candidate 3's recorded launch
    (`../qualify-2026-10-13-c1/manifest.json`): the same `config_hash`
    (`…-eac45232288f`, over argv and env), containment, model path,
    tokenizer, topology, admission, floor and encoder. Otherwise the driver
    refuses.
  - The render is kept as `observe/render-manifest.json`.
- **The code must be candidate 3's.** The render compares strings, so the
  driver also refuses unless:
  - every path candidate 3 froze, except its profile, is unchanged from the
    freeze `2a75f1fe`;
  - so is the accuracy path: `31_bench_accuracy.py`, the `mimo26` encoder,
    the GSM8K, MMLU-Pro and HumanEval evalsets and their pins,
    `verification/MANIFEST.sha256` and the HumanEval runtime pin;
  - the profile differs from the freeze only in `status`, `safety.note` and
    `memory_model`;
  - candidate 3's `probe-c1.json` and `manifest.json` are unchanged from its
    evidence commit `6c695c6a`.
- `RuntimeMaxSec` is 12 h, and the kit is cut at 11.5 h.
  - **GSM8K and MMLU-Pro** should take 1–3 h:
    - on GSM8K dev, MiMo wrote 0.57× Flash-Next's tokens, at about 48 tok/s
      per answer;
    - Flash-Next's MMLU-Pro draw took 2.0 h.
  - **HumanEval is the uncertain part.** It is a raw completion with no stop
    sequences and a 16,384-token budget.
    - Flash-Next ran 11 of its 164 items to the full budget, 180K of its 194K
      HumanEval tokens.
    - MiMo's raw-continuation behaviour has never been measured.
    - If MiMo runs to the budget on most items, HumanEval alone could take
      far longer than the cut (164 × 16,384 tokens at 22–48 tok/s is 16–34
      h).
  - Flash-Next's token counts at MiMo's 21.84 tok/s context-0 decode take
    6.05 h. The cut is a budget, not a safety limit; "If a suite does not
    finish" below says what a cut means.
- After ready, only these two steps run:
  1. **`probe-holdout.json`:** the three greedy prompts. The holdout is drawn
     only if all three hashes equal candidate 3's `probe-c1.json`. Otherwise
     the driver logs `FAILED_ATTEMPT` and the ledger is untouched.
  2. **The kit's accuracy cell:** `--cells accuracy --accuracy-split holdout
     --reasoning-effort ''`. MiMo has no reasoning-effort control, so none is
     sent.
     - The settings are the kit's: the `mimo26` encoder with thinking
       rendering (MiMo's template default), a 16,384-token budget, a 2,700 s
       request timeout, and `--config-evidence` written by the kit.
     - It covers GSM8K holdout (100), MMLU-Pro holdout (247) and HumanEval
       (all 164).
     - `31_bench_accuracy.py` appends the GSM8K and MMLU-Pro draws to
       `results/holdout-ledger.json`, stack
       `mimo-v2.6-flash-cuda-spark-128g-262k`, namespace `""`.
     - The ledger is a tracked file per checkout. Its new entries are
       committed with this evidence and merge with the claim PR, so the
       production checkout's copy also refuses a second MiMo draw.

## Comparison with the default (fixed before the draw)

Flash-Next's published row
(`results/qwen38fn-gates/holdout-2026-10-11/RESULT.md`):

| suite | Flash-Next |
|---|---|
| GSM8K holdout | 98/100 |
| MMLU-Pro holdout | 218/247 |
| HumanEval | 138/164 |
| **sum** | **454/511** |

The holdout rows come from a fixed seed, so they are the same questions.
This is checked: a suite is compared only if its `rowset_sha256` equals
Flash-Next's (GSM8K `e4eec5808d38…`, MMLU-Pro `4b2151a146c1…`, HumanEval
`a8571e687aee…`).

**The rule.** For each *completed* suite (next section), report MiMo's
count, the delta and the one-sided pooled two-proportion z against
Flash-Next's count. A suite is *significantly worse* if z < −1.645. Every
delta is reported with its sign. The label is:
- **more accurate on text** if all three suites completed, the sum exceeds
  454 of 511, and no suite is significantly worse;
- **not more accurate on text** if any completed suite is significantly
  worse, or all three completed and the sum is at most 454; the result names
  each significantly worse suite;
- **not determined** otherwise, when a suite did not complete and no
  completed suite is significantly worse.

**The protocols differ on GSM8K and MMLU-Pro**, where each model runs its
own template's default. Flash-Next ran thinking at effort `low` through the
`qwen38` encoder. MiMo runs its default thinking through the `mimo26`
encoder, which has no effort level. **HumanEval is the same for both**: a
raw completion, with no encoder, thinking or effort level. All three suites
use the same 16,384-token budget, 2,700 s timeout and scorer.

## If a suite does not finish (fixed before the draw)

Each suite is judged on its own. The kit runs GSM8K, then MMLU-Pro, then
HumanEval, and stops further holdout draws after a failed suite. So at most
one suite's rows can be lost, and the spend-once suites run first.

A suite is **completed** if both hold:
- its `acc-<suite>.json` was written before any death evidence;
- its transcripts hold no run of transport errors.

**Server death evidence** is any of:
- a memwatch breach or a kill line;
- the scope dying, or the 12 h cap firing;
- a run of transport errors in a suite's transcripts: two or more
  consecutive rows whose reason is `invalid:` from a refused, reset or
  dropped connection.

`31_bench_accuracy.py` scores the remaining items of a suite as invalid when
the server goes away, and still writes `completed`. So a suite in progress at
a server death is a `FAILED_ATTEMPT`, even if its file was written.

**The kit's 11.5 h cut** (`kit exit nonzero rc=124`) while the server stayed
healthy is a **budget cut**, not a server death. The suite in progress has
no result file, and is reported as `BUDGET_CUT` with its partial transcript
count (diagnostic only). Suites completed before the cut keep their
comparison.

**HumanEval is not a holdout draw** (split `all`, no ledger entry). If it is
cut or fails, it can be measured later in its own window, without spending
anything.

A single request that hits the 2,700 s timeout while the server stays up is
scored invalid, like any other unanswered item; it is not a server death.

## How it is launched

The driver runs detached in its own session, not as a child of an agent's
tool call, so it outlives the session that started it:
`setsid -f nohup bash run-holdout.sh >~/.frontier-at-home/logs/mimo26-holdout-driver.out 2>&1 </dev/null`.

**Manual abort:** `systemctl --user stop mimo26-holdout-<pid>.scope`. The kit
then fails fast, and the driver restores production itself.

## Safety

Any kill line, a failed restore, or a low below the 8 GiB floor stops the
work and is reported to the owner. The draw is not retried.

## Review (one focused round, before the draw)

**Reviewer A (measurement):**
- **H1 (high):** the first rule counted the kit's cut as a server death and
  voided the whole comparison. A long HumanEval could therefore throw away
  GSM8K and MMLU-Pro rows already spent. Closed by the per-suite rule above.
- **M1 (medium):** the render check compared strings, not the code behind
  them. Closed by the code-identity refusals.
- **L1 (low):** the protocol wording and an unchecked "same questions".
  Closed by the wording and the `rowset_sha256` precondition.
- **L2 (low):** the worst-case estimate. Corrected to 6.05 h, and HumanEval's
  risk is stated.
- **L3 (low):** the ledger lives per checkout. Closed by committing and
  merging its entries.

**Reviewer B (run safety):** no critical, high or medium findings.
- **H-1 (low):** the kit's and the probe's exit codes were not logged. They
  now are.
- **H-2 (low):** a 12 h driver under an agent tool could be killed with its
  session. Closed by the detached launch above.
