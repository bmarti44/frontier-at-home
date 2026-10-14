# MiMo-V2.6-Flash holdout accuracy (owner-authorized spend)

On 2026-10-12 the owner authorized one holdout draw for MiMo-V2.6-Flash. It
is drawn on the configuration that M3 candidate 3 qualified
(`../qualify-2026-10-13-c1/`), so MiMo and the default, Flash-Next, are
compared on the same questions. The draw is **reported, not gating**: MiMo is
already qualified as a switchable alias, and promotion stays the owner's
decision.

This file and `run-holdout.sh` were committed before the draw.

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
- `RuntimeMaxSec` is 12 h, and the kit is cut at 11.5 h. The estimate is
  3–6 h:
  - Flash-Next's draw generated about 476K tokens over 3.4 h.
  - On GSM8K dev, MiMo wrote 0.57× Flash-Next's tokens, at about 48 tok/s
    per answer.
  - The worst plausible case, Flash-Next's token counts at MiMo's 21.84 tok/s
    context-0 decode, is about 5.7 h.
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

**The rule.** For each suite, report MiMo's count, the delta and the
one-sided pooled two-proportion z against Flash-Next's count. A suite is
*significantly worse* if z < −1.645. MiMo is labeled **more accurate on
text** only if both hold:
1. its sum over the three suites exceeds 454 of 511;
2. no suite is significantly worse.

Otherwise the label is **not more accurate on text**, and the result names
each significantly worse suite. Every per-suite delta is reported with its
sign.

**The models run different protocols**, each its own template's default.
Flash-Next ran thinking at effort `low` through the `qwen38` encoder. MiMo
runs its default thinking through the `mimo26` encoder, which has no effort
level. Both use the same 16,384-token budget, timeout and scorer.

## If the server dies mid-suite (fixed before the draw)

`31_bench_accuracy.py` scores the remaining items of a suite as invalid when
the server goes away, and still writes `completed`. Such a run is labeled
`FAILED_ATTEMPT`, and the comparison rule is not applied. Evidence of a server
death is any of:
- a memwatch breach or a kill line;
- the scope dying, the 12 h cap firing, or the kit's 11.5 h cut;
- a run of transport errors in a suite's transcripts: two or more
  consecutive rows whose reason is `invalid:` from a refused, reset or
  dropped connection.

A single request that hits the 2,700 s timeout while the server stays up is
scored invalid, like any other unanswered item; it is not a server death.

The kit stops further holdout draws after a failed suite, so at most one
suite's rows can be lost.

## Safety

Any kill line, a failed restore, or a low below the 8 GiB floor stops the
work and is reported to the owner. The draw is not retried.
