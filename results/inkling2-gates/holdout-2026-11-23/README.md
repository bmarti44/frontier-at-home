# Inkling 2 Mini holdout accuracy and the promotion verdict (owner-authorized spend)

The owner pre-approved one holdout draw for the final Inkling configuration on 2026-11-22. The same instruction set the promotion rule: if Inkling 2 Mini is both faster and more accurate than the serving default `qwen38fn` (Qwen3.8-Flash-Next), it becomes the default.

The draw runs on the configuration that passed M3: candidate 2, `../qualify-2026-11-23-c2/` (PASS, evidence `ee2ce2fe`). That is the 1M profile with screen carrier a6 (DSpark ×7, humming MoE, #69288, batch 2,048), BF16 KV, images on, and reasoning effort low.

Candidate 2's values were committed before this draw: prefill 3,332.5 tok/s at 28,672, decode 36.14 tok/s at 0 and 44.16 at 28,672, MMMU 77/100, tool-call 20/20. Each meets its threshold below, so the holdout decides the verdict.

This file, `run-holdout.sh` and `holdout_verdict.py` were committed before the draw. The commit that last changes this file is the draw's freeze.

**Freeze history.** The first freeze (`ace07690`) was reviewed and never launched. Nothing was loaded and nothing was drawn. Reviewer A found one high (A-M5-H1): the 27,000 s kit clock could expire in HumanEval after both ledgered suites were spent. Reviewer B rated the same issue medium (B-M5-M1). This second freeze changes only what those reviews named: the clocks, the sandbox-infrastructure rule, the namespace filter, the freeze's resolution from the head, the scoring-time tree check, and the reported per-prompt decode and TTFT.

## What runs (`run-holdout.sh`, one fresh start)

**Containment.** This is candidate 2's driver with its cells replaced (`make_holdout_driver.py` in the agent's scratchpad derives it mechanically):
- the same refusals, production stop, inference lock and fd handling;
- `verify_identity.py` (interpreter, vLLM tree, patches, package freeze, weights);
- the MemFree ≥ 100 / MemAvailable ≥ 110 GiB start gate;
- a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0` and `OOMPolicy=kill`, with `RuntimeMaxSec` 90,000 s (25 h);
- the profile memwatch at 10 GiB, the 1 Hz meminfo sampler, the kernel log, and the restore.

**Refusals before the stop.** Each one starts nothing:
- candidate 2's `gate_verdict.py --preflight`, so the serving configuration and kit are its freeze;
- this draw's `holdout_verdict.py --preflight`, so this file, the driver, the verdict, `31_bench_accuracy.py`, the encoders, the HumanEval runtime pin and the default draw's `accuracy/` files (the questions) are this freeze's;
- candidate 2's `probe-c2.json` must exist;
- the HumanEval sandbox runtime must resolve. Its suite fails closed only after the draw has started, so it is checked first.

**After ready:**
- **Restart identity.** `probe-holdout.json` holds the three greedy prompts, rendered at effort low through candidate 2's `probe_low.py`. vLLM's KV capacity at 262,144 per request must be at least 1,048,576 tokens. The holdout is drawn only if all three hashes equal candidate 2's. Otherwise the driver logs `FAILED_ATTEMPT` and the ledger is untouched.
- **The draw.** The kit runs `--cells accuracy --accuracy-split holdout --reasoning-effort low` under an 86,400 s clock, under the default's holdout contract:
  - the `inkling2` encoder;
  - a 16,384-token budget and a 2,700 s timeout;
  - `--config-evidence` written by the kit;
  - GSM8K holdout (100), MMLU-Pro holdout (247) and HumanEval (all 164).

  `31_bench_accuracy.py` appends the GSM8K and MMLU-Pro draws to `results/holdout-ledger.json`, namespace `""`.

**Why the clock is long (reviews A-M5-H1, B-M5-M1).** The kit runs GSM8K, MMLU-Pro, then HumanEval, so both ledgered suites are spent before HumanEval starts. HumanEval has no stop sequence and a 16,384-token budget, and it has never run on Inkling. Under the same protocol qwen38fn ran 11 of 164 items to the limit and GLM-5.3 ran all of them. A kill would leave no `summary.json` and void the draw after its rows are spent. The clock covers the worst case:
- GSM8K and MMLU-Pro take about 4,500 s at the effort pick's dev rates;
- every HumanEval item at 16,384 tokens and the slowest measured decode (33.5 tok/s) takes about 489 s, so 164 × 489 = 80,200 s.

The window ends when the kit does. Production is down for the whole window, which could last up to about 25 h if every HumanEval item runs to the limit.

## The rule (fixed before the draw; `holdout_verdict.py`)

**More accurate.** The default's row is `results/qwen38fn-gates/holdout-2026-10-11/RESULT.md`: GSM8K 98/100, MMLU-Pro 218/247, HumanEval 138/164, a sum of 454/511. The verdict requires each suite's `rowset_sha256` to equal that draw's, read from its committed `accuracy/acc-<suite>.json`, so both models answer the same questions. Inkling is more accurate only if all four hold:
1. its sum over the three suites exceeds 454 of 511;
2. no suite is significantly worse: the one-sided pooled two-proportion z against the default's count must be ≥ −1.645;
3. MMMU-val-100 is at least 64, the default's, from candidate 2's vision cell. Inkling's answer budget is 4,096 tokens and the default's 64 was measured at 256. Both budgets are reported;
4. tool-call is at least 19/20, from candidate 2.

**Faster.** These come from candidate 2's kit speed cell, block-aware at K = 8. Each leg must be at least the default's published c10 value:
- prefill at 28,672 tokens ≥ 2,711.5 tok/s;
- decode at 0 ≥ 34.07 tok/s;
- decode at 28,672 ≥ 40.92 tok/s.

These are the rounded values the owner's rule names. c10's unrounded kit values are 2,711.49, 34.0745 and 40.9182 (review A-M5-L2); the difference is immaterial at candidate 2's values.

**Reported, not gating, in `holdout-summary.json`:**
- TTFT at 0 and 28,672 for both models.
- Decode per prompt (review A-M5-M2). Both kits seed each rep's prompt from the same counter, so the reps pair by `prompt_sha256`:
  - at 0, Inkling is faster on both prompts: 33.48 vs 31.25 and 38.80 vs 36.90;
  - at 28,672, it is slower on one prompt (39.21 vs 40.04) and faster on the other (49.12 vs 41.79).
  - The screen's single-start runs of the same configuration gave 49.05, 37.53 and 38.25 at 28,672. So the decode-at-28,672 leg passes on a two-rep median that one prompt carries.
  - The rule is applied as preregistered, and the caveat is carried with any PROMOTE.

**Verdicts:**
- `PROMOTE` if Inkling is faster and more accurate.
- `KEEP_DEFAULT` for a valid draw that misses any leg. Every leg's value is reported against its threshold.
- `NO_RESULT` if candidate 2's gate summary is not `PASS`, or if the draw is missing, malformed or not the declared run. That covers any of:
  - a split, count, encoder, budget, model, stack or namespace that is not the declared one;
  - a transcript count short of the suite size;
  - a GSM8K or MMLU-Pro row not rendered as `official-encoder-inkling2-thinking-effort-low`, or a HumanEval row not sent `raw` (`31_bench_accuracy.py` sends HumanEval as a raw code continuation, without the chat wrapper or an effort, as in the default's draw);
  - a ledger without exactly one `started` and one `completed` entry per ledgered suite, with the completed count matching;
  - a kit accuracy cell that is not `OK`;
  - a HumanEval row whose sandbox never ran the code: `sandbox launch failed`, or Docker's own exit 125, 126 or 127 (review A-M5-M1). Exit 1 (a failing test) and the sandbox timeouts are wrong answers, as in the default's draw, and host timeouts are counted in the summary;
  - a head that is dirty or differs from either freeze;
  - a working tree whose copy of this draw's frozen files, the scorer included, differs from the freeze at scoring time.

The freeze is the last commit, at or before the run's logged head, that changes this file. A result commit therefore cannot move it (review A-M5-L3).

The self-test has 30 synthetic cases, and each must give its expected verdict before the real one is computed. Because synthetic bundles share the verdict's assumptions about field names, it also runs the readers on two real bundles: the default's own draw (`results/qwen38fn-gates/holdout-2026-10-11`) must read back as 98 / 218 / 138 with every validity check passing under that draw's declared run, and candidate 2's committed bundle must give its kit speed, MMMU and tool-call values. A last check requires `run-holdout.sh` to log, verbatim, the lines the verdict requires.

Memwatch's `BREACH`, `EMERGENCY_STOP` and `FAIL_CLOSED` lines are kill lines, as in candidate 2's scorer. This driver disarms memwatch before the engine stops, as candidate 2's did. The default's driver did not, so its draw logged `FAIL_CLOSED signal=TERM armed_target_unreadable` at teardown; the real-evidence replay of that bundle tolerates that one line, and this draw tolerates none.

**What follows:**
- `PROMOTE`: once M4 lands the `inkling2mini` alias (profile `qualified`, switch tests, CI green), `sudo -n scripts/52_engine_switch.sh inkling2mini` becomes the serving default, and the README and AGENTS default text change with it.
- `KEEP_DEFAULT` or `NO_RESULT`: `qwen38fn` stays the default. M4 still lands `inkling2mini` as a qualified, switchable engine. Inkling's README rows come from candidate 2's bundle and this draw.

## If the server dies mid-suite (fixed before the draw)

`31_bench_accuracy.py` scores the remaining items of a suite as invalid when the server goes away. Such a run is labeled `FAILED_ATTEMPT` and the rule is not applied (`NO_RESULT`). Evidence of a server death is any of:
- a memwatch breach or kill line;
- a kernel OOM, NVRM or Xid line;
- a transport-error row in a suite's transcripts. A timed-out request is scored as wrong, not as an error, as in the default's draw.

**Known false-failure path (review B-M5-L1, accepted).** HumanEval's sandbox containers run in Docker's cgroups, each capped at 512 MiB. If a generated program exceeds the cap, the kernel logs a memory-cgroup OOM kill, which matches the kernel kill pattern. The draw would then score `NO_RESULT` even though the engine was healthy. It cannot produce a wrong `PROMOTE`. There were no exit-137 sandbox kills in about 1,300 earlier sandbox runs in this repository.

**Abort.** An abort spends only the suite in progress. The next ledgered suite fails its model check before it writes its ledger `started` entry.

The kit stops later holdout draws after a failed suite (`464effe2`), so at most one suite's rows can be lost.
