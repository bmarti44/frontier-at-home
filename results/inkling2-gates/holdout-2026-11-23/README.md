# Inkling 2 Mini holdout accuracy and the promotion verdict (owner-authorized spend)

The owner pre-approved one holdout draw for the final Inkling configuration on 2026-11-22. The same instruction set the promotion rule: if Inkling 2 Mini is both faster and more accurate than the serving default `qwen38fn` (Qwen3.8-Flash-Next), it becomes the default.

The draw runs on the configuration that passed M3: candidate 2, `../qualify-2026-11-23-c2/` (PASS, evidence `ee2ce2fe`). That is the 1M profile with screen carrier a6 (DSpark ×7, humming MoE, #69288, batch 2,048), BF16 KV, images on, and reasoning effort low.

Candidate 2's values were committed before this draw: prefill 3,332.5 tok/s at 28,672, decode 36.14 tok/s at 0 and 44.16 at 28,672, MMMU 77/100, tool-call 20/20. Each meets its threshold below, so the holdout decides the verdict.

This file, `run-holdout.sh` and `holdout_verdict.py` were committed before the draw. The commit that last changes this file is the draw's freeze.

## What runs (`run-holdout.sh`, one fresh start)

**Containment.** This is candidate 2's driver with its cells replaced (`make_holdout_driver.py` in the agent's scratchpad derives it mechanically):
- the same refusals, production stop, inference lock and fd handling;
- `verify_identity.py` (interpreter, vLLM tree, patches, package freeze, weights);
- the MemFree ≥ 100 / MemAvailable ≥ 110 GiB start gate;
- a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0` and `OOMPolicy=kill`, with `RuntimeMaxSec` 8 h;
- the profile memwatch at 10 GiB, the 1 Hz meminfo sampler, the kernel log, and the restore.

**Refusals before the stop.** Each one starts nothing:
- candidate 2's `gate_verdict.py --preflight`, so the serving configuration and kit are its freeze;
- this draw's `holdout_verdict.py --preflight`, so this file, the driver, the verdict, `31_bench_accuracy.py`, the encoders, the HumanEval runtime pin and the default draw's `accuracy/` files (the questions) are this freeze's;
- candidate 2's `probe-c2.json` must exist;
- the HumanEval sandbox runtime must resolve. Its suite fails closed only after the draw has started, so it is checked first.

**After ready:**
- **Restart identity.** `probe-holdout.json` holds the three greedy prompts, rendered at effort low through candidate 2's `probe_low.py`. vLLM's KV capacity at 262,144 per request must be at least 1,048,576 tokens. The holdout is drawn only if all three hashes equal candidate 2's. Otherwise the driver logs `FAILED_ATTEMPT` and the ledger is untouched.
- **The draw.** The kit runs `--cells accuracy --accuracy-split holdout --reasoning-effort low`, under the default's holdout contract:
  - the `inkling2` encoder;
  - a 16,384-token budget and a 2,700 s timeout;
  - `--config-evidence` written by the kit;
  - GSM8K holdout (100), MMLU-Pro holdout (247) and HumanEval (all 164).

  `31_bench_accuracy.py` appends the GSM8K and MMLU-Pro draws to `results/holdout-ledger.json`, namespace `""`.

## The rule (fixed before the draw; `holdout_verdict.py`)

**More accurate.** The default's row is `results/qwen38fn-gates/holdout-2026-10-11/RESULT.md`: GSM8K 98/100, MMLU-Pro 218/247, HumanEval 138/164, a sum of 454/511. The verdict requires each suite's `rowset_sha256` to equal that draw's, read from its committed `accuracy/acc-<suite>.json`, so both models answer the same questions. Inkling is more accurate only if all four hold:
1. its sum over the three suites exceeds 454 of 511;
2. no suite is significantly worse: the one-sided pooled two-proportion z against the default's count must be ≥ −1.645;
3. MMMU-val-100 is at least 64, the default's, from candidate 2's vision cell. Inkling's answer budget is 4,096 tokens and the default's 64 was measured at 256. Both are reported;
4. tool-call is at least 19/20, from candidate 2.

**Faster.** These come from candidate 2's kit speed cell, block-aware at K = 8. Each leg must be at least the default's published c10 value:
- prefill at 28,672 tokens ≥ 2,711.5 tok/s;
- decode at 0 ≥ 34.07 tok/s;
- decode at 28,672 ≥ 40.92 tok/s.

TTFT is reported but does not gate.

**Verdicts:**
- `PROMOTE` if Inkling is faster and more accurate.
- `KEEP_DEFAULT` for a valid draw that misses any leg. Every leg's value is reported against its threshold.
- `NO_RESULT` if candidate 2's gate summary is not `PASS`, or if the draw is missing, malformed or not the declared run. That covers any of:
  - a split, count, encoder, budget, model, stack or namespace that is not the declared one;
  - a transcript count short of the suite size;
  - a GSM8K or MMLU-Pro row not rendered as `official-encoder-inkling2-thinking-effort-low`, or a HumanEval row not sent `raw` (`31_bench_accuracy.py` sends HumanEval as a raw code continuation, without the chat wrapper or an effort, as in the default's draw);
  - a ledger without exactly one `started` and one `completed` entry per ledgered suite, with the completed count matching;
  - a kit accuracy cell that is not `OK`;
  - a head that is dirty or differs from either freeze.

The self-test has 26 synthetic cases, and each must give its expected verdict before the real one is computed. Because synthetic bundles share the verdict's assumptions about field names, it also runs the readers on two real bundles: the default's own draw (`results/qwen38fn-gates/holdout-2026-10-11`) must read back as 98 / 218 / 138 with every validity check passing under that draw's declared run, and candidate 2's committed bundle must give its kit speed, MMMU and tool-call values. A last check requires `run-holdout.sh` to log, verbatim, the lines the verdict requires.

Memwatch's `BREACH`, `EMERGENCY_STOP` and `FAIL_CLOSED` lines are kill lines, as in candidate 2's scorer. This driver disarms memwatch before the engine stops, as candidate 2's did. The default's driver did not, so its draw logged `FAIL_CLOSED signal=TERM armed_target_unreadable` at teardown; the real-evidence replay of that bundle tolerates that one line, and this draw tolerates none.

**What follows:**
- `PROMOTE`: once M4 lands the `inkling2mini` alias (profile `qualified`, switch tests, CI green), `sudo -n scripts/52_engine_switch.sh inkling2mini` becomes the serving default, and the README and AGENTS default text change with it.
- `KEEP_DEFAULT` or `NO_RESULT`: `qwen38fn` stays the default. M4 still lands `inkling2mini` as a qualified, switchable engine. Inkling's README rows come from candidate 2's bundle and this draw.

## If the server dies mid-suite (fixed before the draw)

`31_bench_accuracy.py` scores the remaining items of a suite as invalid when the server goes away. Such a run is labeled `FAILED_ATTEMPT` and the rule is not applied (`NO_RESULT`). Evidence of a server death is any of:
- a memwatch breach or kill line;
- a kernel OOM, NVRM or Xid line;
- a transport-error row in a suite's transcripts. A timed-out request is scored as wrong, not as an error, as in the default's draw.

The kit stops later holdout draws after a failed suite (`464effe2`), so at most one suite's rows can be lost.
