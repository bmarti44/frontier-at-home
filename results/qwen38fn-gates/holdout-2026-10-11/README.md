# Qwen3.8-Flash-Next holdout accuracy (owner-authorized spend, 2026-10-11)

On 2026-10-11 the owner authorized one holdout spend, so that Flash-Next and
the default (`qwen38-1m`, Qwen3.8-27B) are compared on the same questions. It
is drawn on the configuration that passed the direct-1M gate: candidate 5,
`../qualify-2026-10-11-c4/`, text-only. Image input at 1M is `NO_GO`
(`../diag-vision-2026-10-11/RESULT.md`).

This file and `run-holdout.sh` were committed before the draw.

## What runs (`run-holdout.sh`, one fresh start)

- The containment is candidate 5's driver:
  - a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill`;
  - the profile memwatch at 10 GiB;
  - a 1 Hz `/proc/meminfo` sampler and the kernel log.

  The changes from candidate 5:
  - the guard is 114 GiB;
  - `RuntimeMaxSec` is 8 h;
  - the steps after the KV check.
- vLLM must log an attention block size of 3,200 and a KV capacity of at
  least 1,048,576 tokens at 262,144 per request.
- `probe-holdout.json` holds the three greedy prompts. The holdout is drawn
  only if all three hashes equal candidate 5's. Otherwise the driver logs
  `FAILED_ATTEMPT` and the ledger is untouched.
- The kit then runs `--cells accuracy --accuracy-split holdout --reasoning-effort low`.
  The settings are the default's holdout contract:
  - `qwen38` encoder, thinking rendering;
  - 16,384-token budget, 2,700 s timeout;
  - `--config-evidence` written by the kit.

  It covers:
  - GSM8K holdout (100);
  - MMLU-Pro holdout (247);
  - HumanEval, all 164.

  This re-runs candidate 5's HumanEval (139/164) on a fresh start.
  `31_bench_accuracy.py` appends the GSM8K and MMLU-Pro draws to
  `results/holdout-ledger.json`, namespace `""`.

## Comparison with the default (fixed before the draw)

The default's published row: GSM8K holdout 98/100, MMLU-Pro holdout 210/247,
HumanEval 130/164. The holdout rows come from a fixed seed, so they are the
same questions.

Flash-Next is **more accurate on text** only if both hold:
1. Its correct count summed over the three suites exceeds 438 of 511.
2. No suite is significantly worse: one-sided two-proportion z-test against
   the default's count, z < −1.645.

Every per-suite delta is reported with its sign. This run uses HumanEval
from this start, not candidate 5's.

**Images are a capability difference, not a score.** The default serves
images (MMMU-val-100 64/100). This Flash-Next profile does not. The
diagnostic chat-mode MMMU was 63/100 with images on, which did not fit at 1M.

**Speed** is candidate 5's block-aware bundle: decode 25.80 / 35.54 tok/s,
prefill 2,684 tok/s, TTFT 12.93 s at 28K. All of these beat the default.
Short-prompt TTFT is 0.569 s against the default's 0.39 s (0.505 s under the
kit). That is a miss.

Promotion stays the owner's decision. The owner's condition was "more
accurate and faster", and they asked for vision to be qualified first, which
it could not be at 1M.

## If the server dies mid-suite (fixed before the draw)

`31_bench_accuracy.py` scores the remaining items of a suite as invalid when
the server goes away, and still writes `completed`. Such a run is labeled
`FAILED_ATTEMPT` and the comparison rule is not applied. Evidence of a server
death is any of:
- a memwatch breach or kill line;
- the scope dying;
- a run of transport errors in a suite's transcripts.

The shared tooling (`464effe2`) keeps any later holdout suite from being
spent: the bench reaches `/v1/models` before the ledger's `started` entry, and
the kit stops holdout draws after a failed suite. So at most one suite's rows
can be lost. If the HumanEval preflight fails before any load, `observe/holdout`
is kept under a new name and the run retried. Nothing has been drawn at that
point.
