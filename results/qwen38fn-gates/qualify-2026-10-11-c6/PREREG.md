# Qwen3.8-Flash-Next Phase 2: direct-1M qualification with vision, candidate 6 (preregistered)

This is an acceptance run of profile `qwen3.8-flash-next/cuda-spark-128g-1m`
on the Spark. It was committed before any weights were loaded for it. The
gate verdict comes from `gate_verdict.py` in this directory, which writes
`gate-summary.json`. The qualification kit writes its own bundle
(`manifest.json`, `summary.json`, `SUMMARY.md`, one directory per cell) into
this same directory.

Candidate 5 (`../qualify-2026-10-11-c4/`) qualified the text-only profile.
That directory stays frozen as its record. Before promotion, the owner asked
(2026-10-11) for two more things:

- **Images:** qualify Flash-Next's image input, because the default
  (`qwen38-1m`) serves images.
- **Holdout:** spend the GSM8K and MMLU-Pro holdout once, so both models are
  compared on the same questions.

Candidate 6 makes only those changes, plus the interpreter pin below. Its
driver, scorer and mutation check are candidate 5's with these differences:
- the directory path;
- the memory guard at 114 GiB;
- the vision cell in start 1 and gate clause 9;
- the holdout accuracy draw in start 2, in place of the dev split.

Clauses 1–8 are unchanged.

## What changed in the profile (test-first)

- `--limit-mm-per-prompt {"image":4,"video":0}` replaces `{"image":0,"video":0}`.
  Video stays off, as on the default. `serving.max_images` is 4.
- Images take no width/height override. vLLM's startup `profile_run` runs
  the vision encoder once on the processor's largest image (Qwen3-VL fast
  processor, `longest_edge` 16,777,216 pixels, so 16,384 tokens). A real
  request cannot need a larger encoder pass than the one startup measured.
- `memory_model.resident_weights_gib` is 77. The text-only load reported
  75.65 GiB, and the BF16 vision tower (`model.visual.*`) adds 897,862,112 bytes.
- `model.json` pins the runtime interpreter itself (`runtime/bin/python3.12`)
  in place of its `python3` symlink. The bytes and sha256 are the same. The
  generic switch opens the pin with `O_NOFOLLOW` and checks it with
  `S_ISREG`, so the symlink would fail promotion.
- `qualification_targets.vision_min` is 0.64, the default's MMMU-val-100
  score under this same kit and chat mode
  (`results/qwen38-gates/qualify-2026-09-11-kit-check/vision`).
- **Risk beyond memory.** vLLM logged, in text-only mode, "Disabled
  mm_prefix attention mode because multimodal inputs are
  configuration-disabled". With images on, attention backend selection can
  change. Clause 5 (restart identity) compares the two starts of this
  candidate. It does not compare against candidate 5. The diagnostic below
  checks the text path against candidate 5 once, before this freeze.

## Diagnostic before this freeze (not evidence for this gate)

`../diag-vision-2026-10-11/` loaded this profile once on the commit that
enables images. It recorded:
- startup low and idle `MemAvailable`;
- the KV capacity line and the encoder profiling line;
- the three greedy probe hashes against candidate 5;
- the first 10 MMMU cases.

Its numbers are diagnostic only.

## What runs

Both starts are fresh servers. Each start runs only after
`03_memory_guard.py --required-gib 114` passes (above the 113.2 GiB admission
sum of 77 + 8.2 + 18.0 + 10; candidate 5 used 112), inside a user scope with
`MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill` and a
`RuntimeMaxSec` cap. `93_profile_serve.sh` serves the profile on `:8017` and
arms its memwatch at the profile's 10 GiB floor. An external sampler logs
`/proc/meminfo` (including `MemFree`) and the scope's `memory.current` at
1 Hz, and `journalctl -k` is captured for the whole start.

After the server is ready, the driver checks vLLM's own startup log. It needs
an attention block size of 3,200 and `GPU KV cache size` of at least
1,048,576 tokens at 262,144 per request. Otherwise it logs `FAILED_ATTEMPT`
and stops before any cell runs. On every exit it stops the scope if anything
is still running in it.

**Start 1** (`run-phase2.sh start1`, `RuntimeMaxSec` 5 h):

1. `probe-start1.json`: three greedy prompts, reusing the smoke's `probe.py`
   by path.
2. Kit, `--no-launch --cells speed,toolcall,vision,teacher,context --with-soak`.
   The kit runs cells in its fixed order:
   - speed: the block-aware wrapper (`qualification_options.speed_max_tokens_per_event = 4`);
   - tool-call: 20 cases;
   - vision: MMMU-val-100, chat (non-thinking) mode, 256-token answers;
   - teacher: expected SKIPPED, since `model.json` has no BF16 reference logits;
   - context: direct fill of 4 × 250,128 = 1,000,512 tokens with needles and
     controls, memory floor 10 GiB. It runs after the vision cell, so any
     encoder memory the allocator still holds counts against the fill's low
     point;
   - soak: 30 minutes.

**Start 2** (`run-phase2.sh start2`, `RuntimeMaxSec` 16 h):

1. `probe-start2.json`: the same three prompts.
2. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model default --prefix-block-tokens 3200 --spec-block-drop`.
3. Only if both hold — the three probe hashes equal start 1's, and all four
   agent-gate tests pass — the kit runs
   `--cells accuracy --accuracy-split holdout --reasoning-effort low`:
   - GSM8K holdout (100);
   - MMLU-Pro holdout (247);
   - HumanEval (all 164).

   This is the owner-authorized holdout spend (2026-10-11). It runs under the
   default's holdout contract: `qwen38` encoder, thinking rendering, effort
   low, 16,384-token budget, 2,700 s timeout. `31_bench_accuracy.py` records
   it in `results/holdout-ledger.json` (namespace `""`). Otherwise the driver
   logs `holdout not drawn` and the ledger is untouched.

The media cell is not run. Video is off, and the kit's media cell needs both
an image and a video maximum.

## Gate formula (fixed)

`PASS` requires every clause below. Any failed clause gives `FAIL`. Missing or
malformed evidence, or a `FAILED_ATTEMPT` line in a driver log, gives
`NO_RESULT` unless a clause has already failed. So do these:
- a log that is missing the line proving it saw the run (the server log's
  `session start`, memwatch's `ARMED`);
- driver `head` lines that show tracked edits, or that differ between starts
  in any frozen file (`gate_verdict.py` `FROZEN`):
  - the profile and the runtime manifest;
  - the composed scripts, now including `38_bench_vision.py`;
  - the encoder and fixtures, now including `evalsets/mmmu-val-100`;
  - this driver and this scorer.

  Committing start 1's evidence before start 2 is fine.

1. **No kill lines** in either start. The patterns are the smoke's:
   - kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive;
   - server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch log: `BREACH|EMERGENCY_STOP`.
2. **Host swap growth ≤ 0.25 GiB** in each start: `max(SwapTotal − SwapFree)`
   over the 1 Hz samples, minus the first sample.
3. **External low point ≥ 10 GiB** `MemAvailable` across both starts.
4. **Direct 1M:**
   - vLLM logs a KV capacity of at least 1,048,576 tokens at 262,144 per request in both starts;
   - the context cell verdict is `PASS`;
   - total tokens ≥ 1,000,000;
   - the kit manifest's resolved context cap is exactly 1,048,576.
5. **Restart identity:** start 1 and start 2 greedy outputs are byte-identical
   (all three sha256 values equal).
6. **Tool-call ≥ 19/20**, the `qwen38-1m` target.
7. **Agent gate:** every test passes.
8. **Soak:** the soak cell passes.
9. **Vision:** `vision/summary.json` has `n` = 100, `error_count` = 0 and
   `correct` ≥ 64, the default's score. Unparseable answers count as wrong,
   as they did for the default. A transport error fails the clause. A
   summary that is malformed or not 100 cases is `NO_RESULT`.

## Promotion comparison (fixed before the holdout is drawn; reported, not this gate)

The owner's instruction: promote to default and merge if Flash-Next runs
better than `qwen38-1m`, meaning both more accurate and faster. The
comparison uses this run's numbers and the default's published row. It is
computed only if the gate above is `PASS`.

- **Faster:** all four of these must beat the default's README row:
  - block-aware decode at 0 and at 28,672 tokens (17.46 and 26.71 tok/s);
  - prefill at 28,672 (698.7 tok/s);
  - TTFT at 28,672 (49.75 s).

  Short-prompt TTFT is reported. The kit target is 0.5 s. The default's
  README value is 0.39 s, and the kit re-measured it at 0.505 s. If it
  misses, the miss is named in every summary. Speed targets are general, so a
  miss is reported, not blocking (owner decision 2026-09-11).
- **More accurate:** Flash-Next's correct count summed over the four suites
  must exceed the default's:
  - GSM8K holdout 98/100;
  - MMLU-Pro holdout 210/247;
  - HumanEval 130/164;
  - MMMU 64/100.

  That is 502 of 611. Also, no suite may be significantly worse: a one-sided
  two-proportion z-test against the default's count, where z < −1.645 is
  worse. Every per-suite delta is reported with its sign.

  The GSM8K and MMLU-Pro holdout rows are drawn from a fixed seed, so they are
  the questions the default answered. The default's HumanEval and MMMU
  numbers come from the same 164 and 100 items.
- **If either test fails,** Flash-Next is not promoted. The result goes to the
  owner with the numbers.

## Reported, not gating this verdict

- **Speed:** block-aware decode and prefill against the `qwen38-1m` targets.
  These are the only Flash-Next numbers eligible for the README, and only if
  this gate passes.
- **Accuracy:** holdout, as above.
- **Fidelity:** `NO_RESULT`. There is no BF16 teacher reference on this host.
- **Kit verdict:** reported. It is `FAIL` whenever any declared target misses,
  including the general speed targets.

## Running start 2

Check clauses 1–4, 6, 8 and 9 in start 1's evidence before booking start 2's
long window. If start 1 has already failed any clause, the verdict is `FAIL`
whatever start 2 shows. Start 2 is then skipped, and the holdout stays
unspent.

## Freeze and randomness

- The commit that adds this file freezes:
  - the profile, the kit and the wrapper;
  - the agent-gate adapter and the vision bench and its eval set;
  - this driver and this scorer.

  The kit's manifest records git HEAD and the sha256 of every script it
  composes.
- Once a start has run, any change to these files is a new candidate in a
  new directory.
- There is one arm and no fixture choice. The kit's fixtures are fixed by its
  scripts:
  - speed seed 42;
  - the ledger's fixed-seed holdout rows;
  - the pinned MMMU-val-100 set (`evalsets/mmmu-val-100/pins.json`);
  - deterministic context needles.

  So no public randomness is drawn. This follows the precedent of candidate 5
  and the GLM-5.3 qualification bundle.
- `gate_mutations.py` checks that the scorer rejects malformed or failing
  evidence: 29/29 cases, of which 4 are new vision cases. Run it on synthetic
  copies before scoring.
