# Qwen3.8-Flash-Next candidate 7: images on at 1M under an 8 GiB floor (preregistered)

This is an acceptance run of profile `qwen3.8-flash-next/cuda-spark-128g-1m`
on the Spark. This file was committed before any weights were loaded for this
candidate. The verdict comes from `gate_verdict.py` in this directory, which
writes `gate-summary.json`. The qualification kit writes its own bundle
(`manifest.json`, `summary.json`, `SUMMARY.md`, one directory per cell) into
this same directory.

## Why

Candidate 5 (`../qualify-2026-10-11-c4/`) qualified the text-only profile,
which became the optional `qwen38fn` alias. Images at 1M were `NO_GO` above
the 10 GiB floor (`../diag-vision-2026-10-11/RESULT.md`). Text-only, the fill
bottomed at 12.25 GiB, and images cost 3.1–3.3 GiB.

**Owner decision (2026-10-11):**
- turn image input on;
- keep the full 1M topology;
- move the kill floor to **8 GiB**;
- make Flash-Next the default only if this run holds the measured low point
  above 8 GiB with no NVRM or OOM line.

DeepSeek's 1M-fast profile set the precedent for an 8 GiB floor. AGENTS.md's
1M gate asks for 10 GiB; this profile records the owner's exception.

## What changed in the profile (test-first, `a8e9dd9e`)

- `--limit-mm-per-prompt {"image":4,"video":0}` replaces `{"image":0,"video":0}`.
  Video stays off, as on the default, and `serving.max_images` is 4. Every
  other argv and env value is candidate 5's (`test_launch_is_candidate_5_plus_image_input`).
- **Floors:** `safety.kill_floor_gib` and `memory_model.floor_gib` go to 8.
  The memwatch arms at 8, and the kit's context cell takes the same floor.
- **Admission:** `safety.minimum_start_gib` goes to 115. This is the switch's
  only admission, and the quiet host reads 115.2–115.6 GiB.
- **Memory model:** `resident_weights_gib` 77 and `overhead_gib` 11.3, so
  weights + overhead + extra = 106.3 GiB. That is candidate 5's 102.97 GiB
  peak plus 3.3 GiB for images.
- **Status:** `estimated` until this gate passes, because candidate 5's PASS
  covers only the text-only launch.

## What runs (one fresh start, `run-phase2.sh start1`, `RuntimeMaxSec` 5 h)

The start runs only after `03_memory_guard.py --required-gib 114.5` passes.
`93_profile_serve.sh` admits only at 114.3 GiB (77 + 11.3 + 18 + 8).

Containment:
- a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0` and
  `OOMPolicy=kill`;
- `93_profile_serve.sh` serves on `:8017` and arms its memwatch at 8 GiB;
- an external sampler logs `/proc/meminfo` and the scope's `memory.current`
  at 1 Hz;
- `journalctl -k` is captured for the whole start.

After the server is ready, the driver checks vLLM's startup log. It needs
attention block size 3,200 and a KV capacity of at least 1,048,576 tokens at
262,144 per request. Otherwise it logs `FAILED_ATTEMPT` and stops before any
cell runs.

Then, in order:
1. `probe-start1.json`: the smoke's three greedy prompts (`probe.py`).
2. Kit, `--no-launch --cells speed,toolcall,vision,teacher,context --with-soak`,
   in the kit's fixed order:
   - **speed:** the block-aware wrapper;
   - **tool-call:** 20 cases;
   - **vision:** MMMU-val-100, chat mode, 256-token answers;
   - **teacher:** expected SKIPPED (no BF16 reference);
   - **context:** a direct fill of 4 × 250,128 = 1,000,512 tokens with
     needles and controls, at an 8 GiB floor. It runs **after** the vision
     cell, so any encoder memory the allocator still holds counts against the
     fill's low point;
   - **soak:** 30 minutes.
3. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model default --prefix-block-tokens 3200 --spec-block-drop`.

The holdout is not drawn again. It was spent once on this stack
(`../holdout-2026-10-11/`). Clause 5 is what lets that text accuracy carry
over.

## Gate formula (fixed)

`PASS` requires every clause below, and any failed clause gives `FAIL`.
Missing or malformed evidence gives `NO_RESULT` unless a clause has already
failed. So does a `FAILED_ATTEMPT` line, a log that never saw the run (the
server log's `session start`, memwatch's `ARMED`), a dirty `head` line, or any
frozen file (`gate_verdict.py` `FROZEN`) that differs between the freeze
commit and the run's head. The freeze commit is the last commit that touched
this file.

1. **No kill lines.** These are the smoke's patterns:
   - kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive;
   - server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch log: `BREACH|EMERGENCY_STOP`.
2. **Host swap growth ≤ 0.25 GiB:** `max(SwapTotal − SwapFree)` over the 1 Hz
   samples, minus the first.
3. **External low point ≥ 8 GiB** `MemAvailable` (the owner's floor).
4. **Direct 1M:**
   - vLLM's KV capacity is at least 1,048,576 tokens at 262,144;
   - the context cell verdict is `PASS`;
   - total tokens ≥ 1,000,000;
   - the kit manifest's context cap is exactly 1,048,576.
5. **Text path unchanged:** the three greedy probe sha256 values equal
   candidate 5's (`../qualify-2026-10-11-c4/probe-start1.json`). The
   diagnostic saw 3/3 identical with images on. A mismatch means the holdout
   accuracy cannot be carried over.
6. **Tool-call ≥ 19/20.**
7. **Agent gate:** all four tests pass.
8. **Soak:** the soak cell passes.
9. **Vision not significantly worse than the default:**
   - `vision/summary.json` has `n` = 100 and `error_count` = 0;
   - the one-sided pooled two-proportion z of `correct`/100 against the
     default's 64/100 must be ≥ −1.645, which means `correct` ≥ 53.

   Unparseable answers count as wrong. This is the "no suite significantly
   worse" rule the owner's holdout comparison used.

   Known before this freeze: the diagnostic scored 63/100. The owner chose
   this candidate knowing that, and decoding is greedy, so the score is
   reported with its delta whatever it is.
10. **Switch admission holds:** peak use plus the 8 GiB floor fits inside the
    115 GiB `minimum_start_gib`. Peak use is the first sample's
    `MemAvailable` minus the low point. Otherwise a boot restore could start
    an engine that its own floor would kill.

`gate_mutations.py` runs 33 synthetic cases, and all must match before
scoring:
- PASS: clean, an evidence-only later commit, and MMMU at 53 and at 63;
- FAIL: one case per failing clause;
- NO_RESULT: each missing or malformed input, including a frozen file changed
  after the freeze.

## If it passes

1. One reviewed commit flips the status to `qualified`, with this directory as
   evidence. The README's Flash-Next rows take this bundle's speed and vision
   numbers; the text accuracy rows stay from the holdout.
2. The switch makes `qwen38fn` the default:
   `sudo -n scripts/52_engine_switch.sh qwen38fn`, left running, in a window
   agreed with the co-tenant. README and AGENTS.md name it as the default.
   The switch's boot-restore fallback stays `dsv4`.

If the verdict is `FAIL` or `NO_RESULT`, nothing changes: `qwen38-1m` stays
the default, the text-only `qwen38fn` on main stays as it is, and the result
goes to the owner.

## Reported, not gating

- **Speed:** block-aware decode, prefill and TTFT against the `qwen38-1m` row.
  Short-prompt TTFT has missed so far (0.57 s against 0.39 s); speed targets
  are general.
- **MMMU score**, with its delta against 64.
- **Fidelity:** `NO_RESULT`. There is no BF16 teacher on this host.
- **Kit verdict:** reported. It is `FAIL` whenever any declared target misses.

## Freeze and randomness

The commit that adds this file freezes the profile, the kit, the wrapper, the
agent-gate adapter, the vision bench and its eval set, this driver and this
scorer. Once the start has run, a change to any of these is a new candidate in
a new directory.

There is one arm and no fixture choice. The kit's fixtures are fixed by its
scripts:
- speed seed 42;
- the pinned MMMU-val-100 set;
- deterministic context needles;
- the smoke's three probes.

So no public randomness is drawn. This follows candidate 5 and the GLM-5.3
bundle.
