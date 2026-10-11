# Qwen3.8-Flash-Next Phase 2: direct-1M qualification (preregistered)

This is an acceptance run of profile `qwen3.8-flash-next/cuda-spark-128g-1m`
on the Spark. It was committed before any weights were loaded for it. The
gate verdict comes from `gate_verdict.py` in this directory, which writes
`gate-summary.json`. The qualification kit writes its own bundle
(`manifest.json`, `summary.json`, `SUMMARY.md`, one directory per cell) into
this same directory.

## What runs

Both starts are fresh servers. Each start runs only after
`03_memory_guard.py --required-gib 110` passes, inside a user scope with
`MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill` and a
`RuntimeMaxSec` cap. `93_profile_serve.sh` serves the profile on `:8017` and
arms its memwatch at the profile's 10 GiB floor. An external sampler logs
`/proc/meminfo` and the scope's `memory.current` at 1 Hz, and `journalctl -k`
is captured for the whole start.

After the server is ready, the driver checks vLLM's own startup log. It needs
an attention block size of 3,200 and `GPU KV cache size` of at least
1,048,576 tokens at 262,144 per request. Otherwise it logs `FAILED_ATTEMPT`
and stops before any cell runs. On every exit it stops the scope if anything
is still running in it.

**Start 1** (`run-phase2.sh start1`, `RuntimeMaxSec` 5 h):

1. `probe-start1.json`: three greedy prompts, reusing the smoke's `probe.py`
   by path.
2. Kit, `--no-launch --cells speed,toolcall,teacher,context --with-soak`:
   - speed: the block-aware wrapper (`qualification_options.speed_max_tokens_per_event = 4`);
   - tool-call: 20 cases;
   - teacher: expected SKIPPED, since `model.json` has no BF16 reference logits;
   - context: direct fill of 4 × 250,128 = 1,000,512 tokens with needles and
     controls, memory floor 10 GiB;
   - soak: 30 minutes.

**Start 2** (`run-phase2.sh start2`, `RuntimeMaxSec` 16 h):

1. `probe-start2.json`: the same three prompts.
2. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model default --prefix-block-tokens 3200 --spec-block-drop`.
   MTP makes vLLM drop one trailing cache block on every prefix hit
   (`drop_eagle_block`), so the reuse floors allow two blocks.
3. Kit, `--cells accuracy --accuracy-split dev` on the same bundle. Holdout
   accuracy stays owner-run (docs/PROFILE-SCHEMA.md). HumanEval always runs
   all 164 problems. If the owner resumes the co-tenant, this step may be
   stopped and resumed later; the kit keeps finished suites.

Vision and media are not run: the profile is text-only (`--limit-mm-per-prompt` 0).

## Gate formula (fixed)

`PASS` requires every clause below. Any failed clause gives `FAIL`. Missing or
malformed evidence, or a `FAILED_ATTEMPT` line in a driver log, gives
`NO_RESULT` unless a clause has already failed. So do these:
- a log that is missing the line proving it saw the run (the server log's
  `session start`, memwatch's `ARMED`);
- driver `head` lines that differ between starts or show tracked edits.

1. **No kill lines** in either start. The patterns are the smoke's:
   - kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive;
   - server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch log: `BREACH|EMERGENCY_STOP`.
2. **Host swap growth ≤ 0.25 GiB** in each start: `max(SwapTotal − SwapFree)`
   over the 1 Hz samples, minus the first sample. This re-measures the smoke's
   only failed clause on a host with no co-tenant load.
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

## Reported, not gating this verdict

- **Speed:** block-aware decode and prefill against the `qwen38-1m` targets.
  Speed targets are general; misses are reported (owner decision 2026-09-11).
  These are the only numbers eligible for the README, and only if this gate
  passes.
- **Accuracy:** dev split.
- **Fidelity:** `NO_RESULT`. There is no BF16 teacher reference on this host.
  NVFP4 experts, fp8 KV and the PLE table are as published by NVIDIA. The
  smoke's 20/20 tool-call result is diagnostic only.
- **Kit verdict:** reported. It is `FAIL` whenever any declared target misses,
  including the general speed targets.

Promotion stays the owner's decision (Phase 3).

## Memory sizing (candidate 2)

- **Smoke data.** Every KV-usage reading in the smoke's server log is a whole
  number of blocks over 91. So its 4.2 GB arena held 92 blocks, and a
  131,072-token request reserves 62 of them:
  - one group grows by `cdiv(L, 3200)`, which is 41 blocks at that length;
  - 21 are fixed: GDN state (4 groups × 5) plus the QSA ring (1).
- **Arena.** A 262,144-token slot reserves 103 blocks, so four slots need 412
  blocks, at most 18,808,695,688 bytes. The arena is 18.85 GB.
- **Footprint.** The engine's non-KV footprint is about 84.2 GiB. Projected
  idle on a quiet 114 GiB host is about 12.2 GiB.
- **Correction.** The smoke's RESULT.md said "13–15 GiB on a quiet host" for
  a 22.7 GB arena. That subtracted from MemTotal rather than MemAvailable, so
  it was wrong. That file is frozen evidence and is left unchanged.
- **Risk to clause 2.** Running about 12 GiB above the floor may push idle
  pages of other processes to swap. If it does, clause 2 fails, and that
  failure is the measurement. It is not waived.

## Candidates

- **Candidate 1** (`21ff75df`): reviewed and never run. The review found:
  - high: the KV bound and memory model (admission impossible on a quiet
    host), and the agent gate under MTP block drop;
  - medium: scope cleanup, code identity, and the swap premise;
  - low: first-event bound and empty logs.
- **Candidate 2:** the commit that changes this section. It closes those
  findings with the 18.85 GB arena, the measured memory model,
  `--spec-block-drop`, fail-closed driver checks, and the scorer's identity
  and marker checks. `gate_mutations.py` passes 23/23.

## Freeze and randomness

- The commit that adds this file freezes the profile, the kit, the wrapper,
  the agent-gate adapter, this driver and the scorer. The kit's manifest
  records git HEAD and the sha256 of every script it composes.
- Once a start has run, any change to these files is a new candidate in a
  new directory.
- There is one arm and no fixture choice: the kit's fixtures are fixed by its
  scripts (speed seed 42, ledger-recorded dev rows, deterministic context
  needles). So no public randomness is drawn. This follows the precedent of
  the GLM-5.3 qualification bundle.
- `gate_mutations.py` checks that the scorer rejects malformed or failing
  evidence. Run it on synthetic copies before scoring.
