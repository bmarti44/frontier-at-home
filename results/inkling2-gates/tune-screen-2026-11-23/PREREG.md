# Inkling 2 Mini lossless lever screen (DIAGNOSTIC, preregistered)

This screen decides which lossless speed levers join the Inkling 2 Mini acceptance candidate (M3, `qualify-*-c1`), and picks the served reasoning effort. **It is diagnostic only.** Its numbers are never README, headline or acceptance evidence.

The verdict comes from `screen_verdict.py`. That script, `arms.json`, `run-arm.sh`, `code_decode.py` and this file are fixed by the commit that adds them, before any arm loads. The arm profiles are pinned by `scripts/tests/test_inkling2_tune_contract.py`: each is the carrier plus exactly one lever.

## Arms (one fresh start each, each lever against a0 alone)

| Arm | Profile (`inkling-2-mini/…`) | Lever |
|---|---|---|
| a0 | `cuda-spark-128g-1m` | none: the carrier (4 × 262,144, BF16 KV, 68e9-byte arena, `--max-num-batched-tokens 2048`, effort low) |
| a1 | `cuda-spark-128g-1m-tune-b4096` | `--max-num-batched-tokens 4096` |
| a2 | `cuda-spark-128g-1m-tune-b8192` | `--max-num-batched-tokens 8192` |
| a3 | `cuda-spark-128g-1m-tune-pow2` | `INKLING2_MAX_CTAS_POW2=1` (patch 0001 = vLLM PR #69288, behind 0002) |
| a4 | `cuda-spark-128g-1m-tune-dspark` | DSpark ×7, `draft_sample_method` probabilistic, standard rejection sampling (the vLLM DGX Spark recipe) |
| a5 | `cuda-spark-128g-1m-tune-humming` | `--moe-backend humming` |
| a6 | combined (authored later) | every kept lever; only if two or more are kept |

**Run order:** a0, a3, a4, a5, a1, a2, then a6 if it is due.

**a6 rules:**
- It is a new profile leaf plus an `arms.json` entry with `"levers": [...]`, committed before it loads.
- Its levers must equal the scorer's kept set, or the screen is `NO_RESULT`.
- `--may-run` admits it only after a1–a5 all exist and two or more levers are kept.

**Lossless.** None of these levers changes weights, KV precision or the sampled distribution.
- The batch size changes only the prefill chunking, so float summation order can differ.
- #69288 changes only which compiled kernel variant runs.
- DSpark's rejection sampling keeps the target's distribution: at temperature 0 the target's argmax is emitted.
- Humming is a different NVFP4 GEMM kernel over the same weights. Because its numerics may differ, it is kept only if its greedy probes are byte-identical to a0's. Otherwise it would need the fidelity gate, and that fidelity is not spent.

**Amendments.** A DSpark arm that fails readiness from the wheel's FA2 toolchain (`cudaErrorUnsupportedPtxVersion`, as the vision encoder did in smoke run1) may be retried once as `a4b`. That arm uses a new default-off patch, committed with its test before it loads.

Defects found by running an arm follow AGENTS.md:
- the fix targets only the observed failure;
- the failed arm directory is preserved under a new arm id;
- the failure is recorded in `ATTEMPTS.md` first.

## What runs per arm (`run-arm.sh <arm>`)

**Precondition:** production is stopped with `sudo -n scripts/52_engine_switch.sh stop` and restored after the last arm, through `window.sh` in `../smoke-2026-11-23`, which restores on exit.

**Preflight.** These checks run before the arm directory exists, so an abort leaves no arm behind:
- a clean tracked tree;
- `screen_verdict.py --may-run`;
- the system inference lock is taken; the engine starts with that fd closed;
- the page cache is released under the Inkling and Flash-Next trees;
- `03_memory_guard.py --required-gib 110` passes with 3 stable samples;
- MemFree ≥ 100 GiB.

**Start:**
- `93_profile_serve.sh --profile <arm> --port 8018` inside a user scope: `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill`, `RuntimeMaxSec=7200`. The cgroup cannot see CUDA allocations on GB10; run2 peaked at 35.8 GiB of cgroup memory against a host peak of 46.2 GiB.
- The profile's memwatch kills at the 10 GiB floor.
- A 1 Hz meminfo sampler and kernel-log capture run throughout.

**Readiness:** the launch reaches ready, and vLLM logs ≥ 4.0x concurrency at 262,144 tokens per request. Otherwise the driver logs `FAILED_ATTEMPT` and stops.

**Measured, in order:**
1. `probe.json`: the 3 greedy probes (`../smoke-2026-11-23/probe.py`, 1,024 tokens).
2. `cold.json`: the 6 unseen-length prompts of `../smoke-2026-11-23/cold_ttft.py`, the same seeds in every arm (each arm is a fresh server, so no prefix hits). This gives the aggregate prefill tok/s.
3. `speed.json`: `30_bench_speed_blocks.py` with **K = 8 for every arm**, so one bench and one decode formula apply throughout.
   - Contexts 0 and 28,672; 3 reps after 1 warm-up.
   - Seed 42, 384 max tokens, `min_tokens` 320, ignore-eos.
   - vLLM `/metrics` is scraped before and after, which gives the DSpark acceptance length (reported only).
4. `code.json`: `code_decode.py`, three greedy code-writing requests (768 max tokens). Decode is `(completion_tokens − 8) / (t_last − t_first)`, and each request needs ≥ 200 tokens.
5. `toolcall.json`: `39_bench_toolcall.py`, 20 cases, temperature 0.
6. a0 and a6 only: `38_bench_vision.py`, 10 MMMU cases in chat mode with `--max-tokens 4096`. This is the live check of patch 0004 and is reported only.

**Close:** stop the engine, confirm the scope is empty, and copy the server and memwatch logs.

**Memory:**
- *Launch baseline* is the last MemAvailable sample at or before the `launching` line.
- *Low* is the minimum after it, and *peak use* = baseline − low.
- *Swap growth* is the baseline SwapFree minus the minimum.
- *Projected 1M low* = low − 3 GiB. The arena is preallocated, so a 1M fill differs from this 28K fill only by its activations.

## Keep rule (fixed)

**An arm is safe only if all of these hold:**
- no `FAILED_ATTEMPT`;
- no kernel `out of memory|oom-kill|oom_reaper|NVRM|Xid`;
- no server `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
- no memwatch `BREACH|EMERGENCY_STOP`;
- swap growth ≤ 0.25 GiB;
- projected 1M low ≥ 12 GiB;
- KV concurrency ≥ 4.0;
- tool-call ≥ 19/20;
- the scope is empty after the stop.

**a0 must be safe,** or the screen is `NO_RESULT`.

**Integrity: any of these makes the arm `NO_RESULT`, and its lever is not kept:**
- evidence is missing or malformed;
- the non-default args show a `max_num_batched_tokens` other than the arm's (2048 unless it is a batch arm);
- `INKLING2_MAX_CTAS_POW2=1`, `'method': 'dspark'` or `'moe_backend': 'humming'` appears in the server log without its lever, or is missing with it;
- the patch 0004 line `INKLING2_MAX_IMAGE_SIDE=4096` is missing.

**Ratios against a0:**
- r0 and r28 are the decode medians at 0 and 28,672;
- rp is the median prefill at 28,672;
- rc is the cold aggregate prefill;
- rt0 is the median TTFT at 0;
- rcode is the code decode median.

**Per lever:**
- **Batch (a1, a2):** kept if rp ≥ 1.03, rc ≥ 1.0 and min(r0, r28) ≥ 0.98. If both are kept, the one with the larger rp stays.
- **#69288 (a3):** kept only if the probes are byte-identical to a0's, (rc ≥ 1.03 or rt0 ≤ 0.97), min(r0, r28) ≥ 0.98 and rp ≥ 0.97.
- **DSpark (a4):** kept if min(r0, r28) ≥ 0.98, rp ≥ 0.95, and either √(r0·r28) ≥ 1.03 or rcode ≥ 1.25. Probe hashes are reported, not gating: verification batches can change float order.
- **Humming (a5):** kept only if the probes are byte-identical to a0's, (√(r0·r28) ≥ 1.03 or rp ≥ 1.03), min(r0, r28) ≥ 0.98 and rp ≥ 0.95.

**Combined arm (a6):** it carries if it is safe, √(r0·r28) ≥ 1.0 and rp ≥ 0.97 against a0. Otherwise the kept single-lever arm with the largest √(r0·r28) carries. With zero levers kept, a0 carries; with one, that lever's arm carries.

**Verdicts:**
- `COMPLETE`: a carrier is named.
- `INCOMPLETE`: an arm has not run, a lever is `NO_RESULT`, or a6 is due but absent.
- `NO_RESULT`: a0 is unsafe or malformed, or a6's levers disagree with the kept set.

`screen_verdict.py --self-test` runs 29 synthetic cases, which must all match before it scores. They cover:
- each safety clause and each integrity clause;
- every keep branch;
- the a6 rules;
- `--may-run`.

## Reasoning-effort pick (after the carrier is named; fixed now)

One fresh load of the carrier runs `31_bench_accuracy.py` on the **dev** split, which spends no ledger rows:
- GSM8K: 100 rows;
- MMLU-Pro dev: about 250 rows;
- `--encoder inkling2 --thinking-mode thinking`, max tokens 16,384;
- once at `--reasoning-effort low` and once at `medium`.

The effort with the higher dev sum (GSM8K + MMLU-Pro correct) is served. A tie goes to `low`, the repository convention. That effort goes into `--default-chat-template-kwargs` for the M3 candidate and into the holdout.

**How it runs.** `run-effort.sh <carrier>` (via `window.sh … effort:<carrier>`) reuses run-arm.sh's preflight, containment and monitoring, with `RuntimeMaxSec` 8 h. `--effort-due` admits only the carrier of a `COMPLETE` screen. The four suite runs (2 suites × 2 efforts) go to the one server concurrently, each with its own transcripts. Both efforts therefore share the same batching conditions.

Mean completion tokens per effort are reported, not gating. HumanEval has no dev split, so it is not used here.

## Reported, not gating

- load time and idle MemAvailable;
- KV tokens;
- DSpark acceptance length (1 + Δaccepted / Δdrafts from `/metrics`);
- code decode for every arm;
- the probe hashes against a0 and against smoke run2 (`170c63d11040`, `c7fe1a930bcb`, `217aaabaf808`; run2 used 2 × 131,072);
- vision on a0 and a6;
- projected 1M low.
