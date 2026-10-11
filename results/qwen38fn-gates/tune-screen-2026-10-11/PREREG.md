# Qwen3.8-Flash-Next lossless lever screen (DIAGNOSTIC, preregistered)

This screen decides which lossless speed levers go into the next acceptance candidate (c10). It is **diagnostic only**. Its numbers are never headline, README or acceptance evidence. The verdict comes from `screen_verdict.py`, which is in this directory and fixed by this file's commit, together with `arms.json` and `run-arm.sh`. This file was committed before any arm was loaded.

## Why

On 2026-10-11 the owner chose to tune Flash-Next with **lossless levers only**. Community recipes run the same NVFP4 checkpoint 1.4–2.6× faster than our 25.8 / 34.9 tok/s.

- **The runtime.** `runtime-002` is runtime-001 plus the Python-only patch series in `patches/qwen38fn/` (README there). Its pins are in `configs/build-manifests/qwen38-flash-next-runtime-002.json`.
- **The arms.** The arm profiles `cuda-spark-128g-1m-tune-a1..a5` are pinned by `scripts/tests/test_qwen38fn_tune_contract.py`.

| Arm | Profile | Builds on | Adds |
|---|---|---|---|
| a0 | `cuda-spark-128g-1m` (runtime-001, qualified) | — | baseline, re-measured here |
| a1 | `tune-a1` | a0 | L0: runtime-002 with no switch set. Always-on patches: #59528 null-block fix, #58449 fused QSA draft metadata, #58737 fused mamba-align |
| a2 | `tune-a2` | a1 | L1: `QWEN38FN_PLE_WILLNEED=1` |
| a3 | `tune-a3` | a2 | L2: `QWEN4EXP_DRAFT_VOCAB=65536` + `draft_lm_head_quantization: nvfp4` |
| a4 | `tune-a4` | a3 | L3: k=4 at one request, k=3 at two to four (bench K = 5) |
| a5 | `tune-a5` | a4 | L4: BF16 MTP-layer KV (`--kv-cache-dtype-skip-layers 48`), 20.6 GB arena. Runs only if its base arm's peak ≤ 104.7 GiB |

**If the keep rule drops a lever,** the next arm builds on the last *kept* arm instead. That arm is a new profile leaf and a new `arms.json` entry, committed before it loads; the levers are unchanged. The scorer refuses an `arms.json` whose bases disagree with the keep decisions.

## What runs per arm (`run-arm.sh <arm>`, one fresh start each)

**Precondition.** Production is stopped first with `sudo -n scripts/52_engine_switch.sh stop`, and restored after the last arm. arlab is paused by the owner.

**Start.**
- `03_memory_guard.py --required-gib 114.5`, 3 stable samples.
- Then `93_profile_serve.sh --profile <arm profile> --port 8017` inside a user scope: `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill`, `RuntimeMaxSec=7200`.
- The profile's memwatch is armed at its 8 GiB floor.
- A 1 Hz `/proc/meminfo` sampler and `journalctl -k` run for the whole arm.

**Readiness.** Attention block 3,200 tokens, and KV capacity ≥ 1,048,576 tokens at 262,144 per request. Otherwise the driver logs `FAILED_ATTEMPT`.

**Measured, in order:**
1. `probe.json`: the three greedy probes (`../smoke-2026-10-11/probe.py`). Their sha256 is compared with candidate 5's (`../qualify-2026-10-11-c4/probe-start1.json`), which candidate 9 matched. This is recorded, not gating.
2. `metrics-before.txt`, then `speed.json`, then `metrics-after.txt`.
   - `30_bench_speed_blocks.py` runs at contexts 0 and 28,672, with 3 reps after 1 warm-up.
   - The arguments are the kit's (seed 42, 384 max tokens, `min_tokens` 320, ignore-eos), with K = the profile's `speed_max_tokens_per_event`.
   - MTP acceptance length, reported only, is 1 + Δaccepted / Δdrafts from vLLM's `/metrics` counters across the speed cell.
3. `toolcall.json`: `39_bench_toolcall.py`, 20 cases, temperature 0.

**Close.** Stop, check that the scope is empty, and copy the server and memwatch logs.

**Memory.**
- *Launch baseline* is the last MemAvailable sample at or before the driver's `launching` line.
- *Low* is the minimum MemAvailable from launch until the stop.
- *Peak use* is baseline − low.
- *Swap growth* is SwapFree at launch minus the minimum SwapFree.

## Keep rule (fixed)

**Arm safety.** An arm is safe only if all of these hold:
- no `FAILED_ATTEMPT`;
- no kernel line matching `out of memory|oom-kill|oom_reaper|NVRM|Xid`;
- no server-log `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
- no memwatch `BREACH|EMERGENCY_STOP`;
- swap growth ≤ 0.25 GiB;
- peak use ≤ 106 GiB;
- KV capacity ≥ 1,048,576;
- tool-call ≥ 19/20;
- both speed cells `valid`;
- TTFT at 0 ≤ 0.5 s.

Ratios are taken against the arm's base: r0 and r28 are decode medians at 0 and 28,672, and rp is the prefill median at 28,672.

- **a1 (L0, the runtime change) passes** if it is safe and min(r0, r28) ≥ 0.97 and rp ≥ 0.95. Otherwise the screen ends `NO_GO (runtime-002)`, and nothing goes to c10.
- **A lever (a2 to a5) is kept** if its arm is safe, √(r0 · r28) ≥ 1.03, min(r0, r28) ≥ 0.98 and rp ≥ 0.95. Otherwise it is dropped.
- **a5 is `SKIPPED`** unless its base arm's peak use is ≤ 104.7 GiB.
- **a0 must itself be safe.** Otherwise the screen is `NO_RESULT`: the host or the harness is wrong, not a lever.

The result is the kept set and the arm that carries it. That arm's profile becomes candidate c10's launch, under its own preregistration (plan A2: 5 matched ABBA blocks against the main profile, the full c9 clause set, and a holdout spend if the probe hashes differ).

**Missing or malformed evidence for an arm makes it `NO_RESULT`.** Its lever counts as not kept, and the screen reports it.

`screen_verdict.py --self-test` runs synthetic cases before scoring, and all must match:
- a clean screen keeps every lever;
- each safety clause dropping a lever;
- a1 failing;
- a5 skipped;
- a missing file;
- an `arms.json` base that disagrees with the decisions.

## Reported, not gating

- MTP acceptance length per arm;
- whether `Fused multi-step draft decode is not supported` appears in the server log (expected absent from a1 on);
- the startup switch lines;
- load time;
- idle MemAvailable;
- the probe hash match against candidate 5.
