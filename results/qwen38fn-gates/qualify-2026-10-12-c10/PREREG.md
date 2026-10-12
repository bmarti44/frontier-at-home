# Qwen3.8-Flash-Next candidate 10: runtime-002 + PLE WILLNEED + reduced-vocabulary NVFP4 draft head (preregistered)

This is an acceptance run of profile `qwen3.8-flash-next/cuda-spark-128g-1m-tune-c10` on the Spark. This file was committed before any weights were loaded for this candidate.
- **The verdict** comes from `gate_verdict.py` in this directory, which writes `gate-summary.json`.
- **Run 1:** the qualification kit writes its own bundle (`manifest.json`, `summary.json`, `SUMMARY.md`, one directory per cell) into this directory.
- **Run 2:** `scripts/97_matched_speed_blocks.py` writes `plan.json` and `blocks/`.

## Why

On 2026-10-11 the owner chose to tune Flash-Next with lossless levers only.

The diagnostic lever screen (`../tune-screen-2026-10-11/RESULT.md`) kept two levers:
- **L1:** PLE `MADV_WILLNEED`;
- **L2:** a 65,536-row draft vocabulary plus an NVFP4 draft head.

Both run on runtime-002, which is runtime-001 plus the Python-only series in `patches/qwen38fn/`.

In single starts against candidate 9's launch, the carrier arm a3 measured:
- decode 1.354× at 0 and 1.158× at 28,672;
- prefill 1.013×;
- TTFT 0.274 → 0.212 s;
- a peak 0.56 GiB lower;
- byte-identical probes.

**Rejected:**
- **L3 (k=4):** dropped, because it held only 1,033,876 KV tokens at the fixed arena.
- **L4 (BF16 MTP KV):** refused at membudget admission and not measured.

**Candidate 10 is a3 plus one startup change.** The screen's first a1 attempt met its KV arena at MemFree 19.85 GiB, short of the 19.90 needed, because runtime-001's libraries were still cached (`ATTEMPTS.md` there). The promotion switch (runtime-001 → runtime-002) recreates exactly that host state.

## What changed (test-first: RED `c63f8381`, `92c07f9a`, `21b849a7`)

- **`qwen38fn_worker.HostCacheReleaseWorker`** reads `QWEN38FN_HOST_RELEASE_ROOTS` once at import (absolute paths joined by `:`, empty by default).
  - Before the weights load, it releases each named root after the model root.
  - Before the KV arena, it releases each named root first, logging `host page cache released under <root> (<n> files) before the KV arena`. Then comes candidate 9's unchanged model-root release, MemFree line and refusal.
  - Every existing profile leaves the variable unset and is unchanged.
- **Profile `cuda-spark-128g-1m-tune-c10`** is the screen's a3 with `QWEN38FN_HOST_RELEASE_ROOTS={cache_root}/qwen38-flash-next` and nothing else (`test_candidate_10_is_the_screen_carrier_plus_the_runtime_release_root`).
  - Engine `vllm-qwen38fn-r002`.
  - Speculative config `{"method":"mtp","num_speculative_tokens":3,"draft_lm_head_quantization":"nvfp4"}`, plus `QWEN38FN_PLE_WILLNEED=1` and `QWEN4EXP_DRAFT_VOCAB=65536`.
  - Every other argv and env value, the arena (19,220,000,000 bytes), `memory_model` and `safety` are candidate 9's.
- **`scripts/97_matched_speed_blocks.py`**, the shared tool, gained:
  - `--release-tree` (before each start);
  - the arena guard's refusal not counted as a kill line;
  - `plan --drand-after` (below);
  - after review round 1 (RED `56a393e1`, `9edb516d`):
    - kernel kill lines are matched ignoring case, as the smoke scorer does;
    - SIGTERM unwinds a start through its cleanup (engine stopped, scope waited for) instead of orphaning it;
    - `run` stops the campaign at the first start that failed or has a kill line, swap growth over 0.25 GiB or a low point under `--abort-low-gib` (8);
    - `score` rejects a start whose `profile` is not the plan's for its arm, or whose low point is not finite.
  - after review round 2 (RED `deeaf8c7`, `17f45895`):
    - the sampler's fields and line format are named (`MEMINFO_KEYS`, `meminfo_line`), so scorers and their mutation suites use exactly what it writes;
    - `--resume` reads each start already on disk and stops at a failed or unsafe one, instead of skipping it;
    - the journal follower starts inside a start's `try`, and a further SIGTERM during cleanup waits until the engine is stopped and the scope is gone.

## Run 1: one fresh start (`run-c10.sh start1`, `RuntimeMaxSec` 5 h)

**Preflight**, before `observe/start1` exists, so a refusal leaves nothing behind and the start can be retried:
- no tracked changes;
- the driver takes the system inference lock (`/run/lock/frontier-at-home/inference.lock`), with the engine and memwatch started with that fd closed. Production must already be stopped (`sudo -n scripts/52_engine_switch.sh stop`);
- `03_memory_guard.py --required-gib 114.5` must pass (its output goes into `driver.log` once the directory exists).

**Runtime identity.** After the head line, the driver logs `runtime-tree <name> <16 hex> <files>` for `native-runtime-001` and `native-runtime-002`:
- the `site-packages/vllm` digest by the runtime-002 manifest's `vllm_tree` method (`scripts/tests/test_qwen38fn_tune_contract.py` `tree_sha256`);
- each tree's page cache is released after it is read;
- the lines are logged again after the engine stops.

**Adverse precondition** (after the runtime-tree lines):
- **c9's prime:** qwen38-1m's `Qwen3.8-27B-Q4_K_M.gguf` (16.55 GiB) and `mmproj-Qwen3.8-27B-f16.gguf` (0.86 GiB) are read into page cache.
- **Runtime-001's nine largest libraries (2.98 GiB)** are also read: what a production `qwen38fn` start leaves cached.
- **Then** one `precondition <GiB> GiB <path>` line per primed file comes from `../qualify-2026-10-11-c7/pagecache_residency.py` (mmap + `mincore`), immediately before launch.

**Containment** is candidate 9's:
- user scope `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill`;
- `93_profile_serve.sh` on `:8017` with memwatch at 8 GiB;
- a 1 Hz `/proc/meminfo` and `memory.current` sampler;
- `journalctl -k` for the whole start.

**Readiness:** attention block 3,200 and KV capacity ≥ 1,048,576 tokens at 262,144. Otherwise the driver logs `FAILED_ATTEMPT`.

**Then, in order:**
1. `probe-start1.json`: the three greedy text probes.
2. The kit, `--no-launch --cells speed,toolcall,vision,teacher,context --with-soak`. These are the cells candidate 9 ran:
   - speed (block-aware, K = 4);
   - tool-call, 20 cases;
   - vision MMMU-val-100;
   - teacher, expected SKIPPED;
   - context: the direct fill of 4 × 250,128 tokens at an 8 GiB floor;
   - soak: 30 minutes at the repository 5 GiB soak floor.
3. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model default --prefix-block-tokens 3200 --spec-block-drop`.
4. `cjk-decode-start1.json` (reported, not gating; see below): `cjk_decode.py --k 4`.

**Early stop.** A clause-1 line during run 1 already means FAIL, and the operator may stop the engine. If run 1 has failed any clause, run 2 is not run. `run-c10.sh blocks` first runs `gate_verdict.py --run2-preflight`, which exits 0 only when all of these hold:
- run 1 has no failed clause and no `FAILED_ATTEMPT` line;
- nothing is missing except clause 5 (a changed text path sends accuracy to the holdout, not to a new run 1);
- `plan.json` passes clause 13's plan checks for the current freeze;
- the working tree is the freeze's under `FROZEN`, with no untracked file there.

Both modes' preflight also refuse untracked files under `scripts`, `configs`, `patches`, `fixtures` and `evalsets`, since `scripts/lib` is the engine's `PYTHONPATH`.

## Run 2: matched speed (`run-c10.sh blocks`, after run 1)

**Randomness.** After the freeze, `97_matched_speed_blocks.py plan --drand-after <freeze commit> --domain qwen38fn-c10-blocks --blocks 5` produces the plan:
- A = `qwen3.8-flash-next/cuda-spark-128g-1m`, the main profile, candidate 9's launch on runtime-001;
- B = `qwen3.8-flash-next/cuda-spark-128g-1m-tune-c10`;
- the drand round is the first one published after the freeze commit's committer time;
- three relays (api, api2, api3.drand.sh) must agree, and the bundled verifier checks its BLS signature;
- each block's order is ABBA when `sha256(domain NUL randomness block)[0]` is even, else BAAB.

**Starts.** `run-c10.sh blocks` first refuses if the inference lock is held, before reading anything. It then logs its head line and runtime-tree lines, and runs `97_matched_speed_blocks.py run --plan plan.json --out blocks --release-tree ~/.cache/qwen38-flash-next --runtime-max-sec 3600 --resume`, then logs the runtime-tree lines again. Rerunning it after a refused guard or a held lock resumes the campaign: each invocation logs its own head and runtime-tree lines, and completed starts are kept. A resume first re-checks every start already on disk and refuses to continue past a failed or unsafe one. At exit, the driver releases the page cache under `~/.cache/qwen38-flash-next` in both modes. That is 20 fresh starts on `:8017`, each:
- releasing the runtime trees' page cache;
- passing `03_memory_guard.py --required-gib 114.5`;
- running in a user scope with the profile's containment;
- sampled at 1 Hz with the kernel log;
- measured with `30_bench_speed_blocks.py` at 0 and 28,672 (3 reps after 1 warm-up, seed 42, 384 max tokens, `min_tokens` 320, K from the profile);
- then stopped.

97 stops the campaign after the first start that failed or has a kill line, swap growth over 0.25 GiB or a low point under 8 GiB. No further large load follows such an event, and the candidate is already `FAIL` or `NO_RESULT`.

Run 2's ratios are fresh-start ratios with the PLE pages cold (both workers evict the model root before the arena), measured on the bench's English-prose fixture. They are cited with that label.

**Scoring.** `97 score` reduces each block to its A and B means and takes paired log ratios B/A. It reports ratio and one-sided 95% bounds `exp(mean ∓ t95(df 4) · sd/√5)`. Any missing, failed, unsafe, invalid, non-finite, out-of-order or unexpected start is `NO_RESULT`.

## Gate formula (fixed)

`PASS` requires every clause below, and any failed clause gives `FAIL`. Missing or malformed evidence gives `NO_RESULT` unless a clause has already failed. So do:
- a `FAILED_ATTEMPT` line in run 1;
- a run-1 log that never saw the run (no `session start` in the server log, no `ARMED` in the memwatch log);
- a dirty head line, or head lines naming more than one commit, in either run (the same head logged by each invocation of a resumed run 2 is one commit);
- any `FROZEN` file that differs between the freeze commit and either run's head. The freeze commit is the last commit that touched this file;
- a runtime that is not the frozen one: each run's driver log must have `runtime-tree` lines for `native-runtime-001` and `native-runtime-002`, and every one must equal the runtime-002 manifest's `base_runtime.base_vllm_tree` and `vllm_tree` (first 16 hex digits and file count) at the freeze commit;
- an arm that is not the one named. Run 1's server log and every B start's must show all four lever lines:
  - `checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED=1`;
  - `Qwen4Exp MTP draft vocabulary: token ids [0, 65536)`;
  - `Quantized a nvfp4 copy of the lm_head`;
  - `'draft_lm_head_quantization': 'nvfp4'` in the non-default args.

  No A start may show any of them. Run 1 and every run-2 start must show `Setting attention block size to 3200 tokens`, and every run-2 start must show KV capacity ≥ 1,048,576 at 262,144; run 1's capacity is clause 4. Every `start.json` must name its arm's profile, which 97's score checks;
- a scorer or frozen tree that changed after the freeze. At scoring time the working tree must equal the freeze under `FROZEN`, with no untracked file there. `gate-summary.json` records the scorer's digest (first 16 hex digits), the scoring head and any `GATE_REPO` or `GATE_REFERENCE_PROBE` override.

1. **No kill lines,** in run 1's logs, in every run-2 start's `kill_lines`, and in the raw logs of every run-2 start directory, including one whose driver died before writing `start.json`. This scorer applies the smoke scorer's patterns, ignoring case, to both runs. Run-2 safety (clauses 1–3) is scored even when `plan.json` is missing or defective. In run 2 only, the arena guard's `refusing to allocate the KV arena` line is not a kill line; clause 12 scores it.
   - Kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`.
   - Server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`.
   - Memwatch log: `BREACH|EMERGENCY_STOP`.
2. **Swap growth ≤ 0.25 GiB,** for run 1 and for every run-2 start (from `start.json` and from the start's raw `meminfo.log`).
3. **External low point ≥ 8 GiB `MemAvailable`,** for run 1 and for every run-2 start (from `start.json` and from every raw sample).

   Samples must cover each run, otherwise `NO_RESULT`:
   - Run 1 needs at least 600 samples, from at or before the launch line to within 15 s of the driver's `start1: scope empty after stop` line, with no gap over 5 s.
   - In run 2, samples are 97's format (`MemAvailable` and `SwapFree` used; no `SwapTotal`). Each start's `start.json` must hold finite `swap_growth_gib`, `low_mem_available_gib` and `ready_s`. Its samples must number at least 0.9 × `ready_s`, run from at or before `launched_at` to within 15 s of the start's result line in `run.log`, and have no gap over 5 s.
4. **Direct 1M:**
   - KV capacity ≥ 1,048,576 at 262,144;
   - the context cell is `PASS`;
   - total tokens ≥ 1,000,000;
   - the manifest context cap is exactly 1,048,576.
5. **Text path unchanged:** the three greedy probe sha256 values equal candidate 5's (`../qualify-2026-10-11-c4/probe-start1.json`), so the holdout's text accuracy carries over.
   - The screen measured them equal.
   - If they differ, the verdict is `NO_RESULT` (holdout required). The owner-authorized holdout spend then runs as its own preregistered step before any promotion.
6. **Tool-call ≥ 19/20.**
7. **Agent gate:** all four tests pass, each reported exactly once (otherwise `NO_RESULT`).
8. **Soak** passes at the repository 5 GiB floor. `soak.json` records `mem_floor_gib` 5.0 and `mem_floor_source` `default`; otherwise `NO_RESULT`.
9. **Vision:**
   - `n` = 100 and `error_count` = 0;
   - the one-sided pooled z of `correct`/100 against the default's 64/100 is ≥ −1.645, meaning `correct` ≥ 53.
10. **Switch admission holds:** run 1's peak use + 8 GiB ≤ 115 GiB. Peak use is the highest `MemAvailable` sample at or before the one launch line, minus the low point.
11. **The adverse precondition held.** The driver logged `precondition` residency for exactly the eleven primed files, with:
    - the two GGUF files ≥ 15 GiB in total;
    - the nine runtime-001 libraries ≥ 2.5 GiB in total.

    Otherwise the run did not test the fix: `NO_RESULT`.
12. **The fixed worker ran.** In run 2, `refusing to allocate the KV arena` in a B start's server log is `FAIL`; in an A start it fails that start (`NO_RESULT`). In run 1, the server log has exactly one runtime-root line:
    - `host page cache released under /home/bmarti44/.cache/qwen38-flash-next (<n> files) before the KV arena`, with n ≥ 1.

    It also has exactly one model-root line:
    - under `/home/bmarti44/models`, with at least one file, for an arena of at least 17 GiB.

    `refusing to allocate the KV arena` anywhere is `FAIL`. Any other shape is `NO_RESULT`.
13. **Matched speed (run 2)** must hold all of these:
    - `plan.json` names A and B as above, the domain `qwen38fn-c10-blocks` and 5 blocks;
    - its drand receipt names the freeze commit and the first round after its committer time, and passes BLS verification;
    - its orders equal the orders that receipt implies;
    - `97 score` on `blocks/` is `COMPLETE`.

    The thresholds:
    - decode at 0: point ratio ≥ 1.10 and lower-95 ≥ 1.0;
    - decode at 28,672: lower-95 ≥ 1.0;
    - prefill at 28,672: lower-95 ≥ 0.95.

    Any threshold missed is `FAIL`. Any other defect is `NO_RESULT`.
14. **TTFT at depth 0 ≤ 0.5 s** in run 1's kit speed cell. The suite and its depth-0 cell must be valid; otherwise `NO_RESULT`.

`gate_mutations.py` runs 91 synthetic cases and 12 `--run2-preflight` cases, and all must match before scoring. Run-2 samples are built with 97's own `MEMINFO_KEYS` and `meminfo_line`:
- every candidate-9 case;
- one per new clause (runtime precondition cold, runtime release missing, zero files or wrong root, run-2 kill line, swap and low point, run-2 `NO_RESULT`, each speed threshold, TTFT, a wrong drand round, a receipt for another commit, orders that disagree with the receipt, a wrong arm profile, a dirty run-2 head, a probe mismatch scoring `NO_RESULT`);
- review round 1, reviewer 1: a resumed run 2 (same head twice) `PASS`, two heads, missing or wrong runtime trees in either run, a null run-2 low point, a raw run-2 sample below 8 GiB, short sample coverage, and a crashed start's kernel OOM line;
- review round 1, reviewer 2:
  - a missing lever line in run 1 or a B start, a lever line in an A start;
  - a missing block line in run 1 or run 2, a short run-2 KV capacity, a start naming another profile;
  - a frozen file edited or an untracked file added at scoring time;
  - run-2 kill lines with plan.json missing or with a wrong round (`FAIL`);
  - a duplicate agent-gate row, an invalid kit speed suite, run-1 samples ending early;
  - an A refusal (`NO_RESULT`) and a B refusal (`FAIL`) in run 2;
  - preflight refusals for a missing, wrong-round, wrong-commit or wrong-order plan;
- review round 2: raw run-2 swap growth in 97's sample format (`FAIL`).

The drand cases use a real published round, so the bundled verifier runs for real.

## If it passes (plan A3)

1. **One reviewed, test-first commit promotes it.**
   - The main profile `cuda-spark-128g-1m` takes candidate 10's engine, env (`QWEN38FN_PLE_WILLNEED`, `QWEN4EXP_DRAFT_VOCAB`, `QWEN38FN_HOST_RELEASE_ROOTS`, runtime-002 `PATH`) and argv (`draft_lm_head_quantization`) exactly. A contract test asserts that the rendered launch equals `tune-c10`'s, except for names.
   - `memory_model.overhead_gib` and `safety.minimum_start_gib` come from clause 10's peak, as candidate 9 did.
   - The README Flash-Next speed row comes from this kit bundle, and run 2's ratios and bounds are cited.
   - The text accuracy rows stay from the holdout.
2. **After merge,** `sudo -n scripts/52_engine_switch.sh qwen38fn` runs with `journalctl -k` captured over the switch window. Any NVRM or Xid line means switching back to the candidate 9 revision and reporting.
3. **The reverse direction must be safe as well** (reviewer 2, M5). Candidate 9's launch has no release root, so after a runtime-002 start its worker meets runtime-002's cached libraries: the mirror of the screen's a1 refusal. This affects a switch back to candidate 9, and the switch's own restore after a failed validation.
   - The promotion commit covers that path test-first, and its review checks it.
   - Until then, every start of candidate 9's launch after a runtime-002 start is preceded by an unprivileged `release_tree` of `~/.cache/qwen38-flash-next`. For this campaign, `run-c10.sh` does that at exit in both modes, and the operator checks it before restoring production.
4. **Before promotion, the owner gets the CJK decode diagnostic** (below) next to run 2's ratios.

If the verdict is `FAIL` or `NO_RESULT`, nothing changes: `qwen38fn` stays on candidate 9's launch.

## Reported, not gating

- run 1's kit speed numbers and MTP acceptance;
- the release lines' MemFree against arena + 2 GiB;
- MMMU's delta against 64;
- load time;
- run 2's per-block means;
- the soak's decode median against candidate 9's 30.87 tok/s;
- **decode on CJK output** (`cjk_decode.py`, reviewer 2 M6).
  - **Why:** L2's draft can propose only ids below 65,536. Reviewer 2 measured the share of ids at or above that with the model's tokenizer:

    | Text | Share of ids ≥ 65,536 |
    |---|---|
    | Chinese sample | 86.8% |
    | Japanese sample | 76.3% |
    | Speed fixture | 3.9% |
    | Python, JSON | 1.6–1.7% |

    So CJK decode may fall below candidate 9's, and run 2's English-prose ratios do not cover it.
  - **Method:** the same greedy prompts in English, Chinese and Japanese (thinking off, 384 tokens, 3 reps after a warm-up, block-aware with K = 4).
  - **Candidate 9:** measured on production before the freeze (2026-10-12, pid 124611, `cjk-decode-production-c9.json`):

    | Language | Decode | Ratio to English | Output ids ≥ 65,536 |
    |---|---|---|---|
    | English | 25.67 tok/s | 1 | 4% |
    | Chinese | 22.27 tok/s | 0.87 | 79% |
    | Japanese | 19.56 tok/s | 0.76 | 69% |

  - **Candidate 10:** measured by run 1 into `cjk-decode-start1.json`.
  - **The comparison is by each engine's ratio to its own English decode,** not by absolute tok/s. Candidate 9 was measured on live production (warm PLE, possible concurrent traffic), and candidate 10 at the end of run 1.
  - These are single-start diagnostic numbers, never headline ones.
- fidelity: `NO_RESULT`, because there is no BF16 teacher on this host.
  - L1 is I/O only. L2 acts on the draft side, and the target verifies every token with its untouched BF16 head.
  - L0 also changes target-path code: 0002's null-block mask, 0004's fused gather and 0005's lookup placement. Its losslessness rests on clause 5's three greedy probes, the direct-1M retrieval, the tool-call and agent gates (which include concurrent and long-prefix requests), and the screen's byte-identical probes in every runtime-002 arm.

## Freeze

The last commit that touches this file freezes:
- the profiles directory and both runtime build manifests;
- `patches/qwen38fn/`;
- `scripts/lib`;
- the kit, the serve wrapper, memwatch, membudget, the guard, the benches and the agent-gate adapter;
- the vision and tool-call eval sets and the fixtures;
- `97_matched_speed_blocks.py` and the drand verifier bundle;
- the contract test that defines the runtime tree digest;
- candidate 5's probe and the residency script;
- this driver, this scorer, the mutation suite, `cjk_decode.py` and the host file.

A change to any of these after a run starts is a new candidate in a new directory. Gate execution defects follow the AGENTS.md convergence rule.

## Review history

**Round 1** (freeze `e00663dc`; two persistent reviewers): no critical or high finding from either reviewer. Everything below was closed before the next freeze, each change test-first.

| Finding | Severity | Closed by |
|---|---|---|
| R1 M1: a resumed run 2 logs a second head and is `NO_RESULT` | medium | The gate takes the set of heads |
| R1 M2, R2 M1: runtime-002 tree identity not checked | medium | `runtime-tree` lines in both runs, against the manifest |
| R2 M1: lever, readiness and profile identity per start | medium | Lever and block lines, KV per run-2 start, profile in 97 score |
| R2 M2: scorer identity at scoring time | medium | Working tree equals the freeze under `FROZEN`, no untracked file; digest recorded |
| R2 M3: a plan defect masks run-2 safety | medium | Safety scan runs regardless; `--run2-preflight` checks the plan |
| R2 M4: 97 continues after an unsafe start | medium | `run_steps` stops the campaign |
| R2 M5: rollback to candidate 9 recreates a1 | medium | Plan A3 item 3, plus an operator release before restoring production |
| R2 M6: decode gain measured on English prose only | medium | Reported CJK diagnostic and English-prose label; owner informed before promotion |
| R1 L1: run 1's guard after mkdir | low | Guard before `observe/start1` |
| R1 L2, R2 L6: run-2 kill pattern case-sensitive | low | 97 `re.I`; the gate scans run-2 raw logs with the smoke patterns |
| R1 L3, R2 L1: run-2 low point fails open | low | Finite check, plus raw samples |
| R1 L4, R2 L4: sample coverage | low | Coverage and gap checks in both runs |
| R1 L5: run order not enforced | low | `--run2-preflight` |
| R1 L6: editing PREREG moves the freeze | low | This history is committed before the drand draw |
| R1 L7: SIGTERM orphans the engine | low | `exit_on_sigterm` |
| R1 L8: a crashed start's kill lines are not read | low | Raw scan of every start directory |
| R2 L2: duplicate agent-gate rows | low | Each test exactly once |
| R2 L3: clause 14 ignores validity | low | Valid suite and cell required |
| R2 L5: resume unscoreable | low | As R1 M1 |
| R2 L7: a B-arm refusal in run 2 is only `NO_RESULT` | low | `FAIL` (clause 12) |
| R2 L8: losslessness reasoning omits L0 | low | "Reported, not gating" names L0's target-path code and its evidence |
| R2 L9: hygiene | low | Untracked check in preflight; overrides recorded; fresh-start, cold-PLE label |

**Round 2** (freeze `b1a1d85e`, the gate's second candidate; same two reviewers):
- **Both reviewers found the same new HIGH,** introduced by round 1's coverage fix: the gate read run-2 samples requiring `SwapTotal`, which 97's sampler never writes. Every real run-2 start would have scored `NO_RESULT`. It failed closed; it could not produce a false PASS. The mutation fixtures had used the scorer's format, not the producer's.
- **Convergence: blocking findings went from 0 (round 1) to 1 (round 2).** That is not a strict decrease. The owner pause rule's "across two consecutive candidates" can be read as either one non-decrease or two in a row; this candidate is published under the second reading.
  - Candidate 3 changes only what round 2 named.
  - **If round 3 has any critical or high finding, the gate pauses and the open findings go to the owner.**

| Finding | Severity | Closed by |
|---|---|---|
| R1 H1 = R2 H1: run-2 samples unparseable (`SwapTotal`) | high | Run 2 needs `MemAvailable` and `SwapFree` only; 97 names its format, and the mutation suite builds run-2 samples with it (RED: every PASS case turned `NO_RESULT`) |
| R1 M1 = R2 M1: resume skips an unsafe start | medium | `existing_start` returns the record on disk; `run_steps` stops at it |
| R1 L1: SIGTERM windows (journal follower, second signal during cleanup) | low | Follower inside the `try`; `sigterm_deferred` around cleanup |
| R1 L2 = R2 L1: blocks mode hashes trees before the lock check | low | `flock -n` probe first |
| R2 L2: the M5 mitigation is manual | low | `release_runtime_root` at exit in both modes |
| R2 L3: CJK comparison by absolute tok/s | low | Compared by ratio to English |
