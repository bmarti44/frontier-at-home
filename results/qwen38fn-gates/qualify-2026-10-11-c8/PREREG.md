# Qwen3.8-Flash-Next candidate 8: images on at 1M, host page-cache release (preregistered)

This is an acceptance run of profile `qwen3.8-flash-next/cuda-spark-128g-1m`
on the Spark. This file was committed before any weights were loaded for this
candidate. The verdict comes from `gate_verdict.py` in this directory, which
writes `gate-summary.json`. The qualification kit writes its own bundle
(`manifest.json`, `summary.json`, `SUMMARY.md`, one directory per cell) into
this same directory.

## Why

Candidate 7 (`../qualify-2026-10-11-c7/RESULT.md`) ran images on at 1M under
the owner's 8 GiB floor. It failed on clause 1: the kernel logged 38 `NVRM ...
NV_ERR_NO_MEMORY` lines while vLLM allocated its 17.9 GiB KV arena.

- **The host passed admission anyway.** It read 115.45 GiB MemAvailable but
  carried 20.6 GiB of page cache, most likely qwen38-1m's GGUF left by the
  earlier switch run.
- **So the arena was short.** MemFree was only 10.2 GiB when it was
  allocated, and the driver failed the allocation until the kernel reclaimed
  the cache.
- **Admission is blind to this.** MemAvailable counts that cache as free.
  Every switch from `qwen38-1m` to `qwen38fn` leaves the host in this state.

The owner asked to fix it and qualify again (2026-10-11).

**CPU-only falsifier (before this freeze):**
- Reading qwen38-1m's GGUF and mmproj into page cache gave MemFree 91.08 GiB
  against MemAvailable 115.50 (Cached 23.11).
- One `posix_fadvise(DONTNEED)` pass over the 496 files in `~/models`, run as
  this user, gave MemFree 108.58 GiB (Cached 5.59).

## What changed (test-first, RED `fd22a0d0`)

- **`checkpoint_cache.release_tree(root)`** advises `DONTNEED` on every regular
  file under `root`.
  - It skips symlinks, files it cannot open, and files deleted during the
    walk. It raises any other error.
  - It refuses `/` and missing directories.
- **`qwen38fn_worker.HostCacheReleaseWorker`** subclasses the text-only worker,
  which stays unchanged on main. It adds two startup-only steps:
  - **Before the weights load:** it releases the model root, the parent of the
    model directory (`/home/bmarti44/models`), which holds every engine's
    model files.
  - **Before the KV arena** (`initialize_from_config`): it releases the model
    root again, reads MemFree, and logs one line: `host page cache released
    under ... MemFree ... KV arena ... + margin ...`. It raises (`refusing to
    allocate the KV arena`) unless MemFree ≥ arena + 2 GiB.
  - The arena is the single backing allocation every `KVCacheTensor` views
    (vLLM's `worker/utils.py` asserts one size), or `kv_cache_memory_bytes` if
    that is larger.
- **The profile** selects `--worker-cls qwen38fn_worker.HostCacheReleaseWorker`.
  Every other argv and env value is candidate 7 revision 2's: images 4, video
  0, `--mm-processor-cache-gb 0`, floor 8 GiB, admission 115 GiB
  (`test_launch_is_candidate_5_plus_image_input`).

## What runs (one fresh start, `run-phase2.sh start1`, `RuntimeMaxSec` 5 h)

**Adverse precondition.** After logging the code head, the driver reads
qwen38-1m's `Qwen3.8-27B-Q4_K_M.gguf` (16.55 GiB) and
`mmproj-Qwen3.8-27B-f16.gguf` (0.86 GiB) into page cache. A switch from
`qwen38-1m` leaves exactly this.

Then:
- `03_memory_guard.py --required-gib 114.5` must pass.
- The driver logs one `precondition <GiB> GiB <path>` line per file from
  `../qualify-2026-10-11-c7/pagecache_residency.py` (mmap + `mincore`, reads no
  data), immediately before launch.

Everything after launch is candidate 7's.

**Containment:**
- a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0` and
  `OOMPolicy=kill`;
- `93_profile_serve.sh` serves on `:8017` with its memwatch armed at 8 GiB;
- a 1 Hz `/proc/meminfo` and `memory.current` sampler;
- `journalctl -k` captured for the whole start.

**Readiness check:** attention block 3,200 and a KV capacity of at least
1,048,576 tokens at 262,144. Otherwise the driver logs `FAILED_ATTEMPT`.

Then, in order:
1. `probe-start1.json`: the three greedy text probes.
2. The kit, run as `--no-launch --cells speed,toolcall,vision,teacher,context --with-soak`:
   - speed (block-aware);
   - tool-call, 20 cases;
   - vision: MMMU-val-100, chat mode, 256-token answers;
   - teacher: expected SKIPPED;
   - context: the direct fill of 4 × 250,128 = 1,000,512 tokens at an 8 GiB
     floor, after vision;
   - soak: 30 minutes.
3. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model default --prefix-block-tokens 3200 --spec-block-drop`.

The holdout is not drawn again. It was spent once on this stack
(`../holdout-2026-10-11/`), and clause 5 carries its text accuracy over.

**Early stop.** If a clause-1 line appears during the run, the verdict is
already FAIL, and the operator may stop the engine to protect the host. Cells
that then fail or never run are consequences of the stop, not measurements.
`RESULT.md` says so.

## Gate formula (fixed)

`PASS` requires every clause below, and any failed clause gives `FAIL`.
Missing or malformed evidence gives `NO_RESULT` unless a clause has already
failed. So do:
- a `FAILED_ATTEMPT` line;
- a log that never saw the run (no `session start` in the server log, no
  `ARMED` in the memwatch log);
- a dirty `head` line;
- any `FROZEN` file that differs between the freeze commit and the run's
  head. The freeze commit is the last commit that touched this file.

1. **No kill lines.**
   - Kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`.
   - Server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`.
   - Memwatch log: `BREACH|EMERGENCY_STOP`.
2. **Host swap growth ≤ 0.25 GiB.**
3. **External low point ≥ 8 GiB** `MemAvailable`.
4. **Direct 1M:**
   - KV capacity ≥ 1,048,576 tokens at 262,144;
   - the context cell is `PASS`;
   - total tokens ≥ 1,000,000;
   - the manifest context cap is exactly 1,048,576.
5. **Text path unchanged:** the three greedy probe sha256 values equal
   candidate 5's (`../qualify-2026-10-11-c4/probe-start1.json`).
6. **Tool-call ≥ 19/20.**
7. **Agent gate:** all four tests pass.
8. **Soak:** the soak cell passes.
9. **Vision not significantly worse than the default:**
   - `n` = 100 and `error_count` = 0;
   - the one-sided pooled two-proportion z of `correct`/100 against 64/100 is
     ≥ −1.645, which means `correct` ≥ 53.

   Candidate 7's diagnostic scored 63/100.
10. **Switch admission holds:** peak use + 8 GiB ≤ 115 GiB. Peak use is the
    highest `MemAvailable` sample at or before the one launch line, minus the
    low point. No launch line, or no sample at or before it, gives
    `NO_RESULT`.
11. **The adverse precondition held.**
    - The driver logged `precondition` residency for exactly the two primed
      files, totalling ≥ 15 GiB at launch.
    - Otherwise the run did not test the fix: `NO_RESULT`.
12. **The fixed worker ran.**
    - The server log has exactly one release line, under
      `/home/bmarti44/models`, covering at least one file, for an arena of at
      least 17 GiB.
    - If it has `refusing to allocate the KV arena`, that is a `FAIL`: the
      release did not free enough.
    - Any other shape is `NO_RESULT`.

`gate_mutations.py` runs 42 synthetic cases, and all must match before
scoring:
- PASS: clean, an evidence-only later commit, and MMMU at 53 and at 63;
- FAIL: one case per failing clause, including `admissionlate` and `refused`;
- NO_RESULT: each missing or malformed input, including a missing launch
  line, missing or cold precondition lines, a missing release line, a release
  under the wrong root, a release covering zero files, and frozen files
  changed after the freeze.

On the c7 scorer, which has no clause 11 or 12, the five new cases score PASS.
That is the RED.

## If it passes

1. **One reviewed commit flips the status to `qualified`,** with this
   directory as evidence.
   - The README's Flash-Next rows take this bundle's speed and vision numbers.
     The text accuracy rows stay from the holdout.
   - Admission comes from this run. `memory_model.overhead_gib` becomes the
     measured peak (clause 10) minus `resident_weights_gib` and `extra_gib`,
     rounded up to 0.1 GiB. `safety.minimum_start_gib` becomes
     `ceil(peak + 8)`, which is ≤ 115. A contract test derives both from
     `observe/start1/meminfo.log`.
2. **The merged change makes `qwen38fn` the default:**
   `sudo -n scripts/52_engine_switch.sh qwen38fn`, left running, in a window
   agreed with the co-tenant.
   - `journalctl -k` is captured over the switch window. Any NVRM or Xid line
     there means switching back to `qwen38-1m` and reporting, not a default
     change.
   - README and AGENTS.md name the new default.
   - The boot-restore fallback stays `dsv4`.
   - Known risk, the reverse direction: the release lives only in this
     profile's worker. When `qwen38fn` stops, its own checkpoint and PLE
     pages stay cached in front of the next engine's load. The PR #13 switch
     run logged one NVRM line in exactly that case (`qwen38-1m` after
     `qwen38fn`, `../switch-2026-10-11/`).
     - So the fallback above, the switch's automatic rollback, and any later
       switch away from a `qwen38fn` default can log an NVRM line in the
       other engine.
     - The general fix is a switch-level release of `~/models` before every
       start. That is switch code and a separate change, and it is reported
       to the owner as the follow-up.
     - The promotion switch itself starts `qwen38fn` from a stopped host, so
       it is not exposed.
   - Known risk: boot restore waits 180 s for three stable samples at
     `minimum_start_gib`. A co-tenant holding memory at boot can make it fall
     back to `dsv4`. The operator recovers with
     `sudo scripts/52_engine_switch.sh qwen38fn`, and the README's default note
     says so.

This branch is not merged before a `PASS`. It sets the profile to
`estimated`, which would make the switch refuse the `qwen38fn` alias that is
qualified on main.

If the verdict is `FAIL` or `NO_RESULT`, nothing changes:
- `qwen38-1m` stays the default;
- the text-only `qwen38fn` on main stays as it is;
- the result goes to the owner.

## Reported, not gating

- **Speed:** block-aware decode, prefill and TTFT against the `qwen38-1m` row.
- **MMMU score**, with its delta against 64.
- **The release line's MemFree** against the arena + 2 GiB.
- **Fidelity:** `NO_RESULT`; there is no BF16 teacher on this host.
- **Kit verdict:** reported. It is `FAIL` whenever any declared target misses.

## Freeze and randomness

The last commit that touches this file freezes everything that shapes the
measurement:
- the profile and `scripts/lib`, which includes the new worker and
  `checkpoint_cache.py`;
- the kit, the wrapper and the agent-gate adapter;
- the vision bench and its eval set, and the tool-call cases;
- the residency script;
- this driver and this scorer.

After the start runs, any change to these files is a new candidate in a new
directory.

There is one arm and no fixture choice. The kit's fixtures are fixed by its
scripts (speed seed 42, the pinned MMMU-val-100, deterministic needles, the
three probes), so no public randomness is drawn. This follows candidates 5
and 7.

## Review history

- **Round 1 on `61689dc2`:** no critical or high findings; one MEDIUM and
  two LOW.
  - MEDIUM: the reverse switch direction is still exposed. Recorded as a
    known risk under "If it passes", with the switch-level follow-up.
  - LOW: `release_tree` failed on a file deleted mid-walk. It now skips
    `FileNotFoundError` (regression test
    `test_a_file_deleted_during_the_walk_is_skipped`).
  - LOW: clause 12 accepted a zero-file release. It now requires at least
    one file (mutation `zerofiles`).
  - RED in the commit after `61689dc2`. This file's commit is the new freeze.
