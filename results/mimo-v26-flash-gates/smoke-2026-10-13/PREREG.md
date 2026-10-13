# MiMo-V2.6-Flash M1 smoke and fidelity: preregistered rule (DIAGNOSTIC)

This file is committed before the first MiMo load. The window is a bounded
smoke and fidelity measurement. Its numbers are **diagnostic only**. They are
never headline, README or acceptance evidence. The plan's M3 qualification
produces those.

## Why

The owner chose MiMo-V2.6-Flash (MOPD) as a switchable alias on 2026-10-12.
They asked to test the existing post-fix pack first and convert our own only if
it does not fit. This window decides between those two branches. It also
measures the pack's fidelity against the release weights. That loss is
reported, not waived.

## What runs

Production `qwen38fn` is stopped with `sudo -n scripts/52_engine_switch.sh
stop` and restored with `... qwen38fn` at the end. The owner pre-approved the
window on 2026-10-12.

### Start 1

- **Profile:** `mimo-v2.6-flash/cuda-spark-128g-262k`, estimated. This is the
  production launch: the TabbyAPI shim, one 262,144-token slot, Q8 K/V, the
  DFlash drafter and vision on.
- **Launch:** through `scripts/93_profile_serve.sh` on :8019, with
  `host-worktree.json`. That file differs from `configs/hosts/spark-aba1.json`
  only in `paths.repo`, which points at this worktree, so `PYTHONPATH` reaches
  the worktree's `tabby_launch`.
- **Scope:** a user scope with `MemorySwapMax=0`, `OOMPolicy=kill`,
  `MemoryHigh=16G`, `MemoryMax=48G` and `RuntimeMaxSec=7200`.
  - The low `MemoryHigh` keeps the pack's page cache from eating MemFree while
    the weights load. GB10 CUDA allocations come from MemFree and are not
    charged to the cgroup. From launch until ready, the driver also runs
    `release_tree` over the pack and the drafter every 5 s.
  - `memory.peak` and `memory.stat` (anon versus file) are recorded, to size
    the profile's containment.
- **Pre-load gate:**
  - `release_tree` (`posix_fadvise DONTNEED`) over the Flash-Next model and
    runtimes, the MiMo pack, the drafter and the official tree;
  - then `MemFree >= 108 GiB` over 3 samples.
  - If the gate fails, the run is `NO_RESULT`. Production is restored and the
    owner is asked about `drop_caches`.
- **Preconditions:** the driver refuses to start while the download scope is
  active or any `*.incomplete` file remains. It also refuses when the official
  tree does not match the manifest's `mopd-release` file list and sizes.
- **Observers**, started before the load:
  - 1 Hz `/proc/meminfo` (MemFree, MemAvailable, Cached, AnonPages, Swap);
  - `journalctl -k -f`, kept as two logs: `kernel-smoke.log` from the driver
    start to the "smoke window closed" line, and `kernel-tail.log` for the
    fidelity run and the restore;
  - memwatch at the profile's 8 GiB floor (via 93).
- **On the live server:** `probe.py`, then the kit cells `speed,toolcall`
  with `--no-launch --reasoning-effort ''`.
  - `probe.py` runs 3 greedy prompts at temperature 0, measures SSE tokens per
    event, and reads `/props`, `/v1/models` and the 404 on `/v1/model/list`.

### Start 2

A fresh start of the same launch. It runs `probe.py` only, to record greedy
identity across restarts.

### Fidelity

exllamav3 `eval/model_diff.py` from the pinned runtime source runs through
`fidelity/run_model_diff.py`, with the offline wikitext shim.

- **Models:** A = the served pack; B = the official MOPD release, which is
  the reference. This follows the tool's convention: `-cq` simulates on A,
  and the confidence buckets are B's.
- **Cache:** `-cq 8,8` simulates the served Q8 K/V cache on A, so the
  numbers cover the served configuration: weights plus cache.
- **Rows:** 100 × 2,048 tokens of wikitext-2-raw-v1 test (Salesforce/wikitext
  @ `b08601e0`, parquet sha256 `5f1bea06…6d91`, row export sha256
  `5c8877ee…3e4c`).
- **Scope:** `MemoryHigh=16G`, `MemoryMax=64G`, `MemorySwapMax=0`,
  `OOMPolicy=kill`, `RuntimeMaxSec=14400`. It starts after start 2 is stopped,
  and only after the same MemFree gate.
- **Watchdog:** a driver-side watchdog kills the scope if MemAvailable drops
  below 10 GiB. Fidelity failures are logged as `FIDELITY_FAILED` and never
  as `FAILED_ATTEMPT`.
- **Recorded:** the raw log, PPL A and B, ΔNLL = ln(PPL_A / PPL_B) (pack
  minus reference), KLD both ways, and top-K agreement.

## Fixed decision rule

**Peak definition:** the MemAvailable sample at or before the start-1 launch
line, minus the lowest MemAvailable sample from that line to the empty-scope
line. `load_peak` uses the window from launch to ready; `run_peak` uses the
whole start-1 window.

The pack is `FITS` when all of these hold:

1. there is no OOM, cgroup, memwatch, `NVRM`, `Xid` or CUDA out-of-memory line
   in the smoke window:
   - `kernel-smoke.log`: `out of memory|oom-kill|oom_reaper|NVRM|Xid`,
     case-insensitive;
   - server logs: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch log: `BREACH|EMERGENCY_STOP|FAIL_CLOSED`;
2. swap growth is at most 0.25 GiB (max of `SwapTotal − SwapFree` minus the
   first sample);
3. `load_peak <= 106` GiB and `run_peak <= 106` GiB;
4. the tool-call cell passes at least 19 of 20;
5. both starts become ready, and the kit's speed cell has at least one
   eligible rep at ctx 0.

The verdict is:

- **`FITS`:** the candidate pack is MikeRoz 2.52 bpw and M2 is skipped.
- **`DOES_NOT_FIT`:** clause 1, 2 or 3 fails. M2 converts our own pack with
  target weights `W* = 104 − (run_peak − W)` GiB, where `W` is the served
  pack's `.safetensors` bytes from `weights/mimo-v2.6-flash/manifest.json`,
  divided by 2^30.
- **`NO_RESULT`:** missing evidence, a server that never becomes ready, or a
  `FAILED_ATTEMPT` in the driver log. Start 1 also needs meminfo samples
  covering launch through the empty-scope line at most 3 s apart. One bounded
  fix is allowed, and a rerun uses a fresh evidence directory.
- **Clause 4 failing alone** is reported as `FITS_TOOLCALL_LOW`. The pack fits
  memory and the parser needs work, so M2 is not triggered by it.

Fidelity has its own status. It is `MEASURED` when the driver logged
`fidelity rc=0` and no `FIDELITY_FAILED` line, and `model_diff.log` holds
exactly one of each final metric with every value finite. Otherwise it is
`NO_RESULT`. Kernel lines in `kernel-tail.log` are recorded. Fidelity status never changes the pack verdict. A fidelity
`NO_RESULT` gets one bounded rerun in a later window.

## Restore

The restore runs on every exit path. It waits up to 180 s for MemAvailable
≥ 110 GiB, then runs the switch, retrying once after 30 s. A `RESTORE_FAILED`
line makes the driver exit nonzero. That is a stop-and-report event for the
owner.

## Recorded, not gating

- load time;
- idle MemAvailable;
- MemFree at the KV allocation;
- cgroup `memory.peak`, and anon/file at idle. Before any qualification run,
  the production profile's `memory_high` and `memory_max` are set from the
  measured anon plus margin, so the switch unit bounds page cache the same
  way;
- `/props` (slots, `n_ctx`, vision);
- route allowlist behavior;
- block-aware decode at 0 and 28,672, and the prefill lower bound;
- SSE tokens per event;
- greedy identity across the two starts;
- the fidelity metrics.

Decode is `(N − 8) / (tN − t1)`. DFlash drafts `block_size − 1 = 7` tokens,
so one event carries at most 8.

## Scorer

`verdict.py` in this directory implements exactly this rule and writes
`summary.json`. `verdict_mutations.py` shows that malformed evidence is
rejected.
