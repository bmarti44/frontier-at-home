# Candidate 7 result: FAIL (NVRM lines at KV allocation)

Candidate 7, revision 2 (freeze `94ae2f6e`), ran images on at 1M under the
owner's 8 GiB floor. The frozen scorer (`gate_verdict.py`, mutations 36/36)
returns **FAIL** in `gate-summary.json`.

- **Clause 1 decided the result:** the kernel logged 38 `NVRM ...
  NV_ERR_NO_MEMORY` lines at 17:09:04–05Z, while vLLM allocated its KV arena.
- **Clauses 4, 6 and 7 also show FAIL**, and the soak and vision files are
  missing, but these are not measurements. Each one follows from the operator
  stop below, which came after clause 1 had already failed.

The owner's condition for a default change was "the measured low stays above
8 GiB with no NVRM/OOM lines". It is not met, so nothing changes:
- `qwen38-1m` stays the default;
- the text-only `qwen38fn` alias on main stays as it is;
- this branch's `estimated` profile is not merged.

## What the run measured before the stop

| Item | Value |
|---|---|
| Start guard | `03_memory_guard` pass at 115.42 GiB MemAvailable (needed 114.5) |
| Host at launch (16:58:18Z) | MemAvailable 115.45 GiB, **MemFree 94.37 GiB, Cached 20.59 GiB** |
| Launch to ready | 670 s; vLLM: 76.49 GiB weights, attention block 3,200 |
| KV capacity | 1,079,117 tokens at 262,144 per request |
| NVRM lines | 38, all at 17:09:04–05Z, during the 17.9 GiB KV-arena allocation |
| Xid, OOM, memwatch BREACH | none |
| Idle MemAvailable at ready | 12.92 GiB |
| Low point | MemAvailable 11.02 GiB (17:09:55Z), MemFree 0.90 GiB (17:09:54Z). This is the idle low; no cell had run yet. |
| Peak use (clause 10 rule) | 104.43 GiB, from the launch baseline. This is not a fill peak. |
| Swap growth | 0.0 GiB |
| Text probe vs candidate 5 | 3/3 greedy sha256 identical |

## Operator actions (outside the driver)

1. **17:10:06Z: stopped the engine** with `93_profile_serve.sh ... stop`.
   - Clause 1 had already failed.
   - The remaining cells would have pushed the host to about 8–9 GiB
     available with about 1 GiB MemFree, where more allocation failures are
     likely.
   - AGENTS.md: never trade whole-system stability for benchmark progress.
2. **About 17:21Z: terminated the kit's soak client** (`35_soak.py`, SIGTERM).
   - With the server down, its RSS had reached 8.9 GiB and was growing about
     13 MiB/s.
   - That is a separate client defect: it accumulates memory when the server
     is unreachable. `35_soak.py` is frozen here, so the defect is recorded,
     not fixed.

The driver then ran the agent gate against the stopped server (four `URLError`
rows). Its final `serve stop` exited nonzero because the engine was already
down. `set -e` therefore took it into the cleanup trap before the
`check_no_survivors` and `postrun` lines. The cleanup trap's scope check
logged no `FAILED_ATTEMPT`.

**End state:**
- no GPU compute processes;
- the scope is gone;
- MemAvailable 115.51 GiB;
- SwapFree unchanged at 15.79 GiB.

## Cause: page cache from the previous engine, invisible to admission

The table shows each 1M load's first sample and its KV-allocation step. The
step is the 1 Hz sample where MemFree fell by more than 6 GiB, taken from each
run's `meminfo.log`.

| Load | First: Avail / Free / Cached (GiB) | KV step: MemFree before → after | Cached at KV step | NVRM |
|---|---|---|---|---|
| c5 start1 (`../qualify-2026-10-11-c4`) | 115.22 / 110.90 / 5.35 | 28.16 → 10.28 | 5.42 | 0 |
| c5 start2 | 115.19 / 108.88 / 7.30 | 28.23 → 21.71 | 5.58 | 0 |
| diag run 1 (`../diag-vision-2026-10-11/run`) | 115.56 / 103.60 / 12.54 | 25.21 → 15.37 | 6.05 | 0 |
| diag run 2 (`.../run2-nocache-fill`) | 115.15 / 110.15 / 6.03 | 25.19 → 15.90 | 6.15 | 0 |
| **c7 start1** | **115.45 / 94.37 / 20.59** | **10.20 → 1.38** | **19.48 → 18.78** | **38** |

Every load passed the same MemAvailable admission. c7, though, started with
about 15 GiB more page cache.
- **What the evictor misses:** the profile's checkpoint-cache evictor drops
  only checkpoint pages. Cached stayed near 21 GiB all through c7's load,
  where diag run 1's 12.5 GiB fell to 5.4 GiB.
- **What failed:** the KV arena needed more free pages than MemFree held. The
  NVIDIA driver failed those allocations until the kernel reclaimed about
  8 GiB of the cache, and then startup completed.

**The cache was most likely qwen38-1m's model file.**
- The switch run (`../switch-2026-10-11`, ending 13:26Z) loaded `qwen38-1m`,
  which maps `Qwen3.8-27B-Q4_K_M.gguf` (16.55 GiB) and its mmproj (0.86 GiB).
- After this run, `pagecache_residency.py` still finds 4.13 GiB of that GGUF
  and 0.46 GiB of the mmproj resident. Every other model directory is at
  0.08 GiB or less.
- All four NVRM-free loads ran before that switch run.
- The same mechanism produced the one NVRM line in the switch run's
  `qwen38-1m` load: MemFree was 0.74–2.7 GiB, behind about 21 GiB of page
  cache.

What this means for promotion:
- **It is not about image memory.** The 3.1–3.3 GiB image cost and the idle
  low (11.02 vs diag run 2's 12.06 GiB) behaved as measured.
- **It is about a realistic default switch.** A switch from `qwen38-1m` to
  `qwen38fn` produces exactly this host state. Neither the run's start guard
  nor the switch's `minimum_start_gib` can see it, because both read
  MemAvailable.

## Bounded alternatives (not run; owner decision)

1. **A new candidate that releases or waits on clean page cache before the
   KV allocation.** Two ways:
   - admission on MemFree as well as MemAvailable;
   - a launcher step that drops the previous profile's model files from page
     cache (`POSIX_FADV_DONTNEED` on files the operator can read).

   This is a production-path switch and launcher change, so it needs
   test-first RED, a new freeze, and a fresh run that starts from the
   post-`qwen38-1m` state.
2. **Re-running this frozen configuration on a cache-clean host** would only
   retry until host conditions are favourable. It would not show that the
   default survives a normal switch. It is not justified without (1).

## Files

- `gate-summary.json`: frozen scorer output.
- Kit bundle: `manifest.json`, `summary.json`, `SUMMARY.md`, and one directory
  per cell. Every cell's FAIL is a consequence of the stop.
- `probe-start1.json`: the text probe.
- `agent-gate.json`: connection-refused rows, run after the stop.
- `observe/start1/`: `driver.log`, `meminfo.log` (1 Hz), `kernel.log`,
  `server.log`, `memwatch.log`, `nvidia-smi-end.txt`.
- `pagecache_residency.py`: the mmap + `mincore` residency check behind the
  attribution above. It reads no data.
