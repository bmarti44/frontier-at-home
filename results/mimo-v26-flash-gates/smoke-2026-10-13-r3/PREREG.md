# MiMo-V2.6-Flash M1 smoke r3 (DIAGNOSTIC)

This file is committed before r3's first load. The numbers are **diagnostic
only**: never headline, README or acceptance evidence.

## Why

`smoke-2026-10-13-r2` scored `DOES_NOT_FIT` on clause 1. The kernel logged
NVRM `NV_ERR_NO_MEMORY` 53 s into start 1's weight load. The pack's page cache
refilled to 12–14 GiB between the driver's 5 s release passes and took
MemFree to 1.2 GiB, while MemAvailable was still 13–16 GiB. The steady state
was within the rule: run peak 104.76 GiB, MemAvailable low 10.83 GiB.

**Owner decision (2026-10-13):** fix the load path and rerun. The cause is
page cache crowding MemFree during the load, not the pack's size, so the rule's
M2 target (W* 93.53 GiB, 0.77 GiB smaller) does not apply to it. This
overrides that mapping for this cause only.

## What changed since r2

All commits were made before this PREREG:

- **d77944f2 (RED), 91640a51:** the profile's switch unit now uses
  `MemoryHigh 8G` and `MemoryMax 16G`, down from 100G / 104G. exllamav3's
  CUDA memory is not charged to the unit, which held 3.1–3.8 GB anon.
- **57a8f265 (RED), 481577df:** the focused review's two lows are closed.
  Split terminal fields are kept once, and the completions usage docstring is
  fixed.
- **`run-smoke.sh`** is r2's driver with two changes:
  - the engine scope uses `MemoryHigh=8G` and `MemoryMax=16G`, mirroring the
    profile, instead of 16G / 48G;
  - the load-time `release_tree` runs every 1 s instead of every 5 s.

Everything else is r2's: the probe and host file (attempt 1's), the
preconditions, the MemFree ≥ 108 GiB gate, the observers, both starts, kit
`speed,toolcall` on start 1, and the restore. Fidelity is not rerun; its
status of record is attempt 1's `MEASURED`.

## Rule and scorer

The rule is attempt 1's PREREG ("Fixed decision rule"), unchanged. The scorer
is `../smoke-2026-10-13/verdict.py`, run unchanged against this directory.
Its `fidelity_status: NO_RESULT` is by construction and unused.

- **`FITS`:** MikeRoz 2.52 bpw is the candidate pack. M2 is skipped and M3
  qualification follows.
- **Any other verdict** (`DOES_NOT_FIT`, `NO_RESULT` or `FITS_TOOLCALL_LOW`),
  or any kill line or failed restore: stop and report to the owner with the
  exact evidence. There is no further autonomous rerun.

## Recorded, not gating

- MemFree's low point during the load (r2: 1.20 GiB; attempt 1: 1.89 GiB);
- the load time, against r2's 61 s and 64 s, in case the 8G bound throttles
  reads;
- cgroup `memory.peak` and anon/file at ready;
- block-aware decode at 0 and 28,672, and prefill;
- the probe's `stream_seconds`;
- greedy identity with attempts 1 and r2.
