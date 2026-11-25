# MiMo-V2.6-Flash M3 preflight r2 (DIAGNOSTIC)

This file is committed before r2's load. Its numbers are **diagnostic only**:
never headline, README or acceptance evidence.

## Why

The first preflight (`../preflight-2026-10-13/RESULT.md`) ran every M3 path
successfully on TabbyAPI. Its peak use was 106.09 GiB, over the 106 GiB
limit: MemAvailable went from 115.35 GiB to a 9.25 GiB low at the start of
the 262K fill. Over the run, the engine scope's anon grew from 3.09 to
4.56 GB, mostly from the sliding-window checkpoint cache (1024 MB cap) and
the image cache.

**Owner decision (2026-10-13):** halve the checkpoint cache to 512 MB, which
is about 10 checkpoints of 49.25 MiB (agent turn reuse needs one to three).
Re-run the preflight to measure memory again. M3 starts only if the peak is
≤ 106 GiB.

## What changed since the first preflight

All of these were committed before this PREREG:

- **`50385027` (RED), `4ee2acf5`:** the profile passes
  `--sysmem-recurrent-cache 512` instead of 1024. This is a server-path
  change, so it gets this rerun.
- **`1ef0be30` (RED), `ff4d6c0a`:** the kit passes the profile's
  `vision_max_tokens`, 4,096, to the MMMU cell. The first preflight's 256
  cut 4 of 20 chat-mode answers. This rerun is the first to run vision at
  4,096.
- **`run-preflight.sh`** is `../preflight-2026-10-13/run-preflight.sh` (one
  focused review, its high closed) with only the evidence path changed.

Everything else is the first preflight's:
- containment: scope MemoryHigh 8G / MemoryMax 16G, a 1 s load-time release,
  and MemFree ≥ 108 GiB;
- observers;
- order: probe, agent gate (256 slack), kit vision (first 20) and context
  (direct 250,128-token fill, with the 4,096-budget fallback only if the
  full fill does not run), then GSM8K dev cut at 1,200 s.

## Fixed decision rule

The rule is the first preflight's PREREG, unchanged:

- **Kill line or failed restore:** stop and report to the owner.
- **Memory:** peak use (MemAvailable at launch minus the low) above
  106 GiB, or a low below 8 GiB: M3 does not start. Stop and report to the
  owner with the numbers.
- **Harness mismatch:** close it in the shared tooling, RED-first, before the
  M3 PREREG. The preflight is rerun only if the fix changes the server path.
- **Path not exercised:** that path is `NO_RESULT`, and M3 waits until it has
  run.
- **Model result:** reported. M3 waits until the cause is understood.
- **Otherwise:** write the M3 c1 PREREG with this configuration.

## Recorded, not gating

The first preflight's list, plus:
- vision at 4,096: `finish_reason` counts and how long the answers ran;
- scope `anon` against the first preflight (checkpoint-cache growth at
  512 MB).
