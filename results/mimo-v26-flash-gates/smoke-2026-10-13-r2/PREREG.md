# MiMo-V2.6-Flash M1 smoke rerun (DIAGNOSTIC)

This file is committed before the rerun's first load. The numbers are
**diagnostic only**: never headline, README or acceptance evidence.

## Why

Attempt 1 (`../smoke-2026-10-13`, RESULT.md) scored `NO_RESULT`. The kit's
speed cell could not run, so clause 5 had no eligible rep. Every memory and
safety clause passed. Fidelity was `MEASURED`.

The run found two defects, and both stop the speed cell from measuring a
TabbyAPI stream. Under that PREREG's `NO_RESULT` clause, they are fixed
together as the one bounded fix:

- bc305dbf: the kit takes the model from `--model-name`, not Python's `-m`.
- 087b5814: the shim forwards TabbyAPI's CRLF-framed SSE events as they are
  produced, and splits the `include_usage` chunk into a terminal event and a
  usage event.

## What runs

The run is identical to attempt 1's smoke part. `run-smoke.sh` here is
`../smoke-2026-10-13/run-smoke.sh` with three changes, all visible in a diff:

- the evidence directory is this one;
- the probe and host file are attempt 1's, unchanged;
- the fidelity step is removed.

Otherwise it is the same:
- the profile and launch;
- the scope limits;
- the MemFree gate, preconditions and observers;
- start 1 (probes plus kit `speed,toolcall`) and start 2 (probes);
- the restore.

## Rule and scorer

The rule is attempt 1's PREREG ("Fixed decision rule"), unchanged: `FITS`,
`DOES_NOT_FIT` with `W* = 104 − (run_peak − W)`, `NO_RESULT`, or
`FITS_TOOLCALL_LOW`. The scorer is `../smoke-2026-10-13/verdict.py`, run
unchanged against this directory.

- **Fidelity:** this directory has none, so its `summary.json` reads
  `fidelity_status: NO_RESULT` by construction. That field is not used. The
  fidelity status of record is attempt 1's `MEASURED`.
- **Another `NO_RESULT`:** the bounded fix is spent. The branch stops, and the
  exact failure is reported to the owner.

## Recorded, not gating

- the probe's `stream_seconds`, a direct check that the stream now arrives
  incrementally;
- block-aware decode at 0 and 28,672 tokens, and prefill;
- greedy identity with attempt 1's probes;
- peaks against attempt 1's (103.61 / 104.50 GiB).
