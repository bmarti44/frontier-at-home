# Qwen3.8-Flash-Next Phase-1 smoke: preregistered falsifier (DIAGNOSTIC)

Committed before the first load. This run is a bounded falsifier. Its numbers
are **diagnostic only**: never headline, README, or acceptance evidence.

## What runs

- Profile: `qwen3.8-flash-next/cuda-spark-128g-smoke`. This is the production
  serving path with the KV cache cut down to 2 x 131,072 slots, to leave room
  for the co-tenant workload.
- Launch: `scripts/94_qualify_profile.py --profile qwen3.8-flash-next/cuda-spark-128g-smoke --port 8017 --cells speed,toolcall`.
  The kit serves through `93_profile_serve.sh` (digest check, residency lock,
  release gate, memwatch at a 15 GiB floor).
- External observers, started before load and stopped after unload:
  - a 1 Hz `/proc/meminfo` sampler (MemAvailable, SwapFree, Cached);
  - `journalctl -k -f`;
  - `nvidia-smi` at the end.
- Extra probes on the same live server:
  - greedy byte-identity: 3 fixed prompts at temperature 0, compared across two
    fresh starts;
  - SSE granularity with MTP on: count of content deltas carrying more than one
    token.
- PLE arm: vLLM PR #58439 `checkpoint_mapped` only. The pread "staged gather"
  lane runs only if this arm fails the swap or speed clauses below. At most 2
  arms.

## Fixed NO_GO formula

NO_GO if any of the following holds:

1. any OOM kill, cgroup kill, memwatch kill, `NVRM`, `Xid`, or CUDA
   out-of-memory line in the server log or kernel log;
2. host swap growth: `SwapTotal - SwapFree` rises by more than 0.25 GiB between
   pre-load and the post-run low point (on every PLE arm tried);
3. tool-call cell below 17/20;
4. decode @0 below 17.46 tok/s **and** prefill @28,672 below 698.7 tok/s, so it
   is no better than the current default `qwen38-1m` on either axis.

Anything else is GO to Phase 2. A server that never becomes ready, or a cell
that cannot run, is `NO_RESULT`: the cause is recorded, and one bounded fix is
allowed. Short output, a timeout, or a missing process identity counts as a
failed attempt.

## Recorded alongside (no gate)

- load time;
- resident footprint (MemAvailable before load vs. at idle after load);
- MemAvailable low point;
- engine-reported KV pool tokens;
- MTP acceptance;
- greedy identity;
- SSE granularity;
- the speed cell's `suite_valid`.

If streaming is block-granular with MTP on, record it, and propose a block-aware
shared speed bench to the owner as a new file. `30_bench_speed.py` stays
byte-frozen.
