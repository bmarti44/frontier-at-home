# MiMo-V2.6-Flash M1 smoke rerun: result (DIAGNOSTIC)

**Pack verdict: `DOES_NOT_FIT` (clause 1).** The kernel logged an NVRM
allocation failure while start 1 loaded its weights. Every other clause
passed. Under the owner's standing instruction (stop on any kill line), the
branch is paused for the owner. It does not proceed to M2 automatically.

These numbers are diagnostic only. They are never headline, README or
acceptance evidence. The scorer, `../smoke-2026-10-13/verdict.py`, ran
unchanged and wrote `summary.json`. Its `fidelity_status: NO_RESULT` is by
construction (PREREG); the fidelity of record is attempt 1's `MEASURED`.

## The kill line

```
2026-10-13T01:28:58-04:00 spark-aba1 kernel: NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359
```

That is 05:28:58Z, 53 s into start 1's load (launched 05:28:05Z, ready
05:29:01Z). Around it, from `observe/meminfo.log`:

| time (UTC) | MemFree | MemAvailable | Cached |
|---|---|---|---|
| 05:28:52 | 16.51 | 28.01 | 12.05 |
| 05:28:53 | 22.36 | 25.53 | 3.73 |
| 05:28:57 | 9.53 | 18.82 | 9.85 |
| **05:28:58** | **4.20** | 15.65 | 12.02 |
| 05:28:59 | 1.20 | 12.89 | 12.79 |
| 05:29:00 | 11.67 | 13.01 | 1.88 |

All values are GiB.

- The pack streams in at about 2 GB/s. Between the driver's 5 s
  `release_tree` passes, its page cache climbs back to 12–14 GiB. Near the
  end of the load, that cache took MemFree down to 1.2 GiB while
  MemAvailable was still 13–16 GiB.
- GB10 CUDA allocations come from MemFree, so the driver refused one. The
  server logged no error and became ready. The steady-state footprint is not
  the problem: MemAvailable's low point was 10.83 GiB and the run peak
  104.76 GiB.
- Attempt 1 showed the same crowding (MemFree 1.89 GiB during the load)
  without a kernel line.

The rule's M2 target is `W* = 104 − (run_peak − W)` = **93.53 GiB**. That is
0.77 GiB below the served pack's 94.30 GiB. A smaller pack of that size does
not address page cache filling MemFree during the load.

## Other clauses

| clause | measured | status |
|---|---|---|
| 2. swap growth | 0.0001 GiB | pass |
| 3. load peak / run peak (≤ 106 GiB) | 102.70 / 104.76 GiB | pass |
| 4. tool call | 20 / 20 | pass |
| 5. both ready; eligible rep at ctx 0 | 61 s / 64 s; 2 valid reps | pass |

## Recorded, not gating

- **Decode, block-aware** (`(N − 8)/(tN − t1)`, two reps each, 382 client
  tokens):

  | context | median | reps | TTFT |
  |---|---|---|---|
  | 0 | **21.93 tok/s** | 23.94, 19.92 | 0.65 s |
  | 28,672-token fixture (34,482 prompt tokens) | **36.26 tok/s** | 35.76, 36.77 | 45.3 s |

  Prefill at 28,672 was 760 tok/s. SSE events averaged 2.2 tokens at ctx 0
  and 4.2 at ctx 28,672.
- **Streaming is incremental now:** the probe streams took 1.16–3.79 s,
  against 0.0001 s in attempt 1 before 087b5814.
- **Greedy outputs are identical** across both starts and to attempt 1:
  `c933d4cb…`, `afe5a855…`, `68da9e93…`.
- **Idle MemAvailable:** 13.0 GiB on start 1 and 12.9 GiB on start 2. cgroup
  `memory.peak` was 17.18 GB; anon at idle was 3.1 GB.
- **Window:** `qwen38fn` stopped at 05:27:56Z and was restored at 05:46:29Z
  (rc 0, first attempt), about 19 min. `kernel-tail.log` holds only the
  switch's `drop_caches`.

## Open review lows (focused review of bc305dbf / 087b5814: no high or critical)

1. The chat terminal split copies `logprobs` and `timings` to both halves,
   and `eos_reason` to the first, so a streaming-logprobs client without
   `include_usage` would count the last chunk's logprobs twice. No repository
   harness hits this.
2. The `split_usage_event` docstring does not mention that `/v1/completions`
   usage chunks split the same way.
