# MiMo-V2.6-Flash M1 smoke r3: result (DIAGNOSTIC)

**Pack verdict: `FITS`.** MikeRoz MOPD 2.52 bpw (94.30 GiB of weights) is the
candidate pack. M2 (our own conversion) is skipped, and M3 qualification
follows.

These numbers are diagnostic only. They are never headline, README or
acceptance evidence. The scorer, `../smoke-2026-10-13/verdict.py`, ran
unchanged and wrote `summary.json`. Its `fidelity_status: NO_RESULT` is by
construction; the fidelity of record is attempt 1's `MEASURED` (ΔNLL 0.0503,
KLD 0.0836, top-1 −0.94 pp against the release weights).

## Rule clauses

| clause | measured | status |
|---|---|---|
| 1. kill lines | none (`kernel-tail.log`: only the switch's `drop_caches`) | pass |
| 2. swap growth | 0.00 GiB | pass |
| 3. load peak / run peak (≤ 106 GiB) | 103.17 / **105.59** GiB | pass |
| 4. tool call | 20 / 20 | pass |
| 5. both ready; eligible rep at ctx 0 | 61 s / 59 s; 2 valid reps | pass |

- The launch reading was MemAvailable 115.58 GiB, with 1,083 meminfo samples.
- **MemAvailable's low point was 9.99 GiB, at 11:51:12Z during the tool-call
  cell**, when anon rose to 4.35 GB. r2's low came at the same point, 10.83
  GiB. The run peak is 0.41 GiB under the 106 GiB limit, so M3's 262K
  context fill will show whether long prompts push it higher.

## The load-path fix worked

The owner decided on 2026-10-13 to bound page cache instead of converting
(91640a51; this driver):

| | r2 (MemoryHigh 16G, 5 s release) | r3 (MemoryHigh 8G, 1 s release) |
|---|---|---|
| MemFree low during load | 1.20 GiB, then an NVRM line | **6.38 GiB**, no kernel line |
| page cache peak during load | 12–14 GiB | 6.0 GiB |
| cgroup `memory.peak` | 17.18 GB | 7.33 GB |
| load time | 61 s / 64 s | 61 s / 59 s |

Anon at ready was 3.08 GB and rose to 3.83 GB after the kit cells. The 8G
`MemoryHigh` did not slow the load.

## Recorded, not gating

- **Decode, block-aware** (two reps each):

  | context | median | reps | TTFT |
  |---|---|---|---|
  | 0 | **21.85 tok/s** | 23.84, 19.87 | 0.64 s |
  | 28,672-token fixture (34,482 prompt tokens) | **35.71 tok/s** | 35.58, 35.85 | 45.4 s |

  Prefill at 28,672 was 758–760 tok/s. The ctx-0 rate is below the plan's
  research estimate (31–35 tok/s).
- **Streaming is incremental:** the probe streams took 1.16–3.79 s.
- **Greedy outputs** are byte-identical across both starts and to attempts 1
  and r2.
- **Window:** `qwen38fn` stopped at 11:46:12Z and was restored at 12:04:14Z
  (rc 0, first attempt), about 18 min.
