# MiMo-V2.6-Flash M1 smoke and fidelity: result (DIAGNOSTIC)

**Pack verdict: `NO_RESULT`.** The kit's speed cell could not run, so clause 5
has no eligible speed rep. Every memory and safety clause passed. **Fidelity:
`MEASURED`.** These numbers are diagnostic only. They are never headline,
README or acceptance evidence.

The scorer is `verdict.py`, unchanged since the PREREG commit (382e3011). It
wrote `summary.json`.

## Attempts

- **Attempt 0 (2026-10-13T04:38Z):** refused by the driver's precondition
  before production was touched. Eight `*.incomplete` partials remained in
  the MikeRoz download cache, left by the Xet download killed at 02:51Z (the
  incident in PR #24). They were 0 to 366 MB, all timestamped 22:51 local,
  and no process held them. The pack had been re-downloaded over plain HTTP
  and every file sha256-verified, so the partials were removed, along with the
  empty `observe/` directory. Nothing was measured.
- **Attempt 1 (04:41:26Z to 05:13:37Z):** this directory.

## Window

| step | time (UTC) | note |
|---|---|---|
| `qwen38fn` stopped | 04:41:26 | via `sudo -n 52_engine_switch.sh stop` |
| pre-load | 04:41:30 | MemFree 110.6 GiB, MemAvailable 115.7 GiB |
| start 1 ready | 04:42:34 | 64 s load (the 94.3 GiB pack streamed at about 1.9 GB/s) |
| start 2 ready | 04:44:42 | 60 s load |
| smoke window closed | 04:45:03 | |
| fidelity | 04:45:10 to 05:02:33 | rc 0 |
| `qwen38fn` restored | 05:13:32 | rc 0 on the first attempt; about 32 min of downtime |

## Rule clauses (PREREG)

| clause | measured | status |
|---|---|---|
| 1. kill lines (kernel, server, memwatch) | none | pass |
| 2. swap growth | 0.00 GiB | pass |
| 3. load peak / run peak (≤ 106 GiB) | **103.61 / 104.50 GiB** | pass |
| 4. tool call | 20 / 20 | pass |
| 5. both starts ready; eligible speed rep at ctx 0 | ready; **no rep** | **missing** |

- The launch reading was MemAvailable 115.67 GiB. The low point was 11.17 GiB
  over 1,921 meminfo samples.
- **MemFree fell to 1.89 GiB during the load**, with 11 GiB of page cache
  resident. GB10 CUDA allocations come from MemFree, so this is the margin to
  watch. The idle engine kept 12.5 GiB MemAvailable on start 1 and 12.9 GiB on
  start 2.
- **cgroup:** `memory.peak` was 17.18 GB, where the 16G `MemoryHigh`
  throttled page cache. At idle, anon was 3.1 GB and file 8.5 GB. The weights
  and KV are uncharged CUDA memory.

## Probes

- `/props`: 1 slot, `n_ctx` 262,144, vision on. `/v1/models` lists only
  `mimo-v2.6-flash`. The admin route `/v1/model/list` returns 404.
- **Greedy outputs are byte-identical across the two starts:** `c933d4cb…`,
  `afe5a855…`, `68da9e93…`.
- Tokens per SSE event: 5.59, 2.47 and 4.86 (123, 106 and 68 tokens). The
  DFlash drafter accepts multi-token blocks.

## Fidelity (`fidelity/model_diff.log`)

The pack (A) is compared with the official MOPD release (B, FP8 with
MXFP4-stored experts). The served Q8 K/V cache is simulated on A. The data is
100 rows × 2,048 tokens of wikitext-2 test.

| metric | value |
|---|---|
| perplexity pack / official | 5.3112 / 5.0504 |
| ΔNLL = ln(PPL_A / PPL_B) | **0.0503** |
| KLD (A, B) / (B, A) | 0.0836 / 0.0887 |
| KLD per token: median / p90 | 0.0339 / 0.1905 |
| top-1 label: pack / official | 60.73% / 61.67% (−0.94 pp) |
| top-1 agreement | 88.2% |

This loss is far outside the repository's lossy budget (ΔNLL ≤ 0.01). It is
reported, not waived; the owner decides. For scale, our GLM-5.3-Flash alias
measured ΔNLL 0.079 on the same tool.

## Defects found by this run

1. **Kit model path.** `94_qualify_profile.py` read Python's
   `-m tabby_launch` as the model and passed a bare `tokenizer.json` to the
   speed bench, which then could not open it (`kit/speed/speed.json`). Fixed
   test-first in 76a8accf (RED) and bc305dbf.
2. **The shim held the whole SSE stream.** sse_starlette frames events with
   CRLF CRLF. The vLLM splitter the shim reused cuts only on LF LF, so no
   event reached the client until the response ended. The probe's
   `stream_seconds` is 0.0001 s for 22 events. Separately, TabbyAPI's
   `include_usage` chunk carries `finish_reason`, which the speed bench
   rejects. Fixed test-first in 5d59e5a0 (RED) and 087b5814:
   `TabbySseMiddleware` forwards events as they are produced.

Under the PREREG's `NO_RESULT` clause, both defects together are the one
bounded fix: the speed cell could not measure a TabbyAPI stream. The rerun
uses a fresh evidence directory, `smoke-2026-10-13-r2`. Fidelity is
`MEASURED` here and is not rerun.
