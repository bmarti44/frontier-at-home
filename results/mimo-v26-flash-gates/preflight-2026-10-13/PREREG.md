# MiMo-V2.6-Flash M3 preflight (DIAGNOSTIC)

This file is committed before the preflight's load. Its numbers are
**diagnostic only**: never headline, README or acceptance evidence.

## Why

M3 (acceptance) will run kit cells that the M1 smoke never exercised on
TabbyAPI: vision, the direct 262K fill, the agent gate and the accuracy
encoder path. The holdout draw is spend-once. A harness mismatch found during
M3 would cost the draw, or force a new acceptance candidate. This window checks
each path once, cheaply, before the M3 PREREG is written.

It also measures the profile change in `6fb0ab3c` (RED) and `d7c2418a`, which
caps two TabbyAPI host-RAM caches below their defaults:

| cache | default | profile now |
|---|---|---|
| `--sysmem-recurrent-cache` (sliding-window checkpoints, 49.25 MiB each) | 4096 MB | 1024 MB (about 20 checkpoints) |
| `--sysmem-multimodal-cache` (image embeddings) | 1024 MB | 256 MB |

r3's MemAvailable low was 9.99 GiB, 2 GiB above the 8 GiB floor. Its
workload never filled either cache.

**Correction (focused review).** A checkpoint holds MiMo's 39 SWA layers
(29.25 MiB) plus the DFlash drafter's 5 SWA layers (window 1,024: 20 MiB),
because `PairedState` sums both. That is 49.25 MiB. The "about 35" in
`d7c2418a`'s message is therefore about 20. The cap is enforced in bytes,
so the memory bound stands.

## What runs (`run-preflight.sh`, one fresh start)

**Before the load:**
- HumanEval's sandbox resolver is checked (M3 needs it).
- Production is stopped through the switch.
- The containment is r3's: page cache released, MemFree ≥ 108 GiB for 3
  samples, a user scope with `MemoryHigh=8G` and `MemoryMax=16G` (mirroring
  the profile unit), `MemorySwapMax=0`, `OOMPolicy=kill`, a 1 s page-cache
  release while the weights load, and the profile memwatch at 8 GiB.
- Observers: 1 Hz `/proc/meminfo` plus the scope's `anon`/`file`, and
  `journalctl -k`.

**After ready, in this order:**

1. **`probe.json`:** three greedy prompts (`../smoke-2026-10-13/probe.py`).
   The caps should not change text, so identity with r3 is reported.
2. **Agent gate:** `regression-suite.py agent-gate --engine vllm --model
   mimo-v2.6-flash --prefix-block-tokens 256`.
   - `--engine vllm` selects the OpenAI-usage semantics: reuse counts from
     the response (`timings`, which TabbyAPI also returns, else
     `usage.prompt_tokens_details.cached_tokens`) and prefill from the
     client-measured TTFT. The server is TabbyAPI, not vLLM.
   - **Why 256:** exllamav3 restores MiMo's sliding-window state only from a
     checkpoint, and only whole 256-token pages are reused. Checkpoints fall
     every 2,048 tokens (MiMo's `default_recurrent_checkpoint_interval`).
     Prefill also stashes one at the prompt's last full page (`job.py`
     `recurrent_last_page`). So a healthy exact repeat re-runs at most one
     page: one block of slack, the vLLM rule's.
3. **Kit `--cells vision,context --vision-limit 20`** into `kit/`:
   - vision: MMMU-val, first 20 cases, chat mode;
   - context: the short sanity round, then the direct fill of one slot ×
     250,128 tokens (cap 262,144) with needles and controls, floor 8 GiB.

   The cell asks with a 512-token budget, and MiMo thinks by default. If
   the short round fails, the full fill never runs. In that case only, the
   driver runs `50_probe_context.py` itself with the same fill and
   `--max-tokens 4096` into `context-budget4096/`. That run exercises the
   262K fill, which M3 depends on.
4. **Accuracy encoder path:** `31_bench_accuracy.py --suite gsm8k --split
   dev --encoder mimo26 --max-tokens 16384` (thinking, no effort level), cut
   by `timeout 1200`. The dev split never touches the holdout ledger. The
   transcripts written before the cut are read for prompt rendering,
   reasoning/answer separation and answer extraction.

Then the server stops and `qwen38fn` is restored.

## Fixed decision rule

- **Kill line** (the smoke's patterns: OOM/NVRM/Xid, memwatch breach) **or a
  failed restore:** stop and report to the owner, per the standing rule.
- **Memory:** peak use is the launch-time MemAvailable minus the minimum.
  If it exceeds 106 GiB, or the low point falls below 8 GiB, M3 does not
  start at this configuration. Stop and report to the owner with the numbers.
- **Harness mismatch:** a cell that cannot measure a correct server
  correctly. Examples: a reuse or usage field TabbyAPI reports differently;
  an answer cut off by a fixed token budget while reasoning; a request shape
  TabbyAPI rejects.
  - Close it in the shared tooling, RED-first, for every model, before the
    M3 PREREG.
  - The preflight is rerun only if the fix changes the server path.
- **Path not exercised:** that path is `NO_RESULT`. M3 needs every path
  above to have run once on this server configuration first, after the cause
  is fixed. Examples: no full 262K fill in either context run, zero
  accuracy transcripts before the cut, a cell that never sent a request.
- **Model result** (e.g. wrong needles with a complete answer, an agent-gate
  reuse miss beyond one page): reported. M3 does not start until the cause
  is understood.
- **Otherwise:** write the M3 c1 PREREG with these settings.

## Recorded, not gating

- Per cell: ran or not, compatible or not, with the raw evidence named.
- Memory:
  - launch MemAvailable, low point and time, and peak use;
  - MemFree low during the load;
  - scope `anon` at ready and its maximum (checkpoint-cache growth);
  - swap growth.
- Context: total server prompt tokens, needles, TTFT, prefill time.
- Agent gate: each test's numbers.
- Probe identity with r3.

## Focused review (before the load)

One review found 1 high and 2 mediums. All are closed in the commit that
adds this file:

- **High:** a short round truncated by the thinking budget would skip the
  262K fill, and the rule would still have let M3 spend the holdout first.
  Closed by the fallback fill and the "path not exercised" rule. The
  accuracy cut is raised from 600 s to 1,200 s.
- **Medium:** the reuse slack is one page (256), not 2,048. The interval
  reasoning was wrong: MiMo does set 2,048.
- **Medium:** a checkpoint is 49.25 MiB, not 29.25 MiB.
- **Low:** the driver now refuses uncommitted files in this directory.
