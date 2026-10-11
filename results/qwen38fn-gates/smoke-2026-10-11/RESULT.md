# Qwen3.8-Flash-Next Phase-1 smoke: result (DIAGNOSTIC)

**Verdict (fixed formula, `verdict.py` → `summary.json`): NO_GO for the
PR #58439 `checkpoint_mapped` arm, on clause 2 only.**

Host swap grew 1.67 GiB, against a limit of 0.25 GiB. Every other gating
clause passed. These are diagnostic numbers only: never headline, README, or
acceptance evidence.

| Clause | Measured | Bar |
|---|---|---|
| 1. kill, OOM, NVRM, Xid lines | none | none |
| 2. host swap growth | **1.67 GiB** max, 1.03 GiB at end | ≤ 0.25 GiB |
| 3. tool-call | 20/20 | ≥ 17/20 |
| 4. decode @0, block-aware | 25.0 tok/s (22.3, 27.7) | > 17.46 |
| 4. prefill @28,672-token fixture, lower bound | 2,622 tok/s (2,649, 2,594) | > 698.7 |

The 28,672-token fixture is 34,701 prompt tokens with this tokenizer. Decode
measured at that fixture is 34.1 tok/s. The bench's own event-based decode and
its `suite_valid` field are non-gating under MTP (see amendment 1). Each rep
showed about 2.2–3.1 tokens per SSE event.

## What happened

- **Start time.** The run started at 22:15:01Z from commit `63943b19`, with
  arlab's grant for window A.
- **Load.** Loading took 621 s: weights took 443 s on the first pass and 83 s
  on the second (the MTP draft), then the KV cache was allocated and CUDA graphs
  captured.
  - vLLM reports `Model loading took 75.67 GiB`.
  - The PLE table was mapped in place: 320,001,536 rows × 160 B, with no
    table-sized device or pinned allocation.
- **KV cache.** The 4.2 GB arena holds 194,493 tokens at a 3,200-token block
  size, about 21.6 KB per token. That is above the 15.7 KB/token community
  estimate the profiles were sized with.
- **Memory.** MemAvailable was 112.4 GiB before the load and 24.9 GiB at idle
  after it. The engine's footprint excluding KV is therefore about 83.6 GiB.
  - The low point was 18.1 GiB, during the speed cell, while the co-tenant's
    Lasso agents ran parallel `g++ -O3` builds: 4 × 1.1–1.4 GB `cc1plus`
    processes, with AnonPages going from 5.5 to 12.8 GB in 30 s.
- **Swap.** The engine scope ran with `MemorySwapMax=0`, so the swapped pages
  belong to other processes. The growth came in two steps:
  - 0.66 GiB during the first shard pass (22:16–22:17), coinciding with the
    co-tenant's anon peaks of up to 12.7 GiB;
  - then up to 1.67 GiB at 22:23:40–51, during the draft-load and KV
    allocation, with MemAvailable at about 21 GiB.

  The pressure here comes from the engine's footprint combined with the
  co-tenant, not from any particular way of reading the table. So the pread lane
  is not expected to change clause 2 on a co-tenanted host. Clause 2 needs to be
  re-measured on a quiet host before a second arm is justified.
- **MTP.** vLLM logged `Fused multi-step draft decode is not supported by
  attention backend(s) QWEN4_EXP_EXP_QSA_STATE`. MTP falls back to rebuilding
  attention metadata between draft steps. That fallback, and CPU contention from
  the co-tenant's compiles, are candidate causes of decode running below the
  community's 36.8–45.8 tok/s.
- **Start 2 aborted.** Start 2 (the record-only greedy restart identity) was
  aborted mid-load at 22:30:25Z by the operator, and that abort is logged in
  `observe/driver.log`.
  - The co-tenant asked for at least 25 GiB to stay available, including 8–10
    GiB compile peaks, and the owner kept the co-tenant's registered run going.
  - This engine idles at about 25 GiB available before any co-tenant peaks, so
    no KV sizing satisfies that.
  - `probe-start2.json` is therefore missing, and the scorer lists it.
  - Restart identity moves to Phase 2.

## Consequences for Phase 2 (4 × 262,144 = 1,048,576 tokens)

- At about 21.6 KB/token, 1,048,576 tokens need about 22.7 GB of KV. The 1m
  profile's 16.5 GB arena holds only about 764K tokens, so it has to be resized
  from this measurement.
- Projected resident footprint is about 105 GiB. The idle low point would be
  about 8 GiB with the co-tenant active, or about 13–15 GiB on a quiet host,
  before prefill activations. The direct 1M fill is therefore feasible only in a
  quiet window, and it is marginal against the 10 GiB floor.

## Files

- `PREREG.md`: the formula, with amendment 1.
- `verdict.py` and `summary.json`: the scorer and its output.
- `verdict_mutations.py` and `verdict-mutations.txt`: the mutation check.
- `probe-start1.json`: the probe results from start 1.
- `kit/`: the 94 kit's speed and tool-call cells.
- `observe/`: the 1 Hz meminfo samples, kernel log, driver log and output,
  memwatch log, server logs, and `nvidia-smi`.
