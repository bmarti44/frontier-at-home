# Phase 2 candidate 3: start 1 result

**Verdict: `FAIL`** (`gate_verdict.py`, `gate-summary.json`). The engine was
killed by its own memory watchdog during startup, before the KV arena was
allocated and before any cell ran. Start 2 was not run (PREREG "Running start 2").

## What happened (2026-10-11, `observe/start1/`)

- **Start conditions:** the host was quiet (co-tenant paused). The memory guard passed at 115.28 GiB `MemAvailable`. The run was at head `47e6e3c9` with a clean tree.
- **Target weights:** loaded as in the smoke. About 76 GiB was allocated in roughly 6 s at 02:19:03–09Z, then the shards streamed in for 458 s.
- **NVRM:** 321 kernel lines `NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY]` all appeared at 02:19:09Z, during that allocation. The smoke had none.
- **MTP drafter weights:** loaded in 72 s. At 02:28:06Z `MemAvailable` was 26.26 GiB. The smoke had 21.76 GiB at the same point.
- **The jump:**
  - It came in the drafter's FP8 MoE post-load step, after `DeepGEMM E8M0 enabled` and before `Using MoEPrepareAndFinalizeNoDPEPModular`.
  - `MemAvailable` fell 26.26 → 10.41 GiB in 8 s, measured by the 1 Hz sampler, and was still falling.
  - Neither `AnonPages` (constant at 2.9 GiB) nor page cache (`Cached` fell 17.35 → 7.67 GiB under reclaim) accounts for it, so it is a device allocation.
- **The kill:** memwatch `BREACH mem_available_gib=9.95` at 02:28:15Z killed the engine. The host recovered to about 114.9 GiB.
- **In the smoke:** the same step took 6 s, and `MemAvailable` rose by 8 GiB.
- **Not reached:**
  - vLLM never logged `Model loading took`, so there is no KV capacity line.
  - The driver's KV check and every cell were not reached.

## Clauses

- **Clause 1 (kill lines):** `FAIL`. It matched 321 NVRM lines and the memwatch `BREACH`.
- **Clause 2 (swap growth):** not scored. The run produced 569 samples, under the scorer's 600-sample minimum. Raw growth was 0.32 GiB (from 0.33 to 0.65 GiB used), which is over the 0.25 GiB limit.
- **Clause 3 (low point):** 10.41 GiB by the 1 Hz external sampler. The scorer reports no low point, because the meminfo log has fewer than 600 samples.
- **Clauses 4–8:** not reached.

## What this measures

The 1m profile's startup peak is at least about 16 GiB higher than the memory
model allowed. The smoke run did not show this step's allocation. That points
to one of two causes, and this run cannot tell them apart:
- the 1m launch shape: 4 × 262,144 and the 19.22 GB arena, against the smoke's 2 × 131,072 and 4.2 GB;
- host state at load time: about 31 GiB of page cache here against 90 GiB in the smoke, plus the NVRM allocation failures early in the load.

## Next (diagnostic, not acceptance)

1. One diagnostic load of the same 1m profile under an external stop at 15 GiB.
   - Python stacks are dumped at 22, 19 and 16 GiB through `faulthandler` (SIGUSR1), so the allocating call is named.
   - Nothing in the production profile or code changes.
2. Only if that does not reproduce: one load of the smoke profile on the same quiet host, to split launch shape from host state.
3. A fix becomes a new candidate in a new directory, with its own preregistration. This directory stays frozen as this attempt's record.
