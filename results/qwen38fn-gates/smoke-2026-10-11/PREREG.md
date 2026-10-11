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

## Amendment 1 (2026-10-10, before the first load)

A focused review found that the formula above could not be evaluated as
written on this runtime. These changes are committed before any weights are
loaded. The NO_GO thresholds are unchanged.

- **Reasoning field.** vLLM 0.31 streams `reasoning`, and the benches read
  `reasoning_content`. Both profiles now list
  `vllm_reasoning_compat.ReasoningContentMiddleware` ahead of the terminal SSE
  split. That middleware is a response-only key rename and has its own unit
  tests.
- **Block-aware decode (clause 4).** With MTP, one SSE event carries up to
  `1 + num_speculative_tokens = 4` tokens.
  - The bench's own `decode_tok_s` is computed as (events − 1)/elapsed, and its
    early-stop check counts events rather than tokens. Under MTP both are
    recorded but **non-gating**, and so is `suite_valid`.
  - Clause 4 instead uses, per speed rep:
    `decode = (client_completion_tokens − 4) / ((token_timestamps_ns[-1] − token_timestamps_ns[0]) / 1e9)`.
    It takes the median over eligible reps at ctx 0. Subtracting 4 makes this a
    lower bound on the first event's tokens.
- **Eligible rep:** every one of these holds:
  - the rep is valid, or every recorded reason is the bench's event-count
    `early stop`;
  - `client_completion_tokens >= 200`;
  - `timing_source == sse_content_events`;
  - there are at least 2 strictly ordered timestamps.
- **Prefill @28,672:** the median of `prompt_tokens / ttft_s` over eligible
  reps. This is a lower bound, because TTFT includes the first decode step.
- **No eligible rep** at ctx 0 or 28,672 means `NO_RESULT`.
- **Swap (clause 2):** measured as `max(SwapTotal − SwapFree)` over the 1 Hz
  external samples, minus the first sample. That first sample is taken before
  the load.
- **Kill lines (clause 1):**
  - kernel log: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive;
  - server logs: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
  - memwatch log: `BREACH|EMERGENCY_STOP`.
- **Failed attempts:** a `FAILED_ATTEMPT` in the driver log (surviving
  descendants) or any missing evidence file makes the run `NO_RESULT` unless a
  NO_GO clause fires.
- **Scorer:** `verdict.py` in this directory implements exactly this formula
  and writes `summary.json`.
- **Runtime hardening, also from the review:**
  - `FLASHINFER_DISABLE_JIT=1` over the installed AOT cache, so no nvcc runs at
    load;
  - `--gpu-memory-utilization 0.85`, because vLLM's startup check compares it
    against UMA free memory;
  - a 2 h `RuntimeMaxSec` on each scope;
  - a committed host file that differs from `configs/hosts/spark-aba1.json`
    only in `paths.repo`, which points at this worktree.
