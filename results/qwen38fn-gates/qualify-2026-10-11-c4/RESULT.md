# Phase 2 candidate 5: result

**Gate verdict: `PASS`** (`gate_verdict.py` → `gate-summary.json`;
`gate_mutations.py` 25/25 before scoring). The text-only profile
`qwen3.8-flash-next/cuda-spark-128g-1m` qualifies on every preregistered
clause.

The kit's own verdict is `FAIL` (`SUMMARY.md`), on one declared target.
Short-prompt TTFT is 0.5686 s against the 0.5 s target. Speed targets are
general and a miss is reported, not gating (PREREG "Reported, not gating").

## Starts (2026-10-11, quiet host, co-tenant paused)

| | Start 1 | Start 2 |
|---|---|---|
| Head | `b738d944`, clean | `8cf8af18`, clean (start-1 evidence commit only; frozen files identical) |
| Preload MemAvailable | 115.2 GiB | 115.2 GiB |
| Ready after | 645 s | 652 s |
| vLLM KV capacity at 262,144 | 1,079,117 tokens | 1,079,117 tokens |
| Idle MemAvailable | 13.88 GiB | 13.90 GiB |
| External low point | 12.25 GiB (during the 1M fill) | 12.54 GiB |
| Swap growth | 0.00 GiB | 0.00 GiB |
| Kill lines (OOM/NVRM/Xid/memwatch breach) | 0 | 0 |
| Cells | speed, tool-call, teacher (skipped), context, soak | restart probe, agent gate, dev accuracy |

Memwatch logs `FAIL_CLOSED signal=TERM` at each orderly stop. That is the
watchdog exiting with the engine, not a breach.

## Clauses

1. **No kill lines:** pass, 0 in both starts.
2. **Swap growth ≤ 0.25 GiB:** pass, 0.00 in both starts.
3. **Low point ≥ 10 GiB:** pass, 12.25 GiB.
4. **Direct 1M:** pass.
   - KV capacity was 1,079,117 tokens in both starts.
   - The context cell gave `PASS` at 1,000,672 tokens, and the kit manifest's cap is 1,048,576.
5. **Restart identity:** pass, 3/3 greedy sha256 equal.
6. **Tool-call:** pass, 20/20.
7. **Agent gate:** pass on all four tests: prefix-cache, turn-continuation, slot-thrash and prefill-throughput.
8. **Soak:** pass. It ran 1,807 s with 212 requests and 0 errors.

## Reported, not gating

- **Speed** (block-aware, `speed/speed.json`):

  | Metric | Flash-Next | `qwen38-1m` README |
  |---|---|---|
  | Decode at 0 (tok/s) | 25.80 | 17.46 |
  | Decode at 28,672 (tok/s) | 35.54 | 26.71 |
  | Prefill at 28,672 (tok/s) | 2,683.85 | 698.7 |
  | TTFT at 28,672 (s) | 12.93 | 49.75 |
  | Short-prompt TTFT (s) | 0.5686 | 0.39 (0.505 when the kit re-measured it) |

  Short-prompt TTFT is a miss.
- **Accuracy, dev split** (thinking, effort low, 16,384 budget):
  - GSM8K 99/100;
  - MMLU-Pro 221/253 (87.35%, 1 invalid);
  - HumanEval 139/164 (84.76%).

  The dev split is not comparable with the default's holdout row. The
  holdout is spent once, in candidate 6.
- **Fidelity:** `NO_RESULT`. There is no BF16 teacher on this host.

## Not covered

- This profile is text-only (`--limit-mm-per-prompt` image 0, video 0). The
  default serves images.
- The owner asked for image input to be qualified before promotion. That
  run is candidate 6 (`../qualify-2026-10-11-c6/`).
- These speed numbers stay out of the README until the profile that would be
  promoted has passed its own gate.
