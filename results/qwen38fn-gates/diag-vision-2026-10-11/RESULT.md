# Diagnostic: image input on the 1m profile (DIAGNOSTIC, not acceptance)

The question was whether `qwen3.8-flash-next/cuda-spark-128g-1m` can serve
images and still pass the direct-1M gate (low point ≥ 10 GiB). These two loads
ran before candidate 6 was frozen. Neither run is gate evidence or a
performance number.

**Answer: no, not with a safe margin.** Candidate 6 (images on at 1M) is
`NO_GO` and was never frozen or run.

## Runs

Both runs used the containment of `run-phase2.sh`, a 114 GiB start guard on a
quiet host (co-tenant paused, grant 2026-10-11), and a 10.5 GiB external stop
ahead of the profile's 10 GiB memwatch. The rule was fixed before run 2: a stop
means the margin is too thin for candidate 6.

| | Run 1 (`run/`, `eb6aa4f5`) | Run 2 (`run2-nocache-fill/`, `48e1ec44`) |
|---|---|---|
| Profile | images 4, processor cache on (default 4 GiB) | images 4, `--mm-processor-cache-gb 0` |
| Ready | 659 s | 649 s |
| vLLM weights | 76.49 GiB (text-only: 75.65) | same |
| KV at 262,144 | 1,079,117 | 1,079,117 |
| Encoder profiling | 1 image of 16,384 tokens; vit/MMEncoder on FLASH_ATTN | same |
| Greedy probe vs candidate 5 | identical 3/3 | identical 3/3 |
| Idle MemAvailable (60 s after ready) | 12.11 GiB | 12.06 GiB |
| After the MMMU cases | 10.88 GiB (10 cases) | 10.94 GiB (100 cases) |
| MMMU-val-100, chat mode | 6/10 | **63/100**, 3 invalid, 0 errors |
| 4-slot 1M fill (the kit's argv) | not run | **stopped at 10.37 GiB**, 16 s into the fill, at 08:15:59Z |
| NVRM / swap growth | 0 / 0 | 0 / 0 |

For comparison, candidate 5 (text-only) had 14.56 GiB just after its KV commit
and a 1M-fill low of 12.25 GiB.

## Where the memory goes

Run 2's `procs.log` holds per-process RSS plus nvidia-smi's per-process GPU
use. Against candidate 5, image input costs about 3.1–3.3 GiB of host memory:

- **Vision tower weights:** 0.84 GiB (vLLM: 76.49 against 75.65 GiB).
- **Extra CPU memory:** 1.2–1.5 GiB. Idle AnonPages is 5.47 GiB, against
  4.0–4.3 GiB in candidate 5. In this run the API server holds 2.2 GiB anon
  and EngineCore 2.6 GiB. Candidate 5 has no per-process data, so the
  increase cannot be split between them. The processor cache is not the
  cause: turning it off changed idle by 0.05 GiB.
- **GPU allocator growth while serving images:** about 1.0 GiB (nvidia-smi
  98,565 → 99,591 MiB across the 100 cases). MemFree falls while AnonPages
  barely moves. The fill then adds its own growth on top.

The text-only profile's headroom above the 10 GiB floor at the fill low point
was 2.25 GiB. Keeping a 0.5 GiB margin leaves about 1.75 GiB for vision, which
is short of the 3.1–3.3 GiB it needs.

## Accuracy note (diagnostic)

MMMU-val-100 in chat mode scored 63/100: 3 unparseable, 0 transport errors,
256-token answers, the kit's own setting. The default (`qwen38-1m`) scored
64/100 under the same kit settings
(`results/qwen38-gates/qualify-2026-09-11-kit-check/vision`). Decoding is
greedy, so a gated run would very likely reproduce 63 and fail candidate 6's
clause 9 (≥ 64). Thinking-mode vision was not measured. Switching modes after
seeing this result would not be a like-for-like comparison.

## Bounded alternatives considered

1. **Cap the image size** (`--mm-processor-kwargs`):
   - It changes what the model sees for large images, so it lowers fidelity.
   - It shrinks only the GPU encoder share, about 1 GiB at most. Even then,
     the 0.84 GiB weights plus 1.2–1.5 GiB of CPU memory already exceed the 1.75 GiB budget.
   - Downscaling large images gives no reason to expect MMMU above 63.

   Not run.
2. **Images on a smaller-context profile**, such as 3 × 262,144:
   - It gives up the 1M capability that the default serves alongside images.
   - It is an owner decision, not a fix for this profile.

   Not run.

## Outcome

- The 1m profile goes back to the text-only configuration that candidate 5
  qualified. The interpreter pin fix (`python3.12`, same bytes) is kept.
- The candidate 6 draft (`../qualify-2026-10-11-c6/`) is kept unfrozen as the
  record of what was planned.
