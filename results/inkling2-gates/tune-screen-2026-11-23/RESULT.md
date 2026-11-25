# Inkling 2 Mini lossless lever screen: result (DIAGNOSTIC)

**Verdict: `COMPLETE`, carrier a6.** a6 is `cuda-spark-128g-1m-tune-dspark-humming-pow2`, which carries vLLM PR #69288 (`INKLING2_MAX_CTAS_POW2=1`), DSpark ×7 and `--moe-backend humming`. `screen_verdict.py` (self-test 60/60) applied the rule preregistered in `PREREG.md` and wrote `summary.json`.

**Every number here is a single-start diagnostic.** None is a headline, README or acceptance number. The M3 acceptance bundle measures the carrier's publishable numbers.

## Arms

Windows: s1, 2026-11-23 04:38–05:28Z (a0, a4, a5); s2 from 05:28Z (a3, a1, a2, a0r, a6). Every arm is one fresh start on :8018 that compiles its Triton kernels from an empty cache, at 4 × 262,144 with BF16 KV in a 68e9-byte arena.

Columns:
- decode at 0 and 28,672: the block-aware bench, K = 8, median of 3;
- rp: the fastest valid prefill rep at 28,672;
- TTFT@0: median;
- cold: aggregate prefill over 6 unseen prompt lengths;
- code: code-shaped decode;
- KV: vLLM's concurrency at 262,144;
- low: the external MemAvailable low point in GiB.

| Arm | Levers | Decode @0 | Decode @28K | rp | TTFT@0 (s) | Cold | Code | KV | Low | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 | none | 36.28 | 30.69 | 2,722.4 | 1.767 | 2,116.6 | 36.74 | 5.36x | 27.10 | baseline; probes same |
| a3 | max_ctas_pow2 | 36.25 | 30.90 | 2,722.5 | 0.173 | 2,512.9 | 36.79 | 5.36x | 27.52 | kept; probes same |
| a4 | dspark7 | 36.47 | 36.89 | 2,704.6 | 1.793 | 2,104.0 | 49.66 | 4.72x | 25.59 | kept; probes differs (2nd) |
| a5 | moe_humming | 36.65 | 31.17 | 3,367.4 | 1.364 | 2,551.0 | 37.17 | 5.36x | 26.01 | kept; probes same |
| a1 | batch4096 | 36.24 | 30.87 | 2,715.1 | 1.968 | 1,921.0 | 36.75 | 4.10x | 26.79 | dropped; probes same |
| a2 | batch8192 | — | — | — | — | — | — | — | — | FAILED_ATTEMPT: KV concurrency 2.79x < 4.0 |
| a0r | none | 36.17 | 30.84 | 2,709.1 | 1.914 | 2,120.3 | 36.70 | 5.36x | 27.47 | baseline; probes same |
| a6 | dspark7, moe_humming, max_ctas_pow2 | 37.84 | 38.25 | 3,341.7 | 0.171 | 2,938.3 | 51.66 | 4.72x | 24.42 | kept; probes differs (2nd) |

**Every arm was safe**, apart from a2's preregistered readiness failure:
- 0 kill lines (kernel, server, memwatch), 0 swap growth;
- tool-call 20/20;
- `vllm_tree` `5bcb020b…` in every arm.

## Decisions

**Baseline pair.** Gains are measured against the better of a0 and a0r, floors against the worse. a0r reproduced a0 within 0.5% on decode and prefill, with byte-identical probes.

| Lever | Arm | Decision | Ratios |
|---|---|---|---|
| #69288 | a3 | kept | rt0 0.098 (TTFT@0 0.173 s vs 1.767 s), rc 1.185, probes byte-identical |
| DSpark ×7 | a4 | kept | decode geomean 1.099, code 1.352, rp floor 0.998 |
| humming | a5 | kept | rp 1.237, rc 1.203, probes byte-identical |
| batch 4096 | a1 | dropped | rp 0.997 (needs ≥ 1.03), rc 0.908 |
| batch 8192 | a2 | dropped, unsafe | KV concurrency 2.79x < 4.0 at readiness: the larger batch shrinks the pool |

**Combined arm (a6).** Decode geomean 1.137 against the better start, rp 1.234 against the worse, so a6 carries. Its TTFT@0 reps are all at about 0.17 s: #69288 removes the per-length MoE recompiles that stalled a0's reps 2 and 3 (1.77, 2.04 s), with humming in place too. Vision (10 MMMU cases, chat, 4,096-token budget): 8/10 on a0 and on a6.

**DSpark.**
- Acceptance length during the speed bench: 2.386 tokens per verification step (draft acceptance rate 0.198) in a4 and in a6.
- Probes: DSpark changes the second greedy probe's text (`85f745a0d87f` vs a0's `c7fe1a930bcb`). The PREREG reports this and does not gate on it, because verification batches change float order. The changed text is the same in a4 and a6, two separate starts, so it is deterministic for the configuration.

**Memory.** a6's low point is 24.42 GiB, so the projected 1M low is 21.4 GiB, against ≥ 12 required. It held 4.72x at 262,144, so no arena charge applies, and the 68e9-byte arena stays.

**Against Flash-Next's headline** (diagnostic, single start, reported only): a6 is above the target on prefill @28K (3,342 vs 2,711.5) and decode @0 (37.84 vs 34.07). It is below on decode @28K (38.25 vs 40.92). The M3 bundle measures the qualified numbers.

## Attempts

`ATTEMPTS.md` records them:
- the driver lock race that refused a3, a1, a2 and a0r in window s1, and its fix;
- a6, authored before window s2 and redefined to the three kept levers before it loaded.

## Next

1. The reasoning-effort pick runs on a6 (`effort-a6/`; `screen_verdict.py --effort-pick`).
2. M3: `cuda-spark-128g-1m` takes a6's launch exactly, at the picked effort (test-first). Then comes the acceptance candidate.
