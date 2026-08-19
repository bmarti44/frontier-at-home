# Qwen3.8-27B final production-config cells — 2026-08-19

Exact production profile (configs/qwen38-production-profile.json):
Q4_K_M, b10488, mmproj-f16, -fa on --no-mmap --parallel 1, MTP
draft-mtp n-max 8 p-min 0.6 (code-tuned winner, greedy-exact 3/3),
served -c 40960 for the 28K fixture (34,673 Qwen tokens). Strict-valid
suite, reps 2 + 1 warmup, capped unit, DSV4 stopped/restored.

| ctx | decode tok/s | prefill tok/s | TTFT s |
|---|---|---|---|
| 0 | 17.46 | 173.4 (setup-dominated tiny prompt) | 0.39 |
| 28,672 | **26.71** | 698.7 | 49.75 |

Deep decode exceeds shallow because MTP draft acceptance rises on
long-fixture continuations — consistent with the code-probe finding
(acceptance-driven). These are the README row cells.
