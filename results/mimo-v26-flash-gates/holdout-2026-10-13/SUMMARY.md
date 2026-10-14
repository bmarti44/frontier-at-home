# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/holdout-2026-10-13`
- started/ended: 2026-10-14T00:54:40.575622+00:00 / 2026-10-14T08:02:16.366441+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| accuracy | gsm8k | 0.97 | - | measured | accuracy/acc-<suite>.json |
| accuracy | humaneval | 0.7073 | - | measured | accuracy/acc-<suite>.json |
| accuracy | mmlu-pro | 0.8057 | - | measured | accuracy/acc-<suite>.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
