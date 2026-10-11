# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m

- verdict: **MEASURED**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/holdout-2026-10-11`
- started/ended: 2026-10-11T08:40:42.373745+00:00 / 2026-10-11T12:06:44.637248+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| accuracy | gsm8k | 0.98 | - | measured | accuracy/acc-<suite>.json |
| accuracy | humaneval | 0.8415 | - | measured | accuracy/acc-<suite>.json |
| accuracy | mmlu-pro | 0.8826 | - | measured | accuracy/acc-<suite>.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
