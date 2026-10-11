# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c4`
- started/ended: 2026-10-11T04:38:41.068489+00:00 / 2026-10-11T07:36:33.080260+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 25.80 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 0.5686 | <= 0.5 | FAIL | speed/speed.json |
| speed | prefill_tok_s@0 | 222.95 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 35.54 | >= 26.71 | PASS | speed/speed.json |
| speed | ttft_s@28672 | 12.93 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 2683.85 | >= 698.70 | PASS | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| teacher | - | - | - | SKIPPED |  |
| accuracy | gsm8k | 0.99 | - | measured | accuracy/acc-<suite>.json |
| accuracy | humaneval | 0.8476 | - | measured | accuracy/acc-<suite>.json |
| accuracy | mmlu-pro | 0.8735 | - | measured | accuracy/acc-<suite>.json |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000672 | - | measured | context/summary.json |
| context | min_mem_available_gib | 12.25 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 31.00 | - | measured | soak/soak.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
