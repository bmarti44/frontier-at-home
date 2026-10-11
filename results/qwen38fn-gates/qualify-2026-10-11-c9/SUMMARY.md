# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m

- verdict: **PASS**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c9`
- started/ended: 2026-10-11T19:40:17.245215+00:00 / 2026-10-11T20:22:53.104987+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 25.76 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 0.2749 | <= 0.5 | PASS | speed/speed.json |
| speed | prefill_tok_s@0 | 352.85 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 34.89 | >= 26.71 | PASS | speed/speed.json |
| speed | ttft_s@28672 | 12.93 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 2683.74 | >= 698.70 | PASS | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.63 | - | measured | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| teacher | - | - | - | SKIPPED | model.json declares no reference_logits block (BF16 teacher logits dataset) |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000672 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.63 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 30.87 | - | measured | soak/soak.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
