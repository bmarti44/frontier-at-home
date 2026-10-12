# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m-tune-c10

- verdict: **PASS**
- bundle: `/home/bmarti44/qwen38fn-tune/results/qwen38fn-gates/qualify-2026-10-12-c10`
- started/ended: 2026-10-12T03:31:11.496550+00:00 / 2026-10-12T04:12:50.156174+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m-tune-c10`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 34.07 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 0.2107 | <= 0.5 | PASS | speed/speed.json |
| speed | prefill_tok_s@0 | 460.76 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 40.92 | >= 26.71 | PASS | speed/speed.json |
| speed | ttft_s@28672 | 12.80 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 2711.49 | >= 698.70 | PASS | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.64 | - | measured | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| teacher | - | - | - | SKIPPED | model.json declares no reference_logits block (BF16 teacher logits dataset) |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000672 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.84 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 37.62 | - | measured | soak/soak.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
