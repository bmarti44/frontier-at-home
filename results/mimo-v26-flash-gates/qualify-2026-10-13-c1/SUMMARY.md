# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/qualify-2026-10-13-c1`
- started/ended: 2026-10-13T22:06:04.572277+00:00 / 2026-10-13T23:06:04.701190+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 21.84 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.6511 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 98.29 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 37.37 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 44.23 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 779.66 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.68 | - | measured | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| context | verdict | PASS | - | measured | context/summary.json |
| context | total_tokens | 250135 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.56 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 29.72 | - | measured | soak/soak.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
