# Qualification: inkling-2-mini/cuda-spark-128g-1m

- verdict: **PASS**
- bundle: `/home/bmarti44/inkling2/results/inkling2-gates/qualify-2026-11-23-c2`
- started/ended: 2026-11-23T08:34:25.155731+00:00 / 2026-11-23T09:35:30.194478+00:00
- stack label: `inkling-2-mini-cuda-spark-128g-1m`; served model: `inkling-2-mini`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 36.14 | >= 34.07 | PASS | speed/speed.json |
| speed | ttft_s@0 | 0.1704 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 416.54 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 44.16 | >= 40.92 | PASS | speed/speed.json |
| speed | ttft_s@28672 | 9.72 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 3332.47 | >= 2711.50 | PASS | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.77 | >= 0.64 | PASS | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000572 | - | measured | context/summary.json |
| context | min_mem_available_gib | 24.35 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 37.38 | - | measured | soak/soak.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
