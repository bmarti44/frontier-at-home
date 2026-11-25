# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/smoke-2026-10-13-r3/kit`
- started/ended: 2026-10-13T11:47:33.137058+00:00 / 2026-10-13T11:51:48.690608+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 21.85 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.6404 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 99.91 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 35.71 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 45.42 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 759.18 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
