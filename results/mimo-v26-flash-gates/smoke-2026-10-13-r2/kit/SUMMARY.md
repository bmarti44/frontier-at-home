# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/smoke-2026-10-13-r2/kit`
- started/ended: 2026-10-13T05:29:17.895805+00:00 / 2026-10-13T05:33:34.396125+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 21.93 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.6456 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 99.10 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 36.26 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 45.32 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 760.89 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
