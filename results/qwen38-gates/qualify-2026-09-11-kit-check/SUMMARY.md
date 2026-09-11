# Qualification: qwen3.8-27b/cuda-spark-128g-1m

- verdict: **MEASURED**
- bundle: `/home/bmarti44/glm53-v2/results/qwen38-gates/qualify-2026-09-11-kit-check`
- started/ended: 2026-09-11T09:12:09.409536+00:00 / 2026-09-11T09:12:09.441688+00:00
- stack label: `qwen3.8-27b-cuda-spark-128g-1m`; served model: `qwen3.8-27b`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 22.08 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.5054 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 192.95 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 28.63 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 50.23 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 690.91 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 19 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.64 | - | measured | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
