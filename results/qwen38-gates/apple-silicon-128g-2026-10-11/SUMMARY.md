# Qualification: qwen3.8-27b/apple-silicon-128g

- verdict: **MEASURED**
- bundle: `/Users/bmarti44/github/frontier-at-home/results/qwen38-gates/apple-silicon-128g-2026-10-11`
- started/ended: 2026-10-12T01:12:03.629155+00:00 / 2026-10-12T01:24:25.935171+00:00
- stack label: `qwen3.8-27b-apple-silicon-mac-128g`; served model: `qwen3.8-27b`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 22.34 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.3366 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 288.25 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 13.09 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 159.19 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 220.74 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
