# Qualification: qwen3.8-27b/apple-silicon-128g

- verdict: **MEASURED**
- bundle: `/Users/bmarti44/github/frontier-at-home/results/qwen38-gates/apple-silicon-128g-b11514-2026-10-11`
- started/ended: 2026-10-12T01:48:58.852914+00:00 / 2026-10-12T01:58:50.374007+00:00
- stack label: `qwen3.8-27b-apple-silicon-mac-128g`; served model: `qwen3.8-27b`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 31.44 | - | measured | speed/speed.json |
| speed | ttft_s@0 | 0.3445 | - | measured | speed/speed.json |
| speed | prefill_tok_s@0 | 281.56 | - | measured | speed/speed.json |
| speed | valid@0 | yes | - | measured | speed/speed.json |
| speed | decode_tok_s@28672 | 14.56 | - | measured | speed/speed.json |
| speed | ttft_s@28672 | 136.45 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 254.37 | - | measured | speed/speed.json |
| speed | valid@28672 | yes | - | measured | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
