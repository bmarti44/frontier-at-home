# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c8`
- started/ended: 2026-10-11T18:16:04.188054+00:00 / 2026-10-11T18:58:34.543644+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 25.66 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 0.2997 | <= 0.5 | PASS | speed/speed.json |
| speed | prefill_tok_s@0 | 323.74 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 34.70 | >= 26.71 | PASS | speed/speed.json |
| speed | ttft_s@28672 | 12.95 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 2678.85 | >= 698.70 | PASS | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.63 | - | measured | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| teacher | - | - | - | SKIPPED | model.json declares no reference_logits block (BF16 teacher logits dataset) |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000672 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.62 | - | measured | context/summary.json |
| soak | pass | no | - | FAIL (script exit 1) | soak/soak.json |
| soak | decode_overall_median_tok_s | 30.88 | - | FAIL (script exit 1) | soak/soak.json |

## Failed cells
- **soak**: script exit 1 (exit 1, log `soak/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/35_soak.py --base-url http://127.0.0.1:8017 --stack-label qwen3.8-flash-next-cuda-spark-128g-1m --config-hash qwen3.8-flash-next-cuda-spark-128g-1m-000320e5691e --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c8/soak/soak.json --model qwen3.8-flash-next
soak qwen3.8-flash-next-cuda-spark-128g-1m: pass=False n=211 errors=0 degradation=-0.0732417506246393 decode first/last/overall=29.406050961913415/31.559801613321312/30.883124265901035 tok/s mem_min=9.915748596191406 GiB
  FAILED gates: ['memory_above_floor']
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
