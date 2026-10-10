# Qualification: qwen3.8-flash-next/cuda-spark-128g-smoke

- verdict: **FAIL**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/smoke-2026-10-11/kit`
- started/ended: 2026-10-10T22:25:45.469469+00:00 / 2026-10-10T22:29:11.536387+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-smoke`; served model: `qwen3.8-flash-next`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | ttft_s@0 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | prefill_tok_s@0 | 350.55 | - | FAIL (script exit 1) | speed/speed.json |
| speed | valid@0 | no | - | FAIL (script exit 1) | speed/speed.json |
| speed | decode_tok_s@28672 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | ttft_s@28672 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | prefill_tok_s@28672 | 2621.56 | - | FAIL (script exit 1) | speed/speed.json |
| speed | valid@28672 | no | - | FAIL (script exit 1) | speed/speed.json |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

## Failed cells
- **speed**: script exit 1 (exit 1, log `speed/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/30_bench_speed.py --base-url http://127.0.0.1:8017 --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/smoke-2026-10-11/kit/speed/speed.json --stack-label qwen3.8-flash-next-cuda-spark-128g-smoke --reps 2 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body {"min_tokens":320} --model-id qwen3.8-flash-next --output-tokenizer-path /home/bmarti44/models/qwen3.8-flash-next-nvfp4/tokenizer.json --output-tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
