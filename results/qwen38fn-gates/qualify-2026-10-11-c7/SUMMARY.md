# Qualification: qwen3.8-flash-next/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c7`
- started/ended: 2026-10-11T17:09:53.420089+00:00 / 2026-10-11T17:21:21.616181+00:00
- stack label: `qwen3.8-flash-next-cuda-spark-128g-1m`; served model: `qwen3.8-flash-next`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | - | >= 17.46 | FAIL (script exit 1) | speed/speed.json |
| speed | ttft_s@0 | - | <= 0.5 | FAIL (script exit 1) | speed/speed.json |
| speed | prefill_tok_s@0 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | valid@0 | no | yes | FAIL (script exit 1) | speed/speed.json |
| speed | decode_tok_s@28672 | - | >= 26.71 | FAIL (script exit 1) | speed/speed.json |
| speed | ttft_s@28672 | - | - | FAIL (script exit 1) | speed/speed.json |
| speed | prefill_tok_s@28672 | - | >= 698.70 | FAIL (script exit 1) | speed/speed.json |
| speed | valid@28672 | no | yes | FAIL (script exit 1) | speed/speed.json |
| toolcall | passed | 0 | >= 19 | FAIL | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | - | - | - | FAIL | vision/summary.json (exit 1) |
| teacher | - | - | - | SKIPPED | model.json declares no reference_logits block (BF16 teacher logits dataset) |
| context | verdict | FAIL | PASS | FAIL (script exit 1) | context/summary.json |
| context | total_tokens | 0 | - | FAIL (script exit 1) | context/summary.json |
| context | min_mem_available_gib | - | - | FAIL (script exit 1) | context/summary.json |
| soak | - | - | - | FAIL | soak/soak.json (exit -15) |

## Failed cells
- **speed**: script exit 1 (exit 1, log `speed/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/30_bench_speed_blocks.py --base-url http://127.0.0.1:8017 --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c7/speed/speed.json --stack-label qwen3.8-flash-next-cuda-spark-128g-1m --reps 2 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body {"min_tokens":320} --model-id qwen3.8-flash-next --max-tokens-per-event 4 --output-tokenizer-path /home/bmarti44/models/qwen3.8-flash-next-nvfp4/tokenizer.json --output-tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```

- **vision**: script exit 1 (exit 1, log `vision/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/38_bench_vision.py --base-url http://127.0.0.1:8017 --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c7/vision --thinking-mode chat --model-id qwen3.8-flash-next
38_bench_vision.py: <urlopen error [Errno 111] Connection refused>
```

- **context**: script exit 1 (exit 1, log `context/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/50_probe_context.py --base-url http://127.0.0.1:8017 --model qwen3.8-flash-next --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c7/context --stack-label qwen3.8-flash-next-cuda-spark-128g-1m --slots 4 --tokens-per-slot 250128 --tokenizer-path /home/bmarti44/models/qwen3.8-flash-next-nvfp4/tokenizer.json --tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 --phase both --profile-id qwen3.8-flash-next/cuda-spark-128g-1m --mem-floor-gib 8
{"verdict": "FAIL", "total_tokens": 0, "needles_correct": 0, "needles_total": 0, "min_mem_available_gib": null, "max_ttft_s": null}
```

- **soak**: script exit -15 (exit -15, log `soak/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/qwen38fn/scripts/35_soak.py --base-url http://127.0.0.1:8017 --stack-label qwen3.8-flash-next-cuda-spark-128g-1m --config-hash qwen3.8-flash-next-cuda-spark-128g-1m-6993f8a86f37 --out /home/bmarti44/qwen38fn/results/qwen38fn-gates/qualify-2026-10-11-c7/soak/soak.json --model qwen3.8-flash-next
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
