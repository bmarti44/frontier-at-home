# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **FAIL**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/smoke-2026-10-13/kit`
- started/ended: 2026-10-13T04:43:00.287660+00:00 / 2026-10-13T04:43:38.586926+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | - | - | - | FAIL | speed/speed.json (exit 1) |
| toolcall | passed | 20 | - | measured | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |

## Failed cells
- **speed**: script exit 1 (exit 1, log `speed/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/mimo26/scripts/30_bench_speed_blocks.py --base-url http://127.0.0.1:8019 --out /home/bmarti44/mimo26/results/mimo-v26-flash-gates/smoke-2026-10-13/kit/speed/speed.json --stack-label mimo-v2.6-flash-cuda-spark-128g-262k --reps 2 --warmup 1 --max-tokens 384 --min-completion-tokens 200 --seed 42 --context-levels 0,28672 --ignore-eos-supported --request-timeout 900 --extra-body {"min_tokens":320} --model-id mimo-v2.6-flash --max-tokens-per-event 8 --output-tokenizer-path tokenizer.json --output-tokenizer-sha256 ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
