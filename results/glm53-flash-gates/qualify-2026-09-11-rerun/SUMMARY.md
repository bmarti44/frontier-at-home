# Qualification: glm-5.3-flash/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/spark-deepseek-v4-flash/results/glm53-flash-gates/qualify-2026-09-11-rerun`
- started/ended: 2026-09-11T17:00:18.479036+00:00 / 2026-09-11T18:14:23.120096+00:00
- stack label: `glm-5.3-flash-cuda-spark-128g-1m`; served model: `glm-5.3-flash`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 20.12 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 1.75 | <= 0.5 | FAIL | speed/speed.json |
| speed | prefill_tok_s@0 | 39.59 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 19.84 | >= 26.71 | FAIL | speed/speed.json |
| speed | ttft_s@28672 | 58.93 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 513.27 | >= 698.70 | FAIL | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.77 | >= 0.64 | PASS | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| media | verdict | PASS | PASS | PASS | media/summary.json |
| media | images_at_max_answered | yes | - | measured | media/summary.json |
| media | images_at_max_colours_in_order | yes | - | measured | media/summary.json |
| media | images_over_max_rejected_400 | yes | - | measured | media/summary.json |
| media | video_at_max_answered | yes | - | measured | media/summary.json |
| media | video_direction_right | yes | - | measured | media/summary.json |
| teacher | delta_nll_mean | 0.07924 | <= 0.01 | FAIL (script exit 1 in the previous run of this bundle) | teacher/summary.json |
| teacher | delta_nll_upper_95 | 0.1061 | <= 0.01 | FAIL (script exit 1 in the previous run of this bundle) | teacher/summary.json |
| teacher | top1_loss_pp_mean | 1.49 | - | FAIL (script exit 1 in the previous run of this bundle) | teacher/summary.json |
| teacher | top1_loss_pp_upper_95 | 1.91 | - | FAIL (script exit 1 in the previous run of this bundle) | teacher/summary.json |
| teacher | verdict | FAIL | - | FAIL (script exit 1 in the previous run of this bundle) | teacher/summary.json |
| accuracy | gsm8k | 0.94 | - | FAIL (script exit {'gsm8k': 0, 'mmlu-pro': 0, 'humaneval': 1}) | accuracy/acc-<suite>.json |
| accuracy | mmlu-pro | 0.8178 | - | FAIL (script exit {'gsm8k': 0, 'mmlu-pro': 0, 'humaneval': 1}) | accuracy/acc-<suite>.json |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000560 | - | measured | context/summary.json |
| context | min_mem_available_gib | 13.30 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 20.56 | - | measured | soak/soak.json |

## Failed cells
- **teacher**: script exit 1 in the previous run of this bundle (exit 1, log `teacher/log.txt`)
- **accuracy**: script exit {'gsm8k': 0, 'mmlu-pro': 0, 'humaneval': 1} (exit {'gsm8k': 0, 'mmlu-pro': 0, 'humaneval': 1}, log `accuracy/log.txt`)

```
son=correct
[239/247] index=2283 correct=True reason=correct
[240/247] index=2337 correct=True reason=correct
[241/247] index=2387 correct=True reason=correct
[242/247] index=2415 correct=True reason=correct
[243/247] index=2434 correct=True reason=correct
[244/247] index=2454 correct=True reason=correct
[245/247] index=2527 correct=True reason=correct
[246/247] index=2603 correct=True reason=correct
[247/247] index=2624 correct=True reason=correct
$ /usr/bin/python3 /home/bmarti44/spark-deepseek-v4-flash/scripts/31_bench_accuracy.py --base-url http://127.0.0.1:8015 --out /home/bmarti44/spark-deepseek-v4-flash/results/glm53-flash-gates/qualify-2026-09-11-rerun/accuracy/acc-humaneval.json --stack-label glm-5.3-flash-cuda-spark-128g-1m --suite humaneval --split all --transcripts-dir /home/bmarti44/spark-deepseek-v4-flash/results/glm53-flash-gates/qualify-2026-09-11-rerun/accuracy/transcripts/humaneval --encoder glm53 --max-tokens 16384 --request-timeout 2700 --profile-id glm-5.3-flash/cuda-spark-128g-1m --config-hash glm-5.3-flash-cuda-spark-128g-1m-00bf8818bf52 --reasoning-effort low --config-evidence /home/bmarti44/spark-deepseek-v4-flash/results/glm53-flash-gates/qualify-2026-09-11-rerun/accuracy/config-evidence.json
Traceback (most recent call last):
  File "/home/bmarti44/spark-deepseek-v4-flash/scripts/31_bench_accuracy.py", line 1437, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/bmarti44/spark-deepseek-v4-flash/scripts/31_bench_accuracy.py", line 1235, in main
    resolve_humaneval_runtime_digest() if args.suite == "humaneval" else None
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bmarti44/spark-deepseek-v4-flash/scripts/31_bench_accuracy.py", line 345, in resolve_humaneval_runtime_digest
    raise RuntimeError(f"HumanEval Docker image inspection failed: {detail}")
RuntimeError: HumanEval Docker image inspection failed: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
