# Qualification: glm-5.3-flash/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/glm53-v2/results/glm53-flash-gates/qualify-2026-09-11`
- started/ended: 2026-09-11T07:11:20.377135+00:00 / 2026-09-11T08:28:10.233433+00:00
- stack label: `glm-5.3-flash-cuda-spark-128g-1m`; served model: `glm-5.3-flash`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 20.06 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 5.06 | <= 0.5 | FAIL | speed/speed.json |
| speed | prefill_tok_s@0 | 18.96 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 19.91 | >= 26.71 | FAIL | speed/speed.json |
| speed | ttft_s@28672 | 59.05 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 512.20 | >= 698.70 | FAIL | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.73 | >= 0.64 | PASS | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| media | verdict | PASS | PASS | PASS | media/summary.json |
| media | images_at_max_answered | yes | - | measured | media/summary.json |
| media | images_at_max_colours_in_order | yes | - | measured | media/summary.json |
| media | images_over_max_rejected_400 | yes | - | measured | media/summary.json |
| media | video_at_max_answered | yes | - | measured | media/summary.json |
| media | video_direction_right | yes | - | measured | media/summary.json |
| teacher | delta_nll_mean | 0.0788 | <= 0.01 | FAIL (script exit 1) | teacher/summary.json |
| teacher | delta_nll_upper_95 | 0.1056 | <= 0.01 | FAIL (script exit 1) | teacher/summary.json |
| teacher | top1_loss_pp_mean | 1.53 | - | FAIL (script exit 1) | teacher/summary.json |
| teacher | top1_loss_pp_upper_95 | 1.97 | - | FAIL (script exit 1) | teacher/summary.json |
| teacher | verdict | FAIL | - | FAIL (script exit 1) | teacher/summary.json |
| context | verdict | PASS | PASS | PASS | context/summary.json |
| context | total_tokens | 1000560 | - | measured | context/summary.json |
| context | min_mem_available_gib | 13.38 | - | measured | context/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 20.63 | - | measured | soak/soak.json |

## Failed cells
- **teacher**: script exit 1 (exit 1, log `teacher/log.txt`)

```

candidate final-0023: nll_sum=1115.169752 top1=1821
candidate final-0024: nll_sum=1932.500610 top1=1550

window      domain                              dNLL  top1 loss pp
final-0000  axis1_general                   0.092858        2.4915
final-0001  axis2_legal                     0.198298        2.2960
final-0002  axis3_code_agentic              0.072648        1.8075
final-0003  axis4_reasoning_termination    -0.001548        0.4885
final-0004  axis1_general                   0.320267        5.4714
final-0005  axis2_legal                     0.105254        0.7328
final-0006  axis3_code_agentic              0.095756        2.3937
final-0007  axis4_reasoning_termination    -0.021564        0.5374
final-0008  axis1_general                   0.078735        1.2702
final-0009  axis2_legal                     0.136989        2.3937
final-0010  axis3_code_agentic              0.077376        1.5144
final-0011  axis4_reasoning_termination    -0.009290        0.4885
final-0012  axis1_general                   0.043422        1.1724
final-0013  axis2_legal                     0.136962        2.1006
final-0014  axis3_code_agentic              0.035951        0.4397
final-0015  axis4_reasoning_termination     0.003956        0.3420
final-0016  axis1_general                   0.075097        1.0747
final-0017  axis2_legal                     0.121669        2.5403
final-0018  axis3_code_agentic              0.078262        0.6839
final-0019  axis4_reasoning_termination    -0.023300        0.3420
final-0020  axis1_general                   0.125928        2.5892
final-0021  axis2_legal                     0.158548        3.5173
final-0022  axis3_code_agentic              0.064851        1.5144
final-0023  axis4_reasoning_termination    -0.019015        0.5374
final-0024  axis1_general                   0.021886       -0.4885

FAIL: delta_nll mean=0.07879988224075864 upper95=0.10561757072931384 top1_loss_pp mean=1.5300439667806547 upper95=1.9663146545096817 scored=25 failed=0
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
