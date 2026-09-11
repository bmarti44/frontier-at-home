# Qualification: glm-5.3-flash/cuda-spark-128g-1m

- verdict: **FAIL**
- bundle: `/home/bmarti44/glm53-v2/results/glm53-flash-gates/qualify-2026-09-11-cells`
- started/ended: 2026-09-11T06:21:05.923842+00:00 / 2026-09-11T07:10:14.600545+00:00
- stack label: `glm-5.3-flash-cuda-spark-128g-1m`; served model: `glm-5.3-flash`
- targets: declared

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| speed | decode_tok_s@0 | 20.02 | >= 17.46 | PASS | speed/speed.json |
| speed | ttft_s@0 | 4.99 | <= 0.5 | FAIL | speed/speed.json |
| speed | prefill_tok_s@0 | 22.32 | - | measured | speed/speed.json |
| speed | valid@0 | yes | yes | PASS | speed/speed.json |
| speed | decode_tok_s@28672 | 19.92 | >= 26.71 | FAIL | speed/speed.json |
| speed | ttft_s@28672 | 71.56 | - | measured | speed/speed.json |
| speed | prefill_tok_s@28672 | 422.64 | >= 698.70 | FAIL | speed/speed.json |
| speed | valid@28672 | yes | yes | PASS | speed/speed.json |
| toolcall | passed | 20 | >= 19 | PASS | toolcall/toolcall.json |
| toolcall | total | 20 | - | measured | toolcall/toolcall.json |
| vision | accuracy | 0.73 | >= 0.64 | PASS | vision/summary.json |
| vision | n | 100 | - | measured | vision/summary.json |
| media | - | - | - | FAIL | media/summary.json (exit 2) |
| teacher | delta_nll_mean | 0.08064 | <= 0.01 | FAIL (script exit 1) | teacher/summary.json |
| teacher | delta_nll_upper_95 | 0.1081 | <= 0.01 | FAIL (script exit 1) | teacher/summary.json |
| teacher | top1_loss_pp_mean | 1.57 | - | FAIL (script exit 1) | teacher/summary.json |
| teacher | top1_loss_pp_upper_95 | 2.02 | - | FAIL (script exit 1) | teacher/summary.json |
| teacher | verdict | FAIL | - | FAIL (script exit 1) | teacher/summary.json |
| soak | pass | yes | - | measured | soak/soak.json |
| soak | decode_overall_median_tok_s | 20.66 | - | measured | soak/soak.json |

## Failed cells
- **media**: script exit 2 (exit 2, log `media/log.txt`)

```
$ /usr/bin/python3 /home/bmarti44/glm53-v2/scripts/51_probe_media_max.py --base-url http://127.0.0.1:8015 --model glm-5.3-flash --out /home/bmarti44/glm53-v2/results/glm53-flash-gates/qualify-2026-09-11-cells/media --stack-label glm-5.3-flash-cuda-spark-128g-1m --images 4 --image-size 512 --video-frames 16 --video-size 512
refusing to overwrite non-empty /home/bmarti44/glm53-v2/results/glm53-flash-gates/qualify-2026-09-11-cells/media
```

- **teacher**: script exit 1 (exit 1, log `teacher/log.txt`)

```
611
candidate final-0023: nll_sum=1108.584304 top1=1818
candidate final-0024: nll_sum=1930.640378 top1=1548

window      domain                              dNLL  top1 loss pp
final-0000  axis1_general                   0.088359        1.9541
final-0001  axis2_legal                     0.198625        2.6380
final-0002  axis3_code_agentic              0.072723        2.4426
final-0003  axis4_reasoning_termination    -0.005956        0.3908
final-0004  axis1_general                   0.324406        5.4714
final-0005  axis2_legal                     0.111362        0.9282
final-0006  axis3_code_agentic              0.098351        2.3449
final-0007  axis4_reasoning_termination    -0.022881        0.1954
final-0008  axis1_general                   0.079876        1.1236
final-0009  axis2_legal                     0.140965        2.6869
final-0010  axis3_code_agentic              0.084023        1.5144
final-0011  axis4_reasoning_termination    -0.014197        0.3908
final-0012  axis1_general                   0.045323        0.8305
final-0013  axis2_legal                     0.139217        2.1983
final-0014  axis3_code_agentic              0.034126        0.2443
final-0015  axis4_reasoning_termination     0.004351       -0.0489
final-0016  axis1_general                   0.104462        1.7098
final-0017  axis2_legal                     0.120969        2.5892
final-0018  axis3_code_agentic              0.081743        1.0259
final-0019  axis4_reasoning_termination    -0.023588        0.4885
final-0020  axis1_general                   0.124303        2.6380
final-0021  axis2_legal                     0.165310        3.6150
final-0022  axis3_code_agentic              0.065383        1.5144
final-0023  axis4_reasoning_termination    -0.022232        0.6839
final-0024  axis1_general                   0.020977       -0.3908

FAIL: delta_nll mean=0.08063994237674536 upper95=0.1081169577010214 top1_loss_pp mean=1.567171470444553 upper95=2.020924405109163 scored=25 failed=0
```

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
