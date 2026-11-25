# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/preflight-2026-10-13-r2/kit`
- started/ended: 2026-10-13T21:00:07.535045+00:00 / 2026-10-13T21:14:05.687848+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| vision | accuracy | 0.7 | - | measured | vision/summary.json |
| vision | n | 20 | - | measured | vision/summary.json |
| context | verdict | PASS | - | measured | context/summary.json |
| context | total_tokens | 250135 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.76 | - | measured | context/summary.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
