# Qualification: mimo-v2.6-flash/cuda-spark-128g-262k

- verdict: **MEASURED**
- bundle: `/home/bmarti44/mimo26/results/mimo-v26-flash-gates/preflight-2026-10-13/kit`
- started/ended: 2026-10-13T19:17:51.975569+00:00 / 2026-10-13T19:31:10.567084+00:00
- stack label: `mimo-v2.6-flash-cuda-spark-128g-262k`; served model: `mimo-v2.6-flash`
- targets: none (measured only)

| cell | metric | measured | target | status | evidence |
|---|---|---|---|---|---|
| vision | accuracy | 0.55 | - | measured | vision/summary.json |
| vision | n | 20 | - | measured | vision/summary.json |
| context | verdict | PASS | - | measured | context/summary.json |
| context | total_tokens | 250135 | - | measured | context/summary.json |
| context | min_mem_available_gib | 9.25 | - | measured | context/summary.json |

Every number above is parsed from the composed script's own output; see manifest.json for the exact argv of each cell.
