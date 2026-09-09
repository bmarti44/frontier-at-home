# Additional native-reference lead

A fresh public-source check found another BF16 study with 256 held-out GLM
contexts. It reports exact replay/live agreement for one checked context, but
states that the full captures and per-position arrays are retained privately.
Its public report therefore cannot supply our missing native reference rows.
This is a concrete potential source of qualifying data if those artifacts become
available; no external message, resource purchase, or inference was performed.

Source: [Local Inference Lab distribution report](https://github.com/local-inference-lab/rtx6kpro/blob/7a3e061cac32fea582e93022765d3137f2c87f3d/kld/glm-5.3-flash-bf16-nvfp4.md).
The manifest pins the inspected source and hash. Our 100-case qualification
remains NO_RESULT; published KLD summaries cannot replace aligned native
per-token reference NLL and top-1 outcomes.
