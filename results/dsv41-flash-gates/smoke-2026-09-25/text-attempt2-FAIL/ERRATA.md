# Errata for failure.json (added 2026-09-26 after the GPT-6 sol high review, finding M7)

`failure.json` "diagnosis_so_far" names a "40.88 GiB single cudaMalloc expert cache (ds4_cuda.cu:4210)". That is wrong.

- `ds4_cuda.cu:4210` is a different device-cache slab.
- The SSD-streaming expert cache allocates gate, up and down as three separate buffers, with a 3/4 backoff on failure (`ds4_cuda.cu:27200`–`27212` at 0aaea5a2).
- This run printed no "allocation failed" line and got the full 4410 slots, so no ds4-visible allocation failed.

Corrected diagnosis: the 48 GB cache target reproducibly coincides with one kernel `NVRM NV_ERR_NO_MEMORY` line (this attempt and `diag-cache48gb`). 4, 24, 36 and 42 GB targets are clean. **Which driver request emits it is unresolved.** The 42 GB cap is a bounded safety candidate, not a root-cause fix.

The FAIL verdict is unchanged. The original `failure.json` is kept unedited. `manifest.json`, `raw.jsonl` and `summary.json` in this directory were rebuilt after the fact from the preserved logs by `harness/bundle_attempt.py` and are marked `derived_post_hoc`.
