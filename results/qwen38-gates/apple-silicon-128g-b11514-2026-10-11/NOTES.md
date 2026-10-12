# qwen3.8-27b/apple-silicon-128g on the portable pin b11514 (DIAGNOSTIC)

**Label: diagnostic A/B. Not a qualified result and not a headline number.**
The profile stays `status: estimated`.

- **Host:** `brians-mbp`, an M5 Max with 128 GiB, running macOS 26.6.
- **Engine:** llama.cpp b11514 @ `de7fa0a3`, Metal, built by
  `scripts/14_build_llamacpp_pin.sh --pin llamacpp-b11514-de7fa0a3 --host-class metal`.
  - llama-server sha256 `86b1c03a…322c`
    (`configs/build-manifests/llamacpp-b11514-de7fa0a3-metal.json`).
- **Weights and profile:** identical to the b10488 run in
  `../apple-silicon-128g-2026-10-11/`, with one exception. The profile flag
  `--no-mmap` became `--load-mode none`, because b11514 removed the old flags
  (`configs/pins/llamacpp.json` `removed_args`).
- **Launch:** `93_profile_serve.sh … --port 8015 start`.
  - The release gate passed with 93.78 GiB available.
  - The watchdog ARMED on PID/PGID 66630.
  - Stop wrote an authenticated DISARM: the watchdog logged `DISARMED`, with no FAIL_CLOSED.
- **Concurrent load:** the GPU was quiet. The `ue5` session held Blender and UE
  GPU work for the run window, and a Cycles render had just finished.

## G1 smoke
- `/v1/models` returns `["qwen3.8-27b"]`.
- "17*23" returned `391`.
- The red PNG returned `Red`.
- MTP accepted 51 of 72 drafted tokens.

## A/B against b10488 (same fixtures and kit argv: seed 42, 2 reps + 1 warm-up, single fresh server per arm)

| metric | b10488 | b11514 | ratio |
|---|---:|---:|---:|
| decode tok/s @0 | 22.34 | 31.44 | 1.41 |
| decode tok/s @28,672 | 13.09 | 14.56 | 1.11 |
| prefill tok/s @0 | 288.25 | 281.56 | 0.98 |
| prefill tok/s @28,672 | 220.74 | 254.37 | 1.15 |
| TTFT s @28,672 | 159.19 | 136.45 | 0.86 |
| toolcall | 20/20 | 20/20 | |
| server RSS after cells | 58.2 GiB | 65.1 GiB | |

This is one block per arm and not a matched ABBA campaign. The ratios are
directional only.

## Findings
1. **Decode is +41% at short context and +11% at 28K.** This is consistent with
   the speculative-decoding few-row MMA kernels (#29869/#30065).
2. **Prefill is +15% at 28K but still about 2.7x below the Spark.** The M5
   tensor path could not be confirmed from the log (no `has tensor` line at
   default verbosity). Upstream #30287 (open) reports M5 prefill kernels still
   leave about 68% on the table.
   - Follow-up: a bounded `GGML_METAL_TENSOR_DISABLE=1` A/B, run in a quiet GPU
     window.
3. **Memory is 65.1 GiB RSS against the 46.96 GiB the profile modeled.**
   `memory_model.overhead_gib` is now set to the measured 21.5 for this cell, so
   the fit is 65.46 / 96 GiB. Upstream #30174 (Metal draft-mtp host growth) is
   the likely driver. The smaller Mac tiers also run MTP and should be re-measured
   before they are trusted.
