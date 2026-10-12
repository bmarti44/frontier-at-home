# qwen3.8-27b/apple-silicon-128g — first Mac bring-up (DIAGNOSTIC)

**Label: smoke/diagnostic. Not a qualified result and not a headline number.**
The profile stays `status: estimated`. The full `94_qualify_profile.py`
bundle (dev-split accuracy, vision, context, soak) is a follow-up.

- **Host:** `brians-mbp`, an Apple M5 Max with 128 GiB, running macOS 26.6 (25G72).
  - Metal reports MTL0 with a 110100 MiB recommended working set.
- **Engine:** llama.cpp b10488 @ `9d77fa17`, Metal
  (`configs/build-manifests/llamacpp-qwen38-9d77fa17-metal.json`).
  - llama-server sha256 `91ba79f9…6b32`.
- **Weights:** bartowski Q8_0 `a830e736…362d` plus mmproj-f16 `cb0a71fe…212b`.
  - Both sha256 values match `weights/qwen3.8-27b/manifest.json`, and the
    tokenizer/config digests match too.
- **Launch:** `scripts/93_profile_serve.sh --profile qwen3.8-27b/apple-silicon-128g.json --port 8015 start`.
  - Admission and the release gate passed: 92.55 GiB available stable, 37 required.
  - `06_memwatch_macos.sh` ARMED on the server PID/PGID (`serve/*.memwatch.log`).
  - The model loaded in 5.5 s.
- **Concurrent load:** the UE5 editor and Blender were not running. The other
  Claude sessions, a browser and an IDE were idle.

## G1 smoke (manual, 2026-10-12T01:11Z)
- `/v1/models` returns `["qwen3.8-27b"]`.
- "17*23" returned `391` (temperature 0, reasoning on).
- A solid-red 256x256 PNG returned `Red`, so mmproj works.
- MTP draft is active on Metal: 53 of 72 drafted tokens were accepted on the
  short arithmetic reply. This resolves the profile's
  `verify_on_hardware: metal-mtp-draft-behavior`.
- `scripts/tests/test_encoding_qwen38.py` template fidelity: 6 passed, 1 skipped, 1 xfail.

## G2 speed + toolcall (kit, `--no-launch`; see SUMMARY.md and manifest.json)

| ctx | decode tok/s | prefill tok/s | TTFT s |
|---:|---:|---:|---:|
| 0 | 22.34 | 288.25 | 0.34 |
| 28,672 | 13.09 | 220.74 | 159.19 |

Toolcall scored 20/20.

## Findings
1. **Memory is above the estimate.** Server RSS was 49.2 GiB after load and
   58.2 GiB after the benchmarks. The profile's memory model is 46.96 GiB,
   with `overhead_gib` 3. It still fits in the 96 GiB usable, but the overhead
   must be re-measured (G4 soak) before `memory_model` is trusted.
2. **Prefill is about 3x below the Spark** (Spark at 28K: 698.7 tok/s, Q4_K_M).
   - The M5 Metal tensor-API path could not be confirmed. Device init logs
     before `-lv` takes effect at this pin.
   - Upstream fixed a macOS tensor-API compile issue in b10734 (#27461),
     which is newer than this pin.
   - Follow-up: a bounded A/B of b10488 vs a newer Mac pin on the same fixtures.
3. **`93 stop` never writes the watchdog's DISARM record.** Stopping therefore
   logs `FAIL_CLOSED signal=TERM armed_target_unreadable` after the server has
   already exited. It is harmless here, but noisy.
4. The Metal working set is 107.5 GiB, while the resolver's mac usable formula
   is `min(0.75*ram, ram-8)` = 96 GiB. The formula is conservative on macOS 26.
