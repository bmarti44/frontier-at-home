# Frozen KDA replay001 — PASS

Source `61805ebc`, freeze 1788962504.5512102, predetermined drand round 6451050
published 1788962520, BLS-verified seed 5043861260295210466. All five cases
(4,2,3,2048,1 rows) passed the fixed analytic output/state tolerance with zero
mismatched or nonfinite elements. The 2048-row case is four 512-row synthetic
sequences. No model weights or real input tokens were processed.

The 864-file compiled cache was frozen before the seed and sealed during replay.
Seven finite selected configurations passed; the four historical failed tuning
trials remain byte-identical cache inputs. Neither those failures nor all cached
configurations are qualified. Retuning and compilation were rejected at startup.
Full runtime, source, input, bundle, host, identity and cleanup checks passed.
Minimum MemAvailable was 118576360 KiB; new swap-in/out was zero. The cgroup and
process group were empty after completion. Preparation002 remains NO_RESULT.

`kernels.tar.gz` contains every frozen kernel file and its original manifest,
including permissions and absolute group paths. All 864 payloads and the manifest
were independently checked against frozen hashes after packaging. Restore to
the original bound path to verify directly; relocating is a new preparation.
`state.tar.gz` preserves the post-run state, containing only a zero-byte Humming
lock file. `archive.json` binds these lossless packaging artifacts. The original
uncompressed attempt remains in the local cache.

Qualification is model_free_frozen_KDA_selected_configurations_only. It does not
qualify model fidelity, convolution, serving memory, context capacity, speed,
multimodal behavior or production activation. Diagnostic timings remain in raw
evidence and are not model-performance measurements.
