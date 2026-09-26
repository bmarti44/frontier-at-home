# Fidelity smoke at a 42 GB expert cache: FAIL (containment), fidelity numbers pass

- **Scoring.** All 112 cases were scored. The fixed scorer reproduces the summary: ΔNLL −0.00432 (upper bound −0.00038) and top-1 loss −0.23 pp (upper bound 0.074 pp). Taken alone, the numbers pass.
- **Why the attempt is a FAIL.** At 17:01:19 UTC, 22.5 minutes in, the kernel logged `NVRM ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`. The wrapper converts that into rc 16 (`glm_safe_run.sh:751-755`). The engine itself exited 0, and all output was complete.
- **Host memory was not exhausted.** MemAvailable was 55 GiB at that moment, and the run's low was 47.1 GiB. A driver-side allocation failed while host memory was plentiful. This is the same signature as the 48 GB diagnostic, and the cause is unresolved.
- **Difference from the clean text arm.** This run used ctx 34816 with a prefill cap of 8192. The text arm used ctx 8192 with a cap of 4096. The larger prefill working set is the likely added pressure.
- **Next bounded alternative.** Re-run at a 36 GB cache, which the bisection found clean for short runs.
