# Scope note (added 2026-09-26, review finding M8)

`summary.json` is a **diagnostic estimate**, not a serving ceiling.

- Its NVMe input is borrowed from `results/glm52-gates/NVME-characterization-final-2026-08-03.json`, which is itself `PARTIAL_DIAGNOSTIC_ONLY`.
- The additive formula assumes serial disk and memory service. It omits Engram row reads, copies, compute and overlap.
- `NOT_FALSIFIED` means only that the estimate did not fall below the 2 tok/s stop threshold.
- Decode and prefill claims require measured production-path runs.

The tensor sizes come from `harness/gguf_tensor_sizes.py`. Since 2026-09-26 it validates magic, version, alignment and exact tiling of the data section. The revalidated output is byte-identical to the input this summary used.
