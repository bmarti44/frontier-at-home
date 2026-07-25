# GLM-5.2 streaming: weights-at-rest + full-pipeline bottleneck research (2026-07-25)

Two deep-research agents (weights-at-rest/compression across sciences; full
pipeline bottlenecks), anchored by local measurements on the actual artifacts.
Full agent reports in the session record; this file keeps the verdicts.

## Compression of the 2-bit weights at rest: CLOSED (measured)
- Local measurement: zstd-19/xz on real IQ2_XXS expert data = 99.2% of
  original. Field-level entropy analysis on the sibling IQ2_XS format:
  order-0 lossless CEILING 96.2% (3.8% saving), order-1 95.8%. 86.5% of the
  bits (grid indices + signs) are exactly incompressible; all recoverable
  entropy is in the scale fields. Block dedup across 8.4M blocks: 0.000000%.
  Cross-expert mutual information: ~0 bits.
- Literature agrees: ZipNN states GGUF-quantized models "do not compress at
  all"; ISCA'26 rANS work targets float/linear-int formats (no codebook
  formats, no code release); HyperQuant: lattice-codebook index entropy only
  0.6-5.9% below bit budget. Every float-exponent method (NeuZip/DFloat11)
  inapplicable. DMX (willjriley) investigated separately: real code (PyPI),
  targets FP16/FP32 BFP+entropy coding, useless for 2-bit quants, no
  independent verification.
- ONE live novel lead: RDO recoding (Oodle-Texture lineage) — manufacture
  LZ-matchable redundancy at encode time for controlled imatrix-scored
  distortion; only known path >10%; days-scale probe defined (near-zero-
  lambda tie-biasing, measure zstd ratio + NLL suite).
- Bounded lossy field coarsening: d->fp8 + 3-bit scales ~ 4.5% (near-safe).
- GPU decompression (nvCOMP/DietGPU-class) feasible on GB10 (~30-60 GB/s)
  but moot without a ratio. GB10 lacks the Blackwell Decompression Engine.

## Storage-side compute / exotic paths: ALL CLOSED on GB10 (verified)
SmartSSD dead, no M.2 CSD exists; no CXL root port; BaM/GDS/GPUDirect RDMA
documented unsupported on Spark (and moot under unified memory); stock drive
is a Samsung PM9E1 4TB (best Gen5 2242 shipping) — no upgrade exists; USB
ports are 20 Gbps (~1.9 GB/s) — marginal; RAID striping across asymmetric
tiers hurts. Genuine hardware door: NVMe-oF/RDMA over the ConnectX-7
(StorageReview measured 12.1 GiB/s INTO a Spark at 100G; ~23-24 GB/s at
200G) — requires a second box; peer-Spark DRAM tier ~10-13 tok/s ceiling;
dual-Spark resident inference (37-48 tok/s class, vLLM fork) beats any
storage tier if a second Spark is ever bought.

## The actual win: the I/O submission path (~3x, days of work)
Measured 3.5 GB/s effective is a QD1/synchronous-read-loop signature, NOT a
randomness penalty: the PM9E1 does 11.4 GiB/s at 1M/16T on a Spark
(StorageReview); QD1 large-block math caps ~6-7 GB/s (matches the 6.64
hdparm datapoint). Fixes, ranked (both agents converged):
1. Coalesced >=1 MiB O_DIRECT reads at QD8-32, whole layer's misses in one
   submit (io_uring or 16-32-thread pread pool): 3.5 -> ~10-11 GB/s (~3x).
2. GGUF repack: one expert = one contiguous 4K-aligned slab, layer-grouped.
3. Router-lookahead prefetch 1-2 layers, overlapped with compute (free
   next-layer gate on current hidden state; 85-96% accuracy in literature).
4. Zero-copy hit path CONFIRMED viable on GB10 (ATS/C2C: mmap/malloc GPU-
   accessible; measured ~11% tax vs device memory, llama.cpp datapoint):
   pass arena pointers into GEMV, delete 8.6 GB/token of copy traffic;
   raises high-residency ceiling ~13 -> ~33 tok/s.
5. Cache policy: Belady bound measured on OUR trace = 88.0% at the 67 GiB
   arena vs LRU ~74-79% -> ~9-13pp headroom via gate-weight x recency or
   lookahead-assisted eviction (bounded; not the main lever).
6. Gate-threshold expert skip (k-eff 8->~6): -25% bytes, LOSSY — mandatory
   NLL gate before adoption on a 2-bit artifact.

## Compute wall (for later stages)
GEMV floor ~35 ms/token at GB10 effective bandwidth -> ~28-33 tok/s hard
ceiling at 100% residency; I/O-vs-compute crossover ~97% hit. FlashMLA
un-portable to sm_121; FlashInfer works. MTP/speculation INCREASES I/O ~30%
below ~90-95% residency (23.8% cross-token expert overlap) — stage-2 only.

## Bottom line vs the DSV4 bar
Realistic staged path on current hardware: I/O fix + zero-copy + repack +
prefetch ~ 5.5-9 tok/s faithful decode; +NVMe-oF tier -> 10-13; 18.4 tok/s
faithful remains unreachable without byte reduction (REAP/skip: quality-
measured tradeoffs) or a second machine. Warm TTFT: checkpoint restore is
~90 ms; remaining suffix-prefill cost is addressable (batch small suffixes).
