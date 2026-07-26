# sol xhigh research — how to reduce the 5.8 GB/token decode traffic
Date: 2026-07-26.

HEADLINE: 18.4 tok/s (the DSV4 bar) requires <= 10.7/18.4 = 0.582 GB/token
before compute overhead. That is a 90% cut from 5.8 GB/token. Caching,
prefetch, request packing, GDS and io_uring CANNOT close it individually.
Plain 2-bit VQ reaches only ~3.79 GB/token (2.82 tok/s ceiling); MoBE alone
~3.87 GB/token. Reference-class performance requires making the routed
representation effectively RESIDENT, or committing several verified tokens per
union of loaded experts.

The two top-ranked items need NO fidelity tradeoff (bitwise-identical output),
which makes them qualitatively different from everything measured this session:

1. Decode-protected cache admission / Belady imitation -- +3-15%, LOW cost.
   Treat long-prefill accesses as scan traffic: admit to probation only,
   protect the decode set. Bounded by our own committed Belady simulation
   (results/glm52-gates/logs/g4a/belady-bound.txt): 88.0% clairvoyant at the
   67 GiB arena vs 77.6% measured SLRU, so ~10pp of headroom = 0.6 GB/token.
   Falsifier: replay the committed trace; kill if under 3pp over SLRU.

2. Execute GEMV directly from cache slots -- +5-20%, MEDIUM cost.
   The engine currently COPIES every selected expert into compact gate/up/down
   buffers, which is ~11.6 GB of DRAM traffic per token (5.8 read + 5.8 write)
   against Spark's 273 GB/s, a 42 ms/token floor. Replace with a pointer/slot
   table consumed by GEMV, pin slots until a CUDA completion event, and compute
   each expert as its tensors arrive. Distinct from the REJECTED cross-layer
   prefetch: same-layer producer/consumer, no speculative reads.
   Falsifier: hit-only CUDA-event-timed 600-expert pass, compact-copy vs
   indirect-slot; kill if completed TPOT improves under 5% (NOT enqueue time).

Lower-ranked but higher-upside: full-rank MoBE + 2.02-bit VQ could put the
whole model resident at ~76.9 GiB (zero expert NVMe) but is very high cost with
low-medium gate odds; resident sub-bit + salient IQ2 patches likewise, with a
sobering prior (QMoE ternary is +0.11 nats, 11x our threshold).


codex
## Bottom line

The 18.4 tok/s reference requires a streamed-weight budget below

\[
10.7/18.4 = 0.582\ \text{GB/token},
\]

before compute and software overhead. That is a 90% reduction from 5.8 GB/token. Therefore:

- Cache, prefetch, request packing, GDS, and `io_uring` cannot close the gap individually.
- Plain 2-bit VQ gives about 3.79 GB/token and an all-miss ceiling of only 2.82 tok/s.
- MoBE alone gives roughly 3.87 GB/token and an all-miss ceiling around 2.77 tok/s.
- A credible route to reference-class performance must make the routed representation effectively resident, or commit several verified tokens per union of loaded experts.

The routed pool is approximately:

\[
75 \times 256 \times 9.28\text{ MiB}=174\text{ GiB}.
\]

The existing artifact therefore stores routed weights at about

\[
9.28\text{ MiB}\times 8/(3\times4096\times2048)=3.093\text{ bits/weight}.
\]

## Ranking

Ranked by `(expected gain × chance of passing fidelity) / implementation cost`, rather than by raw upside:

| Rank | Proposal | Plausible result | Gate odds | Cost |
|---:|---|---:|---|---|
| 1 | Decode-protected cache admission / Belady imitation | +3–15%, bounded | Certain | Low |
| 2 | Execute directly from cache slots + same-layer streaming | +5–20% ceiling | Certain | Medium |
| 3 | Full-rank MoBE + 2.02-bit VQ | Entire model resident; potentially several-fold | Low–medium | Very high |
| 4 | Resident 0.8–1-bit safety net + salient IQ2 patches | Zero NVMe if ≤13–21% patches | Low–medium | High |
| 5 | Route-union speculative verification | 0–15% likely; occasionally more | High if verifier exact | Medium–high |
| 6 | CATS-style selective neuron tiles | 33–50% fewer bytes | Low–medium | Very high |
| 7 | Shared centroid + low-rank expert deltas | Zero NVMe if rank ≲550 | Very low | Very high |
| 8 | Packed 2:4 sparsity + low-rank residual | About 41% theoretical reduction | Low | High |
| 9 | Learned depth/layer skipping | 11–25% for 10–20% skipped | Very low without training | Very high |

If ranked purely by upside, 3 and 4 come first.

---

## 1. Decode-protected admission, trained against Belady

**(a) Mechanism.** Keep SLRU, but treat long-prefill accesses as scan traffic: insert them only into probation, protect the pre-existing decode set, and promote only on reuse. Train a tiny layer/position/frequency admission classifier against Belady decisions rather than replacing SLRU wholesale. This is the storage-buffer-pool answer to a 99%-of-experts sequential scan.

**(b) Effect.** Your measured 77.6% hit rate means approximately

\[
5.8(1-0.776)=1.30\text{ GB/token}
\]

from NVMe. Your trace’s Belady bound is 88%, or 0.696 GB/token. The absolute maximum saving is 0.604 GB/token, about 56 ms at 10.7 GB/s. Applied unrealistically perfectly to the 429 ms current TPOT, that gives 373 ms or 2.68 tok/s: a hard +15% ceiling, probably +3–10% in practice. See the local [Belady result](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/logs/g4a/belady-bound.txt:1).

**(c) Fidelity.** None; require bitwise-equivalent logits.

**(d) Cost.** Low. Offline trace simulator plus an admission hook. No kernels or file conversion.

**(e) Fastest falsifier.** Replay the exact prefill→decode trace with byte-weighted hits. Kill it if “prefill probation/no-admit + tail seeding” gains under 3 percentage points over current SLRU.

**(f) Basis.** [TinyLFU](https://arxiv.org/abs/1512.00727), scan-resistant [ARC](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache), and Belady-imitation [Parrot](https://proceedings.mlr.press/v119/liu20f.html). Your local 88% oracle makes this unusually well bounded.

---

## 2. Run GEMV directly from cache slots; pipeline experts within a layer

**(a) Mechanism.** Replace the compact selected-expert buffer with a pointer/slot table consumed by GEMV, pin cache slots until a CUDA completion event, and compute each expert as soon as its three tensors arrive. This is distinct from rejected cross-layer prefetch: it is same-layer producer/consumer execution with no speculative reads.

The current path explicitly copies every selected cached expert into compact gate/up/down buffers in [ds4_cuda.cu](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:2966), through [this D2D copy function](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:1980).

**(b) Effect.** Copying 5.8 GB entails approximately 11.6 GB of DRAM traffic—one read and one write. Against Spark’s 273 GB/s specification, its ideal lower bound is 42 ms/token. [NVIDIA specifies 273 GB/s](https://docs.nvidia.com/dgx/dgx-spark/hardware.html). Same-layer overlap can additionally hide at most the roughly 35 ms expert-compute wall measured locally. The combined mathematical ceiling is therefore around 77 ms: 2.33→2.84 tok/s, +22%; expect 5–15% because indirect kernels and per-expert launches may lose batching efficiency. The local analysis independently estimated [8.6 GB/token of avoidable copy traffic](/home/bmarti44/spark-deepseek-v4-flash/docs/glm52-io-research-2026-07-25.md:52).

**(c) Fidelity.** None. Require bitwise logits and selected IDs.

**(d) Cost.** Medium: pointer-table IQ2 kernels, slot lifetime tracking, event-safe eviction, and grouped accumulation.

**(e) Fastest falsifier.** A hit-only, CUDA-event-timed 600-expert pass comparing compact-copy+GEMV against indirect-slot GEMV. Kill if completed TPOT improves under 5%; do not use enqueue time.

**(f) Basis.** This is out-of-core solver double-buffering applied at expert granularity. It attacks duplicated unified-memory movement, not the compulsory NVMe bytes.

---

## 3. Full-rank MoBE plus 2-bit vector quantization

This is the most interesting representation-level proposal.

**(a) Mechanism.** For gate and up matrices use MoBE

\[
W_i=A_i f\left(\sum_{j=1}^{m}\alpha_{ij}B_j\right),
\]

with full rank \(r=p=2048\), perhaps \(m=4\), resident layer-local bases \(B_j\), and streamed/expert-specific \(A_i\). Leave down matrices intact as in the paper. Then encode all \(A\), bases, and down matrices with 2.02-bit AQLM/VPTQ-style vector codes and resident codebooks.

This is explicitly different from your rejected single common-right-subspace test: multiple bases plus expert-specific coefficients and transformations produce an expert-dependent effective right factor; MoBE also uses a nonlinear elementwise transform and optimized reconstruction. Do not use the paper’s \(k=6\to4\) “dagger” variant—retain all eight experts.

**(b) Effect.** For \(n=256,p=r=2048,d=4096,m=4\), routed parameter retention is

\[
R=\frac13+\frac{2r}{3d}+\frac{2mr}{3np}
  =0.6771.
\]

At 2.02 versus 3.093 bpw:

\[
174\text{ GiB}\times0.6771\times\frac{2.02}{3.093}
=76.9\text{ GiB}.
\]

Depending on whether “211 GB” is decimal or GiB, adding non-routed tensors yields roughly 100–114 GiB—inside 119.7 GiB. That means zero expert NVMe during both decode and prefill.

It does not eliminate unified-memory reads. At these dimensions, the selected \(A\)+down factors consume two-thirds of the old active parameters and the four shared bases consume the remaining third, so the DRAM traffic is still approximately

\[
5.8\times2.02/3.093=3.79\text{ GB/token}.
\]

That is a 14 ms theoretical DRAM floor, leaving enough room for 18.4 tok/s if the fused VQ/MoBE kernels are competent.

MoBE alone or 2-bit VQ alone does not break the NVMe bound; the combination crosses the residency threshold.

**(c) Fidelity.** Risky. MoBE reports 24–30% parameter reduction with roughly 1–2% aggregate benchmark loss, which is encouraging but nowhere near proof of a +0.01 dNLL gate. VQ error compounds it. Use spare memory for sensitive layers/outliers rather than insisting on exactly 2.02 bpw. [MoBE paper](https://arxiv.org/abs/2508.05257), [AQLM](https://arxiv.org/abs/2401.06118), [VPTQ](https://arxiv.org/abs/2409.17066).

**(d) Cost.** Very high: offline optimizer, new GGUF representation, codebook kernels, fused basis-combination/GEMV, and likely access to the original or dequantized teacher weights.

**(e) Fastest falsifier.** Before kernels, emulate a full converted checkpoint and run the paired teacher-forced dNLL test against the current IQ2 teacher. Kill immediately if the upper CI exceeds +0.01. A cheaper preliminary rejection test is activation-weighted output error across early/middle/late layers, but it must never be an acceptance gate.

**(f) Basis.** MoBE is directly targeted at this architecture. VQ codebooks are classical product/additive quantization applied to weight blocks.

---

## 4. Resident sub-bit safety net with sparse high-fidelity replacement blocks

**(a) Mechanism.** Store every routed weight in a resident 0.8–1.0 bpw ternary/binary representation. For Hessian- or activation-sensitive blocks, replace the low-bit block with its original IQ2 block. Unlike HOBBIT’s miss-time choice, make the low-bit representation universal and resident; NVMe is needed only for corrections that cannot fit.

This differs from your lossless IQ2 experiment: the entropy is created by lossy ternarization, followed by selective restoration. It also computes all eight experts, unlike top-k truncation.

**(b) Effect.** A 72 GiB routed arena can hold an average of

\[
3.093\times72/174=1.279\text{ bpw}.
\]

If a fraction \(q\) of blocks is restored to 3.093 bpw:

\[
b_{\rm avg}=b_0+(3.093-b_0)q.
\]

Thus:

- \(b_0=1.0\): \(q_{\max}=13.3\%\).
- \(b_0=0.8\): \(q_{\max}=20.9\%\).

Allowing metadata, practical thresholds are closer to 12% and 19%. If the fidelity gate passes within that budget, all routed weights are resident and NVMe expert traffic becomes zero. DRAM traffic at the 1.279 bpw capacity limit is about

\[
5.8\times1.279/3.093=2.40\text{ GB/token}.
\]

If more patches are required, stream only the nonresident patches rather than 5.8 GB.

**(c) Fidelity.** The prior is sobering. QMoE’s c2048 loss moves from 1.31 BF16 to 1.42 ternary—approximately +0.11, already 11× your threshold; even its 2-bit result is +0.03. Naked sub-bit QMoE is therefore likely dead. The question is whether 12–19% carefully chosen restoration blocks recover almost all of the loss. [QMoE](https://arxiv.org/abs/2310.16795), [HOBBIT](https://arxiv.org/abs/2411.01433), [SqueezeLLM](https://proceedings.mlr.press/v235/kim24f.html), [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), [SpQR](https://openreview.net/pdf?id=Q1u25ahSuy).

**(d) Cost.** High: offline GPTQ/Hessian calibration, patch index, a low-bit GEMV, and a fused replacement-block path.

**(e) Fastest falsifier.** Produce the single curve `paired dNLL versus restored-block fraction q`. Kill if +0.01 requires \(q>0.12\) for a 1-bit base or \(q>0.19\) for a QMoE-like 0.8-bit base.

**(f) Basis.** This combines MoE-specific sub-bit QMoE with dense+sparse/outlier quantization. It is the closest weight-side analogue of a video codec’s low-quality reference frame plus residual enhancement data.

---

## 5. Route-union, utility-gated speculative verification

**(a) Mechanism.** Use a deeper draft, but verify layer-major: at each routed layer load the union of experts required by all draft tokens once, group tokens by expert, and dynamically disable or shorten speculation when `committed tokens / union expert-equivalents` drops below one.

This is different from the measured depth-1 MTP run: deeper blocks, union-deduplicated expert loading, adaptive \(K\), and an exact correcting verifier are the substance—not simply enabling the existing MTP head.

**(b) Effect.** Your 23.8% consecutive-token top-8 overlap means about 1.904 experts persist. Under a first-order approximation, four sequential tokens have union

\[
8+3(8-1.904)=26.29
\]

experts/layer, or 6.57 per token: 17.8% byte amortization. Ignoring drafter cost, four-token verification must commit more than

\[
26.29/8=3.29
\]

tokens—over 82.2%—just to break even. Deeper acceptance will probably be below that given the measured 72.5% depth-1 acceptance. That makes the expected win 0–15%, not 4×.

**(c) Fidelity.** A mathematically exact speculative verifier preserves the target distribution; implementation/RNG ordering may not be bit-identical. Gate with distributional equivalence and the existing paired dNLL test.

**(d) Cost.** Medium–high: multi-token execution, union routing, grouped expert kernels, and rejection rollback.

**(e) Fastest falsifier.** No implementation is initially needed. From route and acceptance traces, calculate per iteration

\[
U=\frac{8\times\text{committed tokens}}
        {\text{union expert count}}.
\]

Kill if median \(U\le1\) before adding drafter overhead.

**(f) Basis.** [Cascade](https://arxiv.org/abs/2506.20675) finds MoE speculation often moves 2–3× more weights and uses adaptive utility to obtain 7–14%; [SpecMoE](https://arxiv.org/abs/2604.10152) coalesces migration using self-assisted speculation, though its reported 4.3× result is not directly transferable.

---

## 6. Two-phase activation-sparse expert loading

**(a) Mechanism.** Fetch and compute the gate projection first. Use its SwiGLU activations to select intermediate channels, then fetch only corresponding up rows and down-column tiles. Down must be stored in a transposed or channel-tiled artifact.

**(b) Effect.** Approximating the three matrices as equal-byte components, retaining fraction \(p\) of intermediate channels costs

\[
B(p)=5.8\frac{1+2p}{3}.
\]

Examples:

- \(p=0.50\): 3.87 GB/token, −33%.
- \(p=0.25\): 2.90 GB/token, −50%.
- \(p\to0\): 1.93 GB/token floor because the full gate is still required.

It cannot reach 0.58 GB/token without also compressing or predicting the gate.

IQ2_XXS stores 256 weights in 66 bytes, as seen in the local [block definition](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:68). A 4096-wide row is only 1,056 bytes, so naïve O_DIRECT row reads suffer nearly 4× amplification. Channel tiles must be 4 KiB-aligned. Down columns are scattered under the current row-major layout, hence the repack requirement.

**(c) Fidelity.** SwiGLU is not ReLU: omitted channels are not exactly zero. TEAL reports 40–50% activation sparsity with modest benchmark loss, but that does not establish +0.01 dNLL. The two-phase I/O also does little for long prefill because the union of active channels over 2048 tokens will approach all channels. [TEAL](https://arxiv.org/abs/2408.14690), [CATS](https://arxiv.org/abs/2404.08763), [LLM in a Flash](https://arxiv.org/abs/2312.11514).

**(d) Cost.** Very high: new down layout, two-stage fetch scheduling, sparse/tiled IQ2 kernels, and likely per-layer thresholds.

**(e) Fastest falsifier.** Run oracle CATS: compute the full expert, then zero the lowest-contribution intermediate channels before down projection. Kill if even oracle \(p=0.5\) fails +0.01 dNLL.

**(f) Basis.** CATS is the closest match for SwiGLU; LLM in a Flash contributes the storage-side row/column bundling idea.

---

## 7. Shared centroid with activation-aware low-rank deltas

**(a) Mechanism.** Represent each matrix as \(W_i=W_b+U_iV_i\), with a Fisher/activation-weighted shared base resident per layer and only expert-specific factors selected. The shared gate/up projections are computed once; the shared down projection can be applied once to the router-weighted sum of expert intermediates.

This differs from the rejected common-right-subspace test: the base is additive and full-rank, while every expert has both unique left and right factors, fitted with activation/Fisher weighting.

**(b) Effect.** For \(p=2048,d=4096\), delta retention is

\[
\rho(r)=\frac{r(p+d)}{pd}.
\]

- \(r=256:\rho=18.75\%\), about 1.09 GB/token of expert deltas.
- \(r=512:\rho=37.5\%\), about 2.18 GB/token.
- A single base for every routed layer costs only \(75\times9.28\text{ MiB}=0.68\) GiB.

Static storage at \(r=512\) is approximately

\[
174\times0.375+0.68=65.9\text{ GiB},
\]

so every delta and base fits in 72 GiB. The zero-NVMe threshold is approximately \(r\le550\).

**(c) Fidelity.** Prior evidence is bad for your gate. At only 20% compression, D²-MoE changes Mixtral WikiText perplexity 3.98→4.65, corresponding to \(\Delta\text{NLL}\approx0.155\); DeepSeek-MoE moves 6.38→6.84, about 0.069. Both are far above +0.01, and \(r\approx512\) represents much stronger compression. [D²-MoE paper and tables](https://arxiv.org/html/2502.17298).

**(d) Cost.** Very high: offline Fisher merge, factorization, factor quantization, and fused base+delta kernels.

**(e) Fastest falsifier.** Measure paired dNLL as a function of activation-aware rank. Kill unless +0.01 is reached by \(r\le550\). Singular-energy retention alone is only a preliminary rejection test.

**(f) Basis.** [D²-MoE](https://proceedings.mlr.press/v267/gu25c.html). Elegant arithmetic, discouraging empirical prior.

---

## 8. Packed 2:4 sparsity plus a low-rank residual

**(a) Mechanism.** Enforce two nonzeros per four weights using second-order pruning, store a compact 2-bit value stream and a six-pattern selector, then recover structured pruning error with a small dense low-rank residual. All 256 experts and all eight routes remain; this is not REAP/expert pruning.

**(b) Effect.** With 2-bit nonzeros:

- Values: \(0.5\times2=1.0\) bpw.
- Pattern: \(\log_2(6)/4=0.646\) bpw.
- Rank-64, 4-bit residual: about \(0.1875\) bpw.

Total: approximately 1.834 bpw before scales, or

\[
5.8\times1.834/3.093=3.44\text{ GB/token},
\]

a 41% reduction. If surviving fidelity requires 4-bit nonzeros, the representation rises to roughly 2.834 bpw before scales—only an 8% reduction. At that point it is dead as a byte optimization.

**(c) Fidelity.** High risk because pruning an already extreme 2-bit artifact compounds errors. SparseGPT’s good 50–60% results were on much less compressed dense models and its 2:4 results were worse than unstructured pruning.

**(d) Cost.** High: prune from a good teacher, new packed format, custom sparse 2-bit kernels, residual kernels.

**(e) Fastest falsifier.** Apply 2:4 SparseGPT plus rank-64/128 residuals to representative layers with real activations. Kill if +0.01 requires 4-bit nonzeros or a residual above roughly rank 128.

**(f) Basis.** [SparseGPT](https://proceedings.mlr.press/v202/frantar23a.html) and sparse+dense decomposition methods such as SqueezeLLM.

---

## 9. Learned depth or layer skipping

**(a) Mechanism.** Train a per-token gate to bypass entire routed FFNs or transformer layers, preferably with distillation and a fixed compute budget.

**(b) Effect.**

\[
B(s)=5.8(1-s).
\]

Skipping 10% of routed layers gives 5.22 GB/token and at most +11%; 20% gives 4.64 GB and at most +25%. Matching 18.4 tok/s through layer skipping alone would require skipping approximately 87% of the routed layers, which is not credible.

**(c) Fidelity.** Very high risk as a post-hoc modification. Mixture-of-Depths and LayerSkip rely on training recipes; they are not evidence that a pretrained GLM checkpoint can safely skip layers. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [LayerSkip](https://arxiv.org/abs/2404.16710).

**(d) Cost.** Very high if trained correctly; moderate for a likely-failing post-hoc gate.

**(e) Fastest falsifier.** Oracle-test the least damaging subset of layers under teacher forcing. Kill if the best 10% subset already exceeds +0.01 dNLL.

**(f) Basis.** Adaptive-depth transformers. This is an incremental lever, not a byte-bound breaker.

---

## Explicit dead ends and bounds

### Four-request batching

For \(B\) independent requests with uniform top-8 routing, expected unique experts per layer are

\[
256\left[1-(1-8/256)^B\right].
\]

At \(B=4\), that is 30.53 experts instead of 32, or 7.63 per output token: only a 4.6% byte reduction, 5.8→5.53 GB/token. It may improve grouped-GEMM utilization, but it barely amortizes NVMe at four requests. At \(B=32\), the reduction becomes meaningful—about 5.1 experts/token—but at substantial batching latency and KV/state cost.

### Expert-pair or layer-group caching

Under the same byte capacity, a single-expert cache weakly dominates a pair cache: it can retain either member independently, while a pair cache must evict both. Pairing helps metadata and request coalescing, not hit-rate capacity. Your 23.8% temporal overlap is already captured by expert-granular caching.

### Router-aware file packing

A 9.28 MiB expert spans roughly 2,376 4 KiB pages; alignment waste is below 0.05%. Co-activation packing can turn several large random reads into a larger read, but cannot reduce payload bytes. It is useful only if the live miss trace fails to reach the measured 10.7 GB/s.

### Reading only “used rows” exactly

A dense gate/up/down GEMV uses every weight. IQ2 block structure allows whole-row reads, but there are no unused rows until activation sparsity is introduced. Down-projection channel selection is especially awkward because the relevant intermediate dimensions are columns in the current layout. Exact row elision is therefore impossible; CATS-style lossy activation sparsity is the actual proposal.

### GPUDirect Storage on Spark

Dead. NVIDIA explicitly states that [DGX Spark supports GDS only in compatibility mode and must not load `nvidia-fs`](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html). GDS P2P also does not support managed/system memory; compatibility mode uses internal bounce buffers. It cannot produce NVMe→GPU peer DMA on this machine.

### `io_uring` with registered buffers

Worth a microbenchmark, not a research program. Registered buffers reduce pinning and CPU overhead, but large 9 MiB expert transfers are bandwidth-dominated. Storage research finds `io_uring` substantially reduces cycles while increasing maximum throughput only slightly once O_DIRECT already saturates the device. [NVMe I/O-stack analysis](https://link.springer.com/chapter/10.1007/978-3-031-74097-8_9). Kill if an exact-miss-trace replay at QD16–32 does not beat the six-thread pool by at least 10%.

### AWQ saliency by itself

AWQ scaling does not reduce stored bytes; it makes a lower-bit representation more accurate. SqueezeLLM/SpQR outliers similarly add sparse high-precision data. They are enabling techniques for proposals 3 and 4, not independent byte levers.

### Video-codec-style expert-output memoization

Likely dead. Reusing an old expert output for a nearby hidden state saves 9.28 MiB, but for a linear projection

\[
Wx_t=Wx_{t-1}+W(x_t-x_{t-1}),
\]

and the residual multiplication still touches all weights. Approximate full-output reuse is lossy and high-dimensional hidden states lack video’s spatial correspondence. Even [DeepCache](https://arxiv.org/abs/1712.01670) explicitly avoids applying video motion heuristics directly to opaque high-dimensional internals.

## The experiment sequence I would actually run

1. **Offline, hours:** replay SLRU with prefill probation, TinyLFU admission, and Belady labels; simultaneously compute speculative union utility from existing route traces.
2. **Fidelity-only, no kernels:** generate three curves:
   - dNLL versus sub-bit patch fraction \(q\);
   - dNLL versus MoBE basis count/factor precision;
   - oracle CATS dNLL versus retained channel fraction.
3. **Exact systems work:** prototype direct cache-slot GEMV before touching `io_uring` or GDS.
4. **Only if the curves pass:** build either the MoBE+VQ resident artifact or the resident sub-bit+patch artifact.

My blunt recommendation: the two serious byte-bound bets are **MoBE+VQ residency** and **resident sub-bit plus salient restoration**. D² low-rank deltas have beautiful capacity arithmetic but published fidelity evidence that is already several times worse than your gate at far milder compression. Everything else is an incremental systems optimization or a trace-dependent amortization technique.
tokens used
467,380
## Bottom line

The 18.4 tok/s reference requires a streamed-weight budget below

\[
10.7/18.4 = 0.582\ \text{GB/token},
\]

before compute and software overhead. That is a 90% reduction from 5.8 GB/token. Therefore:

- Cache, prefetch, request packing, GDS, and `io_uring` cannot close the gap individually.
- Plain 2-bit VQ gives about 3.79 GB/token and an all-miss ceiling of only 2.82 tok/s.
- MoBE alone gives roughly 3.87 GB/token and an all-miss ceiling around 2.77 tok/s.
- A credible route to reference-class performance must make the routed representation effectively resident, or commit several verified tokens per union of loaded experts.

The routed pool is approximately:

\[
75 \times 256 \times 9.28\text{ MiB}=174\text{ GiB}.
\]

The existing artifact therefore stores routed weights at about

\[
9.28\text{ MiB}\times 8/(3\times4096\times2048)=3.093\text{ bits/weight}.
\]

## Ranking

Ranked by `(expected gain × chance of passing fidelity) / implementation cost`, rather than by raw upside:

| Rank | Proposal | Plausible result | Gate odds | Cost |
|---:|---|---:|---|---|
| 1 | Decode-protected cache admission / Belady imitation | +3–15%, bounded | Certain | Low |
| 2 | Execute directly from cache slots + same-layer streaming | +5–20% ceiling | Certain | Medium |
| 3 | Full-rank MoBE + 2.02-bit VQ | Entire model resident; potentially several-fold | Low–medium | Very high |
| 4 | Resident 0.8–1-bit safety net + salient IQ2 patches | Zero NVMe if ≤13–21% patches | Low–medium | High |
| 5 | Route-union speculative verification | 0–15% likely; occasionally more | High if verifier exact | Medium–high |
| 6 | CATS-style selective neuron tiles | 33–50% fewer bytes | Low–medium | Very high |
| 7 | Shared centroid + low-rank expert deltas | Zero NVMe if rank ≲550 | Very low | Very high |
| 8 | Packed 2:4 sparsity + low-rank residual | About 41% theoretical reduction | Low | High |
| 9 | Learned depth/layer skipping | 11–25% for 10–20% skipped | Very low without training | Very high |

If ranked purely by upside, 3 and 4 come first.

---

## 1. Decode-protected admission, trained against Belady

**(a) Mechanism.** Keep SLRU, but treat long-prefill accesses as scan traffic: insert them only into probation, protect the pre-existing decode set, and promote only on reuse. Train a tiny layer/position/frequency admission classifier against Belady decisions rather than replacing SLRU wholesale. This is the storage-buffer-pool answer to a 99%-of-experts sequential scan.

**(b) Effect.** Your measured 77.6% hit rate means approximately

\[
5.8(1-0.776)=1.30\text{ GB/token}
\]

from NVMe. Your trace’s Belady bound is 88%, or 0.696 GB/token. The absolute maximum saving is 0.604 GB/token, about 56 ms at 10.7 GB/s. Applied unrealistically perfectly to the 429 ms current TPOT, that gives 373 ms or 2.68 tok/s: a hard +15% ceiling, probably +3–10% in practice. See the local [Belady result](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/logs/g4a/belady-bound.txt:1).

**(c) Fidelity.** None; require bitwise-equivalent logits.

**(d) Cost.** Low. Offline trace simulator plus an admission hook. No kernels or file conversion.

**(e) Fastest falsifier.** Replay the exact prefill→decode trace with byte-weighted hits. Kill it if “prefill probation/no-admit + tail seeding” gains under 3 percentage points over current SLRU.

**(f) Basis.** [TinyLFU](https://arxiv.org/abs/1512.00727), scan-resistant [ARC](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache), and Belady-imitation [Parrot](https://proceedings.mlr.press/v119/liu20f.html). Your local 88% oracle makes this unusually well bounded.

---

## 2. Run GEMV directly from cache slots; pipeline experts within a layer

**(a) Mechanism.** Replace the compact selected-expert buffer with a pointer/slot table consumed by GEMV, pin cache slots until a CUDA completion event, and compute each expert as soon as its three tensors arrive. This is distinct from rejected cross-layer prefetch: it is same-layer producer/consumer execution with no speculative reads.

The current path explicitly copies every selected cached expert into compact gate/up/down buffers in [ds4_cuda.cu](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:2966), through [this D2D copy function](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:1980).

**(b) Effect.** Copying 5.8 GB entails approximately 11.6 GB of DRAM traffic—one read and one write. Against Spark’s 273 GB/s specification, its ideal lower bound is 42 ms/token. [NVIDIA specifies 273 GB/s](https://docs.nvidia.com/dgx/dgx-spark/hardware.html). Same-layer overlap can additionally hide at most the roughly 35 ms expert-compute wall measured locally. The combined mathematical ceiling is therefore around 77 ms: 2.33→2.84 tok/s, +22%; expect 5–15% because indirect kernels and per-expert launches may lose batching efficiency. The local analysis independently estimated [8.6 GB/token of avoidable copy traffic](/home/bmarti44/spark-deepseek-v4-flash/docs/glm52-io-research-2026-07-25.md:52).

**(c) Fidelity.** None. Require bitwise logits and selected IDs.

**(d) Cost.** Medium: pointer-table IQ2 kernels, slot lifetime tracking, event-safe eviction, and grouped accumulation.

**(e) Fastest falsifier.** A hit-only, CUDA-event-timed 600-expert pass comparing compact-copy+GEMV against indirect-slot GEMV. Kill if completed TPOT improves under 5%; do not use enqueue time.

**(f) Basis.** This is out-of-core solver double-buffering applied at expert granularity. It attacks duplicated unified-memory movement, not the compulsory NVMe bytes.

---

## 3. Full-rank MoBE plus 2-bit vector quantization

This is the most interesting representation-level proposal.

**(a) Mechanism.** For gate and up matrices use MoBE

\[
W_i=A_i f\left(\sum_{j=1}^{m}\alpha_{ij}B_j\right),
\]

with full rank \(r=p=2048\), perhaps \(m=4\), resident layer-local bases \(B_j\), and streamed/expert-specific \(A_i\). Leave down matrices intact as in the paper. Then encode all \(A\), bases, and down matrices with 2.02-bit AQLM/VPTQ-style vector codes and resident codebooks.

This is explicitly different from your rejected single common-right-subspace test: multiple bases plus expert-specific coefficients and transformations produce an expert-dependent effective right factor; MoBE also uses a nonlinear elementwise transform and optimized reconstruction. Do not use the paper’s \(k=6\to4\) “dagger” variant—retain all eight experts.

**(b) Effect.** For \(n=256,p=r=2048,d=4096,m=4\), routed parameter retention is

\[
R=\frac13+\frac{2r}{3d}+\frac{2mr}{3np}
  =0.6771.
\]

At 2.02 versus 3.093 bpw:

\[
174\text{ GiB}\times0.6771\times\frac{2.02}{3.093}
=76.9\text{ GiB}.
\]

Depending on whether “211 GB” is decimal or GiB, adding non-routed tensors yields roughly 100–114 GiB—inside 119.7 GiB. That means zero expert NVMe during both decode and prefill.

It does not eliminate unified-memory reads. At these dimensions, the selected \(A\)+down factors consume two-thirds of the old active parameters and the four shared bases consume the remaining third, so the DRAM traffic is still approximately

\[
5.8\times2.02/3.093=3.79\text{ GB/token}.
\]

That is a 14 ms theoretical DRAM floor, leaving enough room for 18.4 tok/s if the fused VQ/MoBE kernels are competent.

MoBE alone or 2-bit VQ alone does not break the NVMe bound; the combination crosses the residency threshold.

**(c) Fidelity.** Risky. MoBE reports 24–30% parameter reduction with roughly 1–2% aggregate benchmark loss, which is encouraging but nowhere near proof of a +0.01 dNLL gate. VQ error compounds it. Use spare memory for sensitive layers/outliers rather than insisting on exactly 2.02 bpw. [MoBE paper](https://arxiv.org/abs/2508.05257), [AQLM](https://arxiv.org/abs/2401.06118), [VPTQ](https://arxiv.org/abs/2409.17066).

**(d) Cost.** Very high: offline optimizer, new GGUF representation, codebook kernels, fused basis-combination/GEMV, and likely access to the original or dequantized teacher weights.

**(e) Fastest falsifier.** Before kernels, emulate a full converted checkpoint and run the paired teacher-forced dNLL test against the current IQ2 teacher. Kill immediately if the upper CI exceeds +0.01. A cheaper preliminary rejection test is activation-weighted output error across early/middle/late layers, but it must never be an acceptance gate.

**(f) Basis.** MoBE is directly targeted at this architecture. VQ codebooks are classical product/additive quantization applied to weight blocks.

---

## 4. Resident sub-bit safety net with sparse high-fidelity replacement blocks

**(a) Mechanism.** Store every routed weight in a resident 0.8–1.0 bpw ternary/binary representation. For Hessian- or activation-sensitive blocks, replace the low-bit block with its original IQ2 block. Unlike HOBBIT’s miss-time choice, make the low-bit representation universal and resident; NVMe is needed only for corrections that cannot fit.

This differs from your lossless IQ2 experiment: the entropy is created by lossy ternarization, followed by selective restoration. It also computes all eight experts, unlike top-k truncation.

**(b) Effect.** A 72 GiB routed arena can hold an average of

\[
3.093\times72/174=1.279\text{ bpw}.
\]

If a fraction \(q\) of blocks is restored to 3.093 bpw:

\[
b_{\rm avg}=b_0+(3.093-b_0)q.
\]

Thus:

- \(b_0=1.0\): \(q_{\max}=13.3\%\).
- \(b_0=0.8\): \(q_{\max}=20.9\%\).

Allowing metadata, practical thresholds are closer to 12% and 19%. If the fidelity gate passes within that budget, all routed weights are resident and NVMe expert traffic becomes zero. DRAM traffic at the 1.279 bpw capacity limit is about

\[
5.8\times1.279/3.093=2.40\text{ GB/token}.
\]

If more patches are required, stream only the nonresident patches rather than 5.8 GB.

**(c) Fidelity.** The prior is sobering. QMoE’s c2048 loss moves from 1.31 BF16 to 1.42 ternary—approximately +0.11, already 11× your threshold; even its 2-bit result is +0.03. Naked sub-bit QMoE is therefore likely dead. The question is whether 12–19% carefully chosen restoration blocks recover almost all of the loss. [QMoE](https://arxiv.org/abs/2310.16795), [HOBBIT](https://arxiv.org/abs/2411.01433), [SqueezeLLM](https://proceedings.mlr.press/v235/kim24f.html), [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), [SpQR](https://openreview.net/pdf?id=Q1u25ahSuy).

**(d) Cost.** High: offline GPTQ/Hessian calibration, patch index, a low-bit GEMV, and a fused replacement-block path.

**(e) Fastest falsifier.** Produce the single curve `paired dNLL versus restored-block fraction q`. Kill if +0.01 requires \(q>0.12\) for a 1-bit base or \(q>0.19\) for a QMoE-like 0.8-bit base.

**(f) Basis.** This combines MoE-specific sub-bit QMoE with dense+sparse/outlier quantization. It is the closest weight-side analogue of a video codec’s low-quality reference frame plus residual enhancement data.

---

## 5. Route-union, utility-gated speculative verification

**(a) Mechanism.** Use a deeper draft, but verify layer-major: at each routed layer load the union of experts required by all draft tokens once, group tokens by expert, and dynamically disable or shorten speculation when `committed tokens / union expert-equivalents` drops below one.

This is different from the measured depth-1 MTP run: deeper blocks, union-deduplicated expert loading, adaptive \(K\), and an exact correcting verifier are the substance—not simply enabling the existing MTP head.

**(b) Effect.** Your 23.8% consecutive-token top-8 overlap means about 1.904 experts persist. Under a first-order approximation, four sequential tokens have union

\[
8+3(8-1.904)=26.29
\]

experts/layer, or 6.57 per token: 17.8% byte amortization. Ignoring drafter cost, four-token verification must commit more than

\[
26.29/8=3.29
\]

tokens—over 82.2%—just to break even. Deeper acceptance will probably be below that given the measured 72.5% depth-1 acceptance. That makes the expected win 0–15%, not 4×.

**(c) Fidelity.** A mathematically exact speculative verifier preserves the target distribution; implementation/RNG ordering may not be bit-identical. Gate with distributional equivalence and the existing paired dNLL test.

**(d) Cost.** Medium–high: multi-token execution, union routing, grouped expert kernels, and rejection rollback.

**(e) Fastest falsifier.** No implementation is initially needed. From route and acceptance traces, calculate per iteration

\[
U=\frac{8\times\text{committed tokens}}
        {\text{union expert count}}.
\]

Kill if median \(U\le1\) before adding drafter overhead.

**(f) Basis.** [Cascade](https://arxiv.org/abs/2506.20675) finds MoE speculation often moves 2–3× more weights and uses adaptive utility to obtain 7–14%; [SpecMoE](https://arxiv.org/abs/2604.10152) coalesces migration using self-assisted speculation, though its reported 4.3× result is not directly transferable.

---

## 6. Two-phase activation-sparse expert loading

**(a) Mechanism.** Fetch and compute the gate projection first. Use its SwiGLU activations to select intermediate channels, then fetch only corresponding up rows and down-column tiles. Down must be stored in a transposed or channel-tiled artifact.

**(b) Effect.** Approximating the three matrices as equal-byte components, retaining fraction \(p\) of intermediate channels costs

\[
B(p)=5.8\frac{1+2p}{3}.
\]

Examples:

- \(p=0.50\): 3.87 GB/token, −33%.
- \(p=0.25\): 2.90 GB/token, −50%.
- \(p\to0\): 1.93 GB/token floor because the full gate is still required.

It cannot reach 0.58 GB/token without also compressing or predicting the gate.

IQ2_XXS stores 256 weights in 66 bytes, as seen in the local [block definition](/home/bmarti44/spark-deepseek-v4-flash/vendor/ds4/ds4_cuda.cu:68). A 4096-wide row is only 1,056 bytes, so naïve O_DIRECT row reads suffer nearly 4× amplification. Channel tiles must be 4 KiB-aligned. Down columns are scattered under the current row-major layout, hence the repack requirement.

**(c) Fidelity.** SwiGLU is not ReLU: omitted channels are not exactly zero. TEAL reports 40–50% activation sparsity with modest benchmark loss, but that does not establish +0.01 dNLL. The two-phase I/O also does little for long prefill because the union of active channels over 2048 tokens will approach all channels. [TEAL](https://arxiv.org/abs/2408.14690), [CATS](https://arxiv.org/abs/2404.08763), [LLM in a Flash](https://arxiv.org/abs/2312.11514).

**(d) Cost.** Very high: new down layout, two-stage fetch scheduling, sparse/tiled IQ2 kernels, and likely per-layer thresholds.

**(e) Fastest falsifier.** Run oracle CATS: compute the full expert, then zero the lowest-contribution intermediate channels before down projection. Kill if even oracle \(p=0.5\) fails +0.01 dNLL.

**(f) Basis.** CATS is the closest match for SwiGLU; LLM in a Flash contributes the storage-side row/column bundling idea.

---

## 7. Shared centroid with activation-aware low-rank deltas

**(a) Mechanism.** Represent each matrix as \(W_i=W_b+U_iV_i\), with a Fisher/activation-weighted shared base resident per layer and only expert-specific factors selected. The shared gate/up projections are computed once; the shared down projection can be applied once to the router-weighted sum of expert intermediates.

This differs from the rejected common-right-subspace test: the base is additive and full-rank, while every expert has both unique left and right factors, fitted with activation/Fisher weighting.

**(b) Effect.** For \(p=2048,d=4096\), delta retention is

\[
\rho(r)=\frac{r(p+d)}{pd}.
\]

- \(r=256:\rho=18.75\%\), about 1.09 GB/token of expert deltas.
- \(r=512:\rho=37.5\%\), about 2.18 GB/token.
- A single base for every routed layer costs only \(75\times9.28\text{ MiB}=0.68\) GiB.

Static storage at \(r=512\) is approximately

\[
174\times0.375+0.68=65.9\text{ GiB},
\]

so every delta and base fits in 72 GiB. The zero-NVMe threshold is approximately \(r\le550\).

**(c) Fidelity.** Prior evidence is bad for your gate. At only 20% compression, D²-MoE changes Mixtral WikiText perplexity 3.98→4.65, corresponding to \(\Delta\text{NLL}\approx0.155\); DeepSeek-MoE moves 6.38→6.84, about 0.069. Both are far above +0.01, and \(r\approx512\) represents much stronger compression. [D²-MoE paper and tables](https://arxiv.org/html/2502.17298).

**(d) Cost.** Very high: offline Fisher merge, factorization, factor quantization, and fused base+delta kernels.

**(e) Fastest falsifier.** Measure paired dNLL as a function of activation-aware rank. Kill unless +0.01 is reached by \(r\le550\). Singular-energy retention alone is only a preliminary rejection test.

**(f) Basis.** [D²-MoE](https://proceedings.mlr.press/v267/gu25c.html). Elegant arithmetic, discouraging empirical prior.

---

## 8. Packed 2:4 sparsity plus a low-rank residual

**(a) Mechanism.** Enforce two nonzeros per four weights using second-order pruning, store a compact 2-bit value stream and a six-pattern selector, then recover structured pruning error with a small dense low-rank residual. All 256 experts and all eight routes remain; this is not REAP/expert pruning.

**(b) Effect.** With 2-bit nonzeros:

- Values: \(0.5\times2=1.0\) bpw.
- Pattern: \(\log_2(6)/4=0.646\) bpw.
- Rank-64, 4-bit residual: about \(0.1875\) bpw.

Total: approximately 1.834 bpw before scales, or

\[
5.8\times1.834/3.093=3.44\text{ GB/token},
\]

a 41% reduction. If surviving fidelity requires 4-bit nonzeros, the representation rises to roughly 2.834 bpw before scales—only an 8% reduction. At that point it is dead as a byte optimization.

**(c) Fidelity.** High risk because pruning an already extreme 2-bit artifact compounds errors. SparseGPT’s good 50–60% results were on much less compressed dense models and its 2:4 results were worse than unstructured pruning.

**(d) Cost.** High: prune from a good teacher, new packed format, custom sparse 2-bit kernels, residual kernels.

**(e) Fastest falsifier.** Apply 2:4 SparseGPT plus rank-64/128 residuals to representative layers with real activations. Kill if +0.01 requires 4-bit nonzeros or a residual above roughly rank 128.

**(f) Basis.** [SparseGPT](https://proceedings.mlr.press/v202/frantar23a.html) and sparse+dense decomposition methods such as SqueezeLLM.

---

## 9. Learned depth or layer skipping

**(a) Mechanism.** Train a per-token gate to bypass entire routed FFNs or transformer layers, preferably with distillation and a fixed compute budget.

**(b) Effect.**

\[
B(s)=5.8(1-s).
\]

Skipping 10% of routed layers gives 5.22 GB/token and at most +11%; 20% gives 4.64 GB and at most +25%. Matching 18.4 tok/s through layer skipping alone would require skipping approximately 87% of the routed layers, which is not credible.

**(c) Fidelity.** Very high risk as a post-hoc modification. Mixture-of-Depths and LayerSkip rely on training recipes; they are not evidence that a pretrained GLM checkpoint can safely skip layers. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [LayerSkip](https://arxiv.org/abs/2404.16710).

**(d) Cost.** Very high if trained correctly; moderate for a likely-failing post-hoc gate.

**(e) Fastest falsifier.** Oracle-test the least damaging subset of layers under teacher forcing. Kill if the best 10% subset already exceeds +0.01 dNLL.

**(f) Basis.** Adaptive-depth transformers. This is an incremental lever, not a byte-bound breaker.

---

## Explicit dead ends and bounds

### Four-request batching

For \(B\) independent requests with uniform top-8 routing, expected unique experts per layer are

\[
256\left[1-(1-8/256)^B\right].
\]

At \(B=4\), that is 30.53 experts instead of 32, or 7.63 per output token: only a 4.6% byte reduction, 5.8→5.53 GB/token. It may improve grouped-GEMM utilization, but it barely amortizes NVMe at four requests. At \(B=32\), the reduction becomes meaningful—about 5.1 experts/token—but at substantial batching latency and KV/state cost.

### Expert-pair or layer-group caching

Under the same byte capacity, a single-expert cache weakly dominates a pair cache: it can retain either member independently, while a pair cache must evict both. Pairing helps metadata and request coalescing, not hit-rate capacity. Your 23.8% temporal overlap is already captured by expert-granular caching.

### Router-aware file packing

A 9.28 MiB expert spans roughly 2,376 4 KiB pages; alignment waste is below 0.05%. Co-activation packing can turn several large random reads into a larger read, but cannot reduce payload bytes. It is useful only if the live miss trace fails to reach the measured 10.7 GB/s.

### Reading only “used rows” exactly

A dense gate/up/down GEMV uses every weight. IQ2 block structure allows whole-row reads, but there are no unused rows until activation sparsity is introduced. Down-projection channel selection is especially awkward because the relevant intermediate dimensions are columns in the current layout. Exact row elision is therefore impossible; CATS-style lossy activation sparsity is the actual proposal.

### GPUDirect Storage on Spark

Dead. NVIDIA explicitly states that [DGX Spark supports GDS only in compatibility mode and must not load `nvidia-fs`](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html). GDS P2P also does not support managed/system memory; compatibility mode uses internal bounce buffers. It cannot produce NVMe→GPU peer DMA on this machine.

### `io_uring` with registered buffers

Worth a microbenchmark, not a research program. Registered buffers reduce pinning and CPU overhead, but large 9 MiB expert transfers are bandwidth-dominated. Storage research finds `io_uring` substantially reduces cycles while increasing maximum throughput only slightly once O_DIRECT already saturates the device. [NVMe I/O-stack analysis](https://link.springer.com/chapter/10.1007/978-3-031-74097-8_9). Kill if an exact-miss-trace replay at QD16–32 does not beat the six-thread pool by at least 10%.

### AWQ saliency by itself

AWQ scaling does not reduce stored bytes; it makes a lower-bit representation more accurate. SqueezeLLM/SpQR outliers similarly add sparse high-precision data. They are enabling techniques for proposals 3 and 4, not independent byte levers.

### Video-codec-style expert-output memoization

Likely dead. Reusing an old expert output for a nearby hidden state saves 9.28 MiB, but for a linear projection

\[
Wx_t=Wx_{t-1}+W(x_t-x_{t-1}),
\]

and the residual multiplication still touches all weights. Approximate full-output reuse is lossy and high-dimensional hidden states lack video’s spatial correspondence. Even [DeepCache](https://arxiv.org/abs/1712.01670) explicitly avoids applying video motion heuristics directly to opaque high-dimensional internals.

## The experiment sequence I would actually run

1. **Offline, hours:** replay SLRU with prefill probation, TinyLFU admission, and Belady labels; simultaneously compute speculative union utility from existing route traces.
2. **Fidelity-only, no kernels:** generate three curves:
   - dNLL versus sub-bit patch fraction \(q\);
   - dNLL versus MoBE basis count/factor precision;
   - oracle CATS dNLL versus retained channel fraction.
3. **Exact systems work:** prototype direct cache-slot GEMV before touching `io_uring` or GDS.
4. **Only if the curves pass:** build either the MoBE+VQ resident artifact or the resident sub-bit+patch artifact.

My blunt recommendation: the two serious byte-bound bets are **MoBE+VQ residency** and **resident sub-bit plus salient restoration**. D² low-rank deltas have beautiful capacity arithmetic but published fidelity evidence that is already several times worse than your gate at far milder compression. Everything else is an incremental systems optimization or a trace-dependent amortization technique.
