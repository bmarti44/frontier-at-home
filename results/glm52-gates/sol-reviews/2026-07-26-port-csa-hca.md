# sol xhigh -- can DSV4's CSA/HCA sequence compression be ported to GLM-5.2?

VERDICT: NO. The compressor CODE could be transplanted; the compressor BEHAVIOUR
cannot. They are learned, stateful model components and GLM has neither their
tensors nor the attention/indexer co-adaptation.

PROOF IN SOURCE:
  - compression ratios are enabled only for the DeepSeek-4 family (ds4.c:1059)
  - GLM validation explicitly ZEROES the compression-ratio table (ds4.c:5644)
  - compressed DSV4 layers require learned compressor_kv, compressor_gate,
    compressor_ape and compressor_norm tensors; ratio-4 layers need a SECOND
    learned compressor for the indexer (ds4.c:4981)
  - at runtime hidden states go through learned KV and gate matrices, learned
    positional terms, persistent state update, learned-score softmax pooling,
    RMS norm and RoPE (ds4.c:12381). Not pooling, not an interchangeable codec.

COST OF A REFIT ON FROZEN GLM: ~0.9B parameters (38 CSA + 38 HCA over 78 layers;
honest range 0.8-0.95B). ~15 GB for weights+grads+Adam before frozen-model
residency. Captured hidden states cost ~89 GiB per 100k tokens and ~0.89 TiB per
1M. One 1M-token teacher pass is 8.7-12.1 h at GLM's 23-32 tok/s; a 10M corpus
is 3.6-5 DAYS per forward pass. One Spark suffices for a 100K-1M feasibility
pilot; a credible refit needs an 8-32 GPU cluster. A least-squares/SVD fit can
only initialise the KV projection -- gates, softmax pooling, ring state, APE,
normalisation and recurrence make the objective nonlinear, and one 128-token
summary cannot reproduce the query-dependent outputs of 128 addressable rows.

THE DSA CONFLICT IS FUNDAMENTAL. GLM's full indexer runs on 21 of 78 layers,
scoring every visible row then taking top-2048, and that selection is reused by
following layers. With 128:1 HCA there are only three options:
  1. keep the per-token indexer, drop per-token cKV -> DSA can select token
     543,210 but no exact row exists to attend
  2. compress the indexer too -> the learned scorer operates on representations
     it was never trained against
  3. keep exact cKV elsewhere -> that is storage tiering, not compression
AND "compress only where selection is rare" does NOT preserve retrieval:
aggregate rarity is irrelevant to needle retrieval, because the one old row
selected for the first time may hold the answer. Heavy-hitter methods protect
previously important rows, not rows whose importance appears under a future
query.

CORRECTION TO MY EVICTION MATH: reaching 6.9 KiB/token requires retaining at
most 0.84% of GLM rows, not 3.6%, because the 5.34 KiB/token misc term is fixed:
B = 5.34 + 186f KiB/token.

ONLY EXACT TRAINING-FREE ANSWER: DSA-aware cKV offload to NVMe -- 15.84 KiB/token
resident, 175.5 KiB/token on disk, exact arbitrary-row retrieval preserved.


codex
## Bottom line

No direct, training-free port exists. The compressor code can be transplanted; the compressor behavior cannot. DSV4’s CSA/HCA modules are learned, stateful model components, and GLM-5.2 has neither their tensors nor attention/indexer co-adaptation.

A frozen-GLM refit is conceivable, but it is roughly a 0.9B-parameter post-training project, not least-squares plumbing. One Spark could run a small layerwise experiment; producing a credible 128:1 model that passes `dNLL ≤ 0.01` and retrieval tests needs substantial distillation work and probably a GPU cluster.

No currently published training-free eviction/merging method reaches about 6.9 KiB/token while preserving arbitrary distant-token retrieval. The only exact training-free answer is tiering: keep GLM’s complete indexer resident and store every exact cKV row on NVMe. That reduces RAM, not total storage, and appears impractical for dynamically constructed 1M contexts.

## 1. Direct port: no; refit: theoretically yes, practically difficult

Your learned-compressor conclusion is correct.

The inspected engine proves that:

- Compression ratios are enabled only for the DeepSeek-4 family; other families return zero [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:1059).
- GLM-5.2 is defined as 79 layers including one next-token-prediction layer, so 78 ordinary cache layers, with a 576-wide latent and top-2048 DSA [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:613).
- GLM validation explicitly zeroes the compression-ratio table [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:5644).
- Compressed DSV4 layers require learned `compressor_kv`, `compressor_gate`, `compressor_ape`, and `compressor_norm` tensors. Ratio-4 layers also require a second learned compressor for the indexer [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:4981).
- At runtime, hidden states are projected through learned KV and gate matrices; learned positional terms are added; persistent state is updated; then learned-score softmax pooling, RMS normalization and RoPE are applied [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:12381).

That is not average pooling or an interchangeable cache codec.

I could not read `/home/dsv4/llamacpp-project/src/llama.cpp/src/llama-kv-cache-dsv4.cpp`: access was denied. Therefore I cannot independently verify the particular `build_overlap_compressed_kv_from_state` implementation. The accessible engine independently establishes the important conclusion.

### Size of a GLM-shaped refit

Assuming the DSV4 alternating layout is copied onto GLM’s 78 normal layers—two raw, then 38 CSA and 38 HCA—the approximate new parameter count is:

| Component | Parameters/layer |
|---|---:|
| CSA attention compressor, width `2×576` | 14.16M |
| CSA indexer compressor, width `2×128` | 3.15M |
| CSA total | 17.31M |
| HCA attention compressor, width `576` plus 128-position APE | 7.15M |
| **38 CSA + 38 HCA** | **929M parameters** |

That is an estimate from the DSV4 tensor shapes applied to GLM’s `n_embd=6144`; the final topology is undefined, so 0.8–0.95B is the honest range.

BF16 weights, gradients, FP32 master weights and Adam moments would consume roughly 15 GB before frozen-model residency and activations.

Captured BF16 hidden states alone cost:

- About 12 KiB/token/layer.
- About 89 GiB for 100,000 tokens across 78 layers.
- About 0.89 TiB for 1M tokens across 78 layers.
- Comparable additional space if teacher attention outputs are retained.

They can be streamed layer by layer, but the temporal HCA objective needs contiguous sequences, not shuffled independent activations.

A simple least-squares/SVD fit could initialize the KV projection. It cannot fit the whole operation because the learned gates, softmax pooling, ring state, positional embeddings, normalization and downstream recurrence make the objective nonlinear. More importantly, one 128-token summary cannot reproduce all query-dependent attention outputs of 128 independently addressable rows. The frozen downstream network must learn to tolerate that loss—or the compressor and downstream attention/indexer must be jointly distilled.

[CSKV](https://arxiv.org/abs/2409.10593) shows that layerwise reconstruction is a real technique: its 7B experiment used SVD initialization and layerwise MSE, reportedly training with 256 calibration samples in about 90 minutes on one A100. But that was approximately 80% compression, not stateful 128:1 compression inside a 753B MoE with a separately trained sparse indexer.

At your measured 23–32 prefill tok/s:

- One 1M-token teacher pass takes roughly 8.7–12.1 hours.
- A 10M-token corpus takes roughly 3.6–5 days per forward pass.
- Several teacher passes, truncated BPTT, rollout validation and indexer retraining turn this into weeks on one Spark.

Verdict: one Spark is adequate for a 100K–1M-token feasibility pilot. A credible model-quality refit likely needs an 8–32-GPU-class cluster. The exact requirement is unverifiable because no comparable GLM-5.2 compressor-refit recipe has been published.

## 2. DSA versus HCA: the conflict is fundamental

GLM’s full indexer is used in 21 of the 78 normal layers: the first three, then every fourth layer beginning at layer 6 [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:34421). At those layers the engine scores every visible indexer row and only then takes top-2048 [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:45976). That selection is reused by following layers.

With 128:1 HCA, three possibilities exist:

1. Keep the per-token indexer but discard per-token cKV.  
   DSA can select token 543,210, but there is no exact cKV row to attend. Mapping it to a 128-token summary changes the model semantics.

2. Compress the indexer too.  
   The candidate universe changes from individual tokens to summaries. GLM’s learned query/indexer scorer is now operating on representations it was never trained against.

3. Keep exact cKV somewhere else.  
   DSA remains coherent, but this is storage tiering, not sequence compression.

So yes: naïvely adding HCA defeats DSA’s central property.

“Compress only beyond a distance where selection is rare” is coherent as an approximate heuristic, but it does not preserve retrieval. Aggregate rarity is irrelevant to needle retrieval: the one old row selected for the first time may contain the answer. Heavy-hitter exceptions protect previously important rows, not an old row whose importance appears only under a future query.

A coherent exact design is:

- Keep all F32 indexer rows in RAM.
- Score all rows exactly.
- Fetch the selected exact cKV rows from NVMe.
- Reuse the selected/staged rows through that DSA layer group.

That preserves baseline row availability. Without backing storage, no HCA-like design can do so.

Your “co-trained consistency” point is correct globally, with one nuance: in this implementation the separate learned indexer compressor occurs on 4:1 CSA layers; 128:1 HCA layers retain only the attention compressor. DSV4’s complete attention schedule was nevertheless trained jointly around those representations.

## 3. Training-free alternatives, ranked against your fidelity gate

For GLM-specific projections, the useful cost equation is:

\[
B = 5.34 + 175.5f_{\rm cKV} + 10.5f_{\rm indexer}
\quad\text{KiB/token}.
\]

If an eviction method retains the same fraction \(f\) of cKV and indexer rows:

\[
B = 5.34 + 186f.
\]

Therefore reaching 6.9 KiB/token requires retaining at most **0.84%** of GLM rows—not 3.6%—because the 5.34 KiB/token miscellaneous term remains. Published results are on other models and do not establish GLM’s `dNLL ≤ 0.01`.

| Rank | Method | Projected GLM cost | Arbitrary old row? | Retrieval evidence |
|---:|---|---:|---|---|
| 1 | Exact DSA-aware cKV offload | **15.84 KiB/t resident slope**, 175.5 KiB/t on disk | **Yes, exact** | Same computation if implemented correctly; no published GLM result. Practical local working set was estimated at 19.14 GiB plus ~167 GiB disk for 1M decimal tokens. |
| 2 | Full-history low-bit KV | Project estimate **~46 KiB/t** for FP4-class storage | Every row remains, values approximate | [TurboQuant](https://arxiv.org/abs/2504.19874) reported NIAH `0.997`, equal to full cache `0.997`, on Llama-3.1-8B over 4K–104K. GLM DSA-margin stability and dNLL are unverifiable. |
| 3 | KVTC transform coding | Keeping F32 indexer resident: **24.6/21.3/18.6 KiB/t** at 20×/32×/64× cKV coding | Logically yes after decoding; not bit-exact | [KVTC](https://arxiv.org/abs/2511.01815) reported NIAH 99.8 vs 100 at about 32× and 99.6 at 64×, but LITM fell to 90.2 at 64×. It is currently a storage/TTFT codec that decompresses before attention, not online compressed GLM attention. |
| 4 | PyramidKV | **27.66 KiB/t** at 12%; **8.25** at 1.56%; **6.64** at 0.7% | **No** | In [PyramidKV](https://arxiv.org/abs/2406.02069), at 128 rows: Mistral-7B/32K NIAH was 91.6 vs 100 full; Llama-3-8B/8K was 97.4; Llama-3-70B/8K reached 100. Query/model dependent, no guarantee. |
| 5 | CompressKV | **10.92 KiB/t** at 3%; **6.64** at 0.7% | **No** | [CompressKV](https://arxiv.org/abs/2606.24467) reports over 97% of full-cache LongBench QA at 3%, but only about **90% NIAH at 0.7%**. The setting that reaches your memory target fails the retrieval gate. GLM MLA/DSA compatibility is unverified. |
| 6 | SnapKV | About **5.53 KiB/t** with 1,024 retained rows at 1M | **No** | In PyramidKV’s comparative table, SnapKV scored 80.1 on Mistral-7B/32K at a 128-row budget. TurboQuant’s comparison reports 0.858 at a 25% budget. It selects for the current prompt/query; discarded rows cannot serve unforeseen future queries. [SnapKV](https://arxiv.org/abs/2404.14469) |
| 7 | H2O heavy hitters | **42.54 KiB/t** at its common 20% budget; 6.9 requires only 0.84% | **No** | The original [H2O](https://arxiv.org/abs/2306.14048) paper does not publish a needle result sufficient for this gate. PyramidKV’s table gives 64.9 on Mistral-7B/32K and 49.1 on Llama-3-8B/8K at 128 rows. Future-query needles need not already be heavy hitters. |
| 8 | KVMerger / CaM | **70.44 KiB/t** at 35%; **98.34** at 50% | **No** | [KVMerger](https://arxiv.org/abs/2407.08454) includes short-context NIAH heatmaps and reports improvement over CaM/H2O, but no adequate aggregate number for long dynamic contexts. Fidelity at the required 0.84% is **unverifiable**. [CaM](https://proceedings.mlr.press/v235/zhang24n.html) preserves merged value information, not individual keys. |
| 9 | StreamingLLM sinks + window | **5.53 KiB/t** for 1,024 rows; **11.43** for a 32K window at 1M | **No, structurally impossible outside window** | A 128K test in [KV-Fold](https://arxiv.org/abs/2605.12471) found 3/3 retrieval inside StreamingLLM’s window and **0/3** for needles at older tested depths. [StreamingLLM](https://arxiv.org/abs/2309.17453) targets stable streaming perplexity, not old-token retrieval. |

ShadowKV-like host offload is also relevant: [ShadowKV](https://proceedings.mlr.press/v267/sun25b.html) preserves full logical history and reports strong RULER/NIAH results, including GLM-4-9B-1M. But its separate K/V low-rank reconstruction and host-memory assumptions do not map cleanly onto GLM-5.2’s MLA+DSA. On a Spark, CPU and GPU share the same 128 GB physical memory, so “offload to host RAM” does not create capacity. Its natural GLM equivalent is the NVMe cKV design above.

None of these publications reports your exact paired GLM-5.2 `dNLL ≤ 0.01 nat/token` gate. Every such claim is therefore unverifiable until measured locally.

## 4. Honest recommendation

My ranking is:

1. **“GLM-5.2 is architecturally the wrong model for long context on this box; use DSV4 for long context and GLM for what it is better at.”**  
   This ranks first by a wide margin. DSV4 has about 28× lower architectural cache cost and roughly 9–12× faster measured prefill. GLM cannot erase both disadvantages with a cache plugin.

2. **For GLM, build only a bounded-context, full-history quantization path.**  
   Preserve every row, keep the indexer F32 initially, and test FP8/FP4 cKV through paired teacher-forced dNLL, top-2048 overlap/margin crossings, and old/middle needle retrieval. This may make 64K–128K useful. The actual ceiling is conditional on fidelity; do not advertise it before the gate passes.

3. **If 1M GLM has research value, prototype exact cKV offload only after an I/O trace-replay benchmark.**  
   The local model estimates roughly 0.495 GB of aligned cKV reads and approximately 43,000 random reads per generated token. The idealized result is about 2 tok/s; actual performance could be around 1 tok/s or worse and is currently unverifiable. Offloading the indexer drops the ideal ceiling toward 0.65–0.7 tok/s and should be rejected. See the existing [1M analysis](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/sol-reviews/2026-07-26-1M-solve.md:261).

4. **Do not build a StreamingLLM/H2O/SnapKV/PyramidKV “1M mode” under the stated product gate.**  
   Those can display a 1M position counter while having irreversibly deleted most of the middle. That is not the capability you asked for.

For dynamic construction, 1M GLM is especially unattractive: even naïve extrapolation gives **8.7–12.1 hours of prefill**, while the all-row indexer’s cumulative traffic is quadratic and has a modeled 5.5-hour byte-read floor by itself. Exact offload is primarily defensible for restored/prebuilt contexts—the use case you explicitly do not want.

So: a small full-history quantization experiment is worth building. A production GLM 1M path with your fidelity gate is not. DSV4 is the correct long-context architecture on this machine.
tokens used
323,768
## Bottom line

No direct, training-free port exists. The compressor code can be transplanted; the compressor behavior cannot. DSV4’s CSA/HCA modules are learned, stateful model components, and GLM-5.2 has neither their tensors nor attention/indexer co-adaptation.

A frozen-GLM refit is conceivable, but it is roughly a 0.9B-parameter post-training project, not least-squares plumbing. One Spark could run a small layerwise experiment; producing a credible 128:1 model that passes `dNLL ≤ 0.01` and retrieval tests needs substantial distillation work and probably a GPU cluster.

No currently published training-free eviction/merging method reaches about 6.9 KiB/token while preserving arbitrary distant-token retrieval. The only exact training-free answer is tiering: keep GLM’s complete indexer resident and store every exact cKV row on NVMe. That reduces RAM, not total storage, and appears impractical for dynamically constructed 1M contexts.

## 1. Direct port: no; refit: theoretically yes, practically difficult

Your learned-compressor conclusion is correct.

The inspected engine proves that:

- Compression ratios are enabled only for the DeepSeek-4 family; other families return zero [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:1059).
- GLM-5.2 is defined as 79 layers including one next-token-prediction layer, so 78 ordinary cache layers, with a 576-wide latent and top-2048 DSA [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:613).
- GLM validation explicitly zeroes the compression-ratio table [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:5644).
- Compressed DSV4 layers require learned `compressor_kv`, `compressor_gate`, `compressor_ape`, and `compressor_norm` tensors. Ratio-4 layers also require a second learned compressor for the indexer [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:4981).
- At runtime, hidden states are projected through learned KV and gate matrices; learned positional terms are added; persistent state is updated; then learned-score softmax pooling, RMS normalization and RoPE are applied [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:12381).

That is not average pooling or an interchangeable cache codec.

I could not read `/home/dsv4/llamacpp-project/src/llama.cpp/src/llama-kv-cache-dsv4.cpp`: access was denied. Therefore I cannot independently verify the particular `build_overlap_compressed_kv_from_state` implementation. The accessible engine independently establishes the important conclusion.

### Size of a GLM-shaped refit

Assuming the DSV4 alternating layout is copied onto GLM’s 78 normal layers—two raw, then 38 CSA and 38 HCA—the approximate new parameter count is:

| Component | Parameters/layer |
|---|---:|
| CSA attention compressor, width `2×576` | 14.16M |
| CSA indexer compressor, width `2×128` | 3.15M |
| CSA total | 17.31M |
| HCA attention compressor, width `576` plus 128-position APE | 7.15M |
| **38 CSA + 38 HCA** | **929M parameters** |

That is an estimate from the DSV4 tensor shapes applied to GLM’s `n_embd=6144`; the final topology is undefined, so 0.8–0.95B is the honest range.

BF16 weights, gradients, FP32 master weights and Adam moments would consume roughly 15 GB before frozen-model residency and activations.

Captured BF16 hidden states alone cost:

- About 12 KiB/token/layer.
- About 89 GiB for 100,000 tokens across 78 layers.
- About 0.89 TiB for 1M tokens across 78 layers.
- Comparable additional space if teacher attention outputs are retained.

They can be streamed layer by layer, but the temporal HCA objective needs contiguous sequences, not shuffled independent activations.

A simple least-squares/SVD fit could initialize the KV projection. It cannot fit the whole operation because the learned gates, softmax pooling, ring state, positional embeddings, normalization and downstream recurrence make the objective nonlinear. More importantly, one 128-token summary cannot reproduce all query-dependent attention outputs of 128 independently addressable rows. The frozen downstream network must learn to tolerate that loss—or the compressor and downstream attention/indexer must be jointly distilled.

[CSKV](https://arxiv.org/abs/2409.10593) shows that layerwise reconstruction is a real technique: its 7B experiment used SVD initialization and layerwise MSE, reportedly training with 256 calibration samples in about 90 minutes on one A100. But that was approximately 80% compression, not stateful 128:1 compression inside a 753B MoE with a separately trained sparse indexer.

At your measured 23–32 prefill tok/s:

- One 1M-token teacher pass takes roughly 8.7–12.1 hours.
- A 10M-token corpus takes roughly 3.6–5 days per forward pass.
- Several teacher passes, truncated BPTT, rollout validation and indexer retraining turn this into weeks on one Spark.

Verdict: one Spark is adequate for a 100K–1M-token feasibility pilot. A credible model-quality refit likely needs an 8–32-GPU-class cluster. The exact requirement is unverifiable because no comparable GLM-5.2 compressor-refit recipe has been published.

## 2. DSA versus HCA: the conflict is fundamental

GLM’s full indexer is used in 21 of the 78 normal layers: the first three, then every fourth layer beginning at layer 6 [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:34421). At those layers the engine scores every visible indexer row and only then takes top-2048 [here](/tmp/claude-1000/-home-bmarti44-spark-deepseek-v4-flash/3109d29d-cd62-4998-a2e1-bfb6e29288b5/scratchpad/engine-src/ds4.c:45976). That selection is reused by following layers.

With 128:1 HCA, three possibilities exist:

1. Keep the per-token indexer but discard per-token cKV.  
   DSA can select token 543,210, but there is no exact cKV row to attend. Mapping it to a 128-token summary changes the model semantics.

2. Compress the indexer too.  
   The candidate universe changes from individual tokens to summaries. GLM’s learned query/indexer scorer is now operating on representations it was never trained against.

3. Keep exact cKV somewhere else.  
   DSA remains coherent, but this is storage tiering, not sequence compression.

So yes: naïvely adding HCA defeats DSA’s central property.

“Compress only beyond a distance where selection is rare” is coherent as an approximate heuristic, but it does not preserve retrieval. Aggregate rarity is irrelevant to needle retrieval: the one old row selected for the first time may contain the answer. Heavy-hitter exceptions protect previously important rows, not an old row whose importance appears only under a future query.

A coherent exact design is:

- Keep all F32 indexer rows in RAM.
- Score all rows exactly.
- Fetch the selected exact cKV rows from NVMe.
- Reuse the selected/staged rows through that DSA layer group.

That preserves baseline row availability. Without backing storage, no HCA-like design can do so.

Your “co-trained consistency” point is correct globally, with one nuance: in this implementation the separate learned indexer compressor occurs on 4:1 CSA layers; 128:1 HCA layers retain only the attention compressor. DSV4’s complete attention schedule was nevertheless trained jointly around those representations.

## 3. Training-free alternatives, ranked against your fidelity gate

For GLM-specific projections, the useful cost equation is:

\[
B = 5.34 + 175.5f_{\rm cKV} + 10.5f_{\rm indexer}
\quad\text{KiB/token}.
\]

If an eviction method retains the same fraction \(f\) of cKV and indexer rows:

\[
B = 5.34 + 186f.
\]

Therefore reaching 6.9 KiB/token requires retaining at most **0.84%** of GLM rows—not 3.6%—because the 5.34 KiB/token miscellaneous term remains. Published results are on other models and do not establish GLM’s `dNLL ≤ 0.01`.

| Rank | Method | Projected GLM cost | Arbitrary old row? | Retrieval evidence |
|---:|---|---:|---|---|
| 1 | Exact DSA-aware cKV offload | **15.84 KiB/t resident slope**, 175.5 KiB/t on disk | **Yes, exact** | Same computation if implemented correctly; no published GLM result. Practical local working set was estimated at 19.14 GiB plus ~167 GiB disk for 1M decimal tokens. |
| 2 | Full-history low-bit KV | Project estimate **~46 KiB/t** for FP4-class storage | Every row remains, values approximate | [TurboQuant](https://arxiv.org/abs/2504.19874) reported NIAH `0.997`, equal to full cache `0.997`, on Llama-3.1-8B over 4K–104K. GLM DSA-margin stability and dNLL are unverifiable. |
| 3 | KVTC transform coding | Keeping F32 indexer resident: **24.6/21.3/18.6 KiB/t** at 20×/32×/64× cKV coding | Logically yes after decoding; not bit-exact | [KVTC](https://arxiv.org/abs/2511.01815) reported NIAH 99.8 vs 100 at about 32× and 99.6 at 64×, but LITM fell to 90.2 at 64×. It is currently a storage/TTFT codec that decompresses before attention, not online compressed GLM attention. |
| 4 | PyramidKV | **27.66 KiB/t** at 12%; **8.25** at 1.56%; **6.64** at 0.7% | **No** | In [PyramidKV](https://arxiv.org/abs/2406.02069), at 128 rows: Mistral-7B/32K NIAH was 91.6 vs 100 full; Llama-3-8B/8K was 97.4; Llama-3-70B/8K reached 100. Query/model dependent, no guarantee. |
| 5 | CompressKV | **10.92 KiB/t** at 3%; **6.64** at 0.7% | **No** | [CompressKV](https://arxiv.org/abs/2606.24467) reports over 97% of full-cache LongBench QA at 3%, but only about **90% NIAH at 0.7%**. The setting that reaches your memory target fails the retrieval gate. GLM MLA/DSA compatibility is unverified. |
| 6 | SnapKV | About **5.53 KiB/t** with 1,024 retained rows at 1M | **No** | In PyramidKV’s comparative table, SnapKV scored 80.1 on Mistral-7B/32K at a 128-row budget. TurboQuant’s comparison reports 0.858 at a 25% budget. It selects for the current prompt/query; discarded rows cannot serve unforeseen future queries. [SnapKV](https://arxiv.org/abs/2404.14469) |
| 7 | H2O heavy hitters | **42.54 KiB/t** at its common 20% budget; 6.9 requires only 0.84% | **No** | The original [H2O](https://arxiv.org/abs/2306.14048) paper does not publish a needle result sufficient for this gate. PyramidKV’s table gives 64.9 on Mistral-7B/32K and 49.1 on Llama-3-8B/8K at 128 rows. Future-query needles need not already be heavy hitters. |
| 8 | KVMerger / CaM | **70.44 KiB/t** at 35%; **98.34** at 50% | **No** | [KVMerger](https://arxiv.org/abs/2407.08454) includes short-context NIAH heatmaps and reports improvement over CaM/H2O, but no adequate aggregate number for long dynamic contexts. Fidelity at the required 0.84% is **unverifiable**. [CaM](https://proceedings.mlr.press/v235/zhang24n.html) preserves merged value information, not individual keys. |
| 9 | StreamingLLM sinks + window | **5.53 KiB/t** for 1,024 rows; **11.43** for a 32K window at 1M | **No, structurally impossible outside window** | A 128K test in [KV-Fold](https://arxiv.org/abs/2605.12471) found 3/3 retrieval inside StreamingLLM’s window and **0/3** for needles at older tested depths. [StreamingLLM](https://arxiv.org/abs/2309.17453) targets stable streaming perplexity, not old-token retrieval. |

ShadowKV-like host offload is also relevant: [ShadowKV](https://proceedings.mlr.press/v267/sun25b.html) preserves full logical history and reports strong RULER/NIAH results, including GLM-4-9B-1M. But its separate K/V low-rank reconstruction and host-memory assumptions do not map cleanly onto GLM-5.2’s MLA+DSA. On a Spark, CPU and GPU share the same 128 GB physical memory, so “offload to host RAM” does not create capacity. Its natural GLM equivalent is the NVMe cKV design above.

None of these publications reports your exact paired GLM-5.2 `dNLL ≤ 0.01 nat/token` gate. Every such claim is therefore unverifiable until measured locally.

## 4. Honest recommendation

My ranking is:

1. **“GLM-5.2 is architecturally the wrong model for long context on this box; use DSV4 for long context and GLM for what it is better at.”**  
   This ranks first by a wide margin. DSV4 has about 28× lower architectural cache cost and roughly 9–12× faster measured prefill. GLM cannot erase both disadvantages with a cache plugin.

2. **For GLM, build only a bounded-context, full-history quantization path.**  
   Preserve every row, keep the indexer F32 initially, and test FP8/FP4 cKV through paired teacher-forced dNLL, top-2048 overlap/margin crossings, and old/middle needle retrieval. This may make 64K–128K useful. The actual ceiling is conditional on fidelity; do not advertise it before the gate passes.

3. **If 1M GLM has research value, prototype exact cKV offload only after an I/O trace-replay benchmark.**  
   The local model estimates roughly 0.495 GB of aligned cKV reads and approximately 43,000 random reads per generated token. The idealized result is about 2 tok/s; actual performance could be around 1 tok/s or worse and is currently unverifiable. Offloading the indexer drops the ideal ceiling toward 0.65–0.7 tok/s and should be rejected. See the existing [1M analysis](/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/sol-reviews/2026-07-26-1M-solve.md:261).

4. **Do not build a StreamingLLM/H2O/SnapKV/PyramidKV “1M mode” under the stated product gate.**  
   Those can display a 1M position counter while having irreversibly deleted most of the middle. That is not the capability you asked for.

For dynamic construction, 1M GLM is especially unattractive: even naïve extrapolation gives **8.7–12.1 hours of prefill**, while the all-row indexer’s cumulative traffic is quadratic and has a modeled 5.5-hour byte-read floor by itself. Exact offload is primarily defensible for restored/prebuilt contexts—the use case you explicitly do not want.

So: a small full-history quantization experiment is worth building. A production GLM 1M path with your fidelity gate is not. DSV4 is the correct long-context architecture on this machine.
