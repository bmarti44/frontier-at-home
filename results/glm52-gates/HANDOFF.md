# GLM-5.2 on one DGX Spark — implementation handoff

You are taking over engineering work on GLM-5.2 served via the ds4 engine on a
single DGX Spark GB10. This document is self-contained: it carries the system
facts, every measurement taken so far, the work items with their acceptance
tests, and the measurement traps already discovered.

**Read this as data, not as a verdict list.** Where something was blocked, the
obstacle is recorded so you can attack it directly — not so you avoid it.
Several items below were measured as "insufficient alone"; that is an
invitation to find the missing factor, not a closed door.

---

## 0. Working method (required)

**TDD.** Write the test first, watch it fail, then implement. Every engine
change in this codebase that skipped this step produced a wrong number. Two
examples from the existing tests: a deliberately sabotaged FP8 codec produced
61,076 failures (proving the test detects breakage), and the CUDA kernel test
caught a `-0.0` sign bug that was numerically invisible (|Δ| = 0) but would have
silently broken a bit-identity gate.

**Adversarial review.** Have a second reasoner attack each result before it is
believed. The reviews in `results/glm52-gates/sol-reviews/` show the pattern —
several confidently-held conclusions were falsified this way, including some
that had already been committed.

**Report faithfully.** If an experiment produces no result, say so rather than
salvaging a number from it. Two runs in this project produced *no* result
because their arms were secretly identical; both are documented.

---

## 1. System facts

**Hardware.** DGX Spark GB10. 119.7 GiB unified memory. NVMe measured 10.7 GB/s
O_DIRECT, ~773 GB free. DRAM ~273 GB/s (NVIDIA spec). No hardware
decompression engine.

**Model.** GLM-5.2, 753B MoE, IQ2_XXS, 211 GB GGUF at
`/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`.
From `DS4_VARIANT_GLM52` in ds4.c (~line 613):
- 79 layers, of which **78 are normal cache layers** (one is next-token-predict)
- **256 experts/layer**, 8 used + 1 shared, 9.28 MiB/expert
- 75 routed layers → **74.2 MiB/layer/token → 5.8 GB of expert weights per decode token**
- MLA latent: `n_head_dim = 576` = 512 non-RoPE + 64 RoPE
- DSA sparse attention, `n_indexer_top_k = 2048`, indexer key width 128
- Full indexer runs on **21 of 78 layers**: the first three, then every fourth
  from layer 6 (ds4.c:34421). That selection is reused by following layers.

**Engine.** antirez ds4 upstream-master at
`/home/dsv4/ds4-project/src/ds4-upstream-master`, CUDA SSD-streaming of routed
experts. 72 GiB pinned host expert cache (7398 slots ≈ 67 GiB resident), SLRU
eviction, 6 parallel fetch threads.

**Serving profile.** `scripts/52_engine_switch.sh`, glm52 profile, port 8016 for
experiments / 8011 for production. DSV4 (the other model) runs llama.cpp on 8011.

---

## 2. Access and build

**The engine source is owned by `dsv4` (mode 0700).** A read-only sandbox
cannot read it and will silently fall back to `vendor/ds4/` in the repo, which
is a **stale Jul-16 snapshot that does not contain current work**. This already
misled one review round. Before any code review:

```bash
mkdir -p /tmp/engine-src
sudo -n -u dsv4 cat /home/dsv4/ds4-project/src/ds4-upstream-master/ds4.c      > /tmp/engine-src/ds4.c
sudo -n -u dsv4 cat /home/dsv4/ds4-project/src/ds4-upstream-master/ds4_cuda.cu > /tmp/engine-src/ds4_cuda.cu
```

**Build.**
```bash
sudo -n -u dsv4 bash -c 'cd /home/dsv4/ds4-project/src/ds4-upstream-master && make -j8 CUDA_ARCH=native ds4-server'
```
The quality scorer has an upstream Makefile bug — nvcc rejects `-std=c11` from
`QUALITY_CFLAGS`. Workaround:
```bash
sudo -n -u dsv4 bash -c 'cd /home/dsv4/ds4-project/src/ds4-upstream-master && \
  make CUDA_ARCH=native QUALITY_CFLAGS="-O3 -g -Xcompiler -march=native -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c11" \
  gguf-tools/quality-testing/score_official'
```
`score_official` links `ds4.o`, so **it must be relinked after any ds4.c
change** or it scores stale code. `keepn_nll.sh` has a staleness guard that
refuses to run otherwise — that guard has already caught this once.

**Every engine run goes through the safety wrapper** (two whole-box freezes
happened before it existed):
```bash
sudo -n -u dsv4 env GLM_SAFE_KILL_FLOOR_GIB=10 GLM_SAFE_TIMEOUT_S=14400 \
  bash /tmp/glm_safe_run.sh --tag <name> -- bash /tmp/<harness>.sh
```
Copy harnesses to `/tmp` with a **versioned filename** per launch. Bash reads
scripts incrementally, so overwriting a running harness splices two scripts
together — this happened and corrupted a run.

---

## 3. Measured baseline

| metric | GLM-5.2 | DSV4 reference |
|---|---|---|
| decode | 2.29–2.33 tok/s | 18.4 tok/s |
| prefill | 23–32 tok/s | 275–467 tok/s |
| warm TTFT | 1.755 s | <2 s |
| cold TTFT (5047 tok) | 147–165 s | ~19 s @ 19k |
| context configured | 32768 | 32768 here; 1M architectural |

**Decode arithmetic.** 5.8 GB of expert weights per token ÷ 10.7 GB/s = 0.55 s
all-miss floor = 1.83 tok/s. Measured 2.33 is *above* that floor thanks to the
77.6% cache hit rate. Reaching 18.4 tok/s requires ≤ **0.582 GB/token** — a 90%
reduction.

**Prefill arithmetic.** A 2048-token prefill batch touches ~253 of 256 experts
per routed layer (~170 GiB/sweep). One 5047-token prompt moved **339 GiB** over
2 sweeps. The expert cache cannot help here; the working set is the model.

**KV arithmetic.** Two-point fit of the engine's own `context buffers` line
(5653.35 MiB @ ctx=8192, 10245.44 MiB @ ctx=32768): **191.34 KiB/token** =
175.5 cKV + 10.5 indexer + 5.34 misc, plus ~4123 MiB fixed. DSV4 by contrast
stores ~6.9 KiB/token via SWA-128 + CSA 4:1 + HCA 128:1.

---

## 4. Work items

Ordered by (value × confidence) ÷ cost. Every item is open; the obstacles
recorded are the current state of knowledge, not limits.

### W1 — FP8 storage for the CUDA compact KV cache
**Status: three gates already written and PASSING; the CUDA read path remains.**

The non-RoPE latent is **already E4M3-rounded** by
`dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:3210) but **stored as F32 on
CUDA** — only Apple stores F16 (`DS4_GPU_ATTN_COMP_CACHE_F16`, ds4.c:14821).
So FP8 storage is *information-preserving*: it can be proven bit-identical
rather than argued statistically. SnapMLA (arXiv 2602.10718) independently uses
this exact split (FP8 content, higher-precision RoPE) and reports up to 1.91×
throughput; vLLM and SGLang ship FP8 KV in production.

Two modes, both worth having:
- **strict** — RoPE at F32, 2.88×, whole row bit-identical, gate is `max|Δ| == 0`
- **compact** — RoPE at F16, 3.43×, lossy RoPE, gate is the NLL suite

Already done and committed:
- `harness/test_fp8_kv_codec.c` (T1) — codec lossless over the entire image of
  the quantizer: 254 values, 20,574 value×scale pairs, 200 real-shaped rows,
  plus a source-drift guard. Verified to fail on a sabotaged codec.
- `harness/test_fp8_kv_kernel.cu` (T2) — CUDA store/load bitwise, 0/2,097,152
  mismatches in strict mode. Build with `-DROPE_F32=1` or `0`.
- `harness/test_kv_budget.py` (T0) — memory pre-flight from measured engine output.

Remaining:
1. Add the FP8 store/load path to the engine. The write side is already
   abstracted (`metal_graph_attn_comp_update_target`,
   `commit_attn_comp_stage`, and `ds4_gpu_tensor_copy_f32_to_f16` which has a
   CUDA implementation). **The read side is the work**: CUDA casts the cache as
   `(float *)comp_cache->ptr` at ds4.c:14285 and 14298 and the
   compressed-attention kernels assume F32.
2. **T3 (acceptance)**: same prompt, F32-storage vs FP8-storage build, dump
   logits on both, require `max|Δ| == 0` over all 154,880 logits. Reuse
   `harness/f13_pair_analysis.py`.
3. **T4**: the engine's `context buffers` line must fall to the predicted value;
   re-fit and confirm the slope.
4. **T5**: a context size that currently fails to allocate must load and answer
   a needle-in-haystack retrieval correctly.

Note a review correction to fold in: the strict slope is ~76.77 KiB/token (not
72.84) once the 78-layer count and the absence of a current scale array are
accounted for, giving a ~159k context ceiling rather than ~168k. Re-derive it
yourself from the engine's own reporting rather than trusting either number.

### W2 — Decode-protected cache admission (Belady imitation)
**Bitwise-identical output. Low cost. +3–15% projected.**

Treat long-prefill accesses as scan traffic: admit them only to SLRU
probation, protect the pre-existing decode set, promote on reuse. Optionally
train a small layer/position/frequency admission classifier against Belady
decisions.

Bounded by a committed simulation, `results/glm52-gates/logs/g4a/belady-bound.txt`:
clairvoyant 88.0% at the 67 GiB arena versus 77.6% measured — ~10pp of
headroom ≈ 0.6 GB/token ≈ 56 ms. Prior art: TinyLFU, ARC, Parrot
(Belady-imitation).

Falsifier: replay the committed decode trace with byte-weighted hits; if
prefill-probation + tail seeding gains under 3pp over current SLRU, drop it.

### W3 — Run GEMV directly from expert-cache slots
**Bitwise-identical output. Medium cost. +5–20% projected.**

The engine currently copies every selected expert into compact gate/up/down
buffers — approximately **11.6 GB of DRAM traffic per token** (5.8 read + 5.8
write) against 273 GB/s, a **42 ms/token floor**. Replace with a pointer/slot
table consumed by GEMV, pin slots until a CUDA completion event, and compute
each expert as its three tensors arrive.

This is *not* the cross-layer prefetch that was measured net-neutral — that was
speculative cross-layer reads. This is same-layer producer/consumer with no
speculation.

Falsifier: hit-only, CUDA-event-timed 600-expert pass comparing compact-copy
GEMV against indirect-slot GEMV. Require ≥5% improvement in **completed** time
— not enqueue time. (An earlier result in this project was wrong precisely
because it measured enqueue time: 4.62 GB in 2.6 ms would be 1.78 TB/s.)

### W4 — Exact top-k replacement in the indexer
**Same selected IDs. Prefill and long-context lever.**

The current path uses a bitonic full sort. At 1M rows one query runs ~492
separate 4096-element sort networks (~78.6M compare-exchanges per layer);
cumulatively ~8×10¹⁴ across 1M queries × 21 layers. An exact radix/select +
gather, ordering the final 2048 with the existing tie rule, would cut this
sharply **without changing which rows are selected**.

### W5 — FP16 indexer key storage
**Likely bitwise-safe on the fast path. Halves indexer K traffic.**

The fast indexer kernel already converts every cached F32 K value to FP16
before WMMA. Storing the identically-rounded FP16 value earlier should
therefore preserve fast-path results bitwise — verify the rounding matches.
Cuts K traffic 336 → 168 TB at 1M. Does not apply to quality mode; A/B it.

### W6 — Widen indexer K-tile reuse
The default CUDA kernel stages one 128-row K tile and reuses it across **16**
query tokens (ds4_cuda.cu:26289). Executing two or four unchanged 16×16 WMMA
query tiles per block would cut K loads another 2–4× with identical MMA
operations.

### W7 — F13 append-resume state bug (fix)
**Root-caused with logit-level evidence; the fix is designed but unimplemented.**

Trigger: BPE re-merge at the generation junction causes a live-cache miss
(`live=5063 prompt=5066 common=5045`), the server falls back to a *shorter*
disk checkpoint and extends by a long suffix. At the same final position for
the same 5066-token prompt this produces **max|Δ| = 5.911 across 154,880
logits (18.2% of range), mean 1.19, different argmax** with healthy margins —
state, not numerics (FP reassociation is ~1e-3; the engine is deterministic at
max|Δ| = 0 cross-process).

Fix (option B): on junction mismatch, truncate the live session to the common
prefix and prefill forward, instead of loading a shorter disk checkpoint.
Acceptance: re-run `harness/f13_regime_logit.sh` then `harness/f13_pair_analysis.py`
and require max|Δ| to collapse to numerics scale (<1e-2) with matching argmax.
The strict resume guard is the current mitigation and stays until this passes.

### W8 — DSA-aware exact cKV offload to NVMe
**The exact route to large context — no quantization, no fidelity gate.**

Keep the all-row indexer resident. After each full-indexer top-k, fetch the
2048 selected cKV rows from NVMe, stage them contiguously, and run attention
against staged IDs 0…2047, reusing the selection across the following
non-indexer layers in that group. Store rows grouped by the 21 selection epochs
so one fetch serves all layers reusing that selection.

Arithmetic: exact F32 cKV on disk 167.4 GiB (773 GB free); resident indexer +
misc 19.14 GiB (14.14 with F16 indexer) leaving 63.8–68.8 GiB for experts.
NVMe payload 0.368 GB/token useful, ~0.495 GB/token with 4 KiB alignment and 21
grouped reads → +34–46 ms on a 429 ms baseline ≈ **2.10–2.16 tok/s**, against
2.33 today at 32k. Arbitrary-row retrieval is preserved exactly.

### W9 — Sub-FP8 KV (FP4 and below)
**Open. The current evidence is discouraging but the analysis has known flaws.**

FP4 on the non-RoPE latent is the tier that makes very large contexts fit.
State of knowledge:
- Published DeepSeek-MLA-with-Q4 is +0.19 PPL (+3.0%) ≈ 0.0296 nat/token,
  ~3× the 0.01 budget. That datum is from a 16B model with 16 separate K/V
  heads — **not** GLM's single 576-wide latent, so its transferability is
  unestablished in both directions.
- A synthetic rotation study (`harness/test_fp4_rotation_snr.py`) measured
  2.74 dB of Hadamard gain on outlier-bearing data. **That test has two known
  defects**: its MSE→NLL inference is unsound (MSE is already second order;
  the correct objective is query-weighted `δkᵀE[qqᵀ]δk`, as SAW-INT4 uses), and
  its 128-wide geometry was taken from the *indexer* path (ds4.c:3288), not the
  512-wide attention latent. Both are fixable — a corrected study may give a
  very different answer.
- Nobody has published 4-bit results on MLA compressed latents at all. This is
  genuinely open ground, not a solved-and-negative question.

Stack worth evaluating: Walsh-Hadamard or randomized block-diagonal rotation
(UltraQuant 2606.20474, SAW-INT4 2604.19157, RotateKV), learned per-channel
linear correction (KVLinC 2510.05373, ~one element-wise multiply at inference),
higher-precision first/last layers, attention-sink preservation (KVSink
2508.04257), Kitty 2-bit with dynamic channel-wise precision boost (2511.18643),
NVFP4-style E2M1 with UE8M0 scales. Note KVLinC's own ablation shows
rotation+correction combined delivers ~64–65% of the sum of isolated gains, so
model overlap, not addition.

**The cheapest decisive experiment is a real KV capture**, which does not exist
yet. Specification: ~8192 rows per layer from 8 stratified layers
(suggested {0,2,10,26,42,58,74,77} — covering early/middle/late, six
full-indexer layers, and the final non-indexer layer). Hook immediately after
`ds4_gpu_glm_store_compact_kv_tensor` (~ds4.c:44418) in
`glm_graph_forward_indexed_tokens`, and after both the fused store (~45817) and
normal store (~45963) in `glm_graph_forward_token`. Also capture `batch_qk_low`
(~44639) for 128 stratified positions per layer to get real queries, plus
`last_indexer_selected`. Dump ~264 MiB minimum, ~456 MiB with pre-store and
outputs. Then measure per-channel RMS/max/kurtosis, block amax/RMS, covariance
spectrum, outlier-channel stability, plain vs rotated FP4 error, query-weighted
score error `δcᵀM_lδc / 320`, attention-output error, and top-k
Jaccard/margin-crossings.

### W10 — Sequence-dimension compression for GLM
**Open research. The obstacles are specific and worth attacking directly.**

DSV4 reaches ~6.9 KiB/token via SWA-128 + CSA 4:1 + HCA 128:1; GLM stores
191.34 KiB/token. Direct reuse of DSV4's modules is not possible as-is — they
are learned components requiring `compressor_kv`, `compressor_gate`,
`compressor_ape`, `compressor_norm` tensors (ds4.c:4981), with a second learned
indexer compressor on 4:1 layers, and at runtime they perform learned
projection, gating, softmax pooling, ring-state update, APE, RMS norm and RoPE
(ds4.c:12381). The engine enables compression ratios only for the DeepSeek-4
family (ds4.c:1059) and zeroes GLM's ratio table (ds4.c:5644).

A refit against frozen GLM is a defined problem: ~0.9B new parameters
(38 CSA + 38 HCA over 78 layers), ~15 GB optimizer state, ~0.89 TiB of captured
hidden states per 1M tokens, and one teacher pass at GLM's current prefill rate
is 8.7–12.1 hours. CSKV (arXiv 2409.10593) is the closest published recipe —
SVD initialization plus layerwise MSE, 256 calibration samples, ~90 minutes on
one A100 for a 7B model at ~80% compression. Scaling that recipe to stateful
128:1 compression inside a 753B MoE with a separately-trained sparse indexer is
unexplored. **Note that W2/W3/W4/W5 all raise GLM's prefill rate, which
directly reduces the teacher-pass cost for this item** — they compound.

The design tension to solve: GLM's indexer scores every row and selects
top-2048, so any past token is retrievable. Hierarchical summaries remove
per-row addressability. Three framings exist (keep indexer + summarized cKV;
compress both; keep exact cKV in tiered storage = W8), and a co-designed
answer — where the indexer is refit alongside the compressor, as DSV4's was —
has not been attempted. Aggregate selection rarity is not a safe compression
criterion for retrieval, because a row selected for the first time may be the
one holding the answer; any scheme needs a retrieval-preserving argument, not
just an average-case one.

### W11 — Raise the context window
`-c 32768` is a configuration choice. Re-derive the ceiling from the engine's
own `context buffers` reporting after W1 lands, then raise it with T5's
needle-in-haystack test as the gate. Also outstanding: the production systemd
unit shows `failed` and needs a root `systemctl start
deepseek-v4-flash-llamacpp.service` (the endpoint itself was restored manually).

---

## 5. Fidelity gate

Paired teacher-forced NLL via `score_official` against the committed
`glm52-openrouter-100` suite, same case subset in every arm:
- **mean ΔNLL ≤ 0.01 nat/token**
- **top-1 agreement loss ≤ 0.5 pp**
- report the paired 95% CI; if the estimate lands near the threshold, raise the
  case count (16 has been used; 100 are available)

This gate has teeth: it rejected expert-skipping at ΔNLL +0.0799 (8× over,
CI excluding zero) *after* that change had passed coherence, UTF-8 and
repetition checks with **identical** first-token agreement. Coherence is a
liveness check, not fidelity.

For long-context work, add **retrieval** testing (needle-in-haystack). Several
KV-compression methods preserve perplexity while degrading middle-of-context
recall, and a long window that cannot recall its middle is not useful.

Where a change is information-preserving (W1 strict, W2, W3, W4, W5), prefer
**bit-identity** (`max|Δ| == 0`) over any statistical gate.

---

## 6. Measurement traps already hit

Each of these produced a wrong or void result in this project.

1. **Prove the arms differ.** Two separate A/Bs ran with both arms in the same
   configuration: a flush-fix test where the storm only occurs on long prompts
   (both arms recorded 0 flushes and byte-identical cache counters), and a
   prefill-chunk test where the env var was ignored (both logged
   `prefill_chunk=4096` and identical byte totals to one decimal). Assert the
   independent variable changed, from the engine's own reporting, before
   measuring anything.
2. **Isolate the phase the lever acts on.** A decode lever was measured with
   5047-token-prompt requests that were ~95% prefill. Decode is isolated
   arithmetically as `64/(t65 − t1)` against one warm server.
3. **Completion time, not enqueue time.** 4.62 GB in 2.6 ms implies 1.78 TB/s.
4. **ABBA order and repeats.** Arm order and machine state drift; a single pair
   proves little. One contrast is not a noise estimate.
5. **Grep for failure strings, not just success strings.** An entire
   investigation stalled on "the KV store never fires" when it was firing and
   failing with `No space left on device` on a 100%-full disk.
6. **Watch disk space.** The above cost days.
7. **Verify fixture edits took effect.** A no-op text replacement was caught
   only because two "different" arms produced impossibly identical hashes.
8. **Deterministic verification over inference.** Unique-expert histograms and
   flush counts read from the trace beat conclusions drawn from timings.

---

## 7. Artifacts

- `results/glm52-gates/CLAIMS.md` — claim-by-claim status with every retraction
- `results/glm52-gates/STATUS.md` — headline metrics against the DSV4 bar
- `results/glm52-gates/loadprof-2026-07-25.json` — the full measurement ledger
- `results/glm52-gates/sol-reviews/` — adversarial reviews, verbatim
- `results/glm52-gates/harness/` — every harness and test, including
  `test_fp8_kv_codec.c`, `test_fp8_kv_kernel.cu`, `test_kv_budget.py`,
  `test_fp4_rotation_snr.py`, `decode_ab.sh`, `slru_ab.sh`, `flushfix_ab.sh`,
  `keepn_nll.sh`, `f13_regime_logit.sh`, `f13_pair_analysis.py`,
  `glm_safe_run.sh`, and the engine patches `ds4-topk-skip-load.patch`,
  `ds4-model-gen-flush-fix.patch`, `ds4-model-gen-close-reopen.patch`
- `results/glm52-gates/logs/g4a/belady-bound.txt` — the cache oracle bound
- `docs/bigctx-plan-fable-2026-07-16.md` — DSV4's cache cost model, the source
  of the 6.9 KiB/token figure

Settled results worth not re-deriving: SLRU is +6.2% over plain LRU
(ABBA n=2/arm, identical access-stream digests across all four arms, so the
difference is attributable to the eviction policy); the model-generation flush
fix eliminates 452 flushes/run but bought no measurable speed on the fixture
tested and is kept as a correctness fix; expert skipping at keep-6 gives ~11–13%
decode but fails the fidelity gate.
