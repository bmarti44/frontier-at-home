# GLM-5.2 on one DGX Spark — where the numbers actually landed

Answering the standing question directly: *are GLM-5.2's performance metrics
similar to or better than DeepSeek-V4-Flash's?*

**One of four metrics meets the bar. The other three are short by roughly an
order of magnitude, and the reason is arithmetic, not tuning.**

| metric | DSV4 bar | GLM-5.2 measured | verdict |
|---|---|---|---|
| warm TTFT | <2 s | **1.755 s** | **MET** (with a config dependency, below) |
| cold TTFT | **~19 s at ~19k tokens** | 147–165 s at 5047 tokens; ~19k would take **10–14 min** | ~35–45x short |
| decode | 18.4 tok/s | 2.33 tok/s | ~8x short |
| prefill | 467 tok/s | ~23–32 tok/s | ~15–20x short |
| context | **1,000,000** | 32768 configured; **~207k is the ceiling** at the current cache size | see below |

## Context: 1M is impossible here, but 32768 is leaving a lot on the table

Two-point fit of the engine's own `context buffers` line (5653.35 MiB at
ctx=8192, 10245.44 MiB at ctx=32768) gives **191.3 KiB/token + 4.1 GiB fixed**.

- **1M context needs 186.5 GiB of KV alone** against 119.7 GiB of unified
  memory — impossible on one Spark even with zero expert cache and no model
  resident. This is a hard structural gap against DSV4's 1M, not a tuning one.
- But the current **32768 is a config choice, not a limit**: ctx=131072 costs
  only 27.9 GiB of KV and still leaves ~85 GiB for the expert cache and model.
  Ceilings: ~207k tokens with the 72 GiB cache, ~383k at 40 GiB, ~558k at 8 GiB
  (the last would wreck decode — the cache is what holds decode above the
  1.83 tok/s all-miss floor).
- **Raising the context window is the cheapest untried improvement available.**

## Why decode cannot be fixed in software

GLM-5.2 has **256 experts per layer**, 8 used, across 75 routed layers, at
9.28 MiB per expert. One decode token therefore requests
**74.2 MiB x 75 = 5.8 GB** of expert weights. The NVMe delivers 10.7 GB/s
measured, so an all-miss step costs ~0.55 s = **1.83 tok/s**. Measured decode
is 2.29 tok/s, i.e. already *above* the all-miss floor thanks to the 72 GB
host expert cache.

Closing an 8x gap would require moving ~5 GB less per token. The levers found
this session are worth 10–13%, not 8x. **A model whose weights fit in memory is
the only path to DSV4-class decode**; a faster streaming path is not.

The same arithmetic kills prefill: a 2048-token prefill batch already touches
~99% of all 256 experts in every routed layer (~170 GiB per sweep, 339 GiB
measured for one 5047-token prompt over two sweeps). No caching or chunking
strategy helps when the working set is the whole model.

## What shipped

- **Persistent host expert cache** (72 GB, pinned arena, SLRU, parallel fetch):
  warmup 508 s -> 51 s; hit rate 77.6% on short-prompt decode.
- **Expert-cache flush fix** — the load-generation counter incremented before
  the "same map" early-return, so every prefill span install wiped the whole
  cache (452 flushes in one run, hit rate 9.3%). Fixed: 452 -> 0.
  **Correctness only — it buys no measurable speed** (see below).
- **Warm TTFT path** via disk-KV checkpointing: 1.755 s.

## What was measured and rejected

| lever | speed | why not shipped |
|---|---|---|
| keep-6 expert skip | +13.0% free-running / -11.3% teacher-forced | **fails fidelity**: paired ΔNLL +0.0799, 95% CI [+0.0135, +0.1463], 8x the 0.01 threshold, CI excludes zero |
| keep-7 expert skip | +10.0% free-running / -2.0% teacher-forced | fidelity inconclusive (ΔNLL +0.0169, CI spans zero) **and** speed benefit unestablished |
| MTP speculation | ~+10% | output not identical; single unrepeated observation |
| cross-layer prefetch | net-neutral | no gain; "46% fewer misses" was a mislabelled counter |
| batchall prefill | — | **fidelity hazard**: repetition loops and invalid UTF-8 on short prompts |
| prefill chunking | none | lever does not exist: GLM's indexed prefill clamps to the 2048 indexer boundary via a compile-time constant |
| flush fix (as a speed lever) | none | cold 158->163 s, second prefill 147->144 s, hit rate 9.3->12.6% |

## Open correctness issue

**Append-resume divergence (F13)** is an open **state bug**, root-caused this
session with logit-level evidence: at the same final position for the same
5066-token prompt, a resume from a disk checkpoint differs from a fresh
evaluation by **max|Δ| = 5.911 across 154,880 logits (18.2% of logit range),
mean 1.19, different argmax with healthy margins**. FP reassociation lands near
1e-3 and the engine is deterministic (max|Δ| = 0 cross-process), so this is
state, not numerics.
Trigger: BPE re-merge at the generation junction causes a live-cache miss
(`live=5063 prompt=5066 common=5045`), the server falls back to a shorter disk
checkpoint and extends by a long suffix.
**Mitigation in place:** the strict resume guard remains default.
**Fix not implemented:** truncate the live session to the common prefix instead
of loading a shorter checkpoint, then require the logit delta to collapse to
numerics scale.

## Caveats that belong on every number here

- **Warm TTFT 1.755 s requires the disk-KV config.** An identical replay against
  the same live server *without* `--kv-disk-dir` re-prefilled in **147 s**.
- Decode figures are one fixture; the trajectory-controlled figure (+11.3%)
  comes from teacher-forced scoring where both arms evaluate identical tokens.
- The fidelity gate ran 16 of 100 available cases.
- Cold TTFT varies 147–165 s run to run; differences under ~5% are noise.

Full claim-by-claim status, including every retraction: `CLAIMS.md`.
Raw evidence: `loadprof-2026-07-25.json`. sol's adversarial reviews:
`sol-reviews/`.
